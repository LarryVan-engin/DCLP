"""
********************************************************************************************************************
General Information
********************************************************************************************************************
Project:       Traffic Violation Detection & License Plate Recognition
File:          api_main.py
Description:   FastAPI backend for real-time traffic violation detection using YOLO and ByteTrack tracking

Author:        Larry Van 
Email:         vanphongtruc1808@@gmail.com
Created:       2025-11-05
Last Update:   2025-11-27
Version:       1.5

Python:        3.10+
Dependencies:  Main runtime dependencies
               - fastapi, uvicorn
               - opencv-python, numpy
               - pandas
               - ultralytics
               - easyocr

Copyright:     (c) 2025 IOE INNOVATION Team
License:       Proprietary

Notes:         Backend service for AI-based traffic violation detection
               - Only Project Leader has permission to modify this file
               - Coordinates video processing, AI models, tracking and WebSocket streaming
               - Handles application lifecycle and configuration
********************************************************************************************************************
"""

#######################################################################################################################
# Imports
#######################################################################################################################
# Standard library imports
import asyncio
import base64
from datetime import datetime
import os
import uuid
from types import SimpleNamespace
from typing import Dict, Any, Generator, List, Optional, Tuple

# Third-party imports
import cv2
import easyocr
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

# Local imports
from module_utils import get_car, read_license_plate, process_license_plate

#######################################################################################################################
# Constants and Configuration
#######################################################################################################################
# Application metadata
APP_NAME: str = "Traffic Violation Detection Service"
APP_VERSION: str = "1.0.0"
APP_DESCRIPTION: str = "Real-time traffic violation detection using YOLO and ByteTrack"

# Paths
DB_PATH: str = "database/owners_sample.csv"
VIOLATION_DIR: str = "violations"
UPLOADS_DIR: str = "uploads"
INDEX_HTML_PATH: str = r"D:\VSCode\DCLP\web_application\index.html"

# Video / frame size (display)
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720

#######################################################################################################################
# Global Variables
#######################################################################################################################
# Display frame (default black screen)
display_frame: np.ndarray = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

# FastAPI application
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Video capture state
VIDEO_PATH: Optional[str] = None
cap: Optional[cv2.VideoCapture] = None
pause_processing: bool = False

# Zones configuration
zones: Dict[str, List[Any]] = {"lines": [], "polygons": []}

# Runtime data
violations: List[Dict[str, Any]] = []
current_vehicles: Dict[int, Dict[str, Any]] = {}
prev_positions: Dict[int, Tuple[int, int]] = {}
prev_inside: Dict[int, bool] = {}  # {track_id: bool} - có đang trong bất kỳ polygon nào không

# Shared state for UI
shared_data: Dict[str, Any] = {
    "stats": {cls_name: 0 for cls_name in ["car", "motorcycle", "bus", "truck"]},
    "lights": {"left": "red", "straight": "green"},
}

# If to use traffic light status for violation detection
use_traffic_light: bool = True

#######################################################################################################################
# Model and Tracker Initialization
#######################################################################################################################
# Ensure required directories exist
os.makedirs(VIOLATION_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


class ByteTrackerWrapper:
    """
    Wrapper cho ByteTrack để dùng giống DeepSort cũ.

    Input:
        dets: list [[x1, y1, x2, y2], conf, cls_id]

    Output:
        list track objects với:
            - track_id
            - to_ltrb()
            - cls_id
            - is_confirmed()
    """

    def __init__(
        self,
        frame_rate: int = 25,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_box_area: int = 10,
        aspect_ratio_thresh: float = 3.0,
    ) -> None:
        args = SimpleNamespace(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            min_box_area=min_box_area,
            aspect_ratio_thresh=aspect_ratio_thresh,
        )
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def update_tracks(self, dets, frame):
        import numpy as np

        H, W = frame.shape[:2]

        # ===== TRƯỜNG HỢP KHÔNG CÓ DETECTION =====
        if not dets:
            # GỌI update BẰNG THAM SỐ POSITIONAL
            _ = self.tracker.update(
                np.empty((0, 5), dtype=np.float32),  # output_results
            )
            return []

        # ===== CHUẨN BỊ DỮ LIỆU CHO BYTETRACK =====
        boxes = np.array([d[0] for d in dets], dtype=np.float32)  # [N, 4] xyxy
        scores = np.array([float(d[1]) for d in dets], dtype=np.float32)  # [N]
        cls_ids = np.array([int(d[2]) for d in dets], dtype=np.int32)  # [N]

        # [x1, y1, x2, y2, score]
        output_results = np.concatenate(
            [boxes, scores.reshape(-1, 1)],
            axis=1,
        )

        # GỌI update BẰNG THAM SỐ POSITIONAL
        online_targets = self.tracker.update(output_results)

        # ===== WRAP TRACKS =====
        class _Track:
            def __init__(self, track_id, ltrb, cls_id):
                self.track_id = int(track_id)
                self._ltrb = ltrb
                self.cls_id = int(cls_id)

            def to_ltrb(self):
                return self._ltrb

            @staticmethod
            def is_confirmed():
                return True

        wrapped_tracks = []
        # (Giả định số track online tương ứng với số dets — đủ dùng cho stats & hiển thị)
        for t, cls_id in zip(online_targets, cls_ids):
            tlwh = t.tlwh  # [x, y, w, h]
            x1, y1, w, h = tlwh[0], tlwh[1], tlwh[2], tlwh[3]
            x2 = x1 + w
            y2 = y1 + h

            wrapped_tracks.append(
                _Track(
                    track_id=t.track_id,
                    ltrb=[x1, y1, x2, y2],
                    cls_id=cls_id,
                )
            )

        return wrapped_tracks



print("[INFO] Đang tải mô hình...")
coco_model = YOLO("yolo12n.pt")
plate_model = YOLO(
    r"D:\VSCode\DCLP\main_code\runs\detect\model_detect_license_plate.pt"
)
traffic_light_model = YOLO(
    r"D:\VSCode\DCLP\main_code\runs\detect\traffic_light\weights\best.pt"
)
reader = easyocr.Reader(["en"], gpu=True)

# Vehicle classes theo COCO
VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Global tracker instance
vehicle_tracker = ByteTrackerWrapper(frame_rate=25)

#######################################################################################################################
# Database and Mapping
#######################################################################################################################
try:
    owner_db = pd.read_csv(DB_PATH)
    owner_db = owner_db.drop_duplicates(subset="plate")
    vehicle_info: Dict[str, Dict[str, Any]] = owner_db.set_index("plate").to_dict("index")
    print(f"[INFO] Loaded {len(vehicle_info)} vehicle records.")
except Exception as exc:  # pragma: no cover - runtime-only
    print(f"[LỖI DB] {exc}")
    vehicle_info = {}

# Mapping detected type sang class_vehicle trong DB (có thể chỉnh nếu cần)
TYPE_MAPPING: Dict[str, List[str]] = {
    "car": ["Ô tô con", "Xe bán tải"],
    "motorcycle": ["Xe máy"],
    "bus": ["Ô tô khách"],
    "truck": ["Ô tô tải", "Xe chuyên dụng"],
}

#######################################################################################################################
# Utility Functions
#######################################################################################################################
def check_line_crossing(
    prev: Tuple[int, int],
    curr: Tuple[int, int],
    line: List[Tuple[int, int]],
) -> bool:
    """
    Kiểm tra xem đoạn (prev -> curr) có cắt đoạn line hay không
    (dùng CCW orientation).
    """

    def ccw(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (
        ccw(line[0], prev, line[1]) != ccw(line[1], prev, line[0])
        and ccw(line[0], line[1], prev) != ccw(line[0], line[1], curr)
    )


def crop_and_encode(img: np.ndarray, bbox: List[float]) -> Optional[str]:
    """
    Crop ảnh theo bbox và encode base64 JPEG.

    Args:
        img: Frame gốc
        bbox: [x1, y1, x2, y2]

    Returns:
        Chuỗi base64 hoặc None nếu crop invalid
    """
    x1, y1, x2, y2 = map(int, bbox)
    if x1 >= x2 or y1 >= y2:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    success, buf = cv2.imencode(".jpg", crop)
    if not success:
        return None

    return base64.b64encode(buf).decode()


#######################################################################################################################
# Video Frame Generator (MJPEG + AI Pipeline)
#######################################################################################################################
def gen_frames() -> Generator[bytes, None, None]:
    """
    Generator trả về từng frame JPEG (multipart/x-mixed-replace)
    kèm theo toàn bộ pipeline AI:
        - Traffic light detection
        - Vehicle detection + ByteTrack
        - Violation detection (line + polygon)
        - License plate detection + OCR
    """
    global violations
    global current_vehicles
    global prev_positions
    global shared_data
    global pause_processing
    global display_frame

    try:
        while True:
            # Nếu chưa có video thì trả về khung đen
            if cap is None or not cap.isOpened():
                blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "Chua co video - vui long upload",
                    (250, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                ret, buffer = cv2.imencode(".jpg", blank)
                if ret:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )
                cv2.waitKey(1)
                continue

            # Tạm dừng xử lý nhưng vẫn gửi frame hiện tại
            if pause_processing:
                if display_frame is not None:
                    ret, buffer = cv2.imencode(".jpg", display_frame)
                    if ret:
                        yield (
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + buffer.tobytes()
                            + b"\r\n"
                        )
                asyncio.sleep(0.1)  # Reduce CPU load
                continue

            # Đọc frame từ video
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            display_frame = frame.copy()

            # === ĐÈN GIAO THÔNG ===
            light_results = traffic_light_model(frame)[0]
            for box in light_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                cls_name = light_results.names[cls_id]
                center_x = (x1 + x2) // 2
                key = "left" if center_x < frame.shape[1] // 2 else "straight"
                shared_data["lights"][key] = cls_name

                color = {
                    "red": (0, 0, 255),
                    "yellow": (0, 255, 255),
                    "green": (0, 255, 0),
                }.get(cls_name, (255, 255, 255))

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display_frame,
                    cls_name,
                    (x1, y2 + 25),
                    0,
                    0.7,
                    color,
                    2,
                )

            # === XE + TRACKING ===
            results = coco_model(frame)[0]

            all_dets: List[Tuple[List[float], float, int]] = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASSES:
                    all_dets.append(([x1, y1, x2, y2], conf, cls_id))

            track_data: List[List[Any]] = []
            current_vehicles.clear()
            shared_data["stats"] = {
                cls_name: 0 for cls_name in VEHICLE_CLASSES.values()
            }

            tracks = vehicle_tracker.update_tracks(all_dets, frame=frame)
            confirmed = [t for t in tracks if t.is_confirmed()]

            for track in confirmed:
                ltrb = track.to_ltrb()
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, ltrb)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cls_name = VEHICLE_CLASSES.get(getattr(track, "cls_id", -1), "car")
                shared_data["stats"][cls_name] = (
                    shared_data["stats"].get(cls_name, 0) + 1
                )

                # Vẽ bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display_frame,
                    f"{cls_name[:3]}-{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                # Crop xe
                img_b64 = crop_and_encode(frame, ltrb)
                current_vehicles[track_id] = {
                    "img": (
                        f"data:image/jpeg;base64,{img_b64}"
                        if img_b64
                        else ""
                    ),
                    "plate": "Đang đọc...",
                    "type": cls_name,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }

                # === VI PHẠM - LINE ===
                if track_id in prev_positions:
                    prev = prev_positions[track_id]

                    # Check lines
                    for line in zones["lines"]:
                        if check_line_crossing(prev, center, line):
                            line_x = (line[0][0] + line[1][0]) // 2
                            light_key = (
                                "left"
                                if line_x < frame.shape[1] // 2
                                else "straight"
                            )

                            should_violate = True
                            if use_traffic_light:
                                should_violate = (
                                    light_key == "straight"
                                    and shared_data["lights"].get(light_key)
                                    == "red"
                                )

                            if should_violate:
                                violations.append(
                                    {
                                        "id": track_id,
                                        "plate": "Đang đọc...",
                                        "type": (
                                            "VƯỢT ĐÈN ĐỎ (đi thẳng)"
                                            if use_traffic_light
                                            else "VƯỢT LINE"
                                        ),
                                        "time": datetime.now().strftime(
                                            "%H:%M:%S"
                                        ),
                                        "img": current_vehicles[track_id]["img"],
                                        "plate_img": "",
                                        "owner": "",
                                        "phone": "",
                                        "class_vehicle": "",
                                        "province": "",
                                        "registration_date": "",
                                        "id_card": "",
                                        "match_type": False,
                                    }
                                )

                # === VI PHẠM - POLYGON ===
                inside_any = False
                violated_poly = None
                for poly in zones["polygons"]:
                    if len(poly) < 3:
                        continue
                    pts = np.array(poly, np.int32)
                    if cv2.pointPolygonTest(pts, center, False) >= 0:
                        inside_any = True
                        violated_poly = poly  # Lưu poly vi phạm để tính light_key
                        break

                if inside_any and not prev_inside.get(track_id, False):
                    should_violate = True
                    if use_traffic_light and violated_poly:
                        # Tính light_key dựa trên trung bình x của poly
                        avg_x = np.mean([p[0] for p in violated_poly])
                        light_key = (
                            "left"
                            if avg_x < frame.shape[1] // 2
                            else "straight"
                        )
                        should_violate = (
                            shared_data["lights"].get(light_key) == "red"
                        )

                    if should_violate:
                        violations.append(
                            {
                                "id": track_id,
                                "plate": "Đang đọc...",
                                "type": (
                                    "VÀO VÙNG CẤM"
                                    if not use_traffic_light
                                    else f"VÀO VÙNG CẤM (đèn đỏ {light_key})"
                                ),
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "img": current_vehicles[track_id]["img"],
                                "plate_img": "",
                                "owner": "",
                                "phone": "",
                                "class_vehicle": "",
                                "province": "",
                                "registration_date": "",
                                "id_card": "",
                                "match_type": False,
                            }
                        )

                prev_inside[track_id] = inside_any
                prev_positions[track_id] = center

                track_data.append(
                    [x1, y1, x2, y2, track_id, cls_name]
                )

            # === BIỂN SỐ ===
            lps = plate_model(frame)[0]
            for box in lps.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car = get_car(box.xyxy[0].tolist(), track_data)
                if len(car) != 6 or car[4] == -1:
                    continue

                _, _, _, _, vid, _ = car
                plate_crop = process_license_plate(frame, x1, y1, x2, y2)
                text, _ = read_license_plate(plate_crop)

                if text:
                    img_b64 = crop_and_encode(frame, [x1, y1, x2, y2])
                    plate_img = (
                        f"data:image/jpeg;base64,{img_b64}"
                        if img_b64
                        else ""
                    )

                    if vid in current_vehicles:
                        current_vehicles[vid]["plate"] = text

                    # Tra DB dựa trên plate và check loại xe
                    info = vehicle_info.get(text, {})
                    match_type = False
                    if info and cls_name in TYPE_MAPPING:
                        if info.get("class_vehicle") in TYPE_MAPPING[cls_name]:
                            match_type = True

                    for v in violations:
                        if v["id"] == vid and v["plate"] == "Đang đọc...":
                            v["plate"] = text
                            v["plate_img"] = plate_img
                            v["owner"] = info.get(
                                "owner",
                                "Không tìm thấy",
                            )
                            v["phone"] = info.get("phone", "")
                            v["class_vehicle"] = info.get(
                                "class_vehicle",
                                "",
                            )
                            v["province"] = info.get("province", "")
                            v["registration_date"] = info.get(
                                "registration_date",
                                "",
                            )
                            v["id_card"] = info.get("id_card", "")
                            v["match_type"] = match_type

                    cv2.putText(
                        display_frame,
                        text,
                        (x1, y2 + 25),
                        0,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            # === VẼ VÙNG ===
            for line in zones["lines"]:
                cv2.line(
                    display_frame,
                    tuple(map(int, line[0])),
                    tuple(map(int, line[1])),
                    (0, 0, 255),
                    3,
                )

            for poly in zones["polygons"]:
                pts = np.array(poly, np.int32)
                cv2.polylines(display_frame, [pts], True, (255, 255, 0), 3)

            # === ENCODE ===
            ret, buffer = cv2.imencode(".jpg", display_frame)
            if ret:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

    except GeneratorExit:
        print("[INFO] Client closed stream (browser reload).")
    except Exception as exc:  # pragma: no cover - runtime-only
        print(f"[LỖI STREAM] {exc}")


#######################################################################################################################
# HTTP API Endpoints
#######################################################################################################################
@app.get("/")
async def index() -> HTMLResponse:
    """Serve main HTML UI."""
    with open(INDEX_HTML_PATH, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/stream")
async def stream() -> StreamingResponse:
    """MJPEG stream endpoint."""
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/pause")
async def set_pause(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bật/tắt tạm dừng xử lý nhưng vẫn giữ frame hiện tại.
    """
    global pause_processing
    pause_processing = bool(data.get("pause", False))
    return {"status": "ok", "pause": pause_processing}


@app.post("/api/zones")
async def set_zones(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cập nhật các line và polygon dùng để check vi phạm.
    """
    global zones

    lines = data.get("lines", [])
    polygons = data.get("polygons", [])

    zones["lines"] = lines
    zones["polygons"] = polygons

    print(f"[ZONES] Lines: {len(lines)}, Polygons: {len(polygons)}")
    return {"status": "ok", "message": "Zones updated successfully"}


@app.post("/api/set_option")
async def set_option(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bật/tắt chế độ phạt theo đèn giao thông.
    """
    global use_traffic_light
    use_traffic_light = bool(data.get("use_traffic_light", True))
    return {"status": "ok", "use_traffic_light": use_traffic_light}


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload video mới và reset toàn bộ state tracking/vi phạm.
    """
    global VIDEO_PATH, cap
    global violations, current_vehicles, prev_positions, prev_inside
    global vehicle_tracker

    try:
        # Generate tên file unique
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "mp4"
        new_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4()}.{file_ext}")

        # Lưu file
        with open(new_path, "wb") as f:
            f.write(await file.read())

        # Reset cap với video mới
        if cap is not None:
            cap.release()

        VIDEO_PATH = new_path
        cap = cv2.VideoCapture(VIDEO_PATH)

        # Xóa video cũ nếu cần (tùy chọn, để tránh đầy disk)
        # for old_file in os.listdir(UPLOADS_DIR):
        #     if old_file != os.path.basename(new_path):
        #         os.remove(os.path.join(UPLOADS_DIR, old_file))

        violations.clear()
        current_vehicles.clear()
        prev_inside.clear()
        prev_positions.clear()

        # Reset ByteTrack tracker
        vehicle_tracker = ByteTrackerWrapper(frame_rate=25)

        return {
            "status": "ok",
            "message": f"Video uploaded and processing: {file.filename}",
        }
    except Exception as exc:  # pragma: no cover - runtime-only
        return {"status": "error", "message": str(exc)}


#######################################################################################################################
# WebSocket Endpoint
#######################################################################################################################
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket gửi realtime:
        - danh sách xe hiện tại
        - danh sách vi phạm (10 gần nhất)
        - thống kê số lượng xe
        - trạng thái đèn
    """
    await websocket.accept()
    try:
        while True:
            data = {
                "vehicles": current_vehicles,
                "violations": violations[-10:],
                "stats": shared_data["stats"],
                "lights": shared_data["lights"],
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


#######################################################################################################################
# Main Execution
#######################################################################################################################
if __name__ == "__main__":
    import uvicorn

    print("[INFO] Server: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

# End of File

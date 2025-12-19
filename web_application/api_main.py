"""
********************************************************************************************************************
General Information
********************************************************************************************************************
Project:       Traffic Violation Detection & License Plate Recognition
File:          api_main.py
Description:   FastAPI backend for real-time traffic violation detection using YOLO and ByteTrack tracking
********************************************************************************************************************
"""

#######################################################################################################################
# Imports
#######################################################################################################################
import asyncio
import base64
from datetime import datetime
import os
import time
import uuid
import traceback
from types import SimpleNamespace
from typing import Dict, Any, Generator, List, Optional, Tuple

import cv2
import easyocr
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

# Local imports (giữ nguyên, nhưng không còn dùng get_car cho plate full-frame)
from module_utils import read_license_plate, process_license_plate

#######################################################################################################################
# Constants and Configuration
#######################################################################################################################
APP_NAME: str = "Traffic Violation Detection Service"
APP_VERSION: str = "1.5"
APP_DESCRIPTION: str = "Real-time traffic violation detection using YOLOv12 + ByteTrack"

DB_PATH: str = "database/owners_sample.csv"
VIOLATION_DIR: str = "violations"
UPLOADS_DIR: str = r"D:\VSCode\DCLP\web_application\uploads"
INDEX_HTML_PATH: str = r"D:\VSCode\DCLP\web_application\index.html"
TRACKER_YAML = r"D:\VSCode\DCLP\web_application\bytetrack.yaml"  # nên để file yaml ở đây

FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720

PLATE_PENDING = "Reading..."

#######################################################################################################################
# Runtime tuning (tối ưu)
#######################################################################################################################
USE_TRACKER_FILE = True

# Traffic light detect không cần mỗi frame
TRAFFIC_LIGHT_EVERY_N_FRAMES = 3
TRAFFIC_LIGHT_MIN_CONF = 0.5

# Plate detect theo ROI xe: chỉ OCR định kỳ để giảm load + ổn định
PLATE_EVERY_N_FRAMES = 5
PLATE_MIN_CONF = 0.35

# Violation de-dup: tránh spam mỗi frame
VIOLATION_COOLDOWN_SEC = 2.0  # cùng track+type cách nhau tối thiểu 2s mới ghi lại


# Debug info
print("[BOOT] __file__ =", __file__)
print("[BOOT] cwd =", os.getcwd())
print("[PATH] UPLOADS_DIR exists:", os.path.exists(UPLOADS_DIR), "writable:", os.access(UPLOADS_DIR, os.W_OK))
print("[PATH] INDEX_HTML_PATH exists:", os.path.exists(INDEX_HTML_PATH))
print("[PATH] plate_model exists:", os.path.exists(r"D:\VSCode\DCLP\main_code\runs\detect\model_detect_license_plate.pt"))
print("[PATH] traffic_light_model exists:", os.path.exists(r"D:\VSCode\DCLP\main_code\runs\detect\traffic_light\weights\best.pt"))
print("[PATH] TRACKER_YAML exists:", os.path.exists(TRACKER_YAML))
if USE_TRACKER_FILE:
    if not os.path.exists(TRACKER_YAML):
        print("[WARN] TRACKER_YAML not found:", TRACKER_YAML, "-> will fallback to default 'bytetrack.yaml'")

#######################################################################################################################
# Global Variables
#######################################################################################################################
display_frame: np.ndarray = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

app = FastAPI(title=APP_NAME, version=APP_VERSION, description=APP_DESCRIPTION)
app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs(VIOLATION_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

VIDEO_PATH: Optional[str] = None
cap: Optional[cv2.VideoCapture] = None
pause_processing: bool = False
use_traffic_light: bool = True

# Runtime state
violations: List[Dict[str, Any]] = []
current_vehicles: Dict[int, Dict[str, Any]] = {}
prev_positions: Dict[int, Tuple[int, int]] = {}
prev_inside: Dict[int, bool] = {}

# Cache biển số theo track để ổn định hiển thị
plate_cache: Dict[int, Dict[str, Any]] = {}  # track_id -> {..., last_frame, last_seen_ts}
frame_idx: int = 0

# De-dup vi phạm
violation_last_ts: Dict[Tuple[int, str], float] = {}  # (track_id, violation_type) -> last_ts

# Shared state for UI
shared_data: Dict[str, Any] = {
    "stats": {cls_name: 0 for cls_name in ["car", "motorcycle", "bus", "truck"]},
    "violations": [],
    "lights": {"left": "red", "straight": "green"},
    "fps": 0.0,
}

zones: Dict[str, Any] = {
    "lines": [],
    "polygons": [],
}

# OCR reader (nếu module_utils dùng easyocr reader riêng thì giữ, còn không có vẫn ok)
reader = easyocr.Reader(["en"], gpu=False)

#######################################################################################################################
# Helpers
#######################################################################################################################
def _norm_plate(s: str) -> str:
    if not s:
        return ""
    s = str(s).upper()
    return "".join(ch for ch in s if ch.isalnum())

def _extract_boxes(result):
    """
    Return (xyxy[N,4], conf[N], cls[N]) from Ultralytics or numpy output.
    """
    import numpy as np

    if result is None:
        return np.empty((0,4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32)

    b = getattr(result, "boxes", None)
    if b is None:
        return np.empty((0,4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32)

    # Ultralytics Boxes
    if hasattr(b, "xyxy") and hasattr(b, "conf") and hasattr(b, "cls"):
        xyxy_raw = b.xyxy
        conf_raw = b.conf
        cls_raw = b.cls

        xyxy = xyxy_raw.detach().cpu().numpy() if hasattr(xyxy_raw, "detach") else np.array(xyxy_raw)
        conf = conf_raw.detach().cpu().numpy() if hasattr(conf_raw, "detach") else np.array(conf_raw)
        cls_ = cls_raw.detach().cpu().numpy() if hasattr(cls_raw, "detach") else np.array(cls_raw)

        return (
            xyxy.astype(np.float32, copy=False),
            conf.astype(np.float32, copy=False),
            cls_.astype(np.int32, copy=False),
        )


    # Boxes.data (Nx6)
    if hasattr(b, "data"):
        data = b.data
        data = data.detach().cpu().numpy() if hasattr(data, "detach") else np.array(data)
        if data.ndim == 2 and data.shape[1] >= 6:
            return (
                data[:, 0:4].astype(np.float32, copy=False),
                data[:, 4].astype(np.float32, copy=False),
                data[:, 5].astype(np.int32, copy=False),
            )

    # b itself is ndarray (Nx6)
    if isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[1] >= 6:
        return (
            b[:, 0:4].astype(np.float32, copy=False),
            b[:, 4].astype(np.float32, copy=False),
            b[:, 5].astype(np.int32, copy=False),
        )

    return np.empty((0,4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32)

def crop_and_encode(img: np.ndarray, bbox: List[float]) -> Optional[str]:
    x1, y1, x2, y2 = map(int, bbox)
    if x1 >= x2 or y1 >= y2:
        return None
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", crop)
    if not ok:
        return None
    return base64.b64encode(buf).decode()

def check_line_crossing(prev: Tuple[int, int], curr: Tuple[int, int], line: List[Tuple[int, int]]) -> bool:
    """
    Segment intersection: prev->curr cắt line[0]->line[1]
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    p1, p2 = prev, curr
    q1, q2 = line[0], line[1]
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

def _now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _should_log_violation(track_id: int, vtype: str) -> bool:
    now = time.time()
    key = (track_id, vtype)
    last = violation_last_ts.get(key, 0.0)
    if now - last < VIOLATION_COOLDOWN_SEC:
        return False
    violation_last_ts[key] = now
    return True

#######################################################################################################################
# Model and Tracker Initialization
#######################################################################################################################
def get_vehicle_tracks(frame: np.ndarray):
    """
    Return arrays: xyxy (N,4), cls_ids (N,), track_ids (N,) or None
    """
    tracker_arg = TRACKER_YAML if (USE_TRACKER_FILE and os.path.exists(TRACKER_YAML)) else "bytetrack.yaml"

    res = coco_model.track(
        source=frame,
        persist=True,
        tracker=tracker_arg,
        conf=0.35,
        iou=0.5,
        verbose=False
    )[0]

    b = getattr(res, "boxes", None)
    if b is None or b.xyxy is None:
        return np.empty((0,4), np.float32), np.empty((0,), np.int32), None

    xyxy = b.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
    cls_ids = b.cls.detach().cpu().numpy().astype(np.int32, copy=False)

    track_ids = None
    if getattr(b, "id", None) is not None:
        track_ids = b.id.detach().cpu().numpy().astype(np.int32, copy=False)

    return xyxy, cls_ids, track_ids


print("[INFO] Đang tải mô hình...")
coco_model = YOLO("yolo12n.pt")
plate_model = YOLO(r"D:\VSCode\DCLP\main_code\runs\detect\model_detect_license_plate.pt")
traffic_light_model = YOLO(r"D:\VSCode\DCLP\main_code\runs\detect\traffic_light\weights\best.pt")

VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


#######################################################################################################################
# Database and Mapping
#######################################################################################################################
try:
    owner_db = pd.read_csv(DB_PATH)
    if "plate" in owner_db.columns:
        owner_db["plate"] = owner_db["plate"].astype(str).map(_norm_plate)
    owner_db = owner_db.drop_duplicates(subset="plate")
    vehicle_info: Dict[str, Dict[str, Any]] = owner_db.set_index("plate").to_dict("index")
    print(f"[INFO] Loaded {len(vehicle_info)} vehicle records.")
except Exception as exc:
    print(f"[WARN] Cannot load DB: {exc}")
    vehicle_info = {}

TYPE_MAPPING: Dict[str, List[str]] = {
    "car": ["Ô tô con", "Xe bán tải"],
    "motorcycle": ["Xe máy"],
    "bus": ["Ô tô khách"],
    "truck": ["Ô tô tải", "Xe chuyên dụng"],
}

#######################################################################################################################
# Video Frame Generator (MJPEG + AI Pipeline)
#######################################################################################################################
def gen_frames() -> Generator[bytes, None, None]:
    global violations, current_vehicles, prev_positions, prev_inside
    global shared_data, pause_processing, display_frame
    global plate_cache, frame_idx

    last_ts = time.time()
    cached_lights = {"left": shared_data["lights"]["left"], "straight": shared_data["lights"]["straight"]}

    try:
        while True:
            # Nếu chưa có video
            if cap is None or not cap.isOpened():
                blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(blank, "Chua co video - vui long upload", (250, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ok, buf = cv2.imencode(".jpg", blank)
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                time.sleep(0.05)
                continue

            # Pause: vẫn stream frame hiện tại
            if pause_processing:
                if display_frame is not None:
                    ok, buf = cv2.imencode(".jpg", display_frame)
                    if ok:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            display_frame = frame.copy()

            # tick
            frame_idx += 1

            # FPS
            now = time.time()
            dt = max(1e-6, now - last_ts)
            shared_data["fps"] = float(1.0 / dt)
            last_ts = now

            # =========================
            # Traffic light (periodic)
            # =========================
            if frame_idx % TRAFFIC_LIGHT_EVERY_N_FRAMES == 0:
                light_res = traffic_light_model(frame)[0]
                l_xyxy, l_conf, l_cls = _extract_boxes(light_res)

                # reset tạm (nếu frame này không thấy đèn, giữ cache cũ)
                lights_tmp = {"left": cached_lights["left"], "straight": cached_lights["straight"]}

                for i in range(len(l_xyxy)):
                    conf = float(l_conf[i])
                    if conf < TRAFFIC_LIGHT_MIN_CONF:
                        continue
                    x1, y1, x2, y2 = map(int, l_xyxy[i])
                    cls_id = int(l_cls[i])
                    cls_name = light_res.names.get(cls_id, str(cls_id)) if hasattr(light_res, "names") else str(cls_id)

                    center_x = (x1 + x2) // 2
                    key = "left" if center_x < frame.shape[1] // 2 else "straight"
                    lights_tmp[key] = cls_name

                    color = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}.get(cls_name, (255, 255, 255))
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, cls_name, (x1, y2 + 25), 0, 0.7, color, 2)

                cached_lights = lights_tmp
                shared_data["lights"] = dict(cached_lights)
            else:
                # vẽ nhẹ theo cache (không bắt buộc)
                shared_data["lights"] = dict(cached_lights)

            # =========================
            # Vehicle detect + ByteTrack
            # =========================
            t_xyxy, t_cls, t_ids = get_vehicle_tracks(frame)

            shared_data["stats"] = {k: 0 for k in ["car", "motorcycle", "bus", "truck"]}
            current_vehicles.clear()
            seen_track_ids = set()

            for i in range(len(t_xyxy)):
                cls_id = int(t_cls[i])
                if cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, t_xyxy[i])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                track_id = int(t_ids[i]) if t_ids is not None else i
                seen_track_ids.add(track_id)

                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                cls_name = VEHICLE_CLASSES[cls_id]
                shared_data["stats"][cls_name] += 1

                # draw bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{cls_name[:3]}-{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                img_b64 = crop_and_encode(frame, [x1, y1, x2, y2])
                current_vehicles[track_id] = {
                    "img": (f"data:image/jpeg;base64,{img_b64}" if img_b64 else ""),
                    "plate": PLATE_PENDING,
                    "type": cls_name,
                    "time": _now_hms(),
                }

                # =========================
                # PLATE on cropped vehicle
                # =========================
                # nếu đã có cache và chưa tới kỳ OCR -> dùng cache
                cached = plate_cache.get(track_id)
                if cached:
                    cached["last_seen_ts"] = time.time()
                    # hiển thị ổn định
                    if cached.get("plate"):
                        current_vehicles[track_id]["plate"] = cached["plate"]

                need_ocr = True
                if cached and (frame_idx - cached.get("last_frame", 0) < PLATE_EVERY_N_FRAMES):
                    need_ocr = False

                if need_ocr:
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size != 0:
                        lp_res = plate_model(vehicle_crop)[0]
                        p_xyxy, p_conf, p_cls = _extract_boxes(lp_res)

                        best_i = -1
                        best_conf = 0.0
                        for i in range(len(p_xyxy)):
                            conf = float(p_conf[i])
                            if conf >= PLATE_MIN_CONF and conf > best_conf:
                                best_conf = conf
                                best_i = i

                        if best_i >= 0:
                            px1, py1, px2, py2 = map(int, p_xyxy[best_i])

                            # toạ độ plate về frame gốc
                            fx1, fy1 = x1 + px1, y1 + py1
                            fx2, fy2 = x1 + px2, y1 + py2

                            # preprocess + OCR
                            plate_crop = process_license_plate(frame, fx1, fy1, fx2, fy2)

                            out = read_license_plate(plate_crop)  # giữ signature bạn đang dùng
                            text = out[0] if isinstance(out, tuple) else out
                            text = _norm_plate(text)

                            if text:
                                plate_img_b64 = crop_and_encode(frame, [fx1, fy1, fx2, fy2])
                                plate_img = f"data:image/jpeg;base64,{plate_img_b64}" if plate_img_b64 else ""

                                info = vehicle_info.get(text, {})
                                detected_type = cls_name 
                                match_type = False
                                if info and detected_type in TYPE_MAPPING:
                                    db_type = (info.get("class_vehicle") or "").strip()
                                    if db_type in TYPE_MAPPING[detected_type]:
                                        match_type = True

                                plate_cache[track_id] = {
                                    "plate": text,
                                    "plate_img": plate_img,
                                    "owner": info.get("owner", "Không tìm thấy") if info else "Không tìm thấy",
                                    "phone": info.get("phone", "") if info else "",
                                    "class_vehicle": info.get("class_vehicle", "") if info else "",
                                    "province": info.get("province", "") if info else "",
                                    "registration_date": info.get("registration_date", "") if info else "",
                                    "id_card": info.get("id_card", "") if info else "",
                                    "match_type": match_type,
                                    "last_frame": frame_idx,
                                    "last_seen_ts": time.time(),
                                }

                                current_vehicles[track_id]["plate"] = text

                                # update các violation record còn đang "Đang đọc..."
                                for v in violations:
                                    if v.get("id") == track_id and v.get("plate") == PLATE_PENDING:
                                        v["plate"] = text
                                        v["plate_img"] = plate_img
                                        v["owner"] = plate_cache[track_id]["owner"]
                                        v["phone"] = plate_cache[track_id]["phone"]
                                        v["class_vehicle"] = plate_cache[track_id]["class_vehicle"]
                                        v["province"] = plate_cache[track_id]["province"]
                                        v["registration_date"] = plate_cache[track_id]["registration_date"]
                                        v["id_card"] = plate_cache[track_id]["id_card"]
                                        v["match_type"] = plate_cache[track_id]["match_type"]

                                # draw plate bbox + text (optional)
                                cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                                cv2.putText(display_frame, text, (fx1, fy2 + 25), 0, 0.7, (0, 0, 255), 2)
                # --- push DB info into vehicles for UI ---
                if track_id in current_vehicles and track_id in plate_cache:
                    info = plate_cache[track_id]
                    current_vehicles[track_id].update({
                        "owner": info.get("owner", ""),
                        "phone": info.get("phone", ""),
                        "class_vehicle": info.get("class_vehicle", ""),
                        "province": info.get("province", ""),
                        "registration_date": info.get("registration_date", ""),
                        "id_card": info.get("id_card", ""),
                        "match_type": bool(info.get("match_type", False)),
                        "plate_img": info.get("plate_img", ""),
                    })

                # =========================
                # VIOLATION - LINE
                # =========================
                if track_id in prev_positions:
                    prev = prev_positions[track_id]
                    for line in zones.get("lines", []):
                        if not line or len(line) != 2:
                            continue
                        if check_line_crossing(prev, center, line):
                            line_x = (line[0][0] + line[1][0]) // 2
                            light_key = "left" if line_x < frame.shape[1] // 2 else "straight"

                            should_violate = True
                            if use_traffic_light:
                                # giữ logic của bạn: chỉ phạt "straight" khi đỏ
                                should_violate = (light_key == "straight" and shared_data["lights"].get(light_key) == "red")

                            if should_violate and _should_log_violation(track_id, "line"):
                                violations.append({
                                    "id": track_id,
                                    "plate": plate_cache.get(track_id, {}).get("plate", PLATE_PENDING),
                                    "type": ("VƯỢT ĐÈN ĐỎ (đi thẳng)" if use_traffic_light else "VƯỢT LINE"),
                                    "time": _now_hms(),
                                    "img": current_vehicles[track_id]["img"],
                                    "plate_img": plate_cache.get(track_id, {}).get("plate_img", ""),
                                    "owner": plate_cache.get(track_id, {}).get("owner", ""),
                                    "phone": plate_cache.get(track_id, {}).get("phone", ""),
                                    "class_vehicle": plate_cache.get(track_id, {}).get("class_vehicle", ""),
                                    "province": plate_cache.get(track_id, {}).get("province", ""),
                                    "registration_date": plate_cache.get(track_id, {}).get("registration_date", ""),
                                    "id_card": plate_cache.get(track_id, {}).get("id_card", ""),
                                    "match_type": plate_cache.get(track_id, {}).get("match_type", False),
                                })

                # =========================
                # VIOLATION - POLYGON (enter)
                # =========================
                inside_any = False
                violated_poly = None

                for poly in zones.get("polygons", []):
                    if not poly or len(poly) < 3:
                        continue
                    pts = np.array(poly, np.int32)
                    if cv2.pointPolygonTest(pts, center, False) >= 0:
                        inside_any = True
                        violated_poly = poly
                        break

                if inside_any and not prev_inside.get(track_id, False):
                    should_violate = True
                    light_key = "straight"
                    if use_traffic_light and violated_poly:
                        avg_x = float(np.mean([p[0] for p in violated_poly]))
                        light_key = "left" if avg_x < frame.shape[1] // 2 else "straight"
                        should_violate = (shared_data["lights"].get(light_key) == "red")

                    if should_violate and _should_log_violation(track_id, "polygon"):
                        violations.append({
                            "id": track_id,
                            "plate": plate_cache.get(track_id, {}).get("plate", "Đang đọc..."),
                            "type": ("VÀO VÙNG CẤM" if not use_traffic_light else f"VÀO VÙNG CẤM (đèn đỏ {light_key})"),
                            "time": _now_hms(),
                            "img": current_vehicles[track_id]["img"],
                            "plate_img": plate_cache.get(track_id, {}).get("plate_img", ""),
                            "owner": plate_cache.get(track_id, {}).get("owner", ""),
                            "phone": plate_cache.get(track_id, {}).get("phone", ""),
                            "class_vehicle": plate_cache.get(track_id, {}).get("class_vehicle", ""),
                            "province": plate_cache.get(track_id, {}).get("province", ""),
                            "registration_date": plate_cache.get(track_id, {}).get("registration_date", ""),
                            "id_card": plate_cache.get(track_id, {}).get("id_card", ""),
                            "match_type": plate_cache.get(track_id, {}).get("match_type", False),
                        })

                prev_inside[track_id] = inside_any
                prev_positions[track_id] = center

            # Cleanup cache (track mất lâu)
            # Nếu track không còn xuất hiện, sau 5s loại cache để tránh phình RAM
            now_ts = time.time()
            dead_ids = []
            for tid, info in plate_cache.items():
                if tid not in seen_track_ids and (now_ts - float(info.get("last_seen_ts", now_ts)) > 5.0):
                    dead_ids.append(tid)
            for tid in dead_ids:
                plate_cache.pop(tid, None)
                prev_positions.pop(tid, None)
                prev_inside.pop(tid, None)

            # =========================
            # Draw zones
            # =========================
            for line in zones.get("lines", []):
                if not line or len(line) != 2:
                    continue
                cv2.line(display_frame, tuple(map(int, line[0])), tuple(map(int, line[1])), (0, 0, 255), 3)

            for poly in zones.get("polygons", []):
                if not poly or len(poly) < 3:
                    continue
                pts = np.array(poly, np.int32)
                cv2.polylines(display_frame, [pts], True, (255, 255, 0), 3)

            # =========================
            # Encode MJPEG
            # =========================
            ok, buf = cv2.imencode(".jpg", display_frame)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    except GeneratorExit:
        print("[INFO] Client closed stream.")
    except Exception as exc:
        traceback.print_exc()
        print(f"[LỖI STREAM] {exc}")
        time.sleep(0.1)

#######################################################################################################################
# HTTP API Endpoints
#######################################################################################################################
@app.get("/")
async def index() -> HTMLResponse:
    with open(INDEX_HTML_PATH, encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.post("/api/pause")
async def set_pause(data: Dict[str, Any]) -> Dict[str, Any]:
    global pause_processing
    pause_processing = bool(data.get("pause", False))
    return {"status": "ok", "pause": pause_processing}

@app.post("/api/zones")
async def set_zones(data: Dict[str, Any]) -> Dict[str, Any]:
    global zones
    zones["lines"] = data.get("lines", [])
    zones["polygons"] = data.get("polygons", [])
    print(f"[ZONES] Lines: {len(zones['lines'])}, Polygons: {len(zones['polygons'])}")
    return {"status": "ok", "message": "Zones updated successfully"}

@app.post("/api/set_option")
async def set_option(data: Dict[str, Any]) -> Dict[str, Any]:
    global use_traffic_light
    use_traffic_light = bool(data.get("use_traffic_light", True))
    return {"status": "ok", "use_traffic_light": use_traffic_light}

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    global VIDEO_PATH, cap
    global violations, current_vehicles, prev_positions, prev_inside
    global plate_cache, frame_idx, violation_last_ts

    try:
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "mp4"
        new_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4()}.{file_ext}")

        with open(new_path, "wb") as f:
            f.write(await file.read())

        if cap is not None:
            cap.release()

        VIDEO_PATH = new_path
        cap = cv2.VideoCapture(VIDEO_PATH)

        # reset runtime state
        violations.clear()
        current_vehicles.clear()
        prev_inside.clear()
        prev_positions.clear()
        plate_cache.clear()
        violation_last_ts.clear()
        frame_idx = 0
        # reset Ultralytics tracker state (quan trọng)
        try:
            coco_model.predictor = None
        except Exception:
            pass

        return {"status": "ok", "message": f"Video uploaded and processing: {file.filename}"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

#######################################################################################################################
# WebSocket Endpoint
#######################################################################################################################
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = {
                "vehicles": current_vehicles,         # UI panel xe
                "violations": violations[-10:],        # UI panel vi phạm
                "stats": shared_data["stats"],         # thống kê
                "lights": shared_data["lights"],       # đèn
                "fps": shared_data.get("fps", 0.0),    # optional
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

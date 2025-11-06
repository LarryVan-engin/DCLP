"""
*******************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTEGRATION
File:          api_main.py
Run:           uvicorn api_main:app --reload → http://127.0.0.1:8000
*******************************************************************************************************************
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr
import base64
import asyncio
from datetime import datetime
import pandas as pd
import os
import uuid


from module_utils import get_car, read_license_plate, process_license_plate

# ==============================
# CẤU HÌNH
# ==============================
display_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # Frame đen mặc định
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

VIDEO_PATH = None
DB_PATH = "database/owners_sample.csv"
VIOLATION_DIR = "violations"    
os.makedirs(VIOLATION_DIR, exist_ok=True)

# --- Models ---
print("[INFO] Đang tải mô hình...")
coco_model = YOLO('yolo12n.pt')
plate_model = YOLO(r'D:\VSCode\DCLP\main_code\runs\detect\model_detect_license_plate.pt')
traffic_light_model = YOLO(r"D:\VSCode\DCLP\main_code\runs\detect\traffic_light\weights\best.pt")
reader = easyocr.Reader(['en'], gpu=True)

# --- Trackers ---
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
trackers = {
    cls: DeepSort(max_age=5, n_init=2, embedder="mobilenet", max_cosine_distance=0.3)
    for cls in VEHICLE_CLASSES.values()
}

# --- DB ---
try:
    owner_db = pd.read_csv(DB_PATH)
    owner_db = owner_db.drop_duplicates(subset='plate')
    vehicle_info = owner_db.set_index('plate').to_dict('index')
    print(f"[INFO] Loaded {len(vehicle_info)} vehicle records.")
except Exception as e:
    print(f"[LỖI DB] {e}")
    vehicle_info = {}

# Mapping detected type sang class_vehicle trong DB (có thể chỉnh nếu cần)
TYPE_MAPPING = {
    'car': ['Ô tô con', 'Xe bán tải'],
    'motorcycle': ['Xe máy'],
    'bus': ['Ô tô khách'],
    'truck': ['Ô tô tải', 'Xe chuyên dụng']
}

# --- Biến toàn cục ---
zones = {"lines": [], "polygons": []}
violations = []
current_vehicles = {}
prev_positions = {}
prev_inside = {}  # {track_id: bool} - có đang trong bất kỳ polygon nào không
cap = None
os.makedirs("uploads", exist_ok=True) #save video upload
pause_processing = False #Pause video


# === BIẾN CHIA SẺ GIỮA CÁC FRAME ===
shared_data = {
    "stats": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
    "lights": {"left": "red", "straight": "green"}
}

use_traffic_light = True  # Default: Phạt dựa đèn

# ==============================
# HÀM HỖ TRỢ
# ==============================
def check_line_crossing(prev, curr, line):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(line[0], prev, line[1]) != ccw(line[1], prev, line[0]) and \
           ccw(line[0], line[1], prev) != ccw(line[0], line[1], curr)

def crop_and_encode(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    if x1 >= x2 or y1 >= y2: return None
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return None
    _, buf = cv2.imencode('.jpg', crop)
    return base64.b64encode(buf).decode()

# ==============================
# MJPEG + AI
# ==============================
def gen_frames():
    global violations, current_vehicles, prev_positions, shared_data, pause_processing, display_frame
    
    try:
        while True:
            # Nếu chưa có video thì trả về khung đen
            if cap is None or not cap.isOpened():
                blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(blank, "Chua co video - vui long upload", (250, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ret, buffer = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                cv2.waitKey(1)
                continue

            if pause_processing:
                if display_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', display_frame)
                    if ret:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                asyncio.sleep(0.1) #Reduce CPU load
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.resize(frame, (1280, 720))
            display_frame = frame.copy()

            # === ĐÈN GIAO THÔNG ===
            light_results = traffic_light_model(frame)[0]
            for box in light_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = box.conf[0]
                if conf < 0.5: continue
                cls_name = light_results.names[cls_id]
                center_x = (x1 + x2) // 2
                key = "left" if center_x < frame.shape[1] // 2 else "straight"
                shared_data["lights"][key] = cls_name

                color = {"red": (0,0,255), "yellow": (0,255,255), "green": (0,255,0)}.get(cls_name, (255,255,255))
                cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(display_frame, cls_name, (x1, y2+25), 0, 0.7, color, 2)

            # === XE + TRACKING ===
            results = coco_model(frame)[0]
            dets_by_class = {cls: [] for cls in trackers.keys()}
            for box in results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASSES:
                    cls_name = VEHICLE_CLASSES[cls_id]
                    dets_by_class[cls_name].append(([x1, y1, x2, y2], conf, cls_name))

            track_data = []
            current_vehicles.clear()
            shared_data["stats"] = {cls: 0 for cls in trackers.keys()}

            for cls_name, dets in dets_by_class.items():
                tracks = trackers[cls_name].update_tracks(dets, frame=frame)
                confirmed = [t for t in tracks if t.is_confirmed()]
                shared_data["stats"][cls_name] = len(confirmed)
                for track in confirmed:
                    ltrb = track.to_ltrb()
                    track_id = track.track_id
                    x1, y1, x2, y2 = map(int, ltrb)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Vẽ
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{cls_name[:3]}-{track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Crop xe
                    img_b64 = crop_and_encode(frame, ltrb)
                    current_vehicles[track_id] = {
                        "img": f"data:image/jpeg;base64,{img_b64}" if img_b64 else "",
                        "plate": "Đang đọc...",
                        "type": cls_name,
                        "time": datetime.now().strftime("%H:%M:%S")
                    }

                    # === VI PHẠM ===
                    if track_id in prev_positions:
                        prev = prev_positions[track_id]
                        
                        # Check lines
                        for line in zones['lines']:
                            if check_line_crossing(prev, center, line):
                                line_x = (line[0][0] + line[1][0]) // 2
                                light_key = "left" if line_x < frame.shape[1]//2 else "straight"
                                should_violate = True
                                if use_traffic_light:
                                    should_violate = (light_key == "straight" and shared_data["lights"].get(light_key) == "red")
                                if should_violate:
                                    violations.append({
                                        "id": track_id,
                                        "plate": "Đang đọc...",
                                        "type": f"VƯỢT ĐÈN ĐỎ (đi thẳng)" if use_traffic_light else "VƯỢT LINE",
                                        "time": datetime.now().strftime("%H:%M:%S"),
                                        "img": current_vehicles[track_id]["img"],
                                        "plate_img": "",
                                        "owner": "",
                                        "phone": "",
                                        "class_vehicle": "",
                                        "province": "",
                                        "registration_date": "",
                                        "id_card": "",
                                        "match_type": False
                                    })

                    # Check polygons
                    inside_any = False
                    violated_poly = None
                    for poly in zones['polygons']:
                        if len(poly) < 3: continue
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
                            light_key = "left" if avg_x < frame.shape[1]//2 else "straight"
                            should_violate = (shared_data["lights"].get(light_key) == "red")
                        if should_violate:
                            violations.append({
                                "id": track_id,
                                "plate": "Đang đọc...",
                                "type": "VÀO VÙNG CẤM" if not use_traffic_light else f"VÀO VÙNG CẤM (đèn đỏ {light_key})",
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "img": current_vehicles[track_id]["img"],
                                "plate_img": "",
                                "owner": "",
                                "phone": "",
                                "class_vehicle": "",
                                "province": "",
                                "registration_date": "",
                                "id_card": "",
                                "match_type": False
                            })

                    prev_inside[track_id] = inside_any
                    prev_positions[track_id] = center

                    track_data.append([x1, y1, x2, y2, track_id, cls_name])

            # === BIỂN SỐ ===
            lps = plate_model(frame)[0]
            for box in lps.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car = get_car(box.xyxy[0].tolist(), track_data)
                if len(car) != 6 or car[4] == -1:
                    continue
                # if car[4] == -1: continue
                _, _, _, _, vid, _ = car

                plate_crop = process_license_plate(frame, x1, y1, x2, y2)
                text, _ = read_license_plate(plate_crop)

                if text:
                    img_b64 = crop_and_encode(frame, [x1, y1, x2, y2])
                    plate_img = f"data:image/jpeg;base64,{img_b64}" if img_b64 else ""

                    if vid in current_vehicles:
                        current_vehicles[vid]["plate"] = text

                    # Tra DB dựa trên plate và check loại xe
                    info = vehicle_info.get(text, {})
                    match_type = False
                    if info and cls_name in TYPE_MAPPING:
                        if info.get('class_vehicle') in TYPE_MAPPING[cls_name]:
                            match_type = True
                    
                    for v in violations:
                            if v['id'] == vid and v['plate'] == "Đang đọc...":
                                v['plate'] = text
                                v['plate_img'] = plate_img
                                v['owner'] = info.get('owner', "Không tìm thấy")
                                v['phone'] = info.get('phone', "")
                                v['class_vehicle'] = info.get('class_vehicle', "")
                                v['province'] = info.get('province', "")
                                v['registration_date'] = info.get('registration_date', "")
                                v['id_card'] = info.get('id_card', "")
                                v['match_type'] = match_type

                    cv2.putText(display_frame, text, (x1, y2 + 25), 0, 0.7, (0, 0, 255), 2)

            # === VẼ VÙNG ===
            for line in zones['lines']:
                cv2.line(display_frame, tuple(map(int, line[0])), tuple(map(int, line[1])), (0, 0, 255), 3)
            for poly in zones['polygons']:
                pts = np.array(poly, np.int32)
                cv2.polylines(display_frame, [pts], True, (255, 255, 0), 3)

            # === ENCODE ===
            ret, buffer = cv2.imencode('.jpg', display_frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    except GeneratorExit:
        print("[INFO] Client closed stream (browser reload).")
    except Exception as e:
        print(f"[LỖI STREAM] {e}")
# ==============================
# API
# ==============================
@app.get("/")
async def index():
    with open(r"D:\VSCode\DCLP\web_application\index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/stream")
async def stream():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
# -------------------------------------------------
# Endpoint bật/tắt tạm dừng
# -------------------------------------------------
@app.post("/api/pause")
async def set_pause(data: dict):
    global pause_processing
    pause_processing = data.get("pause", False)
    return {"status": "ok", "pause": pause_processing}

@app.post("/api/zones")
async def set_zones(data: dict):
    global zones
    # Dữ liệu gửi từ app.js có dạng: {lines: [[...]], polygons: [[...]]}
    lines = data.get('lines', [])
    polygons = data.get('polygons', [])
    
    # Cập nhật biến zones toàn cục
    zones['lines'] = lines
    zones['polygons'] = polygons
    
    # In ra để debug (Tùy chọn)
    print(f"[ZONES] Lines: {len(lines)}, Polygons: {len(polygons)}")
    
    return {"status": "ok", "message": "Zones updated successfully"}

@app.post("/api/set_option")
async def set_zones(data: dict):
    global use_traffic_light
    use_traffic_light = data.get('use_traffic_light', True)
    return {"status": "ok", "use_traffic_light": use_traffic_light}

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    global VIDEO_PATH, cap, violations, current_vehicles, prev_positions, prev_inside, trackers
    try:
        # Generate tên file unique
        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'mp4'
        new_path = f"uploads/{uuid.uuid4()}.{file_ext}"
        
        # Lưu file
        with open(new_path, "wb") as f:
            f.write(await file.read())
        
        # Reset cap với video mới
        if cap is not None:
            cap.release()  # Release cap cũ
        
        VIDEO_PATH = new_path
        cap = cv2.VideoCapture(VIDEO_PATH)
        
        # Xóa video cũ nếu cần (tùy chọn, để tránh đầy disk)
        # for old_file in os.listdir("uploads"):
        #     if old_file != os.path.basename(new_path):
        #         os.remove(os.path.join("uploads", old_file))

        violations.clear()
        current_vehicles.clear()
        prev_inside.clear()
        prev_positions.clear()

        #Reset deepSORT tracker
        for cls_name in trackers.keys():
            trackers[cls_name] = DeepSort(max_age=5, n_init=2, embedder="mobilenet", max_cosine_distance=0.3)
        
        return {"status": "ok", "message": f"Video uploaded and processing: {file.filename}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==============================
# WEBSOCKET
# ==============================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = {
                "vehicles": current_vehicles,
                "violations": violations[-10:],
                "stats": shared_data["stats"],
                "lights": shared_data["lights"]
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass

# ==============================
# CHẠY
# ==============================
if __name__ == "__main__":
    import uvicorn
    print("[INFO] Server: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
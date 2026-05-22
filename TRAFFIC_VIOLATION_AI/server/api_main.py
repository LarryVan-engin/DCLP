"""
********************************************************************************************************************
Project:      Traffic Violation Detection (Pro Version - MQTT Hybrid)
File:         server/api_main.py
Description:  Server trung tâm điều khiển MQTT, xử lý OCR biển số và đồng bộ MongoDB Atlas.
********************************************************************************************************************
"""

import asyncio
import base64
import os
import json
import re
import shutil
from contextlib import asynccontextmanager
import cv2
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from datetime import datetime
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from motor.motor_asyncio import AsyncIOMotorClient

# Import Schemas và Utils
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.schemas import ControlCommand, ViolationPacket, HeartbeatPacket
import imageio_ffmpeg
from module_utils import read_license_plate_vn

# ====================== CONFIGURATION ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_CSV_PATH = os.path.join(BASE_DIR, "database", "owners_sample.csv")
VIOLATION_DIR = os.path.join(BASE_DIR, "violations")
MODEL_PLATE_PATH = os.path.join(BASE_DIR, "models", "model_detect_license_plate.pt")

# MQTT Settings
MQTT_BROKER = "127.0.0.1" 
MQTT_PORT = 1883
MQTT_CLIENT_ID = "TRAFFIC_SERVER_01"

# MongoDB Settings (Atlas)
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://admin:admin123@cluster0.iipaqpd.mongodb.net/?appName=Cluster0")

os.makedirs(VIOLATION_DIR, exist_ok=True)
PROCESSED_VIDEOS_DIR = os.path.join(BASE_DIR, "processed_videos")
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)

# Lock per (camera_id, mode, track_id, video_name) — tránh race condition khi 2 gói tin
# cùng track_id gửi gần nhau và cả hai thấy find_one → None → tạo 2 bản ghi riêng.
_violation_locks: dict = {}

# ====================== INITIALIZATION ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng: Khởi tạo và Giải phóng tài nguyên"""
    global loop, mongodb_connected
    # Lấy CHÍNH XÁC Event Loop đang sống của Uvicorn
    loop = asyncio.get_running_loop()

    # Ẩn lỗi ConnectionResetError (WinError 10054) trên Windows — xảy ra khi browser đóng
    # kết nối HTTP giữa chừng (ví dụ: không phát được codec AVI/XVID), không nguy hiểm.
    def _suppress_connection_reset(loop, context):
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
            return
        loop.default_exception_handler(context)
    loop.set_exception_handler(_suppress_connection_reset)

    # 1. Kiểm tra kết nối MongoDB thực tế
    try:
        await client.admin.command("ping")
        mongodb_connected = True
        print("[SERVER] ✅ Connected to MongoDB Atlas: traffic_db.violations")
        
        # Tạo Index cho MongoDB để tối ưu query
        await violations_col.create_index("camera_id")
        await violations_col.create_index("timestamp")
        await violations_col.create_index("violation_type")
        await violations_col.create_index("plate_read")
        print("[SERVER] ✅ MongoDB indexes created successfully")
    except Exception as e:
        mongodb_connected = False
        print(f"[SERVER] ❌ MongoDB connection failed: {e}")
    
    # 2. Khởi động kết nối MQTT
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print("[SERVER] ✅ Khởi động luồng MQTT nền thành công.")
    except Exception as e:
        print(f"[SERVER] ❌ MQTT connection failed: {e}")

    yield # Tại đây ứng dụng sẽ chạy và phục vụ requests
    
    # 3. Dọn dẹp tài nguyên khi tắt Server
    mqtt_client.loop_stop()
    client.close() # Đóng pool kết nối MongoDB
    print("[SERVER] 🛑 Resources cleaned up (MQTT stopped, MongoDB closed).")

app = FastAPI(title="AI Traffic Monitoring Server", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Load Models & Database
print("[SERVER] Loading Plate Detection model...")
plate_model = YOLO(MODEL_PLATE_PATH)

vehicle_db = {}
try:
    df = pd.read_csv(DB_CSV_PATH)
    for _, row in df.iterrows():
        key = re.sub(r"[^A-Z0-9]", "", str(row.get("plate", "")).upper())
        vehicle_db[key] = row.to_dict()
    print(f"[SERVER] Loaded {len(vehicle_db)} vehicle records.")
except Exception as e:
    print(f"[WARNING] CSV DB load error: {e}")

# Global State
active_ws: List[WebSocket] = []
last_heartbeat = {}  # {camera_id: {stats, fps, ...}}
current_stream_frame = "" # Base64 frame 
connected_cameras = {}  # {camera_id: {id, location, last_seen}}

# ====================== MONGODB CONNECTION ======================
client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client.traffic_db
violations_col = db.violations
mongodb_connected = False

# ====================== MQTT LOGIC ======================
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}")
    # Subscribe tất cả các camera
    res = client.subscribe("status/+/heartbeat")
    print(f"[SERVER MQTT] Kết quả Subscribe Heartbeat: {res}")

    client.subscribe("status/+/files")
    client.subscribe("status/+/roi_preview")
    client.subscribe("stream/+/mjpeg")
    client.subscribe("violation/+")
    client.subscribe("complete/+")  # Nhận gói hoàn thành từ Edge khi xử lý xong video
    print("[SERVER MQTT] Đã subscribe đầy đủ tất cả topics.")

def on_message(client, userdata, msg):
    # Thêm dòng này để theo dõi:
    print(f"[DEBUG MQTT] 📥 Nhận được tin nhắn tại topic: {msg.topic}")
    asyncio.run_coroutine_threadsafe(handle_mqtt_message(msg), loop)

async def handle_mqtt_message(msg):
    global current_stream_frame, connected_cameras
    topic = msg.topic
    payload = msg.payload.decode()

    # Xử lý Stream (MJPEG)
    if "stream/" in topic:
        current_stream_frame = payload 
    
    # Xử lý Heartbeat/Stats
    elif "status/" in topic and "heartbeat" in topic:
        data = json.loads(payload)
        camera_id = data.get('camera_id')
        last_heartbeat[camera_id] = data
        
        # Cập nhật danh sách camera kết nối
        if camera_id not in connected_cameras:
            connected_cameras[camera_id] = {
                "id": camera_id,
                "location": f"Edge Device - {camera_id}",
                "connected_at": datetime.now().isoformat()
            }
            # Gửi danh sách camera cập nhật tới tất cả clients
            await broadcast_camera_list()
            print(f"[SERVER] ✅ Camera mới kết nối: {camera_id}")

    # Xử lý Danh sách Video (Dành cho tính năng chọn file động)
    elif "status/" in topic and "files" in topic:
        try:
            data = json.loads(payload)
            files = data.get("files", [])
            # Đẩy danh sách file qua WebSocket lên giao diện (Browser)
            for ws in active_ws:
                await ws.send_json({"type": "file_list", "files": files})
        except Exception as e:
            print(f"[SERVER] Lỗi parse file list: {e}")

    # 4. Xử lý Violation (Quan trọng nhất)
    elif "status/" in topic and "roi_preview" in topic:
        try:
            data = json.loads(payload)
            for ws in active_ws:
                await ws.send_json({
                    "type": "auto_roi_proposal",
                    "camera_id": data.get("camera_id"),
                    "video_name": data.get("video_name"),
                    "points": data.get("points", []),
                    "right_turn_zone_bottom_y": data.get("right_turn_zone_bottom_y", 0.15)
                })
        except Exception as e:
            print(f"[SERVER] Lỗi parse ROI preview: {e}")

    elif "violation/" in topic:
        data = json.loads(payload)
        await process_violation(data)

async def process_violation(data: dict):
    """Xử lý OCR, lưu ảnh theo thư mục riêng từng vi phạm và lưu trữ MongoDB"""
    lock_key = None
    lock_acquired = False
    try:
        packet = ViolationPacket(**data)

        # Decode ảnh xe từ Edge
        img_bytes = base64.b64decode(packet.vehicle_crop_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ── Plate Detection + OCR ──────────────────────────────────────────
        res = plate_model(frame, verbose=False)[0]
        plate_text    = "UNKNOWN"
        plate_img_b64 = None
        plate_crop    = None

        if len(res.boxes) > 0:
            box = res.boxes[0].xyxy[0].cpu().numpy().astype(int)
            text, success = read_license_plate_vn(frame, box[0], box[1], box[2], box[3])
            if success:
                plate_text = text
                plate_crop = frame[box[1]:box[3], box[0]:box[2]]
                _, p_buf   = cv2.imencode(".jpg", plate_crop,
                                          [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                plate_img_b64 = base64.b64encode(p_buf).decode()

        # Khóa per-vehicle: tránh race condition khi 2 gói tin cùng track_id tới cách nhau ~50ms,
        # cả hai thấy find_one → None và tạo 2 bản ghi + 2 thư mục riêng.
        lock_key = (packet.camera_id, packet.mode, packet.track_id, getattr(packet, 'video_name', None))
        if lock_key not in _violation_locks:
            _violation_locks[lock_key] = asyncio.Lock()
        await _violation_locks[lock_key].acquire()
        lock_acquired = True

        # ── Kiểm tra phương tiện đã vi phạm trước đó chưa ─────────────────────
        # Dùng video_name để phân biệt session: tránh gộp nhầm lỗi từ lần chạy cũ
        # (track_id có thể trùng giữa các lần chạy vì YOLO reset về 1, 2, 3...)
        query_filter = {
            "camera_id": packet.camera_id,
            "mode":      packet.mode,
            "track_id":  packet.track_id,
        }
        if getattr(packet, 'video_name', None):
            query_filter["video_name"] = packet.video_name
        existing_violation = await violations_col.find_one(query_filter)
        is_update = existing_violation is not None

        # Gộp lỗi vi phạm
        if is_update:
            existing_vtype = existing_violation.get("violation_type", "")
            existing_list = [v.strip() for v in existing_vtype.split("+") if v.strip()]
            incoming_list = [v.strip() for v in packet.violation_type.split("+") if v.strip()]
            combined_list = []
            for v in existing_list + incoming_list:
                if v not in combined_list:
                    combined_list.append(v)
            combined_violation_type = "+".join(combined_list)
        else:
            combined_violation_type = packet.violation_type

        # Xử lý biển số và thông tin chủ xe
        if is_update:
            existing_plate = existing_violation.get("plate_read", "UNKNOWN")
            if existing_plate != "UNKNOWN" and plate_text == "UNKNOWN":
                plate_text = existing_plate
                plate_img_b64 = existing_violation.get("plate_img_base64")
            
        clean_plate = re.sub(r"[^A-Z0-9]", "", plate_text.upper())
        owner_info  = vehicle_db.get(clean_plate, {})

        # ── Xử lý thư mục lưu trữ & Đổi tên ──────────────────────────────────
        orig_timestamp = existing_violation.get("timestamp", packet.timestamp) if is_update else packet.timestamp
        ts_clean      = re.sub(r"[:\.]", "-", orig_timestamp)
        combined_vtype_clean = re.sub(r"\s+", "_", combined_violation_type)
        plate_clean   = clean_plate if clean_plate else "UNKNOWN"
        folder_name   = f"ID{packet.track_id}_{ts_clean}_{plate_clean}_{combined_vtype_clean}"
        folder_path   = os.path.join(VIOLATION_DIR, folder_name)

        if is_update:
            old_folder_path = existing_violation.get("image_folder")
            if old_folder_path and os.path.exists(old_folder_path) and old_folder_path != folder_path:
                try:
                    if os.path.exists(folder_path):
                        # Nếu thư mục mới đã tồn tại, di chuyển toàn bộ file từ thư mục cũ sang mới
                        for f in os.listdir(old_folder_path):
                            src_f = os.path.join(old_folder_path, f)
                            dst_f = os.path.join(folder_path, f)
                            if os.path.exists(dst_f):
                                os.remove(dst_f)
                            shutil.move(src_f, dst_f)
                        shutil.rmtree(old_folder_path)
                    else:
                        shutil.move(old_folder_path, folder_path)
                    print(f"[SERVER] 📁 Renamed violation folder: {old_folder_path} -> {folder_path}")
                except Exception as e:
                    print(f"[SERVER] ⚠️ Error renaming violation folder: {e}")
                    os.makedirs(folder_path, exist_ok=True)
            else:
                os.makedirs(folder_path, exist_ok=True)
        else:
            os.makedirs(folder_path, exist_ok=True)

        # ── Lưu ảnh vào thư mục vi phạm ──────────────────────────────────
        # Tính sớm để dùng cho tên file — mỗi gói tin (1 loại lỗi) được đặt tên riêng,
        # tránh ghi đè ảnh khi cùng xe bị phạt nhiều lỗi khác nhau.
        current_vtype_clean = re.sub(r"\s+", "_", packet.violation_type)

        # 1. Ảnh smart crop khít từ Edge — label theo loại lỗi của gói tin này
        vehicle_img_path = os.path.join(folder_path, f"vehicle_crop_{current_vtype_clean}.jpg")
        cv2.imwrite(vehicle_img_path, frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # 2. Ảnh crop rộng 100px (ngữ cảnh rõ hơn, 1 frame sau vi phạm)
        vehicle_wide_img_path = None
        if hasattr(packet, 'vehicle_crop_wide_base64') and packet.vehicle_crop_wide_base64:
            try:
                wide_bytes = base64.b64decode(packet.vehicle_crop_wide_base64)
                wide_img   = cv2.imdecode(np.frombuffer(wide_bytes, np.uint8), cv2.IMREAD_COLOR)
                if wide_img is not None:
                    vehicle_wide_img_path = os.path.join(folder_path, f"vehicle_crop_wide_{current_vtype_clean}.jpg")
                    cv2.imwrite(vehicle_wide_img_path, wide_img,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            except Exception as e:
                print(f"[SERVER] ⚠️ Lỗi lưu vehicle_crop_wide: {e}")

        # 3. Ảnh biển số với label chữ
        plate_img_path = None
        if plate_crop is not None and plate_crop.size > 0:
            labeled_plate = plate_crop.copy()
            label_h = 24
            label_bar = np.zeros((label_h, labeled_plate.shape[1], 3), dtype=np.uint8)
            cv2.putText(label_bar, plate_text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            labeled_plate = np.vstack([label_bar, labeled_plate])
            plate_img_path = os.path.join(folder_path, f"plate_{plate_clean}.jpg")
            cv2.imwrite(plate_img_path, labeled_plate,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 4. Ảnh toàn khung SẠCH — thấy đèn + bối cảnh
        full_frame_a_path = None
        if packet.full_frame_a_base64:
            try:
                fa_bytes = base64.b64decode(packet.full_frame_a_base64)
                fa_img   = cv2.imdecode(np.frombuffer(fa_bytes, np.uint8), cv2.IMREAD_COLOR)
                if fa_img is not None:
                    full_frame_a_path = os.path.join(folder_path, f"full_A_{current_vtype_clean}.jpg")
                    cv2.imwrite(full_frame_a_path, fa_img,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            except Exception as e:
                print(f"[SERVER] ⚠️ Lỗi lưu full_A: {e}")

        # 5. Ảnh toàn khung CÓ CHÚ THÍCH — bbox đỏ + loại vi phạm
        full_frame_b_path = None
        if packet.full_frame_b_base64:
            try:
                fb_bytes = base64.b64decode(packet.full_frame_b_base64)
                fb_img   = cv2.imdecode(np.frombuffer(fb_bytes, np.uint8), cv2.IMREAD_COLOR)
                if fb_img is not None:
                    full_frame_b_path = os.path.join(folder_path, f"full_B_{current_vtype_clean}.jpg")
                    cv2.imwrite(full_frame_b_path, fb_img,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            except Exception as e:
                print(f"[SERVER] ⚠️ Lỗi lưu full_B: {e}")

        print(f"[SERVER] 📁 Đã lưu ảnh vi phạm → {folder_path}")

        # Helper to update paths to new folder
        def update_path_to_new_folder(old_path, old_dir, new_dir):
            if not old_path or old_path == "N/A":
                return "N/A"
            # Chuẩn hóa đường dẫn để so sánh chính xác trên mọi OS (đặc biệt là Windows)
            old_path_norm = os.path.normpath(old_path)
            old_dir_norm = os.path.normpath(old_dir)
            new_dir_norm = os.path.normpath(new_dir)

            if old_path_norm.startswith(old_dir_norm):
                return old_path_norm.replace(old_dir_norm, new_dir_norm, 1)

            old_base = os.path.basename(old_dir_norm)
            new_base = os.path.basename(new_dir_norm)
            if old_base in old_path_norm:
                return old_path_norm.replace(old_base, new_base, 1)
            return old_path_norm

        # Đồng bộ và cập nhật đường dẫn ảnh
        if is_update:
            old_dir = existing_violation.get("image_folder", "")
            # Chỉ cập nhật các đường dẫn từ bản ghi cũ nếu gói tin mới không gửi ảnh mới đè lên
            if vehicle_img_path is None:
                vehicle_img_path = update_path_to_new_folder(
                    existing_violation.get("vehicle_img_path"), old_dir, folder_path
                )
            else:
                vehicle_img_path = os.path.normpath(vehicle_img_path)

            if vehicle_wide_img_path is None:
                vehicle_wide_img_path = update_path_to_new_folder(
                    existing_violation.get("vehicle_wide_img_path"), old_dir, folder_path
                )
            else:
                vehicle_wide_img_path = os.path.normpath(vehicle_wide_img_path)

            if plate_img_path is None:
                old_plate_img = existing_violation.get("plate_img_path")
                if old_plate_img and old_plate_img != "N/A":
                    old_plate_base = os.path.basename(os.path.normpath(old_plate_img))
                    new_plate_base = f"plate_{plate_clean}.jpg"
                    if old_plate_base != new_plate_base:
                        old_p_file = os.path.join(folder_path, old_plate_base)
                        new_p_file = os.path.join(folder_path, new_plate_base)
                        if os.path.exists(old_p_file):
                            try:
                                shutil.move(old_p_file, new_p_file)
                            except Exception as e:
                                print(f"[SERVER] ⚠️ Error renaming plate image file: {e}")
                    plate_img_path = os.path.join(folder_path, new_plate_base)
                else:
                    plate_img_path = "N/A"
            else:
                plate_img_path = os.path.normpath(plate_img_path)

            if full_frame_a_path is None:
                full_frame_a_path = update_path_to_new_folder(
                    existing_violation.get("full_frame_a_path"), old_dir, folder_path
                )
            else:
                full_frame_a_path = os.path.normpath(full_frame_a_path)

            if full_frame_b_path is None:
                full_frame_b_path = update_path_to_new_folder(
                    existing_violation.get("full_frame_b_path"), old_dir, folder_path
                )
            else:
                full_frame_b_path = os.path.normpath(full_frame_b_path)

        # ── Chuẩn bị document lưu MongoDB ────────────────────────────────
        packet_dict = packet.dict()
        # Bỏ các trường base64 nặng — đã lưu vào file, không cần lưu trong MongoDB
        mongo_doc = {k: v for k, v in packet_dict.items()
                     if k not in ("vehicle_crop_base64", "vehicle_crop_wide_base64",
                                  "full_frame_a_base64", "full_frame_b_base64")}
        
        timestamp_to_save = orig_timestamp if is_update else packet.timestamp
        confidence_to_save = max(existing_violation.get("confidence", 0), packet.confidence) if is_update else packet.confidence

        violation_doc = {
            **mongo_doc,
            "timestamp":         timestamp_to_save,
            "violation_type":    combined_violation_type,
            "confidence":        confidence_to_save,
            "plate_read":        plate_text,
            "owner":             owner_info.get("owner",             "Không xác định"),
            "phone":             owner_info.get("phone",             "N/A"),
            "class_vehicle":     owner_info.get("class_vehicle",     "N/A"),
            "province":          owner_info.get("province",          "N/A"),
            "registration_date": owner_info.get("registration_date", "N/A"),
            "id_card":           owner_info.get("id_card",           "N/A"),
            "plate_img_base64":  plate_img_b64,
            "image_folder":           folder_path,
            "vehicle_img_path":       vehicle_img_path,
            "vehicle_wide_img_path":  vehicle_wide_img_path  or "N/A",
            "plate_img_path":         plate_img_path         or "N/A",
            "full_frame_a_path":      full_frame_a_path      or "N/A",
            "full_frame_b_path":      full_frame_b_path      or "N/A",
            "processed_at":           datetime.now().isoformat()
        }

        # Lưu MongoDB Atlas với retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if is_update:
                    result = await violations_col.update_one(
                        {"_id": existing_violation["_id"]},
                        {"$set": violation_doc}
                    )
                    print(f"[SERVER] ✅ Violation updated in MongoDB: {existing_violation['_id']} (modified: {result.modified_count})")
                else:
                    result = await violations_col.insert_one(violation_doc.copy())
                    print(f"[SERVER] ✅ Violation saved to MongoDB: {result.inserted_id}")
                break
            except Exception as mongo_err:
                print(f"[SERVER] MongoDB save attempt {attempt+1} failed: {mongo_err}")
                if attempt == max_retries - 1:
                    # Backup to local JSON if MongoDB fails
                    backup_path = os.path.join(VIOLATION_DIR, f"backup_{packet.track_id}_{ts_clean}.json")
                    with open(backup_path, 'w', encoding='utf-8') as bf:
                        json.dump(violation_doc, bf, ensure_ascii=False)
                    print(f"[SERVER] ⚠️ Backup saved to local: {backup_path}")

        # Đẩy qua WebSocket lên UI — thêm lại vehicle_crop_base64 để Dashboard hiển thị ảnh
        if "_id" in violation_doc:
            del violation_doc["_id"]
        
        # ws_payload: violation_doc (không có base64 nặng) + ảnh crop nhỏ để hiển thị thumbnail
        ws_payload = {
            **violation_doc,
            "vehicle_crop_base64": packet.vehicle_crop_base64,  # ảnh thumbnail cho Luồng vi phạm
        }
        for ws in active_ws:
            await ws.send_json({"type": "violation", "data": ws_payload})

        if lock_acquired and lock_key in _violation_locks:
            _violation_locks[lock_key].release()
            lock_acquired = False

    except Exception as e:
        if lock_acquired and lock_key and lock_key in _violation_locks:
            _violation_locks[lock_key].release()
        print(f"[ERROR] Process violation failed: {e}")

# ====================== BROADCAST FUNCTIONS ======================
async def broadcast_camera_list():
    """Gửi danh sách camera tới tất cả clients đang kết nối"""
    camera_list = list(connected_cameras.values())
    message = {"type": "camera_list", "cameras": camera_list}
    for ws in active_ws:
        try:
            await ws.send_json(message)
        except Exception as e:
            print(f"[SERVER] Loi camera list: {e}")

# Khởi chạy MQTT trong background
# Chuẩn bị MQTT Client
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, MQTT_CLIENT_ID)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
loop = None  # Khởi tạo biến rỗng, sẽ gán sau

# ====================== API ENDPOINTS ======================
@app.get("/")
async def get_dashboard():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))

@app.get("/api/refresh_videos/{camera_id}")
async def refresh_videos(camera_id: str):
    # Gửi lệnh yêu cầu Edge quét thư mục video và báo cáo lại
    mqtt_client.publish(f"control/{camera_id}/command", json.dumps({"action": "list_files"}))
    return {"status": "request_sent"}

@app.post("/api/control_edge")
async def control_edge(request: Request, camera_id: str):
    """Gui lenh MQTT xuong Edge; nhan raw JSON de stop/reset/preview khong bi schema chan."""
    cmd = await request.json()
    cmd.setdefault("mode", "realtime")
    topic = f"control/{camera_id}/command"
    mqtt_client.publish(topic, json.dumps(cmd))
    print(f"[SERVER MQTT] Published command to {topic}: {cmd.get('action')}")
    return {"status": "ok", "message": f"Command sent to {camera_id}"}

async def _convert_avi_to_mp4(avi_path: str, mp4_path: str) -> bool:
    """Chuyển AVI/XVID → MP4 để trình duyệt phát được.
    Thử ffmpeg trước (nhanh, H.264), fallback sang OpenCV mp4v nếu không có ffmpeg."""
    # Thử ffmpeg (dùng binary từ imageio-ffmpeg, không cần cài tay)
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        proc = await asyncio.create_subprocess_exec(
            ffmpeg_exe, "-y", "-i", avi_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart",
            mp4_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=180)
        if proc.returncode == 0 and os.path.exists(mp4_path):
            return True
    except (FileNotFoundError, asyncio.TimeoutError, RuntimeError):
        pass

    # Fallback: OpenCV mp4v (chạy trong thread để không block event loop)
    def _cv2_reencode():
        cap = cv2.VideoCapture(avi_path)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            cap.release()
            return False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        return os.path.exists(mp4_path)

    return await asyncio.get_running_loop().run_in_executor(None, _cv2_reencode)


@app.post("/api/upload_video/{camera_id}")
async def upload_video(camera_id: str, video: UploadFile = File(...),
                       processing_time_seconds: float = Form(...),
                       mode: str = Form(default="video_local")):
    video_bytes = await video.read()

    processed_dir = os.path.join(BASE_DIR, "processed_videos")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, video.filename)
    with open(processed_path, "wb") as f:
        f.write(video_bytes)
    print(f"[SERVER] ✅ Đã lưu video vào processed_videos: {video.filename} (mode={mode})")

    if mode == "video_local":
        # Lưu vào /static/videos để trình duyệt có thể phát trực tiếp
        save_dir = os.path.join(BASE_DIR, "static", "videos")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, video.filename)
        with open(file_path, "wb") as f:
            f.write(video_bytes)

        serve_filename = video.filename

        # AVI/XVID không phát được trên trình duyệt — chuyển sang MP4
        if video.filename.lower().endswith(".avi"):
            mp4_filename = video.filename.rsplit(".", 1)[0] + ".mp4"
            mp4_path     = os.path.join(save_dir, mp4_filename)
            print(f"[SERVER] 🔄 Đang chuyển đổi AVI → MP4: {mp4_filename} ...")
            ok = await _convert_avi_to_mp4(file_path, mp4_path)
            if ok:
                try:
                    os.remove(file_path)
                except OSError:
                    pass
                serve_filename = mp4_filename
                print(f"[SERVER] ✅ Chuyển đổi xong: {mp4_filename}")
            else:
                print(f"[SERVER] ⚠️ Chuyển đổi thất bại — phát AVI trực tiếp (có thể không xem được trên trình duyệt)")

        print(f"[SERVER] ✅ Video Local đã sẵn sàng phát: {serve_filename}")
        for ws in active_ws:
            await ws.send_json({
                "type": "video_ready",
                "video_url": f"/static/videos/{serve_filename}",
                "processing_time": processing_time_seconds
            })
    else:
        # Mode "video" (Chạy xử lý phạt): chỉ lưu trữ, không cần phát lại trên dashboard
        print(f"[SERVER] ✅ Video Phạt Nguội đã lưu trữ (không phát lại): {video.filename}")

    return {"status": "success", "file": video.filename}

import csv
import io
from fastapi.responses import StreamingResponse

@app.get("/api/export_violations")
async def export_violations(start_date: str = None, end_date: str = None, format: str = "csv"):
    """
    Xuất dữ liệu vi phạm với bộ lọc thời gian và định dạng tùy chọn (csv/xlsx).
    Bao gồm đường dẫn thư mục ảnh để dễ tra cứu bằng chứng.
    """
    query = {}
    if start_date and end_date:
        query["timestamp"] = {"$gte": start_date, "$lte": end_date}

    cursor = violations_col.find(query).sort("timestamp", -1)
    violations = await cursor.to_list(length=None)

    if not violations:
        return JSONResponse(status_code=404,
                            content={"detail": "Không có dữ liệu trong khoảng thời gian này."})

    df = pd.DataFrame(violations)

    # Bỏ ObjectId MongoDB và các cột base64 nặng không cần thiết trong file xuất
    drop_cols = ["_id", "vehicle_crop_base64", "plate_img_base64",
                 "full_frame_a_base64", "full_frame_b_base64"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Thứ tự cột: thông tin tra cứu nhanh trước, đường dẫn ảnh ở cuối
    priority_cols = [
        "timestamp",         # Thời điểm vi phạm
        "camera_id",         # Camera ghi nhận
        "violation_type",    # Loại vi phạm
        "plate_read",        # Biển số đọc được
        "owner",             # Chủ xe
        "phone",             # Số điện thoại
        "class_vehicle",     # Loại xe
        "province",          # Tỉnh/thành
        "registration_date", # Ngày đăng ký
        "id_card",           # CMND/CCCD
        "confidence",        # Độ tin cậy AI
        "track_id",          # ID tracking
        "image_folder",      # ← Thư mục chứa ảnh vi phạm
        "vehicle_img_path",  # ← Ảnh smart crop (dùng để OCR)
        "plate_img_path",    # ← Ảnh biển số
        "full_frame_a_path", # ← Ảnh toàn khung SẠCH
        "full_frame_b_path", # ← Ảnh toàn khung CÓ CHÚ THÍCH
        "processed_at",      # Thời điểm server xử lý
    ]
    # Chỉ lấy cột nào thực sự tồn tại trong data
    final_cols = [c for c in priority_cols if c in df.columns]
    # Thêm các cột còn lại chưa được liệt kê (tránh mất dữ liệu)
    extra_cols = [c for c in df.columns if c not in final_cols]
    df = df[final_cols + extra_cols]

    # Đổi tên cột sang tiếng Việt cho dễ đọc
    col_rename = {
        "timestamp":         "Thời điểm vi phạm",
        "camera_id":         "Camera",
        "violation_type":    "Loại vi phạm",
        "plate_read":        "Biển số",
        "owner":             "Chủ xe",
        "phone":             "Điện thoại",
        "class_vehicle":     "Loại xe",
        "province":          "Tỉnh/Thành",
        "registration_date": "Ngày đăng ký",
        "id_card":           "CMND/CCCD",
        "confidence":        "Độ tin cậy",
        "track_id":          "ID Xe",
        "image_folder":      "Thư mục ảnh vi phạm",
        "vehicle_img_path":  "Ảnh phương tiện",
        "plate_img_path":    "Ảnh biển số",
        "full_frame_a_path": "Ảnh full frame (sạch)",
        "full_frame_b_path": "Ảnh full frame (chú thích)",
        "processed_at":      "Thời điểm xử lý",
    }
    df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})

    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    if format == "xlsx":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Vi phạm')

            # Format đẹp cho Excel
            workbook  = writer.book
            worksheet = writer.sheets['Vi phạm']

            # Header style: nền xanh đậm, chữ trắng, bold
            header_fmt = workbook.add_format({
                'bold': True, 'bg_color': '#1e3a5f', 'font_color': '#FFFFFF',
                'border': 1, 'align': 'center', 'valign': 'vcenter'
            })
            # Cell style: border nhẹ
            cell_fmt = workbook.add_format({'border': 1, 'valign': 'vcenter'})
            # Style riêng cho cột đường dẫn: màu xanh nhạt để dễ nhận ra
            path_fmt = workbook.add_format({
                'border': 1, 'valign': 'vcenter',
                'bg_color': '#e8f4fd', 'font_color': '#1565c0'
            })

            # Ghi header với format
            for col_num, col_name in enumerate(df.columns):
                worksheet.write(0, col_num, col_name, header_fmt)

            # Tự động chỉnh độ rộng cột + áp dụng format cho từng cột
            path_keywords = {"Thư mục ảnh vi phạm", "Ảnh phương tiện", "Ảnh biển số",
                             "Ảnh full frame (sạch)", "Ảnh full frame (chú thích)"}
            for col_num, col_name in enumerate(df.columns):
                # Tính độ rộng tự động dựa trên nội dung
                max_len = max(
                    df[col_name].astype(str).map(len).max() if len(df) > 0 else 0,
                    len(col_name)
                )
                col_width = min(max_len + 2, 60)  # tối đa 60 ký tự
                fmt = path_fmt if col_name in path_keywords else cell_fmt
                worksheet.set_column(col_num, col_num, col_width, fmt)

            # Freeze header row
            worksheet.freeze_panes(1, 0)
            # Auto-filter
            worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=violations_{date_str}.xlsx"}
        )
    else:
        # CSV — utf-8-sig để Excel mở đúng tiếng Việt
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=violations_{date_str}.csv"}
        )

@app.post("/api/test_ocr")
async def test_ocr_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JSONResponse(status_code=400, content={"detail": "Không thể đọc được ảnh."})
            
        res = plate_model(frame, verbose=False)[0]
        if len(res.boxes) == 0:
            return JSONResponse(status_code=400, content={"detail": "Không tìm thấy biển số trong ảnh."})
            
        box = res.boxes[0].xyxy[0].cpu().numpy().astype(int)
        
        # Vẽ box lên ảnh gốc để hiển thị
        annotated_frame = frame.copy()
        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        _, img_buf = cv2.imencode(".jpg", annotated_frame)
        image_base64 = base64.b64encode(img_buf).decode()
        
        text, success = read_license_plate_vn(frame, box[0], box[1], box[2], box[3])
        
        if not success:
            return JSONResponse(status_code=400, content={"detail": "Không thể trích xuất chữ từ biển số."})
            
        # Cắt ảnh biển số
        p_crop = frame[max(0, box[1]):box[3], max(0, box[0]):box[2]]
        _, p_buf = cv2.imencode(".jpg", p_crop)
        plate_crop_base64 = base64.b64encode(p_buf).decode()
        
        clean_plate = re.sub(r"[^A-Z0-9]", "", text.upper())
        owner_info = vehicle_db.get(clean_plate, {})
        
        return {
            "plate_text": text,
            "owner_info": {
                "owner": owner_info.get("owner", "Không tìm thấy"),
                "phone": owner_info.get("phone", "N/A"),
                "class_vehicle": owner_info.get("class_vehicle", "N/A"),
                "province": owner_info.get("province", "N/A"),
                "registration_date": owner_info.get("registration_date", "N/A"),
                "id_card": owner_info.get("id_card", "N/A")
            },
            "image_base64": image_base64,
            "plate_crop_base64": plate_crop_base64
        }
        
    except Exception as e:
        print(f"[TEST OCR ERROR] {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    try:
        if not file.filename:
            return JSONResponse(status_code=400, content={"detail": "Không có tên file."})
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(PROCESSED_VIDEOS_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"[SERVER] Đã lưu video xử lý: {filename}")
        return {"status": "success", "filename": filename, "path": file_path}
    except Exception as e:
        print(f"[SERVER] Lỗi khi lưu video: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_ws.append(websocket)
    print(f"[SERVER] ✅ WebSocket client connected. Total: {len(active_ws)}")
    try:
        # Gửi danh sách camera hiện tại ngay khi client kết nối
        camera_list = list(connected_cameras.values())
        await websocket.send_json({"type": "camera_list", "cameras": camera_list})
        
        while True:
            # Đẩy Heartbeat và Stream liên tục lên Dashboard
            await websocket.send_json({
                "type": "realtime_update",
                "stream": current_stream_frame,
                "heartbeats": last_heartbeat
            })
            await asyncio.sleep(0.1) # Tương đương 10 FPS cho UI
    except WebSocketDisconnect:
        active_ws.remove(websocket)
        print(f"[SERVER] ❌ WebSocket client disconnected. Total: {len(active_ws)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

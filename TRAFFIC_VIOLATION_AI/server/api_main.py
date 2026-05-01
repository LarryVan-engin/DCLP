<<<<<<< HEAD
"""
=======
﻿"""
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
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
<<<<<<< HEAD
from contextlib import asynccontextmanager
=======
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
import cv2
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from datetime import datetime
from typing import Dict, List
<<<<<<< HEAD
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
=======
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from motor.motor_asyncio import AsyncIOMotorClient

# Import Schemas và Utils
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.schemas import ControlCommand, ViolationPacket, HeartbeatPacket
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
<<<<<<< HEAD
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://admin:admin123@cluster0.iipaqpd.mongodb.net/?appName=Cluster0")
=======
MONGO_URI = "mongodb+srv://admin:admin123@cluster0.teleibk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913

os.makedirs(VIOLATION_DIR, exist_ok=True)

# ====================== INITIALIZATION ======================
<<<<<<< HEAD
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng: Khởi tạo và Giải phóng tài nguyên"""
    global loop, mongodb_connected
    # Lấy CHÍNH XÁC Event Loop đang sống của Uvicorn
    loop = asyncio.get_running_loop() 

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
=======
app = FastAPI(title="AI Traffic Monitoring Server")
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
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
<<<<<<< HEAD
client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client.traffic_db
violations_col = db.violations
mongodb_connected = False
=======
client = AsyncIOMotorClient(MONGO_URI)
db = client.traffic_db
violations_col = db.violations
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913

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
                    "points": data.get("points", [])
                })
        except Exception as e:
<<<<<<< HEAD
            print(f"[SERVER] Lỗi parse ROI preview: {e}")
=======
            print(f"[SERVER] Loi parse ROI preview: {e}")
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913

    elif "violation/" in topic:
        data = json.loads(payload)
        await process_violation(data)

async def process_violation(data: dict):
    """Xử lý OCR và lưu trữ vi phạm"""
    try:
        packet = ViolationPacket(**data)
        
        # Decode ảnh xe từ Edge
        img_bytes = base64.b64decode(packet.vehicle_crop_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Plate Detection + OCR tại Server
        res = plate_model(frame, verbose=False)[0]
        plate_text = "UNKNOWN"
        plate_img_b64 = None

        if len(res.boxes) > 0:
            box = res.boxes[0].xyxy[0].cpu().numpy().astype(int)
            # Gọi hàm OCR từ module_utils.py
            text, success = read_license_plate_vn(frame, box[0], box[1], box[2], box[3])
            if success:
                plate_text = text
                # Cắt ảnh biển số để lưu
                p_crop = frame[box[1]:box[3], box[0]:box[2]]
                _, p_buf = cv2.imencode(".jpg", p_crop)
                plate_img_b64 = base64.b64encode(p_buf).decode()

        # Tra cứu chủ xe
        clean_plate = re.sub(r"[^A-Z0-9]", "", plate_text.upper())
        owner_info = vehicle_db.get(clean_plate, {})

        # Lưu file cục bộ
        file_name = f"{packet.camera_id}_{packet.track_id}_{packet.timestamp.replace(':', '-')}.jpg" # Sửa lại tên file tránh lỗi ký tự ":" trên Windows
        cv2.imwrite(os.path.join(VIOLATION_DIR, file_name), frame)

        # Chuẩn bị dữ liệu cuối cùng
        violation_doc = {
            **packet.dict(),
            "plate_read": plate_text,
            "owner": owner_info.get("owner", "Không xác định"),
            "phone": owner_info.get("phone", "N/A"),
            "class_vehicle": owner_info.get("class_vehicle", "N/A"),
            "province": owner_info.get("province", "N/A"),
            "registration_date": owner_info.get("registration_date", "N/A"),
            "id_card": owner_info.get("id_card", "N/A"),
            "plate_img_base64": plate_img_b64,
            "processed_at": datetime.now().isoformat()
        }

<<<<<<< HEAD
        # Lưu MongoDB Atlas với retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await violations_col.insert_one(violation_doc.copy())
                print(f"[SERVER] ✅ Violation saved to MongoDB: {result.inserted_id}")
                break
            except Exception as mongo_err:
                print(f"[SERVER] MongoDB insert attempt {attempt+1} failed: {mongo_err}")
                if attempt == max_retries - 1:
                    # Backup to local JSON if MongoDB fails
                    backup_path = os.path.join(VIOLATION_DIR, f"backup_{file_name.replace('.jpg', '.json')}")
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(violation_doc, f, ensure_ascii=False)
                    print(f"[SERVER] ⚠️ Backup saved to local: {backup_path}")
=======
        # Lưu MongoDB Atlas
        result = await violations_col.insert_one(violation_doc.copy())
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913

        # Đẩy qua WebSocket lên UI (Phải bỏ ObjectId đi vì JSON không serialize được)
        if "_id" in violation_doc:
            del violation_doc["_id"]
            
        for ws in active_ws:
            await ws.send_json({"type": "violation", "data": violation_doc})

    except Exception as e:
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

<<<<<<< HEAD
=======
# Sử dụng Lifespan hook của FastAPI để đồng bộ Event Loop
@app.on_event("startup")
async def startup_event():
    global loop
    # Lấy CHÍNH XÁC Event Loop đang sống của Uvicorn
    loop = asyncio.get_running_loop() 
    
    # Bắt đầu kết nối MQTT sau khi Web Server đã chạy
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    print("[SERVER] Khởi động luồng MQTT nền thành công.")

>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
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

<<<<<<< HEAD
@app.post("/api/upload_video/{camera_id}")
async def upload_video(camera_id: str, video: UploadFile = File(...), processing_time_seconds: float = Form(...)):
    save_dir = os.path.join(BASE_DIR, "static", "videos")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, video.filename)
    
    with open(file_path, "wb") as f:
        f.write(await video.read())
        
    print(f"[SERVER] ✅ Nhận thành công Video Local từ Camera {camera_id}: {video.filename}")
    
    # Broadcast tới Dashboard
    for ws in active_ws:
        await ws.send_json({
            "type": "video_ready",
            "video_url": f"/static/videos/{video.filename}",
            "processing_time": processing_time_seconds
        })
        
    return {"status": "success", "file": video.filename}

=======
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
import csv
import io
from fastapi.responses import StreamingResponse

@app.get("/api/export_violations_csv")
async def export_violations_csv():
    # Query all violations from MongoDB
    cursor = violations_col.find({})
    violations = await cursor.to_list(length=None)
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
<<<<<<< HEAD
    # Define headers based on actual schema
    headers = [
        "timestamp", "camera_id", "violation_type", "plate_read",
        "owner", "phone", "province", "confidence"
=======
    # Define headers based on schema
    headers = [
        "timestamp", "camera_id", "violation_type", "license_plate",
        "owner_name", "phone", "address", "confidence"
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
    ]
    writer.writerow(headers)
    
    for v in violations:
        writer.writerow([
            v.get("timestamp", ""),
            v.get("camera_id", ""),
            v.get("violation_type", ""),
<<<<<<< HEAD
            v.get("plate_read", ""),
            v.get("owner", ""),
            v.get("phone", ""),
            v.get("province", ""),
=======
            v.get("license_plate", ""),
            v.get("owner_name", ""),
            v.get("phone", ""),
            v.get("address", ""),
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
            v.get("confidence", "")
        ])
        
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=violations.csv"}
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_ws.append(websocket)
<<<<<<< HEAD
    print(f"[SERVER] ✅ WebSocket client connected. Total: {len(active_ws)}")
=======
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
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
<<<<<<< HEAD
        print(f"[SERVER] ❌ WebSocket client disconnected. Total: {len(active_ws)}")
=======
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

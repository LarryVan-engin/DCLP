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
import cv2
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from datetime import datetime
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
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
MQTT_BROKER = "broker.hivemq.com" # Hoặc Cloud Broker của bạn
MQTT_PORT = 1883
MQTT_CLIENT_ID = "TRAFFIC_SERVER_01"

# MongoDB Settings (Atlas)
MONGO_URI = "mongodb+srv://admin:admin123@cluster0.teleibk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

os.makedirs(VIOLATION_DIR, exist_ok=True)

# ====================== INITIALIZATION ======================
app = FastAPI(title="AI Traffic Monitoring Server")
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
last_heartbeat = {}
current_stream_frame = "" # Base64 frame để đẩy lên UI

# ====================== MONGODB CONNECTION ======================
client = AsyncIOMotorClient(MONGO_URI)
db = client.traffic_db
violations_col = db.violations

# ====================== MQTT LOGIC ======================
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}")
    # Subscribe tất cả các camera
    client.subscribe("status/+/heartbeat")
    client.subscribe("stream/+/mjpeg")
    client.subscribe("violation/+")

def on_message(client, userdata, msg):
    asyncio.run_coroutine_threadsafe(handle_mqtt_message(msg), loop)

async def handle_mqtt_message(msg):
    global current_stream_frame
    topic = msg.topic
    payload = msg.payload.decode()

    # Xử lý Stream (MJPEG)
    if "stream/" in topic:
        current_stream_frame = payload 
    
    # Xử lý Heartbeat/Stats
    elif "status/" in topic and "heartbeat" in topic:
        data = json.loads(payload)
        last_heartbeat[data['camera_id']] = data

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

        # Lưu MongoDB Atlas
        result = await violations_col.insert_one(violation_doc.copy())

        # Đẩy qua WebSocket lên UI (Phải bỏ ObjectId đi vì JSON không serialize được)
        if "_id" in violation_doc:
            del violation_doc["_id"]
            
        for ws in active_ws:
            await ws.send_json({"type": "violation", "data": violation_doc})

    except Exception as e:
        print(f"[ERROR] Process violation failed: {e}")

# Khởi chạy MQTT trong background
mqtt_client = mqtt.Client(MQTT_CLIENT_ID)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()
loop = asyncio.get_event_loop()

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
async def control_edge(cmd: ControlCommand, camera_id: str):
    """Gửi lệnh MQTT xuống Edge"""
    topic = f"control/{camera_id}/command"
    mqtt_client.publish(topic, cmd.json())
    return {"status": "ok", "message": f"Command sent to {camera_id}"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_ws.append(websocket)
    try:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
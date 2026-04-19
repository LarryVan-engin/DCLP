"""
********************************************************************************************************************
Project:      Traffic Violation Detection (Pro Version - MQTT Hybrid)
File:         edge/main_edge.py
Description:  MQTT Client trên Jetson Nano. Chạy AI Inference cục bộ, gửi dữ liệu realtime qua MQTT.
********************************************************************************************************************
"""

import cv2
import json
import base64
import time
import os
import sys
import threading
import paho.mqtt.client as mqtt
# Import Utils & Config
import edge_config as cfg

from utils.capture_utils import smart_crop, encode_for_mqtt
from utils.violation_engine import ViolationEngine
violation_engine = ViolationEngine()
from utils.lane_detection import LaneDetector
lane_detector = LaneDetector() 

from ultralytics import YOLO
from collections import defaultdict

# Thêm đường dẫn để import shared schemas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))
from shared.schemas import ZoneDefinition

# ====================== GLOBAL STATE & INIT ======================
is_running = False
current_mode = "realtime"
active_video = None
zones_config = {}
current_light = {"unknown"}


print("[EDGE] Đang load model YOLOv12n...")
model_vehicle = YOLO(cfg.YOLO_VEHICLE_MODEL) 
model_traffic_light = YOLO(cfg.YOLO_LIGHT_MODEL) 

# ====================== HELPERS ======================
def publish_violation(client, track_id, violation_type, vehicle_crop, conf):
    """Đóng gói và gửi ViolationPacket"""
    packet = {
        "camera_id": cfg.CAMERA_ID,
        "mode": current_mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "track_id": int(track_id),
        "violation_type": violation_type,
        "lane": 1,
        "direction": "straight",
        "confidence": float(conf),
        # Dùng hàm encode_for_mqtt từ capture_utils đã viết
        "vehicle_crop_base64": encode_for_mqtt(vehicle_crop, quality=cfg.JPEG_ENCODE_QUALITY) 
    }
    client.publish(cfg.TOPIC_VIOLATION, json.dumps(packet), qos=1)
    print(f"[EDGE] Đã gửi vi phạm: ID {track_id} | {violation_type}")

# ====================== AI WORKER THREAD ======================
def ai_processing_loop(client):
    global is_running
    
    # 1. Xác định nguồn Video
    if current_mode == "video" and active_video:
        source = os.path.join(cfg.VIDEOS_DIR, active_video)
    else:
        source = 0 # Camera thực tế
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[EDGE] Không thể mở nguồn video: {source}")
        is_running = False
        return

    print(f"[EDGE] Bắt đầu xử lý luồng: Mode={current_mode}, Source={source}")
    
    frame_count = 0
    start_time = time.time()
    total_violations = 0
    track_history = defaultdict(list)

    while is_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[EDGE] 🎬 Video kết thúc.")
            break
            
        frame_count += 1
        
        current_light = parse_light_status(frame, model_traffic_light)
        
        # 2. AI Inference & Tracking
        results = model_vehicle.track(frame, persist=True, tracker=cfg.TRACKER_CONFIG, verbose=False)[0]
        
        # Xác định chế độ vận hành dựa trên zones_config
        # Nếu có bất kỳ polygon nào mang nhãn "forbidden", ta bật Forbidden Mode
        forbidden_zones = [z for z in zones_config.get("polygons", []) if z.label == "forbidden"]
        is_forbidden_mode = len(forbidden_zones) > 0

        cars = motorcycles = 0

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            if not lane_detector.is_ready():
                lane_detector.update_learning_data(boxes, classes)

            for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                
                # Đếm xe
                if int(cls) == 2: cars += 1
                elif int(cls) == 3: motorcycles += 1
                
                # 1. DUY TRÌ LỊCH SỬ QUỸ ĐẠO (TRAJECTORY)
                center_x = int((x1 + x2) / 2)
                bottom_y = int(y2)

                track_history[track_id].append((center_x, bottom_y))
                if len(track_history[track_id]) > cfg.MAX_TRACK_HISTORY:
                    track_history[track_id].pop(0)
                    
                path = track_history[track_id]
                
                new_violations_list = [] # Khởi tạo danh sách lỗi trống cho mỗi xe
                
                if is_forbidden_mode:
                    # ==========================================
                    # CHẾ ĐỘ ĐƯỜNG CẤM (FORBIDDEN MODE)
                    # ==========================================
                    # Chỉ check lỗi đi vào vùng cấm (Forbidden)
                    new_errors = violation_engine.check_violations(
                        track_id=track_id, bbox=[x1, y1, x2, y2], 
                        trajectory=path, light_status=current_light, 
                        zones_config={"polygons": forbidden_zones} # Chỉ truyền vùng cấm
                    )
                    # Lọc đúng lỗi đường cấm
                    new_violations_list = [e for e in new_errors if "ĐƯỜNG CẤM" in e]
                
                else:
                    # ==========================================
                    # CHẾ ĐỘ ĐƯỜNG BÌNH THƯỜNG (NORMAL MODE)
                    # ==========================================
                    # 1. Học phân làn tự động
                    if not lane_detector.is_ready:
                        lane_detector.update_learning_data(boxes, classes)
                        
                    # 2. Check lỗi luật (Vượt đèn, ngược chiều...) từ Engine
                    new_violations_list = violation_engine.check_violations(
                        track_id=track_id, bbox=[x1, y1, x2, y2], 
                        trajectory=path, light_status=current_light, 
                        zones_config=zones_config
                    )

                # --- LOGIC VI PHẠM (Chỉ chạy ở Mode Video) ---
                if current_mode == "video":
                    # Trả về list các lỗi vi phạm (có thể nhiều lỗi cùng lúc)
                    new_violations_list = violation_engine.check_violations(
                        track_id = track_id,
                        bbox = [x1, y1, x2, y2],
                        trajectory = path,
                        light_status = {"straight": "red"},
                        zones_config = zones_config
                    )
                    # ========================================================
                    #  CHECK THÊM SAI LÀN AI TỰ ĐỘNG
                    # ========================================================
                    if lane_detector.is_ready:
                        if lane_detector.check_wrong_lane([x1, y1, x2, y2], cls):
                            if "SAI LÀN" not in violation_engine.recorded_violations.get([track_id]):
                                new_violations_list.append("SAI LÀN")
                                violation_engine.recorded_violations.setdefault([track_id]).append("SAI LÀN")
                    
                    # Nếu có lỗi -> Smart Capture & Publish MQTT
                    if len(new_violations_list) > 0:
                        combo_violation_string = "+".join(new_violations_list)

                        crop_img = smart_crop(frame, box, padding=cfg.SMART_CROP_PADDING)
                        publish_violation(client, track_id, combo_violation_string, crop_img, conf)
                        total_violations += 1

                # 2. VẼ UI (BOUNDING BOX & TRAJECTORY)
                is_violating = len(violation_engine.recorded_violations[track_id]) > 0
                color = (0, 0, 255) if is_violating else (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                for i in range(1, len(path)):
                    cv2.line(frame, path[i-1], path[i], color, 2)

        # 3. Stream Frame realtime
        if frame_count % 3 == 0:
            # Dùng cv2.imencode trả về base64 tự viết lại nhanh ở đây để giảm phụ thuộc
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.STREAM_JPEG_QUALITY]
            _, buffer = cv2.imencode('.jpg', cv2.resize(frame, cfg.STREAM_RESOLUTION), encode_param)
            stream_b64 = base64.b64encode(buffer).decode('utf-8')
            client.publish(cfg.TOPIC_STREAM, stream_b64, qos=0)

        # 4. Gửi Heartbeat & Stats mỗi 30 frame
        if frame_count % 30 == 0:
            current_fps = frame_count / (time.time() - start_time)
            heartbeat = {
                "camera_id": cfg.CAMERA_ID,
                "stats": {"car": cars, "motorcycle": motorcycles, "bus": 0, "truck": 0},
                "lights": {"left": "green", "straight": "red"},
                "fps": round(current_fps, 1),
                "active_video": active_video
            }
            client.publish(cfg.TOPIC_STATUS, json.dumps(heartbeat), qos=0)

    # Dọn dẹp khi kết thúc
    cap.release()
    is_running = False
    
    # 5. Gửi gói Complete khi xử lý xong (Chỉ ở mode Video)
    if current_mode == "video":
        complete_pkt = {
            "camera_id": cfg.CAMERA_ID,
            "video_name": active_video,
            "total_violations": total_violations,
            "processing_time_seconds": round(time.time() - start_time, 2),
            "status": "success"
        }
        client.publish(cfg.TOPIC_COMPLETE, json.dumps(complete_pkt), qos=1)
        print(f"[EDGE] Đã gửi gói Complete. Tổng lỗi: {total_violations}")

# ====================== MQTT CALLBACKS ======================
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Đã kết nối Broker với mã: {rc}")
    client.subscribe(cfg.TOPIC_CMD)
    print(f"[MQTT] Lắng nghe lệnh điều khiển tại: {cfg.TOPIC_CMD}")

def on_message(client, userdata, msg):
    global is_running, current_mode, active_video, zones_config
    
    if msg.topic == cfg.TOPIC_CMD:
        try:
            cmd = json.loads(msg.payload.decode())
            action = cmd.get("action")
            print(f"[MQTT] Nhận lệnh điều khiển: {action}")

            if action == "list_files":
                files = [f for f in os.listdir(cfg.VIDEOS_DIR) if f.endswith(('.mp4', '.avi'))]
                client.publish(f"status/{CAMERA_ID}/files", json.dumps({"files": files}))

            if action == "start":
                if not is_running:
                    current_mode = cmd.get("mode", "realtime")
                    active_video = cmd.get("video_name")
                    
                    # Sửa lỗi 3: Parse Dict sang ZoneDefinition Pydantic Schema
                    zones_config = {
                        "lines": [ZoneDefinition(**z) for z in cmd.get("lines", [])],
                        "polygons": [ZoneDefinition(**z) for z in cmd.get("polygons", [])]
                    }
                    
                    # Sửa lỗi 4: Reset engine cho video mới
                    violation_engine.reset()
                    
                    is_running = True
                    threading.Thread(target=ai_processing_loop, args=(client,), daemon=True).start()
                else:
                    print("[EDGE] Luồng AI đang chạy rồi!")
                    
            elif action == "stop":
                is_running = False
                print("[EDGE] Đang dừng xử lý...")
                
        except Exception as e:
            print(f"[EDGE] Lỗi phân tích lệnh điều khiển: {e}")

# ====================== MAIN ======================
if __name__ == "__main__":
    os.makedirs(cfg.VIDEOS_DIR, exist_ok=True)
    
    client = mqtt.Client(cfg.MQTT_CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message
    
    print("[EDGE] Đang kết nối đến MQTT Broker...")
    client.connect(cfg.MQTT_BROKER, cfg.MQTT_PORT, cfg.MQTT_KEEPALIVE)
    
    client.loop_forever()
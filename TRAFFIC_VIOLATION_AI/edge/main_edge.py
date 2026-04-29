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
import numpy as np
np.bool = bool
np.float = float
np.int = int

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
current_light = "unknown"
stop_reason = None
roi_confirmed = False

# ROI Polygon (sẽ được cập nhật từ config hoặc mặc định)
roi_polygon = np.array(cfg.DEFAULT_ROI_PTS, np.int32)

print("[EDGE] Đang load model YOLOv12n...")
model_vehicle = YOLO(cfg.YOLO_VEHICLE_MODEL) 
model_traffic_light = YOLO(cfg.YOLO_LIGHT_MODEL) 

# ====================== HELPER: VẼ CÁC LÀN ĐƯỜNG (GIỐNG full_main.py) ======================
def draw_lanes_on_frame(frame, lane_detector):
    """
    Vẽ các làn đường đã học lên frame.
    Giống logic trong full_main.py: calculate_data_driven_lanes()
    """
    if not lane_detector.is_ready:
        return frame
        
    display = frame.copy()
    overlay = display.copy()
    
    # Lấy các thông số từ lane_detector
    car_only_zones = lane_detector.car_only_zones
    roi_pts = lane_detector.roi_pts
    
    if not car_only_zones:
        return frame
        
    # Tính toán các đường biên làn để vẽ (giống thuật toán trong full_main.py)
    top_y = roi_pts[0][1]
    bot_y = roi_pts[2][1]
    
    virtual_lines = getattr(lane_detector, "virtual_lines", [])

    # Vẽ từng làn đường
    for idx, (z_start, z_end) in enumerate(car_only_zones):
        # Tính tọa độ các điểm của hình thang làn
        top_l_x = int(roi_pts[0][0] + z_start * (roi_pts[1][0] - roi_pts[0][0]))
        bot_l_x = int(roi_pts[3][0] + z_start * (roi_pts[2][0] - roi_pts[3][0]))
        top_r_x = int(roi_pts[0][0] + z_end * (roi_pts[1][0] - roi_pts[0][0]))
        bot_r_x = int(roi_pts[3][0] + z_end * (roi_pts[2][0] - roi_pts[3][0]))
        
        lane_poly = np.array([[top_l_x, top_y], [top_r_x, top_y], [bot_r_x, bot_y], [bot_l_x, bot_y]], np.int32)
        
        # Màu sắc: Vùng cấm xe máy (Car Only) màu cam
        color = (0, 100, 255)  # BGR - Orange
        
        # Fill làn đường
        cv2.fillPoly(overlay, [lane_poly], color)
        
        # Thêm text label
        M = cv2.moments(lane_poly)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(display, f"CAR ONLY LANE {idx+1}", (cx - 70, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Blend với frame gốc (giống full_main.py: LANE_ALPHA = 0.20)
    display = cv2.addWeighted(overlay, cfg.LANE_ALPHA, display, 1 - cfg.LANE_ALPHA, 0, display)

    for vx1, vy1, vx2, vy2 in virtual_lines:
        for i in range(0, 10):
            start = (
                int(vx1 + (vx2 - vx1) * (i / 10.0)),
                int(vy1 + (vy2 - vy1) * (i / 10.0))
            )
            end = (
                int(vx1 + (vx2 - vx1) * ((i + 0.5) / 10.0)),
                int(vy1 + (vy2 - vy1) * ((i + 0.5) / 10.0))
            )
            cv2.line(display, start, end, (255, 255, 255), 3)
    
    return display


def draw_roi_on_frame(frame, roi_pts):
    """
    Vẽ khung ROI lên frame.
    """
    display = frame.copy()
    roi_poly = np.array(roi_pts, np.int32)
    
    # Vẽ đường viền ROI màu vàng
    cv2.polylines(display, [roi_poly], True, (0, 255, 255), 3)
    
    # Vẽ đường dừng (stop line) màu đỏ ở đỉnh ROI
    cv2.line(display, tuple(roi_poly[0]), tuple(roi_poly[1]), (0, 0, 255), 4)
    cv2.putText(display, "AUTO STOP LINE", (roi_poly[0][0] + 10, roi_poly[0][1] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return display

# ====================== HELPERS ======================
def parse_light_status(frame, light_model):
    """
    Hàm nhận diện màu đèn giao thông.
    Tối ưu Jetson Nano: Chỉ crop nửa trên màn hình (Top-half) để tiết kiệm 50% sức mạnh tính toán.
    """
    h, w = frame.shape[:2]
    # Cắt nửa trên của frame
    top_half = frame[0:int(h/2), 0:w]
    
    # Chạy AI với ngưỡng tự động lấy từ edge_config.py
    results = light_model(top_half, conf=cfg.CONF_TRAFFIC_LIGHT, verbose=False)[0]
    
    # Nếu không thấy đèn nào
    if results.boxes is None or len(results.boxes) == 0:
        return "unknown"
        
    best_conf = 0
    best_color = "unknown"
    
    # Lặp qua các đèn tìm thấy để lấy đèn rõ nhất
    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        
        if conf > best_conf:
            best_conf = conf
            # Lấy tên class (VD: 'Red', 'Green', 'Yellow') và chuyển về chữ thường
            label = results.names[cls_id].lower() 
            
            # Map kết quả
            if "red" in label or "do" in label or "đỏ" in label:
                best_color = "red"
            elif "green" in label or "xanh" in label:
                best_color = "green"
            elif "yellow" in label or "vang" in label or "vàng" in label:
                best_color = "yellow"
                
    return best_color

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

def get_active_source(mode=None, video_name=None):
    mode = mode or current_mode
    video_name = video_name or active_video
    if mode == "video" and video_name:
        return os.path.join(cfg.VIDEOS_DIR, video_name)
    return 0

def get_source_size(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg.FRAME_WIDTH
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg.FRAME_HEIGHT
    cap.release()
    return width, height

def scale_default_roi(width, height):
    sx = width / float(cfg.FRAME_WIDTH)
    sy = height / float(cfg.FRAME_HEIGHT)
    return [[int(x * sx), int(y * sy)] for x, y in cfg.DEFAULT_ROI_PTS]

def normalize_roi_points(points, width, height):
    normalized = []
    for x, y in points:
        normalized.append({
            "x": round(float(x) / max(width, 1), 4),
            "y": round(float(y) / max(height, 1), 4)
        })
    return normalized

def denormalize_points(points, width, height):
    denormalized = []
    for p in points:
        if isinstance(p, dict):
            x, y = float(p.get("x", 0)), float(p.get("y", 0))
        else:
            x, y = float(p[0]), float(p[1])

        if 0 <= x <= 1 and 0 <= y <= 1:
            denormalized.append([int(x * width), int(y * height)])
        else:
            denormalized.append([int(x), int(y)])
    return denormalized

def build_zones_config(cmd, width, height):
    def convert_zone(zone):
        zone_copy = dict(zone)
        zone_copy["points"] = [
            {"x": x, "y": y}
            for x, y in denormalize_points(zone_copy.get("points", []), width, height)
        ]
        return ZoneDefinition(**zone_copy)

    return {
        "lines": [convert_zone(z) for z in cmd.get("lines", [])],
        "polygons": [convert_zone(z) for z in cmd.get("polygons", [])]
    }

def publish_video_roi_preview(client, video_name):
    global roi_polygon, active_video

    if not video_name:
        print("[EDGE] Không có video để preview ROI.")
        return

    source = get_active_source("video", video_name)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[EDGE] Không thể mở Video preview: {source}")
        return

    frame = None
    for _ in range(5):
        ret, next_frame = cap.read()
        if not ret:
            break
        frame = next_frame
    cap.release()

    if frame is None:
        print(f"[EDGE] Không đọc được frame preview: {source}")
        return

    height, width = frame.shape[:2]
    active_video = video_name
    roi_polygon = np.array(scale_default_roi(width, height), np.int32)
    lane_detector.set_roi(roi_polygon.tolist())

    preview_frame = draw_roi_on_frame(frame, roi_polygon.tolist())
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.STREAM_JPEG_QUALITY]
    _, buffer = cv2.imencode('.jpg', cv2.resize(preview_frame, cfg.STREAM_RESOLUTION), encode_param)
    client.publish(cfg.TOPIC_STREAM, base64.b64encode(buffer).decode('utf-8'), qos=0)

    client.publish(
        f"status/{cfg.CAMERA_ID}/roi_preview",
        json.dumps({
            "camera_id": cfg.CAMERA_ID,
            "video_name": video_name,
            "points": normalize_roi_points(roi_polygon.tolist(), width, height)
        }),
        qos=0
    )
    print(f"[EDGE] Đã gửi Preview ROI của Video: {video_name}")

# ====================== AI WORKER THREAD ======================
def ai_processing_loop(client):
    global is_running, roi_polygon, stop_reason, current_light
    
    # 1. Xác định nguồn Video
    source = get_active_source()
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[EDGE] Không thể mở nguồn video: {source}")
        is_running = False
        return

    print(f"[EDGE] Bắt đầu xử lý luồng: Mode={current_mode}, Source={source}")
    
    # KHỞI TẠO VIDEO WRITER
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg.FRAME_WIDTH
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg.FRAME_HEIGHT
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    video_basename = os.path.basename(str(source)) if current_mode == 'video' else 'realtime_output.mp4'
    output_filename = f"processed_{int(time.time())}_{video_basename}"
    output_path = os.path.join(cfg.VIDEOS_DIR, output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print(f"[EDGE] Đang lưu video xử lý tại: {output_path}")
    
    # KHỞI TẠO ROI CHO LANE DETECTOR (giống full_main.py)
    lane_detector.set_roi(roi_polygon.tolist())
    roi_top_y = roi_polygon[0][1]  # Lấy đường stop line từ ROI
    
    frame_count = 0
    start_time = time.time()
    total_violations = 0
    track_history = defaultdict(list)
    completed_naturally = False

    while is_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[EDGE] 🎬 Video kết thúc.")
            completed_naturally = True
            break
            
        frame_count += 1

        # HIỂN THỊ TRẠNG THÁI HỌC LÀN (giống full_main.py)
        if not lane_detector.is_ready:
            # Vẽ overlay thông báo đang học
            overlay = frame.copy()
            cv2.putText(overlay, f"LEARNING DATA-DRIVEN LANES... {frame_count}/{cfg.LANE_LEARNING_FRAMES}", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
            frame = overlay

        if frame_count % 3 == 1:
            new_light = parse_light_status(frame, model_traffic_light)
            if new_light != "unknown":
                current_light = new_light
        
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

            # Cập nhật dữ liệu học làn (giống full_main.py)
            if not lane_detector.is_ready:
                lane_detector.update_learning_data(boxes, classes)

            for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                
                # KIỂM TRA XE CÓ TRONG ROI KHÔNG (giống full_main.py)
                # Lấy tâm điểm bánh xe dưới
                center_x = int((x1 + x2) / 2)
                bottom_y = int(y2)
                
                # Kiểm tra xe có nằm trong ROI không (cv2.pointPolygonTest)
                if cv2.pointPolygonTest(roi_polygon, (center_x, bottom_y), False) < 0:
                    continue  # Bỏ qua xe ngoài ROI
                
                # Đếm xe
                if int(cls) == 2: cars += 1
                elif int(cls) == 3: motorcycles += 1
                
                # 1. DUY TRÌ LỊCH SỬ QUỸ ĐẠO (TRAJECTORY)
                track_history[track_id].append((center_x, bottom_y))
                if len(track_history[track_id]) > cfg.MAX_TRACK_HISTORY:
                    track_history[track_id].pop(0)
                    
                path = track_history[track_id]
                
                new_violations_list = [] # Khởi tạo danh sách lỗi trống cho mỗi xe
                
                # --- LOGIC VI PHẠM (Chỉ xử phạt ở Mode Video hoặc có yêu cầu) ---
                if current_mode == "video":
                    if is_forbidden_mode:
                        # ==========================================
                        # CHẾ ĐỘ ĐƯỜNG CẤM (FORBIDDEN MODE)
                        # ==========================================
                        new_errors = violation_engine.check_violations(
                            track_id=track_id, bbox=[x1, y1, x2, y2], 
                            trajectory=path, light_status={"straight": current_light}, 
                            zones_config={"polygons": forbidden_zones} 
                        )
                        new_violations_list = [e for e in new_errors if "ĐƯỜNG CẤM" in e]
                    
                    else:
                        # ==========================================
                        # CHẾ ĐỘ ĐƯỜNG BÌNH THƯỜNG (NORMAL MODE)
                        # ==========================================
                        if not lane_detector.is_ready: # Sửa lỗi gọi hàm ()
                            lane_detector.update_learning_data(boxes, classes)
                            
                        new_violations_list = violation_engine.check_violations(
                            track_id=track_id, bbox=[x1, y1, x2, y2], 
                            trajectory=path, light_status={"straight": current_light}, 
                            zones_config=zones_config
                        )
                        
                        # CHECK THÊM SAI LÀN AI TỰ ĐỘNG
                        if lane_detector.is_ready:
                            if lane_detector.check_wrong_lane([x1, y1, x2, y2], cls):
                                # Sửa lỗi Hashable List và dùng hàm .add()
                                if "SAI LÀN" not in violation_engine.recorded_violations[track_id]:
                                    new_violations_list.append("SAI LÀN")
                                    violation_engine.recorded_violations[track_id].add("SAI LÀN")
                        
                        # ==========================================
                        # KIỂM TRA VƯỢT ĐÈN DỰA TRÊN ROI (giống full_main.py)
                        # ==========================================
                        # Logic: Nếu xe đi qua đường stop line (roi_top_y) khi đèn đỏ/vàng
                        center_y = (y1 + y2) // 2
                        
                        # Kiểm tra xe vừa đi qua đường stop line (từ dưới lên trên)
                        if len(path) >= 2:
                            prev_y = path[-2][1]
                            curr_y = path[-1][1]
                            
                            # Xe đi từ dưới lên qua stop line
                            if prev_y >= roi_top_y and curr_y < roi_top_y:
                                if current_light == "red":
                                    if "VƯỢT ĐÈN ĐỎ" not in violation_engine.recorded_violations[track_id]:
                                        new_violations_list.append("VƯỢT ĐÈN ĐỎ")
                                        violation_engine.recorded_violations[track_id].add("VƯỢT ĐÈN ĐỎ")
                                elif current_light == "yellow":
                                    if "VƯỢT ĐÈN VÀNG" not in violation_engine.recorded_violations[track_id]:
                                        new_violations_list.append("VƯỢT ĐÈN VÀNG")
                                        violation_engine.recorded_violations[track_id].add("VƯỢT ĐÈN VÀNG")
                    
                    # NẾU CÓ LỖI -> SMART CAPTURE & PUBLISH MQTT
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

        # 2.5. VẼ ROI VÀ CÁC LÀN ĐƯỜNG (giống full_main.py)
        # Vẽ khung ROI trước
        frame = draw_roi_on_frame(frame, roi_polygon.tolist())
        
        # Nếu đã học xong làn, vẽ các làn đường lên frame
        if lane_detector.is_ready:
            frame = draw_lanes_on_frame(frame, lane_detector)

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
                "lights": {"left": "unknown", "straight": current_light},
                "fps": round(current_fps, 1),
                "active_video": active_video
            }
            client.publish(cfg.TOPIC_STATUS, json.dumps(heartbeat), qos=0)

    # Dọn dẹp khi kết thúc
    cap.release()
    out.release()
    is_running = False
    
    # 5. Gửi gói Complete khi xử lý xong (Chỉ ở mode Video)
    should_publish_complete = current_mode == "video" and completed_naturally
    if current_mode == "video" and stop_reason == "reset_roi":
        print("[EDGE] Đã dừng video để xử lý ROI.")
    stop_reason = None

    if should_publish_complete:
        complete_pkt = {
            "camera_id": cfg.CAMERA_ID,
            "video_name": active_video,
            "total_violations": total_violations,
            "processing_time_seconds": round(time.time() - start_time, 2),
            "status": "success"
        }
        client.publish(cfg.TOPIC_COMPLETE, json.dumps(complete_pkt), qos=1)
        print(f"[EDGE] Đã gửi gói Complete. Tổng lỗi: {total_violations}")

def send_idle_heartbeat(client):
    """Gửi heartbeat định kỳ khi Edge đang ở trạng thái chờ (Idle)"""
    def heartbeat_thread():
        while True:
            # Cứ 1 phut gửi 1 lần
            if not is_running: # Chỉ gửi khi AI đang KHÔNG chạy (vì lúc AI chạy đã có heartbeat riêng)
                heartbeat = {
                    "camera_id": cfg.CAMERA_ID,
                    "status": "idle",
                    "stats": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
                    "lights": {"left": "unknown", "straight": "unknown"},
                    "fps": 0,
                    "active_video": active_video
                }
                client.publish(cfg.TOPIC_STATUS, json.dumps(heartbeat), qos=0)
            print(f"[MQTT] Đã gửi message thông báo đến Server")
            time.sleep(60)   
    threading.Thread(target=heartbeat_thread, daemon=True).start()

# ====================== MQTT CALLBACKS ======================
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Đã kết nối Broker với mã: {rc}")
    client.subscribe(cfg.TOPIC_CMD)
    print(f"[MQTT] Lắng nghe lệnh điều khiển tại: {cfg.TOPIC_CMD}")

def on_message(client, userdata, msg):
    global is_running, current_mode, active_video, zones_config, roi_polygon, stop_reason, roi_confirmed
    
    if msg.topic == cfg.TOPIC_CMD:
        try:
            cmd = json.loads(msg.payload.decode())
            action = cmd.get("action")
            print(f"[MQTT] Nhận lệnh điều khiển: {action}")

            if action == "list_files":
                files = [f for f in os.listdir(cfg.VIDEOS_DIR) if f.endswith(('.mp4', '.avi'))]
                client.publish(f"status/{cfg.CAMERA_ID}/files", json.dumps({"files": files}))

            if action == "preview_video":
                was_running = is_running
                is_running = False
                stop_reason = "preview_video"
                roi_confirmed = False
                zones_config = {}
                violation_engine.reset()
                lane_detector.reset_learning()
                current_mode = "video"
                if was_running:
                    time.sleep(0.2)
                publish_video_roi_preview(client, cmd.get("video_name"))
                return

            if action in ("update_roi", "update_zones"):
                source = get_active_source(cmd.get("mode", current_mode), cmd.get("video_name") or active_video)
                frame_w, frame_h = get_source_size(source)
                roi_pts = cmd.get("roi")
                if not roi_pts:
                    polygons = cmd.get("polygons", [])
                    if polygons:
                        roi_pts = polygons[0].get("points", [])
                if roi_pts and len(roi_pts) == 4:
                    roi_pts = denormalize_points(roi_pts, frame_w, frame_h)
                    roi_polygon = np.array(roi_pts, np.int32)
                    lane_detector.set_roi(roi_pts)
                    roi_confirmed = True
                    print(f"[EDGE] Đã cập nhật ROI mới: {roi_pts}")
                zones_config = build_zones_config(cmd, frame_w, frame_h)
                return

            if action == "reset_roi":
                was_running = is_running
                is_running = False
                stop_reason = "reset_roi"
                roi_confirmed = False
                active_video = cmd.get("video_name") or active_video
                current_mode = cmd.get("mode", current_mode)
                if was_running:
                    time.sleep(0.2)
                source = get_active_source(cmd.get("mode", current_mode), cmd.get("video_name") or active_video)
                frame_w, frame_h = get_source_size(source)
                roi_pts = scale_default_roi(frame_w, frame_h)
                roi_polygon = np.array(roi_pts, np.int32)
                zones_config = {}
                violation_engine.reset()
                lane_detector.reset_learning()
                lane_detector.set_roi(roi_pts)
                if active_video:
                    publish_video_roi_preview(client, active_video)
                print("[EDGE] Đã dừng xử lý, reset ROI và đưa về mặc định.")
                return

            if action == "start":
                if not is_running:
                    current_mode = cmd.get("mode", "realtime")
                    active_video = cmd.get("video_name")
                    if current_mode == "video" and not cmd.get("roi"):
                        print("[EDGE] Preview frame, khong xu ly.")
                        publish_video_roi_preview(client, active_video)
                        return
                    source = get_active_source(current_mode, active_video)
                    frame_w, frame_h = get_source_size(source)

                    roi_pts = cmd.get("roi")
                    if roi_pts and len(roi_pts) == 4:
                        roi_pts = denormalize_points(roi_pts, frame_w, frame_h)
                    else:
                        roi_pts = scale_default_roi(frame_w, frame_h)
                    roi_polygon = np.array(roi_pts, np.int32)
                    lane_detector.set_roi(roi_pts)
                    roi_confirmed = True
                    print(f"[EDGE] Đã cấu hình ROI: {roi_pts}")

                    zones_config = build_zones_config(cmd, frame_w, frame_h)
                    violation_engine.reset()

                    is_running = True
                    threading.Thread(target=ai_processing_loop, args=(client,), daemon=True).start()
                else:
                    print("[EDGE] Luồng AI đang chạy")

            elif action == "stop":
                is_running = False
                stop_reason = "stop"
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
    send_idle_heartbeat(client)
    
    client.loop_forever()
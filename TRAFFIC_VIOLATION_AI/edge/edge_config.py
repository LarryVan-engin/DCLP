"""
********************************************************************************************************************
Project:      Traffic Violation Detection (Pro Version - MQTT Hybrid)
File:         edge/edge_config.py
Description:  File cấu hình tập trung cho Edge Node (Jetson Nano). 
              Quản lý định danh camera, thông số MQTT, đường dẫn Model và các Threshold của AI.
********************************************************************************************************************
"""

import os

# ==========================================
# 1. ĐỊNH DANH THIẾT BỊ (NODE IDENTITY)
# ==========================================
# Tên của Camera này. PHẢI LÀ DUY NHẤT cho mỗi Jetson Nano trong hệ thống.
CAMERA_ID = "JETSON_NANO_01"

# ==========================================
# 2. CẤU HÌNH KẾT NỐI MQTT
# ==========================================
# Khuyến nghị dùng IP tĩnh của Server Local hoặc HiveMQ Cloud
MQTT_BROKER = "broker.hivemq.com" 
MQTT_PORT = 1883
MQTT_CLIENT_ID = f"{CAMERA_ID}_CLIENT"
MQTT_KEEPALIVE = 60

# Các Topic giao tiếp (Tự động sinh theo CAMERA_ID)
TOPIC_CMD = f"control/{CAMERA_ID}/command"
TOPIC_STATUS = f"status/{CAMERA_ID}/heartbeat"
TOPIC_STREAM = f"stream/{CAMERA_ID}/mjpeg"
TOPIC_VIOLATION = f"violation/{CAMERA_ID}"
TOPIC_COMPLETE = f"complete/{CAMERA_ID}"

# ==========================================
# 3. ĐƯỜNG DẪN HỆ THỐNG (PATHS)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thư mục chứa Model TensorRT / PyTorch
MODELS_DIR = os.path.join(BASE_DIR, "models")
# Thư mục chứa Video test chạy nội bộ (Zero-Upload mode)
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

# Tên file Models
YOLO_VEHICLE_MODEL = os.path.join(MODELS_DIR, "yolo12n.pt") # Nên đổi thành .engine khi chạy thật
YOLO_LIGHT_MODEL = os.path.join(MODELS_DIR, "model_detect_traffic_light.pt")
TRACKER_CONFIG = "bytetrack.yaml" # Hoặc đường dẫn tuyệt đối đến file yaml

# ==========================================
# 4. THÔNG SỐ AI & XỬ LÝ ẢNH (AI HYPERPARAMETERS)
# ==========================================
# YOLO Confidence thresholds
CONF_VEHICLE = 0.35
CONF_TRAFFIC_LIGHT = 0.45

# Cấu hình Smart Capture
SMART_CROP_PADDING = 40        # Số pixel mở rộng khi cắt ảnh vi phạm
JPEG_ENCODE_QUALITY = 85       # Chất lượng nén ảnh gửi qua MQTT (1-100)
STREAM_JPEG_QUALITY = 50       # Chất lượng nén cho luồng Realtime (giảm để mượt)
STREAM_RESOLUTION = (640, 360) # Độ phân giải luồng Realtime đẩy lên Dashboard

# Cấu hình Violation Engine
MAX_TRACK_HISTORY = 30         # Số lượng frame lưu vết quỹ đạo cho mỗi xe
MIN_FRAMES_WRONG_WAY = 15      # Số frame tối thiểu để xác nhận đi ngược chiều
RED_LIGHT_WAIT_FRAMES = 15     # Số frame chờ để xác nhận đi thẳng hay rẽ phải
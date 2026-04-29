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
CAMERA_ID = "JETSON_01"

# ==========================================
# 2. CẤU HÌNH KẾT NỐI MQTT
# ==========================================
# Khuyến nghị dùng IP tĩnh của Server Local hoặc HiveMQ Cloud
MQTT_BROKER = "192.168.1.5" 
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
YOLO_VEHICLE_MODEL = os.path.join(MODELS_DIR, "yolo12n.engine") # Nên đổi thành .engine khi chạy thật
YOLO_LIGHT_MODEL = os.path.join(MODELS_DIR, "model_detect_traffic_light.engine")
TRACKER_CONFIG = os.path.join(MODELS_DIR, "bytetrack.yaml") # Hoặc đường dẫn tuyệt đối đến file yaml

# ==========================================
# 4. THÔNG SỐ AI & XỬ LÝ ẢNH (AI HYPERPARAMETERS)
# ==========================================
# YOLO Confidence thresholds
CONF_VEHICLE = 0.35
CONF_TRAFFIC_LIGHT = 0.45

# Cấu hình Smart Capture
SMART_CROP_PADDING = 40        # Số pixel mở rộng khi cắt ảnh vi phạm
JPEG_ENCODE_QUALITY = 98       # Chất lượng nén ảnh gửi qua MQTT (1-100)
STREAM_JPEG_QUALITY = 50       # Chất lượng nén cho luồng Realtime (giảm để mượt)
STREAM_RESOLUTION = (640, 360) # Độ phân giải luồng Realtime đẩy lên Dashboard

# Cấu hình Violation Engine
MAX_TRACK_HISTORY = 30         # Số lượng frame lưu vết quỹ đạo cho mỗi xe
MIN_FRAMES_WRONG_WAY = 15      # Số frame tối thiểu để xác nhận đi ngược chiều
RED_LIGHT_WAIT_FRAMES = 15     # Số frame chờ để xác nhận đi thẳng hay rẽ phải

# ==========================================
# 5. CẤU HÌNH ROI (Region of Interest) - Calibration
# ==========================================
# ROI mặc định: Hình thang perspective tương tự full_main.py
# Điểm: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
# Tỷ lệ: 80% chiều rộng đỉnh, 100% chiều rộng đáy
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

DEFAULT_ROI_PTS = [
    [int(FRAME_WIDTH * 0.1), int(FRAME_HEIGHT * 0.3)],   # Top-Left
    [int(FRAME_WIDTH * 0.9), int(FRAME_HEIGHT * 0.3)],   # Top-Right
    [int(FRAME_WIDTH * 1.0), int(FRAME_HEIGHT * 1.0)],   # Bottom-Right
    [int(FRAME_WIDTH * 0.0), int(FRAME_HEIGHT * 1.0)]    # Bottom-Left
]

# Số frame để học phân làn tự động (giống full_main.py: OBSERVATION_FRAMES = 100)
LANE_LEARNING_FRAMES = 100

# Alpha cho việc blend làn đường (giống full_main.py: LANE_ALPHA = 0.20)
LANE_ALPHA = 0.20

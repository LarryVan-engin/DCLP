"""
********************************************************************************************************************
Project:      Traffic Violation Detection (Pro Version - MQTT Hybrid)
File:         shared/schemas.py
Description:  Định nghĩa chuẩn dữ liệu (Pydantic Models) cho giao tiếp qua MQTT.
********************************************************************************************************************
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# ==========================================
# 1. CẤU TRÚC VÙNG QUAN SÁT (ZONES)
# ==========================================
class Point(BaseModel):
    x: float
    y: float

class ZoneDefinition(BaseModel):
    points: List[Point]
    label: str  # VD: 'straight', 'turn_left', 'forbidden'

# ==========================================
# 2. SCHEMA: LỆNH ĐIỀU KHIỂN (Server -> Edge qua MQTT: control/{camera_id}/command)
# ==========================================
class ControlCommand(BaseModel):
    action: str = Field(..., description="'start', 'stop', 'pause', 'update_zones', 'update_roi'")
    mode: str = Field(..., description="'realtime' hoặc 'video'")
    video_name: Optional[str] = Field(None, description="Tên file video local trên Jetson nếu chạy mode 'video'")
    roi: Optional[List[List[float]]] = Field(None, description="ROI polygon points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]")
    lines: List[ZoneDefinition] = []
    polygons: List[ZoneDefinition] = []
    light_zones: List[ZoneDefinition] = []

# ==========================================
# 3. SCHEMA: HEARTBEAT & STATS (Edge -> Server qua MQTT: status/{camera_id}/heartbeat)
# ==========================================
class VehicleStats(BaseModel):
    car: int = 0
    motorcycle: int = 0
    bus: int = 0
    truck: int = 0

class LightStatus(BaseModel):
    left: str = "unknown"
    straight: str = "unknown"

class HeartbeatPacket(BaseModel):
    camera_id: str
    stats: VehicleStats
    lights: LightStatus
    fps: float
    avg_inference_ms: Optional[float] = None  # Rolling avg YOLO vehicle inference (last 30 frames)
    active_video: Optional[str] = None

# ==========================================
# 4. SCHEMA: GÓI TIN VI PHẠM (Edge -> Server qua MQTT: violation/{camera_id})
# ==========================================
class ViolationPacket(BaseModel):
    camera_id: str
    mode: str
    timestamp: str
    track_id: int
    violation_type: str
    lane: Optional[int] = None
    direction: Optional[str] = None
    confidence: float
    vehicle_crop_base64: str                       # Ảnh crop khít (padding nhỏ, dùng để OCR)
    vehicle_crop_wide_base64: Optional[str] = ""   # Ảnh crop rộng 75px (bằng chứng ngữ cảnh)
    full_frame_a_base64: Optional[str] = ""        # Ảnh toàn khung SẠCH — thấy đèn + bối cảnh
    full_frame_b_base64: Optional[str] = ""        # Ảnh toàn khung CÓ CHÚ THÍCH — bbox + loại lỗi
    video_offset: Optional[float] = None           # Khoảng thời gian (giây) xảy ra lỗi tính từ đầu video
    video_name: Optional[str] = None              # Tên file video local (dùng để phân biệt session, tránh gộp nhầm lỗi giữa các lần chạy)

# ==========================================
# 5. SCHEMA: GÓI HOÀN THÀNH (Edge -> Server qua MQTT: complete/{camera_id})
# ==========================================
class CompletePacket(BaseModel):
    camera_id: str
    video_name: str
    total_violations: int
    processing_time_seconds: float
    annotated_video_path: Optional[str] = Field(None, description="Đường dẫn local lưu video đã render bboxes trên Edge")
    status: str = "success"
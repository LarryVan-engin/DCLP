"""
********************************************************************************************************************
Project:      Traffic Violation Detection (Pro Version - MQTT Hybrid)
File:         edge/main_edge.py
Description:  MQTT Client trên Jetson Nano. Chạy AI Inference cục bộ, gửi dữ liệu realtime qua MQTT.
********************************************************************************************************************
"""

import os
import sys
import ctypes

# [FIX TẬN GỐC CHO JETSON NANO]
# Tải libgomp.so.1 vào Global Scope trước khi import cv2 và PyTorch (YOLO).
# Điều này giúp cấp phát đủ bộ nhớ tĩnh (Static TLS block) cho cả PyTorch và GStreamer,
# khắc phục lỗi "cannot allocate memory in static TLS block" khiến libgstlibav (mp4v encoder) bị văng.
try:
    ctypes.CDLL("/usr/lib/aarch64-linux-gnu/libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

import cv2
import json
import base64
import time
import os
import sys
import threading
import queue
import requests
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
from shared.zones_utils import get_normalized_x

from ultralytics import YOLO
from collections import defaultdict

# Thêm đường dẫn để import shared schemas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from shared.schemas import ZoneDefinition

# ====================== GLOBAL STATE & INIT ======================
is_running = False
current_mode = "realtime"
active_video = None
zones_config = {}
current_light = "unknown"
stop_reason = None
roi_confirmed = False

# Ranh giới dưới vùng theo dõi rẽ phải (tỷ lệ [0-1], cập nhật từ Dashboard qua MQTT)
right_turn_zone_bottom_y_ratio = cfg.RIGHT_TURN_ZONE_BOTTOM_Y_RATIO

# ROI Polygon (sẽ được cập nhật từ config hoặc mặc định)
roi_polygon = np.array(cfg.DEFAULT_ROI_PTS, np.int32)

print("[EDGE] Đang load model YOLOv12n...")
model_vehicle = YOLO(cfg.YOLO_VEHICLE_MODEL)
model_traffic_light = YOLO(cfg.YOLO_LIGHT_MODEL)

# =================================================================
# WARM-UP: Chạy inference giả ngay sau khi load để CUDA/TensorRT
# khởi tạo execution context và cache CUDA kernels sẵn.
# → Khi user nhấn "Lưu" trên Dashboard, inference đầu tiên chạy ngay,
#   không mất thêm 5-15s cho lần khởi động context.
# Chạy trong background thread để không block kết nối MQTT.
# =================================================================
_models_ready = threading.Event()   # Set khi warm-up xong

def _warmup_worker():
    """Chạy 3 vòng inference giả để pre-heat CUDA kernel cache."""
    try:
        print("[EDGE] 🔥 Warm-up models (background)...")
        # Dùng ảnh đen — không cần ảnh thật, chỉ cần chạy đúng pipeline
        dummy_vehicle = np.zeros((cfg.FRAME_HEIGHT, cfg.FRAME_WIDTH, 3), dtype=np.uint8)
        dummy_light   = np.zeros((cfg.FRAME_HEIGHT // 2, cfg.FRAME_WIDTH // 2, 3), dtype=np.uint8)
        for _ in range(3):
            model_vehicle(dummy_vehicle, imgsz=640, verbose=False)
        for _ in range(3):
            model_traffic_light(dummy_light, imgsz=640, verbose=False)
        print("[EDGE] ✅ Warm-up hoàn tất — inference sẵn sàng chạy ngay lập tức.")
    except Exception as e:
        print(f"[EDGE] ⚠️ Warm-up lỗi (sẽ tự warm-up khi chạy thật): {e}")
    finally:
        _models_ready.set()

threading.Thread(target=_warmup_worker, daemon=True).start()

# ====================== HELPER: VẼ CÁC LÀN ĐƯỜNG (GIỐNG full_main.py) ======================
def draw_lanes_on_frame(frame, lane_detector):
    """
    Vẽ các làn đường đã học lên frame.
    Tối ưu: Chỉ dùng 1 bản copy (overlay) thay vì 2 - giảm áp lực RAM bus Jetson Nano.
    """
    if not lane_detector.is_ready:
        return frame
        
    # Chỉ cần 1 bản copy làm overlay để blend
    overlay = frame.copy()
    
    # Lấy các thông số từ lane_detector
    car_only_zones = lane_detector.car_only_zones
    roi_pts = lane_detector.roi_pts
    
    if not car_only_zones:
        return frame
        
    # Tính toán các đường biên ROI để vẽ
    top_y = roi_pts[0][1]
    bot_y = roi_pts[2][1]
    
    virtual_lines = getattr(lane_detector, "virtual_lines", [])

    # Vẽ từng làn ô tô (Car Only)
    for idx, (z_start, z_end) in enumerate(car_only_zones):
        top_x_left  = int(roi_pts[0][0] + z_start * (roi_pts[1][0] - roi_pts[0][0]))
        top_x_right = int(roi_pts[0][0] + z_end   * (roi_pts[1][0] - roi_pts[0][0]))
        bot_x_left  = int(roi_pts[3][0] + z_start * (roi_pts[2][0] - roi_pts[3][0]))
        bot_x_right = int(roi_pts[3][0] + z_end   * (roi_pts[2][0] - roi_pts[3][0]))
        
        pts = np.array([[top_x_left, top_y], [top_x_right, top_y], [bot_x_right, bot_y], [bot_x_left, bot_y]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 165, 255))
        
        # Text "CAR ONLY LANE"
        text_x = bot_x_left + int((bot_x_right - bot_x_left) * 0.2)
        cv2.putText(overlay, f"CAR ONLY LANE {idx+1}", (text_x, bot_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Blend overlay với frame gốc (alpha=0.3)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Vẽ vạch nét đứt TRUC TIẬP lên frame (sau blend để luôn nét, không bị mờ bởi alpha)
    for (tx, ty, bx, by) in virtual_lines:
        total_len = ((bx - tx)**2 + (by - ty)**2) ** 0.5
        n_segments = max(1, int(total_len / 30))  # 1 đoạn / 30px
        for seg in range(n_segments):
            t0 = seg / n_segments
            t1 = (seg + 0.5) / n_segments  # Nửa đoạn vẽ, nửa đoạn trống
            p0 = (int(tx + t0 * (bx - tx)), int(ty + t0 * (by - ty)))
            p1 = (int(tx + t1 * (bx - tx)), int(ty + t1 * (by - ty)))
            cv2.line(frame, p0, p1, (255, 255, 255), 3, cv2.LINE_AA)
    
    return frame


def draw_roi_on_frame(frame, roi_pts, rtz_ratio):
    """
    Vẽ khung ROI lên frame.
    """
    display = frame.copy()
    h, w = frame.shape[:2]
    roi_poly = np.array(roi_pts, np.int32)
    
    # Vẽ đường viền ROI màu vàng
    cv2.polylines(display, [roi_poly], True, (0, 255, 255), 3)
    
    # Vẽ đường dừng (stop line) màu đỏ ở đỉnh ROI
    cv2.line(display, tuple(roi_poly[0]), tuple(roi_poly[1]), (0, 0, 255), 4)
    cv2.putText(display, "AUTO STOP LINE", (roi_poly[0][0] + 10, roi_poly[0][1] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
               
    # Vẽ đường Right Turn Zone (nếu ở trên stop line)
    rtz_y = int(h * rtz_ratio)
    if rtz_y < roi_poly[0][1] and rtz_y >= 0:
        right_turn_lane_min = 0.65
        zone_x_start = int(roi_poly[0][0] + right_turn_lane_min * (roi_poly[1][0] - roi_poly[0][0]))
        cv2.line(display, (zone_x_start, rtz_y), (w, rtz_y), (0, 255, 0), 2)
        cv2.putText(display, "RIGHT TURN ZONE", (zone_x_start + 10, rtz_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return display

# ====================== HELPERS ======================
def parse_light_status(frame, light_model, display_frame=None):
    """
    Nhận diện màu đèn giao thông.

    Crop động theo kích thước frame thực tế (giống full_main.py):
        light_roi = frame[0 : h//2,  w//2 : w]   ← nửa trên, nửa bên phải
    Không dùng TRAFFIC_LIGHT_ROI cố định trong config để tránh sai khi camera
    có độ phân giải khác 1280×720.

    Nếu display_frame được truyền vào, vẽ bounding box đèn lên display_frame
    với offset X = w//2 (giống cách full_main.py cộng lại FRAME_WIDTH//2).

    Trả về: "red" | "yellow" | "green" | "unknown", list_of_detected_boxes
    """
    h, w = frame.shape[:2]
    x_offset = w // 2   # offset để quy đổi tọa độ crop → frame gốc

    # Crop: nửa bên phải, nửa phía trên
    light_roi = frame[0 : h // 2, x_offset : w]

    results = light_model(light_roi, conf=cfg.CONF_TRAFFIC_LIGHT, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return "unknown", []

    best_conf  = 0
    best_color = "unknown"
    light_boxes = []

    for box in results.boxes:
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])
        label  = results.names[cls_id].lower()

        if conf > best_conf:
            best_conf = conf
            if   "red"    in label or "do"   in label or "đỏ"  in label:
                best_color = "red"
            elif "green"  in label or "xanh" in label:
                best_color = "green"
            elif "yellow" in label or "vang" in label or "vàng" in label:
                best_color = "yellow"

        # Tọa độ box
        xl, yl, xr, yr = map(int, box.xyxy[0])
        xl += x_offset
        xr += x_offset
        
        if   "red"    in label or "do"   in label or "đỏ"  in label:
            color_l = (0, 0, 255)
        elif "yellow" in label or "vang" in label or "vàng" in label:
            color_l = (0, 255, 255)
        else:
            color_l = (0, 255, 0)
        
        light_boxes.append((xl, yl, xr, yr, color_l, label, conf))

    return best_color, light_boxes

def publish_violation(client, track_id, violation_type, vehicle_crop, conf,
                      frame_a_b64: str = "", frame_b_b64: str = "",
                      vehicle_crop_wide=None, video_offset=None):
    packet = {
        "camera_id": cfg.CAMERA_ID,
        "mode": current_mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "track_id": int(track_id),
        "violation_type": violation_type,
        "lane": 1,
        "direction": "straight",
        "confidence": float(conf),
        "vehicle_crop_base64": encode_for_mqtt(vehicle_crop, quality=cfg.JPEG_ENCODE_QUALITY) if vehicle_crop is not None and vehicle_crop.size > 0 else "",
        "full_frame_a_base64": frame_a_b64,
        "full_frame_b_base64": frame_b_b64,
        # Ảnh crop rộng 75px — bằng chứng ngữ cảnh rõ hơn
        "vehicle_crop_wide_base64": encode_for_mqtt(vehicle_crop_wide, quality=cfg.JPEG_ENCODE_QUALITY) if vehicle_crop_wide is not None and vehicle_crop_wide.size > 0 else "",
        "video_offset": video_offset,
        "video_name": active_video if current_mode in ("video", "video_local") else None,
    }
    client.publish(cfg.TOPIC_VIOLATION, json.dumps(packet), qos=1)
    print(f"[EDGE] Đã gửi vi phạm: ID {track_id} | {violation_type} | Offset: {video_offset}s")

def get_active_source(mode=None, video_name=None):
    mode = mode or current_mode
    video_name = video_name or active_video
    if mode in ("video", "video_local") and video_name:
        return os.path.join(cfg.VIDEOS_DIR, video_name)
    return 0

def get_source_size(source):
    """
    Đo kích thước thực tế của nguồn video — quan trọng cho việc denormalize ROI chính xác.

    Vấn đề gốc rễ trên Jetson Nano:
      cv2.VideoCapture(mp4_file) với CAP_FFMPEG đôi khi trả về 0x0 cho file MP4 FullHD (1920x1080)
      rồi fallback về FRAME_WIDTH×FRAME_HEIGHT (1280x720). Điều này gây sai lệch 60% tọa độ ROI
      (1280/1920 ≈ 0.667) khi video thực tế được mở bằng GStreamer NVDEC (đọc đúng 1920x1080).

    Giải pháp: Dùng đúng pipeline GStreamer mà open_video_capture cũng dùng.
    Thứ tự ưu tiên:
      1. NVDEC H264 (nhanh nhất, Jetson hardware)
      2. decodebin GStreamer (tổng quát)
      3. CAP_FFMPEG (fallback cũ)
      4. Đọc 1 frame thực tế để đo (last resort)
    """
    if isinstance(source, str) and source.endswith(('.mp4', '.avi', '.mkv', '.mov')):
        # Thử GStreamer NVDEC — cùng pipeline với open_video_capture
        gst_h264 = (
            f"filesrc location={source} ! qtdemux ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink sync=false"
        )
        cap = cv2.VideoCapture(gst_h264, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                cap.release()
                print(f"[EDGE] get_source_size (NVDEC): {w}x{h}")
                return w, h
            # CAP_PROP trả về 0 → đọc 1 frame thực để đo
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"[EDGE] get_source_size (NVDEC frame): {w}x{h}")
                return w, h
        else:
            cap.release()

        # Fallback: decodebin
        gst_decode = (
            f"filesrc location={source} ! decodebin ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink sync=false"
        )
        cap = cv2.VideoCapture(gst_decode, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                cap.release()
                print(f"[EDGE] get_source_size (decodebin): {w}x{h}")
                return w, h
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"[EDGE] get_source_size (decodebin frame): {w}x{h}")
                return w, h
        else:
            cap.release()

    # Fallback cuối: CAP_FFMPEG (camera hoặc source không phải file)
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            print(f"[EDGE] get_source_size (FFMPEG): {w}x{h}")
            return w, h
    else:
        cap.release()

    print(f"[EDGE] ⚠️ get_source_size: Không đọc được kích thước, fallback về {cfg.FRAME_WIDTH}x{cfg.FRAME_HEIGHT}")
    return cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT

def open_video_capture(source):
    if isinstance(source, str) and source.endswith(('.mp4', '.avi', '.mkv', '.mov')):
        gst_h264 = (
            f"filesrc location={source} ! qtdemux ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink sync=false"
        )
        cap = cv2.VideoCapture(gst_h264, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, "nvdec"
        cap.release()

        gst_decode = (
            f"filesrc location={source} ! decodebin ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink sync=false"
        )
        cap = cv2.VideoCapture(gst_decode, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, "gstreamer"
        cap.release()

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cap, "software"

# ====================== WORKER THREADS ======================

def _stream_worker(client, q: queue.Queue):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.STREAM_JPEG_QUALITY]
    while True:
        frame = q.get()
        if frame is None:
            q.task_done()
            break
        try:
            resized = cv2.resize(frame, cfg.STREAM_RESOLUTION)
            _, buffer = cv2.imencode('.jpg', resized, encode_param)
            client.publish(cfg.TOPIC_STREAM,
                           base64.b64encode(buffer).decode('utf-8'), qos=0)
        except Exception as e:
            print(f"[STREAM WORKER] ⚠️ {e}")
        finally:
            q.task_done()


def _video_write_worker(out: cv2.VideoWriter, q: queue.Queue):
    while True:
        frame = q.get()
        if frame is None:
            q.task_done()
            break
        try:
            if out is not None and out.isOpened():
                out.write(frame)
        except Exception as e:
            print(f"[VIDEO WRITER] ⚠️ {e}")
        finally:
            q.task_done()



def scale_default_roi(width, height):
    """Scale DEFAULT_ROI_PTS (vốn thiết kế cho FRAME_WIDTH x FRAME_HEIGHT)
    theo tỷ lệ thực tế của video đang xử lý."""
    sx = width / float(cfg.FRAME_WIDTH)
    sy = height / float(cfg.FRAME_HEIGHT)
    return [[int(x * sx), int(y * sy)] for x, y in cfg.DEFAULT_ROI_PTS]

def normalize_roi_points(points, width, height):
    """Chuẩn hoá toạ độ pixel → tỷ lệ [0,1] để gửi về Dashboard/Server."""
    normalized = []
    for x, y in points:
        normalized.append({
            "x": round(float(x) / max(width, 1), 4),
            "y": round(float(y) / max(height, 1), 4)
        })
    return normalized

def denormalize_points(points, width, height):
    """Chuyển đổi ngược: toạ độ tỷ lệ [0,1] → pixel thực tế.
    Hỗ trợ cả dạng dict {"x":…,"y":…} lẫn dạng list [x, y]."""
    denormalized = []
    for p in points:
        if isinstance(p, dict):
            x, y = float(p.get("x", 0)), float(p.get("y", 0))
        else:
            x, y = float(p[0]), float(p[1])

        if 0 <= x <= 1 and 0 <= y <= 1:
            denormalized.append([int(x * width), int(y * height)])
        else:
            # Đã là toạ độ pixel, giữ nguyên
            denormalized.append([int(x), int(y)])
    return denormalized

def build_zones_config(cmd, width, height):
    """Xây dựng zones_config từ lệnh MQTT, denormalize toạ độ theo kích thước frame."""
    global roi_polygon
    def convert_zone(zone):
        zone_copy = dict(zone)
        zone_copy["points"] = [
            {"x": x, "y": y}
            for x, y in denormalize_points(zone_copy.get("points", []), width, height)
        ]
        return ZoneDefinition(**zone_copy)

    zones = {
        "lines":    [convert_zone(z) for z in cmd.get("lines", [])],
        "polygons": [convert_zone(z) for z in cmd.get("polygons", [])]
    }

    # Tự động thêm stop_line từ cạnh trên ROI nếu Dashboard chưa vẽ
    has_stop_line = any(z.label == "stop_line" for z in zones["lines"])
    if not has_stop_line and roi_polygon is not None and len(roi_polygon) >= 2:
        zones["lines"].append(ZoneDefinition(
            label="stop_line",
            points=[{"x": int(roi_polygon[0][0]), "y": int(roi_polygon[0][1])},
                    {"x": int(roi_polygon[1][0]), "y": int(roi_polygon[1][1])}]
        ))
    return zones

def publish_video_roi_preview(client, video_name):
    global roi_polygon, active_video, right_turn_zone_bottom_y_ratio

    if not video_name:
        print("[EDGE] Khong co video de preview ROI.")
        return

    source = get_active_source("video", video_name)
    cap, _ = open_video_capture(source)
    if not cap.isOpened():
        print(f"[EDGE] Khong the mo Video preview: {source}")
        return

    frame = None
    for _ in range(5):
        ret, next_frame = cap.read()
        if not ret:
            break
        frame = next_frame
    cap.release()

    if frame is None:
        print(f"[EDGE] Khong doc duoc frame preview: {source}")
        return

    height, width = frame.shape[:2]
    active_video = video_name

    display_frame = draw_roi_on_frame(frame.copy(), roi_polygon.tolist(), right_turn_zone_bottom_y_ratio)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.STREAM_JPEG_QUALITY]
    _, buffer = cv2.imencode('.jpg', cv2.resize(display_frame, cfg.STREAM_RESOLUTION), encode_param)
    client.publish(cfg.TOPIC_STREAM, base64.b64encode(buffer).decode('utf-8'), qos=0)

    import json
    client.publish(f"status/{cfg.CAMERA_ID}/roi_preview", json.dumps({
        "camera_id": cfg.CAMERA_ID,
        "video_name": video_name,
        "points": normalize_roi_points(roi_polygon.tolist(), width, height),
        "right_turn_zone_bottom_y": round(right_turn_zone_bottom_y_ratio, 4)
    }), qos=0)

# ====================== AI WORKER THREAD ======================
def ai_processing_loop(client):
    global is_running, roi_polygon, stop_reason, current_light, right_turn_zone_bottom_y_ratio
    
    source = get_active_source()
    cap, decode_mode = open_video_capture(source)
    if not cap.isOpened():
        print(f"[EDGE] Không thể mở nguồn video: {source}")
        is_running = False
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg.FRAME_WIDTH
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg.FRAME_HEIGHT
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    video_basename = os.path.basename(str(source)) if current_mode in ('video', 'video_local') else 'realtime_output.mp4'
    output_filename = f"processed_{int(time.time())}_{video_basename}"
    output_path = os.path.join(cfg.VIDEOS_DIR, output_filename)
    os.makedirs(cfg.VIDEOS_DIR, exist_ok=True)
    
    def _open_video_writer(base_path, w, h, f):
        # Ưu tiên mp4v (.mp4) — giúp trình duyệt Web chạy trực tiếp không cần qua phần mềm bên thứ 3
        # Đối với Jetson Nano, nếu mp4v thất bại ta sẽ dùng XVID (.avi) làm dự phòng
        mp4_path = base_path.rsplit('.', 1)[0] + '.mp4'
        writer_mp4 = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), f, (w, h))
        if writer_mp4.isOpened():
            print(f"[EDGE] VideoWriter: mp4v (.mp4) → {mp4_path}")
            return writer_mp4, "mp4v", mp4_path
        writer_mp4.release()
        
        # Fallback: XVID (.avi) — ổn định nhất trên Linux ARM (Jetson Nano)
        avi_path = base_path.rsplit('.', 1)[0] + '.avi'
        writer_avi = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'XVID'), f, (w, h))
        if writer_avi.isOpened():
            print(f"[EDGE] VideoWriter: XVID (.avi) → {avi_path}")
            return writer_avi, "xvid", avi_path
        writer_avi.release()
        # Fallback: mp4v (.mp4)
        mp4_path = base_path
        writer_mp4 = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), f, (w, h))
        if writer_mp4.isOpened():
            print(f"[EDGE] VideoWriter: mp4v (.mp4) → {mp4_path}")
            return writer_mp4, "mp4v", mp4_path
        writer_mp4.release()
        print("[EDGE] ⚠️ VideoWriter: Cả XVID và mp4v đều thất bại — video sẽ không được ghi.")
        return None, "none", base_path

    out, encoder_mode, output_path = _open_video_writer(output_path, frame_width, frame_height, fps)
    if out is None:
        print("[EDGE] ⚠️ VideoWriter không khởi tạo được. Sẽ bỏ qua ghi video.")
    
    lane_detector.set_roi(roi_polygon.tolist())
    roi_top_y = roi_polygon[0][1]
    
    frame_count = 0
    start_time = time.time()
    total_violations = 0
    local_violations_queue = []
    track_history = defaultdict(list)
    completed_naturally = False
    seen_vehicle_ids = set()
    violated_vehicle_ids = set()
    vehicle_state = {}
    last_light_boxes = []

    inference_times = []
    jpeg_encode_times = []
    stream_encode_times = []
    video_write_times = []

    _sq = queue.Queue(maxsize=2)
    _vq = queue.Queue(maxsize=60)
    _stream_t = threading.Thread(target=_stream_worker, args=(client, _sq), daemon=True)
    _video_t  = threading.Thread(target=_video_write_worker, args=(out, _vq), daemon=True)
    _stream_t.start()
    _video_t.start()

    while is_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            completed_naturally = True
            break
            
        display_frame = frame.copy()
        frame_count += 1
        current_offset = round(frame_count / float(fps), 2) if current_mode in ("video", "video_local") else None

        current_roi_list = roi_polygon.tolist()
        roi_bottom_y = int(frame_height * right_turn_zone_bottom_y_ratio)

        forbidden_zones = [z for z in zones_config.get("polygons", []) if z.label == "forbidden"]
        is_forbidden_mode = len(forbidden_zones) > 0

        if is_forbidden_mode:
            lane_detector.is_ready = True
            lane_detector.car_only_zones = []

        if frame_count % 3 == 1:
            new_light, last_light_boxes = parse_light_status(frame, model_traffic_light)
            if new_light != "unknown":
                current_light = new_light
        
        for box in last_light_boxes:
            xl, yl, xr, yr, color_l, label, conf = box
            cv2.rectangle(display_frame, (xl, yl), (xr, yr), color_l, 2)
            cv2.putText(display_frame, f"{label.upper()}:{conf:.2f}",
                        (xl, max(20, yl - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_l, 2)
        
        results = model_vehicle.track(frame, persist=True, tracker=cfg.TRACKER_CONFIG,
                                      verbose=False, iou=0.45, imgsz=640)[0]
        
        cars = motorcycles = 0
        track_ids = []
        frame_roi_boxes = []
        frame_roi_classes = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(cls)

                # Lọc class: chỉ xử lý xe (Car, Bus, Truck, Motorcycle)
                if cls_id not in cfg.VEHICLE_CLASSES:
                    continue

                center_x  = (x1 + x2) // 2
                center_y  = (y1 + y2) // 2
                bottom_y  = y2

                in_roi = cv2.pointPolygonTest(roi_polygon, (center_x, bottom_y), False) >= 0

                # --- TÍNH VÙNG RẼ PHẢI ---
                right_turn_lane_min = 0.65  # fallback
                if lane_detector.is_ready and lane_detector.car_only_zones:
                    right_turn_lane_min = max(
                        [end for (_, end) in lane_detector.car_only_zones] + [0.65]
                    )
                norm_x = get_normalized_x(center_x, bottom_y, current_roi_list)
                in_right_turn_zone = (roi_bottom_y <= center_y < roi_top_y) and (norm_x >= right_turn_lane_min)

                # Cờ kiểm tra sai làn nghiêm ngặt: chỉ khi xe ở trong ROI, không ở vùng rẽ phải và đã vượt qua vạch dừng (center_y >= roi_top_y)
                in_roi_for_lane = in_roi and not in_right_turn_zone and (center_y >= roi_top_y)

                # --- PAST STOPLINE: tiếp tục theo dõi xe đang chờ xử lý lỗi ---
                # Chỉ tiếp tục theo dõi nếu xe này đang ở trạng thái pending (chưa chốt vi phạm)
                past_stopline_active = (track_id in violation_engine.pending_red_lights) and (center_y < roi_top_y)

                if not (in_roi or in_right_turn_zone or past_stopline_active):
                    continue  # Bỏ qua xe hoàn toàn ngoài vùng quan tâm

                # Khởi tạo state cho xe mới
                if track_id not in vehicle_state:
                    vehicle_state[track_id] = {'was_right_lane': False}
                vstate = vehicle_state[track_id]

                # Ghi nhận xe vào thống kê (chỉ đếm xe trong ROI thực sự)
                if in_roi:
                    seen_vehicle_ids.add(track_id)

                # Ghi nhận xe máy đang ở đúng làn rẽ phải (trước vạch)
                if cls_id in cfg.MOTO_CLASSES and in_right_turn_zone:
                    vstate['was_right_lane'] = True

                # Lưu xe hợp lệ trong ROI để học làn
                if in_roi and not lane_detector.is_ready:
                    frame_roi_boxes.append(box)
                    frame_roi_classes.append(cls)

                # Đếm xe
                if cls_id == 2: cars += 1
                elif cls_id == 3: motorcycles += 1

                # 1. DUY TRÌ LỊCH SỬ QUỸ ĐẠO (TRAJECTORY)
                track_history[track_id].append((center_x, bottom_y))
                if len(track_history[track_id]) > cfg.MAX_TRACK_HISTORY:
                    track_history[track_id].pop(0)
                path = track_history[track_id]

                if current_mode in ("video", "video_local"):
                    new_violations_list = []
                    
                    if is_forbidden_mode:
                        # ==========================================
                        # CHẾ ĐỘ ĐƯỜNG CẤM (FORBIDDEN MODE)
                        # ==========================================
                        new_errors = violation_engine.check_violations(
                            track_id=track_id, bbox=[x1, y1, x2, y2],
                            trajectory=path, light_status={"straight": current_light},
                            zones_config={"polygons": forbidden_zones},
                            vehicle_cls=cls_id
                        )
                        new_violations_list = [e for e in new_errors if "ĐƯỜNG CẤM" in e]
                    else:
                        # ==========================================
                        # CHẾ ĐỘ ĐƯỜNG BÌNH THƯỜNG (NORMAL MODE)
                        # ==========================================
                        # Lọc bỏ polygon "forbidden" khỏi zones_config để tránh phạt nhầm ĐI VÀO ĐƯỜNG CẤM
                        # khi chế độ đường cấm chưa được kích hoạt (phòng trường hợp zones_config còn sót từ session cũ)
                        normal_zones_config = {
                            "lines":    zones_config.get("lines", []),
                            "polygons": [z for z in zones_config.get("polygons", []) if z.label != "forbidden"]
                        }
                        new_violations_list = violation_engine.check_violations(
                            track_id=track_id, bbox=[x1, y1, x2, y2],
                            trajectory=path, light_status={"straight": current_light},
                            zones_config=normal_zones_config, stop_line_y=roi_top_y, vehicle_cls=cls_id,
                            was_right_lane=vstate['was_right_lane'], in_roi=in_roi_for_lane
                        )
                        
                        # CHECK SAI LÀN AI TỰ ĐỘNG (Chỉ kiểm tra khi phương tiện nằm trong ROI và không rẽ phải)
                        if in_roi_for_lane and lane_detector.is_ready and not is_forbidden_mode:
                            if lane_detector.check_wrong_lane([x1, y1, x2, y2], cls):
                                if "SAI LÀN" not in violation_engine.recorded_violations[track_id]:
                                    new_violations_list.append("SAI LÀN")
                                    violation_engine.recorded_violations[track_id].add("SAI LÀN")
                    
                    if len(new_violations_list) > 0:
                        combo_violation_string = "+".join(new_violations_list)
                        violated_vehicle_ids.add(track_id)
                        crop_img = smart_crop(frame, box, padding=cfg.SMART_CROP_PADDING)
                        # Ảnh crop rộng hơn (100px) làm bằng chứng ngữ cảnh rõ hơn
                        crop_wide = smart_crop(frame, box, padding=100)
                        
                        ev_param = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.EVIDENCE_JPEG_QUALITY]
                        _, buf_a = cv2.imencode('.jpg', cv2.resize(frame, cfg.EVIDENCE_RESOLUTION), ev_param)
                        
                        ann = display_frame.copy()
                        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        _, buf_b = cv2.imencode('.jpg', cv2.resize(ann, cfg.EVIDENCE_RESOLUTION), ev_param)
                        
                        if current_mode == "video_local":
                            local_violations_queue.append((
                                client, track_id, combo_violation_string, crop_img, conf,
                                base64.b64encode(buf_a).decode('utf-8'), base64.b64encode(buf_b).decode('utf-8'),
                                crop_wide, current_offset
                            ))
                        else:
                            publish_violation(client, track_id, combo_violation_string, crop_img, conf,
                                              base64.b64encode(buf_a).decode('utf-8'), base64.b64encode(buf_b).decode('utf-8'),
                                              vehicle_crop_wide=crop_wide, video_offset=current_offset)
                        total_violations += 1

                color = (0, 0, 255) if len(violation_engine.recorded_violations[track_id]) > 0 else (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                for i in range(1, len(path)):
                    cv2.line(display_frame, path[i-1], path[i], color, 2)

        # HỌC LÀN TỰ ĐỘNG: Chỉ chạy nếu có dữ liệu ROI và không phải Đường cấm
        if not lane_detector.is_ready and frame_roi_boxes and not is_forbidden_mode:
            lane_detector.update_learning_data(
                np.array(frame_roi_boxes), np.array(frame_roi_classes)
            )

        # Kiểm tra mất tracking để phạt nguội lỗi Đèn Đỏ
        active_tracks = set(track_ids) if 'track_ids' in locals() else set()
        lost_violations = violation_engine.cleanup_lost_tracks(active_tracks)
        for v in lost_violations:
            # Cắt từ 'frame' (ảnh sạch gốc)
            crop_img = smart_crop(frame, v["bbox"], padding=cfg.SMART_CROP_PADDING)
            crop_wide = smart_crop(frame, v["bbox"], padding=100)
            # Ảnh bằng chứng: dùng frame hiện tại (xe vừa mất tracking)
            ev_param = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.EVIDENCE_JPEG_QUALITY]
            _, buf_a = cv2.imencode('.jpg', cv2.resize(frame, cfg.EVIDENCE_RESOLUTION), ev_param)
            lv_a_b64 = base64.b64encode(buf_a).decode('utf-8')
            bx1, by1, bx2, by2 = map(int, v["bbox"])
            ann_lv = frame.copy()
            cv2.rectangle(ann_lv, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            cv2.putText(ann_lv, f"VI PHAM: {v['violation_type']}",
                        (max(bx1, 5), max(by1 - 15, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(ann_lv, f"{cfg.CAMERA_ID} | ID:{v['track_id']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            _, buf_b = cv2.imencode('.jpg', cv2.resize(ann_lv, cfg.EVIDENCE_RESOLUTION), ev_param)
            lv_b_b64 = base64.b64encode(buf_b).decode('utf-8')
            if current_mode == "video_local":
                local_violations_queue.append((
                    client, v["track_id"], v["violation_type"], crop_img, 0.8,
                    lv_a_b64, lv_b_b64,
                    crop_wide, current_offset
                ))
            else:
                publish_violation(client, v["track_id"], v["violation_type"], crop_img, 0.8,
                                  frame_a_b64=lv_a_b64, frame_b_b64=lv_b_b64,
                                  vehicle_crop_wide=crop_wide, video_offset=current_offset)
            total_violations += 1

        display_frame = draw_roi_on_frame(display_frame, current_roi_list, right_turn_zone_bottom_y_ratio)
        if lane_detector.is_ready:
            display_frame = draw_lanes_on_frame(display_frame, lane_detector)

        # put_nowait(): không block AI thread; nếu queue đầy (worker bận) → drop frame
        # → stream luôn fresh, không lag buffer
        if current_mode != "video_local" and frame_count % 3 == 0:
            t_stream = time.time()
            try:
                _sq.put_nowait(display_frame.copy())
            except queue.Full:
                pass  # Drop frame cũ — ưu tiên freshness hơn completeness
            stream_encode_times.append((time.time() - t_stream) * 1000)

        # 5. Gửi Heartbeat & Stats mỗi 30 frame
        if frame_count % 30 == 0:
            current_fps = frame_count / (time.time() - start_time)
            heartbeat = {
                "camera_id": cfg.CAMERA_ID,
                "mode": current_mode,
                "stats": {"car": cars, "motorcycle": motorcycles, "bus": 0, "truck": 0},
                "lights": {"left": "unknown", "straight": current_light},
                "fps": round(current_fps, 1),
                "active_video": active_video
            }
            client.publish(cfg.TOPIC_STATUS, json.dumps(heartbeat), qos=0)

        # 5. Ghi video — đưa vào queue, worker thread xử lý I/O ghi đĩa
        # put(timeout=0.1): chờ tối đa 100ms nếu queue đầy, sau đó skip frame
        t_write = time.time()
        if out is not None:  # Chỉ ghi nếu VideoWriter khởi tạo thành công
            try:
                _vq.put(display_frame.copy(), timeout=0.1)
            except queue.Full:
                print("[EDGE] ⚠️ Video write queue đầy, bỏ qua 1 frame.")
        video_write_times.append((time.time() - t_write) * 1000)

    # Dọn dẹp khi kết thúc
    # 1. Gửi sentinel None để báo worker threads dừng
    _sq.put(None)   # stream worker
    if out is not None and out.isOpened():
        _vq.put(None)   # video write worker
    else:
        _vq.put(None)   # luôn gửi sentinel để thread kết thúc
        
    # 2. Chờ video write worker xử lý hết frame còn trong queue rồi mới release
    #    (tránh VideoWriter bị đóng trong khi worker vẫn đang ghi)
    if _video_t.is_alive():
        _video_t.join(timeout=10)
    if _stream_t.is_alive():
        _stream_t.join(timeout=5)

    cap.release()
    if out is not None:
        out.release()
    is_running = False

    # 3. Phát tán toàn bộ vi phạm tích lũy trong mode video_local sau khi giải phóng tài nguyên
    if current_mode == "video_local" and local_violations_queue:
        print(f"[EDGE] Đang phát tán {len(local_violations_queue)} vi phạm tích lũy từ hàng đợi local...")
        for args in local_violations_queue:
            publish_violation(*args)
            time.sleep(0.05)
        total_violations = len(local_violations_queue)

    # ================================================================
    # THỐNG KÊ PHƯƠNG TIỆN & VI PHẠM
    # ================================================================
    total_seen     = len(seen_vehicle_ids)
    total_violated = len(violated_vehicle_ids)
    violation_rate = (total_violated / total_seen * 100) if total_seen > 0 else 0.0

    print("\n" + "="*62)
    print("[EDGE] 🚦 KẾT QUẢ PHÂN TÍCH GIAO THÔNG")
    print("="*62)
    print(f"  Video               : {active_video or 'realtime'}")
    print(f"  Thời gian xử lý     : {round(time.time() - start_time, 1)} s")
    print("-"*62)
    print(f"  Tổng phương tiện    : {total_seen:>6} xe")
    print(f"  Phương tiện vi phạm : {total_violated:>6} xe")
    print(f"  Tỉ lệ vi phạm       : {violation_rate:>6.1f} %")
    if total_violated > 0:
        print("-"*62)
        print("  Chi tiết loại vi phạm:")
        # Đếm từng loại lỗi từ violation_engine
        violation_type_count = {}
        for tid, vset in violation_engine.recorded_violations.items():
            for v in vset:
                violation_type_count[v] = violation_type_count.get(v, 0) + 1
        for vtype, count in sorted(violation_type_count.items(), key=lambda x: -x[1]):
            print(f"    {vtype:<28s}: {count} xe")
    print("="*62 + "\n")

    # IN & LƯU THỐNG KÊ TIMING CHI TIẾT
    if inference_times:
        def _stats(lst): 
            return (sum(lst)/len(lst), min(lst), max(lst)) if lst else (0, 0, 0)

        avg_inf, min_inf, max_inf   = _stats(inference_times)
        avg_jpg, min_jpg, max_jpg   = _stats(jpeg_encode_times)
        avg_str, min_str, max_str   = _stats(stream_encode_times)
        avg_wrt, min_wrt, max_wrt   = _stats(video_write_times)
        avg_fps = 1000.0 / avg_inf if avg_inf > 0 else 0
        total_s = sum(inference_times) / 1000.0

        print("\n" + "="*62)
        print("[EDGE] 📊 DETAILED TIMING REPORT")
        print("="*62)
        print(f"  Decoder Mode        : {decode_mode.upper()}")
        print(f"  Encoder Mode        : {encoder_mode.upper()}")
        print(f"  Tổng frame xử lý  : {len(inference_times)}")
        print(f"  Tổng thời gian     : {total_s:.1f} s")
        print(f"  FPS thực tế        : ~{avg_fps:.1f} FPS")
        print("-"*62)
        print(f"  {'Phân tich':28s} {'TB (ms)':>9} {'Min (ms)':>9} {'Max (ms)':>9}")
        print("-"*62)
        print(f"  {'YOLO Inference':28s} {avg_inf:9.1f} {min_inf:9.1f} {max_inf:9.1f}")
        if jpeg_encode_times:
            print(f"  {'Nén ảnh + Gửi vi phạm':28s} {avg_jpg:9.1f} {min_jpg:9.1f} {max_jpg:9.1f}")
        else:
            print(f"  {'Nén ảnh + Gửi vi phạm':28s} {'(không có vi phạm)':>30}")
        if stream_encode_times:
            print(f"  {'Nén stream Dashboard':28s} {avg_str:9.1f} {min_str:9.1f} {max_str:9.1f}")
        else:
            print(f"  {'Nén stream Dashboard':28s} {'(mode video_local)':>30}")
        print(f"  {'Ghi file video':28s} {avg_wrt:9.1f} {min_wrt:9.1f} {max_wrt:9.1f}")
        print("="*62 + "\n")

        # Lưu ra JSON để automation test đọc
        timing_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "decode_mode": decode_mode,
            "encode_mode": encoder_mode,
            "total_frames": len(inference_times),
            "vehicles_seen": total_seen,
            "vehicles_violated": total_violated,
            "violation_rate_pct": round(violation_rate, 2),
            "total_s": round(total_s, 2),
            "avg_fps": round(avg_fps, 2),
            "inference": {"avg_ms": round(avg_inf,2), "min_ms": round(min_inf,2), "max_ms": round(max_inf,2)},
            "jpeg_violation": {"avg_ms": round(avg_jpg,2), "min_ms": round(min_jpg,2), "max_ms": round(max_jpg,2), "count": len(jpeg_encode_times)},
            "stream_encode": {"avg_ms": round(avg_str,2), "min_ms": round(min_str,2), "max_ms": round(max_str,2), "count": len(stream_encode_times)},
            "video_write":   {"avg_ms": round(avg_wrt,2), "min_ms": round(min_wrt,2), "max_ms": round(max_wrt,2)},
            # backward-compat keys (AI-06 cũ vẫn đọc được)
            "avg_ms":  round(avg_inf, 2),
            "min_ms":  round(min_inf, 2),
            "max_ms":  round(max_inf, 2),
        }
        timing_file = os.path.join(BASE_DIR, "inference_timing.json")
        try:
            with open(timing_file, "w", encoding="utf-8") as f:
                json.dump(timing_data, f, indent=2, ensure_ascii=False)
            print(f"[EDGE] 💾 Đã lưu timing chi tiết vào: {timing_file}")
        except Exception as e:
            print(f"[EDGE] ⚠️ Không thể lưu timing file: {e}")
    
    # 5. Gửi gói Complete khi xử lý xong (Chỉ ở mode Video)
    should_publish_complete = current_mode in ("video", "video_local") and completed_naturally
    if current_mode in ("video", "video_local") and stop_reason == "reset_roi":
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

    # 6. Upload file video nếu đang chạy mode video_local hoặc video
    if current_mode in ("video", "video_local") and completed_naturally:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            try:
                print(f"[EDGE] Đang upload video kết quả lên Server: {output_path}")
                video_filename = os.path.basename(output_path)
                with open(output_path, 'rb') as vf:
                    # Xác định MIME type dựa vào extension
                    ext = os.path.splitext(output_path)[1].lower()
                    mime = 'video/x-msvideo' if ext == '.avi' else 'video/mp4'
                    files = {'video': (video_filename, vf, mime)}
                    form_data = {
                        'processing_time_seconds': str(round(time.time() - start_time, 2)),
                        'mode': current_mode
                    }
                    # Endpoint server: /api/upload_video/{camera_id} với key 'video'
                    api_url = f"http://{cfg.MQTT_BROKER}:8000/api/upload_video/{cfg.CAMERA_ID}"
                    res = requests.post(api_url, files=files, data=form_data, timeout=120)
                    if res.status_code == 200:
                        print(f"[EDGE] ✅ Upload video thành công: {res.json()}")
                    else:
                        print(f"[EDGE] ❌ Lỗi upload video: HTTP {res.status_code} | {res.text[:200]}")
            except Exception as e:
                print(f"[EDGE] ❌ Lỗi kết nối upload video: {e}")
        else:
            size = os.path.getsize(output_path) if os.path.exists(output_path) else -1
            print(f"[EDGE] ⚠️ Bỏ qua upload: File video không tồn tại hoặc rỗng (size={size}B). Path: {output_path}")

def send_idle_heartbeat(client):
    """Gửi heartbeat định kỳ khi Edge đang ở trạng thái chờ (Idle).
    Gửi NGAY LẬP TỨC lần đầu để Dashboard nhận biết camera sớm nhất có thể,
    sau đó lặp lại mỗi 60 giây.
    """
    def _build():
        return {
            "camera_id":   cfg.CAMERA_ID,
            "status":      "idle",
            "stats":       {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
            "lights":      {"left": "unknown", "straight": "unknown"},
            "fps":         0,
            "active_video": active_video
        }

    def heartbeat_thread():
        # Gửi ngay lập tức sau 2s — chờ đủ để MQTT connect xong
        time.sleep(2)
        client.publish(cfg.TOPIC_STATUS, json.dumps(_build()), qos=1)
        print(f"[MQTT] 📡 Heartbeat khởi động gửi: {cfg.CAMERA_ID}")

        while True:
            time.sleep(60)
            # Chỉ gửi khi AI đang KHÔNG chạy (lúc AI chạy đã có heartbeat riêng mỗi 30 frame)
            if not is_running:
                client.publish(cfg.TOPIC_STATUS, json.dumps(_build()), qos=0)
                print(f"[MQTT] 📡 Idle heartbeat: {cfg.CAMERA_ID}")

    threading.Thread(target=heartbeat_thread, daemon=True).start()

# ====================== MQTT CALLBACKS ======================
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Đã kết nối Broker với mã: {rc}")
    client.subscribe(cfg.TOPIC_CMD)
    print(f"[MQTT] Lắng nghe lệnh điều khiển tại: {cfg.TOPIC_CMD}")

def on_message(client, userdata, msg):
    global is_running, current_mode, active_video, zones_config, roi_polygon, stop_reason, roi_confirmed, right_turn_zone_bottom_y_ratio
    
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
                current_mode = "video" if cmd.get("action") == "preview_video" and current_mode != "video_local" else current_mode
                if current_mode not in ("video", "video_local"):
                    current_mode = "video"
                if was_running:
                    time.sleep(0.2)
                    
                # Phân tích ROI được gửi kèm để không làm mất cấu hình của Dashboard
                roi_pts_cmd = cmd.get("roi")
                if not roi_pts_cmd:
                    polygons_cmd = cmd.get("polygons", [])
                    if polygons_cmd:
                        roi_pts_cmd = polygons_cmd[0].get("points", [])
                
                source = get_active_source(cmd.get("mode", current_mode), cmd.get("video_name") or active_video)
                frame_w, frame_h = get_source_size(source)
                
                if roi_pts_cmd and len(roi_pts_cmd) == 4:
                    roi_pts_cmd = denormalize_points(roi_pts_cmd, frame_w, frame_h)
                    roi_polygon = np.array(roi_pts_cmd, np.int32)
                    lane_detector.set_roi(roi_pts_cmd)
                    roi_confirmed = True

                rtz_y_cmd = cmd.get("right_turn_zone_bottom_y")
                if rtz_y_cmd is not None:
                    right_turn_zone_bottom_y_ratio = float(rtz_y_cmd)

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
                # Cập nhật ranh giới vùng rẽ phải nếu Dashboard gửi kèm
                rtz_y = cmd.get("right_turn_zone_bottom_y")
                if rtz_y is not None:
                    right_turn_zone_bottom_y_ratio = float(rtz_y)
                    print(f"[EDGE] Cập nhật Right Turn Zone bottom Y: {right_turn_zone_bottom_y_ratio:.3f}")
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
                # Reset ranh giới vùng rẽ phải về mặc định config
                right_turn_zone_bottom_y_ratio = cfg.RIGHT_TURN_ZONE_BOTTOM_Y_RATIO
                if active_video:
                    publish_video_roi_preview(client, active_video)
                print("[EDGE] Đã dừng xử lý, reset ROI và đưa về mặc định.")
                return

            if action == "start":
                if not is_running:
                    # Trích xuất cấu hình ROI nếu Dashboard đính kèm trong lệnh start
                    roi_pts_cmd = cmd.get("roi")
                    if not roi_pts_cmd:
                        polygons_cmd = cmd.get("polygons", [])
                        if polygons_cmd:
                            roi_pts_cmd = polygons_cmd[0].get("points", [])
                    if roi_pts_cmd and len(roi_pts_cmd) == 4:
                        source = get_active_source(cmd.get("mode", "realtime"), cmd.get("video_name"))
                        frame_w, frame_h = get_source_size(source)
                        roi_pts_cmd = denormalize_points(roi_pts_cmd, frame_w, frame_h)
                        roi_polygon = np.array(roi_pts_cmd, np.int32)
                        lane_detector.set_roi(roi_pts_cmd)
                        roi_confirmed = True
                        rtz_y_cmd = cmd.get("right_turn_zone_bottom_y")
                        if rtz_y_cmd is not None:
                            right_turn_zone_bottom_y_ratio = float(rtz_y_cmd)
                        zones_config = build_zones_config(cmd, frame_w, frame_h)
                        print(f"[EDGE] Đã cấu hình ROI trước khi Start: {roi_pts_cmd}")

                    # Đảm bảo warm-up đã xong trước khi chạy inference thật
                    # Nếu warm-up chạy xong rồi thì is_set()=True → không block
                    # Nếu chưa xong → chờ tối đa 60s (thường chỉ vài giây)
                    if not _models_ready.is_set():
                        print("[EDGE] ⏳ Models đang warm-up, vui lòng chờ giây lát...")
                        _models_ready.wait(timeout=60)
                        print("[EDGE] ✅ Models đã sẵn sàng, tiếp tục khởi động.")

                    current_mode = cmd.get("mode", "realtime")
                    active_video = cmd.get("video_name")
                    if current_mode in ("video", "video_local") and not cmd.get("roi"):
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

                    # Cập nhật ranh giới vùng rẽ phải nếu Dashboard gửi kèm
                    rtz_y = cmd.get("right_turn_zone_bottom_y")
                    if rtz_y is not None:
                        right_turn_zone_bottom_y_ratio = float(rtz_y)
                        print(f"[EDGE] Vùng rẽ phải bottom Y = {right_turn_zone_bottom_y_ratio:.3f}")

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
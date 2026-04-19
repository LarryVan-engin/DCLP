"""
********************************************************************************************************************
Project:      Traffic Violation Detection (Pro Version - MQTT Hybrid)
File:         edge/utils/capture_utils.py
Description:  Module xử lý cắt ảnh thông minh (Smart Crop) và mã hóa (Base64) để gửi qua MQTT.
********************************************************************************************************************
"""

import cv2
import base64
import numpy as np
from typing import List

def smart_crop(frame: np.ndarray, bbox: List[int], padding: int = 40) -> np.ndarray:
    """
    Cắt ảnh phương tiện với một khoảng padding (mở rộng) xung quanh.
    Mục đích: 
    - Không làm mất rìa biển số nếu bounding box của YOLO quá sát.
    - Lấy thêm được bối cảnh (vạch kẻ đường, một phần không gian) làm bằng chứng pháp lý.
    
    Args:
        frame: Khung hình gốc (numpy array).
        bbox: Tọa độ bounding box [x1, y1, x2, y2] của phương tiện.
        padding: Số pixel mở rộng ra các hướng (mặc định 40px).
        
    Returns:
        Ảnh đã cắt (numpy array).
    """
    if frame is None or len(bbox) != 4:
        return np.array([])

    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # Tính toán tọa độ mới có padding, dùng max/min để đảm bảo không tràn viền ảnh (Out of bounds)
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w_frame, x2 + padding)
    y2_pad = min(h_frame, y2 + padding)

    # Cắt ảnh theo vùng đã tính toán
    cropped_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
    
    return cropped_img

def encode_for_mqtt(img: np.ndarray, quality: int = 90) -> str:
    """
    Nén ảnh sang định dạng JPEG và mã hóa thành chuỗi Base64.
    
    Args:
        img: Ảnh cần mã hóa (thường là ảnh lấy từ smart_crop).
        quality: Chất lượng nén JPEG (1-100). Đặt 90 để cân bằng giữa dung lượng thấp và độ nét OCR cao.
        
    Returns:
        Chuỗi Base64 để nhúng vào payload JSON. Trả về chuỗi rỗng nếu có lỗi.
    """
    if img is None or img.size == 0:
        return ""

    try:
        # Nén ảnh JPEG với chất lượng được chỉ định
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode('.jpg', img, encode_param)
        
        if success:
            return base64.b64encode(buffer).decode('utf-8')
        return ""
    except Exception as e:
        print(f"[CAPTURE UTILS] ❌ Lỗi mã hóa ảnh Base64: {e}")
        return ""
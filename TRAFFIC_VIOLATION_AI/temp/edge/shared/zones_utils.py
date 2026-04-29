"""
********************************************************************************************************************
Project:      Traffic Violation Detection (Pro Version - MQTT Hybrid)
File:         shared/zones_utils.py
Description:  Các hàm tiện ích xử lý hình học: kiểm tra xe trong vùng, cắt vạch kẻ (Line/Polygon Intersections).
********************************************************************************************************************
"""

import cv2
import numpy as np
from typing import List, Tuple
from .schemas import ZoneDefinition

# =====================================================================
# 1. CHUYỂN ĐỔI DỮ LIỆU (CONVERTERS)
# =====================================================================
def zone_to_numpy(zone: ZoneDefinition) -> np.ndarray:
    """
    Chuyển đổi danh sách Point từ schema Pydantic sang numpy array cho OpenCV.
    Dùng để vẽ hoặc tính toán với cv2.pointPolygonTest, cv2.polylines.
    """
    pts = [[pt.x, pt.y] for pt in zone.points]
    return np.array(pts, np.int32).reshape((-1, 1, 2))

# =====================================================================
# 2. XỬ LÝ BOUNDING BOX CỦA YOLO
# =====================================================================
def get_bottom_center(bbox: List[float]) -> Tuple[int, int]:
    """
    Lấy tọa độ điểm giữa cạnh dưới của Bounding Box.
    Rất quan trọng trong Traffic AI vì đây chính là điểm tiếp xúc của bánh xe với mặt đường.
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))

def get_center(bbox: List[float]) -> Tuple[int, int]:
    """
    Lấy tọa độ điểm tâm của Bounding Box.
    Dùng cho Đèn giao thông hoặc biển báo.
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

# =====================================================================
# 3. THUẬT TOÁN HÌNH HỌC (GEOMETRY ALGORITHMS)
# =====================================================================
def is_point_in_polygon(point: Tuple[float, float], polygon_pts: np.ndarray) -> bool:
    """
    Kiểm tra xem một điểm (VD: bánh xe) có nằm trong đa giác (VD: vùng cấm, light_zone) không.
    Sử dụng cv2.pointPolygonTest.
    """
    if polygon_pts is None or len(polygon_pts) < 3:
        return False
    # Trả về > 0 nếu nằm trong, = 0 nếu nằm trên cạnh, < 0 nếu nằm ngoài
    return cv2.pointPolygonTest(polygon_pts, point, False) >= 0

def check_line_intersection(p1: Tuple[float, float], p2: Tuple[float, float], 
                            p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """
    Kiểm tra xem quỹ đạo di chuyển của xe có cắt ngang vạch kẻ đường hay không.
    - (p1, p2): Tọa độ frame trước và frame hiện tại của bánh xe (Vector di chuyển).
    - (p3, p4): Tọa độ 2 đầu của vạch kẻ (Vạch dừng đèn đỏ, vạch phân làn).
    
    Thuật toán: Kiểm tra hướng (Counter-Clockwise) của các tam giác tạo bởi các điểm.
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    # Nếu hai đoạn thẳng cắt nhau, hai điểm của đoạn này phải nằm về hai phía của đoạn kia.
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def check_vehicle_crossed_line(trajectory: List[Tuple[int, int]], line_zone: ZoneDefinition) -> bool:
    """
    Kiểm tra xem lịch sử di chuyển (trajectory) của một xe có cắt qua một LineZone hay không.
    """
    if len(trajectory) < 2 or len(line_zone.points) < 2:
        return False

    # Lấy điểm đầu và điểm cuối của vạch kẻ
    p3 = (line_zone.points[0].x, line_zone.points[0].y)
    p4 = (line_zone.points[-1].x, line_zone.points[-1].y)

    # Lấy vị trí frame trước và frame hiện tại của xe
    p1 = trajectory[-2]
    p2 = trajectory[-1]

    return check_line_intersection(p1, p2, p3, p4)

# =====================================================================
# 4. KIỂM TRA ĐI NGƯỢC CHIỀU (WRONG WAY DETECTION)
# =====================================================================
def check_wrong_way(trajectory: List[Tuple[int, int]], allowed_direction: str, min_dist: int = 20) -> bool:
    """
    Kiểm tra xe đi ngược chiều (Wrong Way) dựa trên vector di chuyển.
    
    Tham số:
    - trajectory: Danh sách lịch sử tọa độ tâm bánh xe.
    - allowed_direction: "up" (đi xa dần camera - y giảm) hoặc "down" (đi lại gần - y tăng).
    - min_dist: Khoảng cách tối thiểu bằng pixel để loại bỏ nhiễu rung lắc của YOLO.
    """
    if len(trajectory) < 2:
        return False
        
    start_pt = trajectory[0]
    end_pt = trajectory[-1]
    
    # Tính khoảng cách di chuyển thực tế (Chống nhiễu)
    distance = ((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)**0.5
    if distance < min_dist:
        return False  # Xe chỉ nhích nhẹ (nhiễu), bỏ qua
        
    # Tính hướng di chuyển theo trục Y
    dy = end_pt[1] - start_pt[1]
    
    # So sánh với hướng cho phép
    if allowed_direction == "up" and dy > 0:
        # Lẽ ra phải đi lên (y giảm), nhưng lại đi xuống (y tăng)
        return True
    elif allowed_direction == "down" and dy < 0:
        # Lẽ ra phải đi xuống (y tăng), nhưng lại đi lên (y giảm)
        return True
        
    return False

# =====================================================================
# 5. KIỂM TRA SAI LÀN THỦ CÔNG (MANUAL WRONG LANE BY POLYGON)
# =====================================================================
def check_wrong_lane_by_polygon(bbox: List[float], trajectory: List[Tuple[int, int]], 
                                zones_config: dict, class_id: int) -> bool:
    """
    Kiểm tra sai làn dựa trên các Polygon vẽ từ Dashboard.
    """
    if len(trajectory) < 3 or "polygons" not in zones_config:
        return False
    
    vehicle_point = get_bottom_center(bbox)
    lane_zones = zones_config.get("polygons", [])
    if not lane_zones:
        return False
    
    # Kiểm tra xe có nằm trong bất kỳ vùng làn nào không
    in_any_lane = False
    for zone in lane_zones:
        if zone.label and "lane" in zone.label.lower():
            zone_poly = zone_to_numpy(zone)
            if is_point_in_polygon(vehicle_point, zone_poly):
                in_any_lane = True
                break
    
    # Nếu di chuyển từ làn này sang làn khác quá nhanh -> Vi phạm
    if len(trajectory) >= 3:
        prev_point = trajectory[-3]
        curr_point = trajectory[-1]
        y_diff = abs(curr_point[1] - prev_point[1])
        if y_diff > 30:
            return True
            
    return not in_any_lane
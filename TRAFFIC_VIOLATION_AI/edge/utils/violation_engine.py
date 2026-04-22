"""
********************************************************************************************************************
Project:      Traffic Violation Detection (Pro Version - MQTT Hybrid)
File:         edge/utils/violation_engine.py
Description:  Động cơ kiểm tra vi phạm đa luồng (Hỗ trợ bắt Combo nhiều lỗi cùng lúc).
              Đã tối ưu hóa hiệu năng (đưa import ra ngoài) và chuẩn hóa logic Đường Cấm.
********************************************************************************************************************
"""

import sys
import os
import numpy as np  # Chuyển import lên đầu file để cứu FPS cho Jetson Nano
from typing import List, Tuple, Dict
from collections import defaultdict

# Thêm đường dẫn để import shared module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from shared.zones_utils import (
    check_vehicle_crossed_line, 
    check_wrong_way, 
    is_point_in_polygon, 
    get_bottom_center
)

class ViolationEngine:
    def __init__(self):
        # Bộ nhớ theo dõi Combo lỗi: { track_id: set("VƯỢT ĐÈN ĐỎ", "SAI LÀN", ...) }
        self.recorded_violations = defaultdict(set)
        
        # Bộ nhớ chờ xác nhận rẽ phải (Tránh bắt nhầm xe rẽ phải khi đèn đỏ)
        self.pending_red_lights = {}

    def check_violations(self, 
                         track_id: int, 
                         bbox: List[int], 
                         trajectory: List[Tuple[int, int]], 
                         light_status: Dict[str, str], 
                         zones_config: dict) -> List[str]:
        """
        Quét toàn bộ luật vi phạm. Trả về danh sách các lỗi mới phát hiện ở frame hiện tại.
        """
        detected_new_violations = []
        bottom_center = get_bottom_center(bbox)
        
        # =================================================================
        # LUẬT 1: VƯỢT ĐÈN ĐỎ / VÀNG
        # =================================================================
        current_light = light_status.get("straight", "unknown").lower()
        
        if track_id in self.pending_red_lights:
            suspect_data = self.pending_red_lights[track_id]
            suspect_data["frames_waited"] += 1
            
            # Đợi 15 frame để xem xe đi thẳng hay rẽ phải
            if suspect_data["frames_waited"] >= 15:
                start_p = suspect_data["cross_point"]
                dx = bottom_center[0] - start_p[0]
                dy = bottom_center[1] - start_p[1]
                
                # Logic rẽ phải: Trục X thay đổi nhiều hơn trục Y
                is_turning_right = dx > 15 and dx > abs(dy) * 0.35
                
                if not is_turning_right:
                    if "VƯỢT ĐÈN ĐỎ" not in self.recorded_violations[track_id]:
                        detected_new_violations.append("VƯỢT ĐÈN ĐỎ")
                        self.recorded_violations[track_id].add("VƯỢT ĐÈN ĐỎ")
                        
                del self.pending_red_lights[track_id]
        else:
            for line_zone in zones_config.get("lines", []):
                if line_zone.label == "stop_line":
                    if check_vehicle_crossed_line(trajectory, line_zone):
                        if current_light == "red":
                            self.pending_red_lights[track_id] = {
                                "cross_point": bottom_center,
                                "frames_waited": 0
                            }
                        elif current_light == "yellow":
                            if "VƯỢT ĐÈN VÀNG" not in self.recorded_violations[track_id]:
                                detected_new_violations.append("VƯỢT ĐÈN VÀNG")
                                self.recorded_violations[track_id].add("VƯỢT ĐÈN VÀNG")

        # =================================================================
        # LUẬT 2: ĐI NGƯỢC CHIỀU
        # =================================================================
        if "ĐI NGƯỢC CHIỀU" not in self.recorded_violations[track_id]:
            default_allowed_direction = "up" 
            if check_wrong_way(trajectory, default_allowed_direction, min_dist=20):
                detected_new_violations.append("ĐI NGƯỢC CHIỀU")
                self.recorded_violations[track_id].add("ĐI NGƯỢC CHIỀU")

        # =================================================================
        # LUẬT 3: SAI LÀN (Do User tự vẽ thủ công)
        # =================================================================
        if "SAI LÀN" not in self.recorded_violations[track_id]:
            for poly_zone in zones_config.get("polygons", []):
                if poly_zone.label == "wrong_lane":
                    pts = np.array([[p.x, p.y] for p in poly_zone.points], np.int32).reshape((-1, 1, 2))
                    if is_point_in_polygon(bottom_center, pts):
                        detected_new_violations.append("SAI LÀN")
                        self.recorded_violations[track_id].add("SAI LÀN")
                        break 

        # =================================================================
        # LUẬT 4: ĐI VÀO ĐƯỜNG CẤM (Forbidden Mode)
        # =================================================================
        if "ĐI VÀO ĐƯỜNG CẤM" not in self.recorded_violations[track_id]:
            for poly_zone in zones_config.get("polygons", []):
                if poly_zone.label == "forbidden":
                    pts = np.array([[p.x, p.y] for p in poly_zone.points], np.int32).reshape((-1, 1, 2))
                    if is_point_in_polygon(bottom_center, pts):
                        detected_new_violations.append("ĐI VÀO ĐƯỜNG CẤM")
                        self.recorded_violations[track_id].add("ĐI VÀO ĐƯỜNG CẤM")
                        break

        # Trả về các lỗi VỪA MỚI PHÁT HIỆN để main_edge gửi MQTT
        return detected_new_violations

    def reset(self):
        """Xóa trắng bộ nhớ khi chuyển đổi video hoặc reset hệ thống"""
        self.recorded_violations.clear()
        self.pending_red_lights.clear()
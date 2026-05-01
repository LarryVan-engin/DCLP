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
<<<<<<< HEAD
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
=======
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
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
        self.wrong_way_candidates = defaultdict(int)

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
            if self._confirm_wrong_way(track_id, trajectory, allowed_direction="up"):
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

    def _confirm_wrong_way(self, track_id: int, trajectory: List[Tuple[int, int]], allowed_direction: str) -> bool:
        """Chi phat nguoc chieu khi xe da di chuyen du xa va sai huong lien tiep."""
        min_points = 12
        min_total_dist = 55
        min_axis_dist = 35
        min_consecutive_frames = 5

        if len(trajectory) < min_points:
            self.wrong_way_candidates[track_id] = 0
            return False

        start_pt = trajectory[0]
        end_pt = trajectory[-1]
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        total_dist = (dx ** 2 + dy ** 2) ** 0.5

        if total_dist < min_total_dist or abs(dy) < min_axis_dist:
            self.wrong_way_candidates[track_id] = 0
            return False

        wrong_direction = (allowed_direction == "up" and dy > 0) or (allowed_direction == "down" and dy < 0)
        mostly_vertical = abs(dy) > abs(dx) * 1.2

        recent = trajectory[-min_points:]
        recent_dy = recent[-1][1] - recent[0][1]
        recent_wrong = (allowed_direction == "up" and recent_dy > 0) or (allowed_direction == "down" and recent_dy < 0)

        if wrong_direction and mostly_vertical and recent_wrong:
            self.wrong_way_candidates[track_id] += 1
        else:
            self.wrong_way_candidates[track_id] = 0

        return self.wrong_way_candidates[track_id] >= min_consecutive_frames

    def reset(self):
        """Xóa trắng bộ nhớ khi chuyển đổi video hoặc reset hệ thống"""
        self.recorded_violations.clear()
        self.pending_red_lights.clear()
        self.wrong_way_candidates.clear()
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import edge_config as cfg
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
                         zones_config: dict,
                         stop_line_y: int = None,
                         vehicle_cls: int = None,
                         was_right_lane: bool = False,
                         in_roi: bool = True) -> List[str]:
        """
        Quét toàn bộ luật vi phạm.
        stop_line_y   : Tọa độ Y pixel của vạch dừng (đỉnh ROI).
        vehicle_cls   : Class ID YOLO của xe (3=motorcycle, 2/5/7=car/bus/truck).
        was_right_lane: True nếu xe máy đã từng ở trong làn rẽ phải trước khi vượt vạch.
                        Chỉ xe máy có cờ này mới được xóa lỗi khi xác nhận rẽ phải.
        """
        detected_new_violations = []
        x1, y1, x2, y2 = bbox
        bottom_center = get_bottom_center(bbox)

        # Xe máy từ đúng làn rẽ phải mới được miễn lỗi đèn đỏ khi rẽ phải
        is_moto = (vehicle_cls in cfg.MOTO_CLASSES) if vehicle_cls is not None else False
        can_exempt_right_turn = is_moto and was_right_lane

        # =================================================================
        # LUẬT 1: VƯỢT ĐÈN ĐỎ / VÀNG
        # =================================================================
        current_light = light_status.get("straight", "unknown").lower()

        if track_id in self.pending_red_lights:
            suspect_data = self.pending_red_lights[track_id]
            suspect_data["frames_waited"] += 1
            suspect_data["bbox"] = bbox  # Cập nhật bbox mới nhất để crop đúng vị trí khi mất tracking

            start_p = suspect_data["cross_point"]
            dx = bottom_center[0] - start_p[0]
            dy = bottom_center[1] - start_p[1]

            # Giống full_main.py: dx > 15 AND dx > abs(dy)*1.5 (chặt hơn 0.35 cũ)
            is_turning_right = dx > cfg.RIGHT_TURN_DX_THRESHOLD and dx > abs(dy) * cfg.RIGHT_TURN_RATIO_THRESHOLD

            if is_turning_right and can_exempt_right_turn:
                # Xe máy từ làn phải, xác nhận rẽ phải → xóa án chờ, không phạt
                del self.pending_red_lights[track_id]
            elif suspect_data["frames_waited"] >= cfg.RED_LIGHT_WAIT_FRAMES:
                # Hết thời gian chờ mà không rẽ phải → chốt lỗi
                if "VƯỢT ĐÈN ĐỎ" not in self.recorded_violations[track_id]:
                    detected_new_violations.append("VƯỢT ĐÈN ĐỎ")
                    self.recorded_violations[track_id].add("VƯỢT ĐÈN ĐỎ")
                del self.pending_red_lights[track_id]
        else:
            # So sánh Y trực tiếp — Điểm kiểm tra: tâm bbox (center_y)
            crossed = False
            if stop_line_y is not None:
                center_y = (y1 + y2) // 2
                crossed = (center_y < stop_line_y)
            else:
                for line_zone in zones_config.get("lines", []):
                    if line_zone.label == "stop_line":
                        if check_vehicle_crossed_line(trajectory, line_zone):
                            crossed = True
                            break

            if crossed:
                if current_light == "red":
                    if can_exempt_right_turn:
                        if track_id not in self.pending_red_lights:
                            self.pending_red_lights[track_id] = {
                                "cross_point": bottom_center,
                                "frames_waited": 0,
                                "bbox": bbox,
                                "can_exempt": can_exempt_right_turn  # lưu lại để cleanup_lost_tracks dùng
                            }
                    else:
                        if "VƯỢT ĐÈN ĐỎ" not in self.recorded_violations[track_id]:
                            detected_new_violations.append("VƯỢT ĐÈN ĐỎ")
                            self.recorded_violations[track_id].add("VƯỢT ĐÈN ĐỎ")
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
        if in_roi and "SAI LÀN" not in self.recorded_violations[track_id]:
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

    def cleanup_lost_tracks(self, active_tracks: set) -> List[dict]:
        """Kiểm tra xe đang chờ phạt Đèn Đỏ nhưng mất tracking -> Chốt lỗi."""
        lost_violations = []
        lost_tracks = [t for t in self.pending_red_lights.keys() if t not in active_tracks]
        
        for track_id in lost_tracks:
            # Nếu mất tracking -> Coi như đi thẳng vượt đèn đỏ
            if "VƯỢT ĐÈN ĐỎ" not in self.recorded_violations[track_id]:
                lost_violations.append({
                    "track_id": track_id,
                    "violation_type": "VƯỢT ĐÈN ĐỎ",
                    "bbox": self.pending_red_lights[track_id]["bbox"]
                })
                self.recorded_violations[track_id].add("VƯỢT ĐÈN ĐỎ")
            del self.pending_red_lights[track_id]
            
        return lost_violations

    def reset(self):
        """Xóa trắng bộ nhớ khi chuyển đổi video hoặc reset hệ thống"""
        self.recorded_violations.clear()
        self.pending_red_lights.clear()
        self.wrong_way_candidates.clear()
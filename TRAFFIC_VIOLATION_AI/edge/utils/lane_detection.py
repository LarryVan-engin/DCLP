"""
********************************************************************************************************************
Project:      Traffic Violation Detection (Pro Version - MQTT Hybrid)
File:         edge/utils/lane_detection.py
Description:  Thuật toán Data-Driven Lane Detection (Tác giả: Larry Van).
              Tự động gom cụm 1D, chia làn ô tô độc lập với làn xe máy.
              Đã tích hợp tối ưu RAM (Memory Leak Prevention) và ROI Reset.
********************************************************************************************************************
"""

import numpy as np

class LaneDetector:
    def __init__(self, observation_frames=100, frame_width=1280, frame_height=720):
        self.observation_frames = observation_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Khởi tạo các biến trạng thái và bộ nhớ
        self.reset_learning()
        
        # Vùng ROI mặc định (vùng màu xanh trên hình)
        self.roi_pts = np.array([
            [0, int(frame_height * 0.3)], 
            [frame_width, int(frame_height * 0.3)], 
            [frame_width, frame_height], 
            [0, frame_height]
        ], np.int32)

    def reset_learning(self):
        """Xóa trắng bộ nhớ, yêu cầu AI học lại từ đầu (dùng khi thay đổi ROI)"""
        self.frame_count = 0
        self.is_ready = False
        self.car_frames = []           # Danh sách per-frame: [[box,...], [box,...], ...]
        self.moto_boxes_learning = []
        self.car_only_zones = []
        self.virtual_lines = []
        print("[LANE DETECTION] 🔄 Đã reset bộ nhớ. Bắt đầu thu thập dữ liệu phân làn mới...")

    def set_roi(self, roi_pts_list):
        """Cập nhật ROI mới từ Dashboard. Nếu có thay đổi, bắt buộc học lại."""
        if roi_pts_list and len(roi_pts_list) == 4:
            new_roi = np.array(roi_pts_list, np.int32)
            if not np.array_equal(self.roi_pts, new_roi):
                self.roi_pts = new_roi
                self.reset_learning()

    def get_normalized_x(self, x, y):
        """Chuyển đổi toạ độ X thực tế sang tỷ lệ 0.0 -> 1.0 dựa trên hình thang perspective"""
        top_y, bot_y = self.roi_pts[0][1], self.roi_pts[2][1]
        
        if y <= top_y: y = top_y + 1
        if y >= bot_y: y = bot_y - 1
        
        ratio = (y - top_y) / float(bot_y - top_y)
        left_x = self.roi_pts[0][0] + ratio * (self.roi_pts[3][0] - self.roi_pts[0][0])
        right_x = self.roi_pts[1][0] + ratio * (self.roi_pts[2][0] - self.roi_pts[1][0])
        
        # Tránh chia cho 0
        width_at_y = max(right_x - left_x, 1e-6)
        return np.clip((x - left_x) / width_at_y, 0.0, 1.0)

    def update_learning_data(self, boxes, clss):
        """Thu thập dữ liệu theo từng frame riêng biệt.
        Mỗi frame lưu danh sách xe riêng để tìm frame có nhiều xe nhất.
        """
        if self.is_ready: return
        
        self.frame_count += 1
        frame_cars = []
        frame_motos = []

        for box, cls_id in zip(boxes, clss):
            cls_id = int(cls_id)
            x1, y1, x2, y2 = map(int, box)
            if cls_id in [2, 5, 7]:
                frame_cars.append((x1, y1, x2, y2))
            elif cls_id == 3:
                frame_motos.append((x1, y1, x2, y2))

        # Lưu per-frame để sau này tìm frame nhiều xe nhất
        if frame_cars:
            self.car_frames.append(frame_cars)
        # Xe máy vẫn tích lũy để lấy thống kê ranh giới
        self.moto_boxes_learning.extend(frame_motos)

        if self.frame_count >= self.observation_frames:
            self._calculate_lanes()

    def _calculate_lanes(self):
        """Logic đơn giản:
        - Làn ô tô 1 : [0.0 → MAX(mép phải xe trái nhất)] qua 100 frame
        - Vùng giữa : phần còn lại đến ranh giới xe máy (không chia nhỏ)
        - Làn xe máy : [m_left → 1.0] nếu có
        """
        top_y = self.roi_pts[0][1]
        bot_y = self.roi_pts[2][1]

        if not self.car_frames:
            self.car_only_zones.append((0.0, 0.35))
            boundaries = [0.0, 0.35, 1.0]
            print("[LANE DETECTION] Không phát hiện ô tô, dùng cấu hình mặc định.")
            self._finalize(boundaries, top_y, bot_y)
            return

        # ── BƯỚC 1: Tìm MAX mép phải của xe ô tô trái nhất qua tất cả frame ──────
        w_car_1_max = 0.0

        for frame_cars in self.car_frames:
            frame_norm = []
            for (x1, y1, x2, y2) in frame_cars:
                nx1 = self.get_normalized_x(x1, y2)
                nx2 = self.get_normalized_x(x2, y2)
                if (nx2 - nx1) > 0.02:  # loại box quá hẹp (xe ở rìa ROI)
                    frame_norm.append({'left': nx1, 'right': nx2, 'center': (nx1 + nx2) / 2})

            if frame_norm:
                leftmost = min(frame_norm, key=lambda c: c['center'])
                w_car_1_max = max(w_car_1_max, leftmost['right'])

        if w_car_1_max < 0.05:
            w_car_1_max = 0.35  # fallback an toàn

        # ── BƯỚC 2: m_left từ xe máy (ranh giới phải vùng ô tô) ──────────────
        motos_norm = []
        for (x1, y1, x2, y2) in self.moto_boxes_learning:
            nx1 = self.get_normalized_x(x1, y2)
            nx2 = self.get_normalized_x(x2, y2)
            motos_norm.append({'left': nx1, 'right': nx2, 'center': (nx1 + nx2) / 2})

        if motos_norm:
            right_motos = [m for m in motos_norm if m['center'] > w_car_1_max]
            if right_motos:
                m_left = float(np.percentile([m['left'] for m in right_motos], 10))
            else:
                m_left = 1.0
        else:
            m_left = 1.0

        m_left = max(w_car_1_max + 0.05, min(m_left, 0.95))

        # ── BƯỚC 3: Ranh giới 2 vùng đơn giản ───────────────────────────────
        # [0.0, w_car_1_max]   = Làn ô tô 1 (Car Only)
        # [w_car_1_max, m_left] = Vùng giữa (không chia nhỏ)
        # [m_left, 1.0]         = Làn xe máy
        boundaries = [0.0, w_car_1_max]
        if m_left < 1.0:
            boundaries.append(m_left)
        boundaries.append(1.0)

        # ── BƯỚC 4: Chỉ đánh dấu làn ô tô 1 là vùng cấm xe máy ───────────────
        self.car_only_zones.append((0.0, w_car_1_max))

        print(f"[LANE DETECTION] Đã học xong!")
        print(f"[LANE DETECTION] Mép phải MAX làn ô tô 1 : {w_car_1_max:.3f}")
        print(f"[LANE DETECTION] Ranh giới xe máy (m_left): {m_left:.3f}")
        print(f"[LANE DETECTION] Boundaries : {[round(b,3) for b in boundaries]}")
        self._finalize(boundaries, top_y, bot_y)

    def _finalize(self, boundaries, top_y, bot_y):
        """Tính virtual_lines và giải phóng RAM."""
        self.virtual_lines = []
        for boundary in boundaries[1:-1]:
            top_x = int(self.roi_pts[0][0] + boundary * (self.roi_pts[1][0] - self.roi_pts[0][0]))
            bot_x = int(self.roi_pts[3][0] + boundary * (self.roi_pts[2][0] - self.roi_pts[3][0]))
            self.virtual_lines.append((top_x, top_y, bot_x, bot_y))

        self.is_ready = True
        self.car_frames.clear()
        self.moto_boxes_learning.clear()




    def check_wrong_lane(self, bbox, cls_id) -> bool:
        """
        Kiểm tra lỗi đi sai làn.
        Chỉ kiểm tra xe máy (cls_id == 3). Nếu lọt vào Vùng cấm (Làn ô tô) -> Vi phạm.
        """
        if not self.is_ready or int(cls_id) != 3: 
            return False 
            
        x1, y1, x2, y2 = map(int, bbox)
        center_x, bottom_y = (x1 + x2) // 2, y2
        
        norm_x = self.get_normalized_x(center_x, bottom_y)
        
        # Quét xem tâm bánh xe máy có nằm trong vùng chỉ dành cho ô tô không
        for (z_start, z_end) in self.car_only_zones:
            if z_start <= norm_x <= z_end:
                return True
                
        return False
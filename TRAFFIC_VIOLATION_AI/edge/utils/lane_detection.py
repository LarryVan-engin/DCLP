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
        """Port trực tiếp từ calculate_data_driven_lanes() trong demo gốc:
        1. Greedy cluster tâm ô tô (gap ≥ 0.20 → làn mới) → N làn.
        2. r_boundary = 90th-percentile cạnh phải cụm cuối,
           nếu đủ xe máy bên phải thì lấy điểm giữa r_car_max và moto_left_15th.
        3. Chia đều: lane_width = r_boundary / N → ranh giới từng làn.
        4. car_only_zones: làn có ≤5 xe máy = Làn Ô Tô thuần.
        """
        top_y = self.roi_pts[0][1]
        bot_y = self.roi_pts[2][1]

        # Flatten car_frames → danh sách phẳng (như demo dùng car_boxes_learning)
        all_car_boxes = [box for frame in self.car_frames for box in frame]

        # ── Chuẩn hoá toạ độ ────────────────────────────────────────────────
        cars_norm = []
        for (x1, y1, x2, y2) in all_car_boxes:
            nx1 = self.get_normalized_x(x1, y2)
            nx2 = self.get_normalized_x(x2, y2)
            if (nx2 - nx1) > 0.02:
                cars_norm.append({'left': nx1, 'right': nx2, 'center': (nx1 + nx2) / 2.0})

        motos_norm = []
        for (x1, y1, x2, y2) in self.moto_boxes_learning:
            nx1 = self.get_normalized_x(x1, y2)
            nx2 = self.get_normalized_x(x2, y2)
            if (nx2 - nx1) > 0.01:
                motos_norm.append({'left': nx1, 'right': nx2, 'center': (nx1 + nx2) / 2.0})

        if not cars_norm:
            self.car_only_zones = []
            boundaries = [0.0, 0.75, 1.0]
            print(f"[LANE DETECTION] ✅ Không có ô tô → 1 vạch tại 0.75")
            self._finalize(boundaries, top_y, bot_y)
            return

        # ── Greedy cluster tâm ô tô (gap ≥ 0.20 → làn mới) ─────────────────
        car_centers = sorted([c['center'] for c in cars_norm])
        clusters = []
        for c in car_centers:
            if not clusters:
                clusters.append([c])
            else:
                if c - np.mean(clusters[-1]) < 0.20:
                    clusters[-1].append(c)
                else:
                    clusters.append([c])

        N_car_lanes = len(clusters)

        # ── Ranh giới phải vùng ô tô ─────────────────────────────────────────
        last_cluster_set = set(clusters[-1])
        clast_rights = [c['right'] for c in cars_norm if c['center'] in last_cluster_set]
        r_car_max = float(np.percentile(clast_rights, 90)) if clast_rights else 0.6

        right_motos = [m for m in motos_norm if m['center'] > r_car_max - 0.05]

        if len(right_motos) > 5:
            m_left = float(np.percentile([m['left'] for m in right_motos], 15))
            r_boundary = (r_car_max + m_left) / 2.0
            r_boundary = max(r_car_max + 0.02, r_boundary)
        else:
            r_boundary = r_car_max

        r_boundary = max(0.2, r_boundary)

        # ── Chia đều làn, đánh dấu car_only_zones ───────────────────────────
        lane_width = r_boundary / N_car_lanes
        boundaries = [0.0]
        car_only_zones = []

        for i in range(N_car_lanes):
            b_left  = i * lane_width
            b_right = (i + 1) * lane_width
            motos_in_zone = sum(1 for m in motos_norm if b_left <= m['center'] <= b_right)
            if motos_in_zone <= 5:
                car_only_zones.append((b_left, b_right))
            boundaries.append(b_right)

        boundaries.append(1.0)  # đảm bảo _finalize vẽ đường tại r_boundary

        # Loại ranh giới quá gần nhau (< 0.05)
        merged = [boundaries[0]]
        for b in boundaries[1:]:
            if b - merged[-1] >= 0.05:
                merged.append(b)
        if merged[-1] < 1.0:
            merged.append(1.0)
        boundaries = merged

        self.car_only_zones = car_only_zones

        print(f"[LANE DETECTION] ✅ Đã học xong! Số làn ô tô: {N_car_lanes}")
        print(f"[LANE DETECTION] r_boundary: {r_boundary:.3f}")
        print(f"[LANE DETECTION] Boundaries vẽ: {[round(b, 3) for b in boundaries]}")
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
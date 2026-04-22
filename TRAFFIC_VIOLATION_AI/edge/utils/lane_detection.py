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
        self.car_boxes_learning = []
        self.moto_boxes_learning = []
        self.car_only_zones = []  # [(min_norm_x, max_norm_x)] -> Vùng chỉ dành cho ô tô
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
        """Thu thập dữ liệu trong giai đoạn Warm-up (Mặc định 100 frames)"""
        if self.is_ready: return
        
        self.frame_count += 1
        for box, cls_id in zip(boxes, clss):
            cls_id = int(cls_id)
            x1, y1, x2, y2 = map(int, box)
            
            # 2: Car, 5: Bus, 7: Truck | 3: Motorcycle
            if cls_id in [2, 5, 7]:
                self.car_boxes_learning.append((x1, y1, x2, y2))
            elif cls_id == 3:
                self.moto_boxes_learning.append((x1, y1, x2, y2))
                
        # Đủ frame -> Chạy thuật toán chốt làn
        if self.frame_count >= self.observation_frames:
            self._calculate_lanes()

    def _calculate_lanes(self):
        """Thuật toán gom cụm (Clustering) 1D thông minh của Larry Van"""
        cars_norm = []
        for (x1, y1, x2, y2) in self.car_boxes_learning:
            nx1 = self.get_normalized_x(x1, y2)
            nx2 = self.get_normalized_x(x2, y2)
            cars_norm.append({'left': nx1, 'right': nx2, 'center': (nx1 + nx2)/2})

        motos_norm = []
        for (x1, y1, x2, y2) in self.moto_boxes_learning:
            nx1 = self.get_normalized_x(x1, y2)
            nx2 = self.get_normalized_x(x2, y2)
            motos_norm.append({'left': nx1, 'right': nx2, 'center': (nx1 + nx2)/2})

        if not cars_norm:
            # Fallback nếu video không có ô tô nào chạy qua trong 100 frame đầu
            self.car_only_zones.append((0.0, 0.4))
            N_car_lanes = 0
        else:
            # 1. Gom cụm theo mật độ tâm xe ô tô
            car_centers = [c['center'] for c in cars_norm]
            car_centers.sort()
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
            last_cluster = clusters[-1]
            
            # 2. Tìm điểm ngoài cùng bên phải của dòng ô tô
            clast_rights = [c['right'] for c in cars_norm if c['center'] in last_cluster]
            r_car_max = np.percentile(clast_rights, 90) if clast_rights else 0.8
            
            # 3. Tìm các xe máy chạy sát rìa phải của ô tô
            right_motos = [m for m in motos_norm if m['center'] > r_car_max - 0.05]
            
            # 4. Xác định biên giới tuyệt đối giữa vùng ô tô và xe máy
            if len(right_motos) > 5:
                m_left = np.percentile([m['left'] for m in right_motos], 15)
                r_boundary = max(r_car_max + 0.02, (r_car_max + m_left) / 2) 
            else:
                r_boundary = r_car_max

            r_boundary = max(0.2, r_boundary)
            
            # 5. Chia đều không gian ô tô thành các làn bằng nhau
            lane_width = r_boundary / max(N_car_lanes, 1)
            
            # 6. Kiểm tra từng làn ô tô, nếu vắng xe máy -> Chốt làm vùng cấm xe máy
            for i in range(N_car_lanes):
                b_left = i * lane_width
                b_right = (i + 1) * lane_width
                
                motos_in_zone = sum(1 for m in motos_norm if b_left <= m['center'] <= b_right)
                
                if motos_in_zone <= 5:
                    self.car_only_zones.append((b_left, b_right)) 

        # ==========================================================
        # ĐÓNG GÓI KẾT QUẢ & GIẢI PHÓNG RAM (CRITICAL FIX)
        # ==========================================================
        self.is_ready = True
        
        # Xóa mảng lưu tọa độ vì đã tính toán xong -> Chống Memory Leak trên Jetson
        self.car_boxes_learning.clear()
        self.moto_boxes_learning.clear()
        
        print(f"[LANE DETECTION] Đã học xong! Số làn ô tô: {N_car_lanes if cars_norm else 0}")
        print(f"[LANE DETECTION] Vùng cấm xe máy (Car Only Zones): {self.car_only_zones}")

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
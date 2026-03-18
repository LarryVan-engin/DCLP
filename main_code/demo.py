"""
DEMO TRAFFIC VIOLATION - STRICT LEFT-LANE & DYNAMIC MIDPOINT
- Làn trái cùng (Car Only): Rìa bouding box phải của nhóm ô tô đầu tiên.
- Làn phải cùng (Moto): Midpoint giữa khối ô tô và khối xe máy.
- Làn giữa (Tổng hợp): Toàn bộ không gian còn lại (kể cả chứa các cụm ô tô song song).
- Vi phạm: Phạt gắt gao xe máy lấn vào Làn trái cùng.
- Tự động hóa: Cạnh trên của khối ROI chính là Vạch Dừng (Stop Line).
- Bắt 3 loại vi phạm độc lập: 
    1. Vượt đèn đỏ (Red Light)
    2. Đi sai làn (Wrong Lane)
    3. Đi ngược chiều (Wrong Way - Tính toán dựa trên vector Y hướng về camera)
"""

import cv2
import numpy as np
import os
import json
from ultralytics import YOLO
from collections import deque
import easyocr
import re

# ==================== CONFIG CƠ BẢN ====================
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
OBSERVATION_FRAMES = 100
LANE_ALPHA = 0.20
PATH_HISTORY = 30
PLATE_CONF = 0.35

VIDEO_NAME = r"E:\Video\train\video_test.mp4" 
CONFIG_FILE = f"config_{os.path.splitext(os.path.basename(VIDEO_NAME))[0]}.json"

# MODELS
coco_model = YOLO("yolo12n.pt")
plate_model = YOLO(r"D:\VSCode\DCLP\main_code\runs\detect\model_detect_license_plate.pt")
ocr_reader = easyocr.Reader(["en"], gpu=True)

# Tách biệt Class
CAR_CLASSES = {2, 5, 7} # Car, Bus, Truck
MOTO_CLASSES = {3}      # Motorcycle
VEHICLE_CLASSES = CAR_CLASSES.union(MOTO_CLASSES)

# ==================== HELPERS ====================
def nothing(x): pass

def extract_boxes(res):
    if res is None or res.boxes is None: return np.empty((0,4)), np.empty((0,)), np.empty((0,))
    return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()

def read_plate(crop):
    if crop.size == 0: return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    res = ocr_reader.readtext(gray, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.", detail=1)
    if not res: return ""
    return re.sub(r"[^A-Z0-9]", "", "".join([r[1] for r in res]).upper())

def get_normalized_x(x, y, roi_pts):
    top_y, bot_y = roi_pts[0][1], roi_pts[2][1]
    if y <= top_y: y = top_y + 1
    if y >= bot_y: y = bot_y - 1
    
    ratio = (y - top_y) / (bot_y - top_y)
    left_x = roi_pts[0][0] + ratio * (roi_pts[3][0] - roi_pts[0][0])
    right_x = roi_pts[1][0] + ratio * (roi_pts[2][0] - roi_pts[1][0])
    
    return np.clip((x - left_x) / (right_x - left_x + 1e-6), 0.0, 1.0)

# ==================== GIAI ĐOẠN 1: GUI ====================
def run_calibration_gui(video_path, config_file):
    print("KHỞI ĐỘNG SETUP:...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cv2.namedWindow("CALIBRATION_TOOL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CALIBRATION_TOOL", 1000, 600)
    
    cv2.createTrackbar("Frame", "CALIBRATION_TOOL", 415, total_frames - 1, nothing)
    cv2.createTrackbar("ROI_Center_X", "CALIBRATION_TOOL", FRAME_WIDTH//2, FRAME_WIDTH, nothing)
    cv2.createTrackbar("ROI_Top_Y", "CALIBRATION_TOOL", int(FRAME_HEIGHT*0.3), FRAME_HEIGHT, nothing)
    cv2.createTrackbar("ROI_Top_W", "CALIBRATION_TOOL", 300, FRAME_WIDTH, nothing)
    cv2.createTrackbar("ROI_Bot_W", "CALIBRATION_TOOL", 800, FRAME_WIDTH, nothing)

    last_frame_idx = -1
    frame = None

    while True:
        current_frame_idx = cv2.getTrackbarPos("Frame", "CALIBRATION_TOOL")
        if current_frame_idx != last_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if ret: frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            last_frame_idx = current_frame_idx
            
        if frame is None: break
        display = frame.copy()
        
        center_x = cv2.getTrackbarPos("ROI_Center_X", "CALIBRATION_TOOL")
        roi_top_y = cv2.getTrackbarPos("ROI_Top_Y", "CALIBRATION_TOOL")
        roi_top_w = cv2.getTrackbarPos("ROI_Top_W", "CALIBRATION_TOOL")
        roi_bot_w = cv2.getTrackbarPos("ROI_Bot_W", "CALIBRATION_TOOL")
        
        pts = np.array([[center_x - roi_top_w//2, roi_top_y], [center_x + roi_top_w//2, roi_top_y],
                        [center_x + roi_bot_w//2, FRAME_HEIGHT], [center_x - roi_bot_w//2, FRAME_HEIGHT]], np.int32)
        
        cv2.polylines(display, [pts], True, (0, 255, 255), 2)
        cv2.line(display, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 4)
        cv2.putText(display, "AUTO STOP LINE", (pts[0][0] + 10, pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.putText(display, "Chinh ROI -> Bam 'S' de Luu", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("CALIBRATION_TOOL", display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            config = {"roi_pts": pts.tolist()}
            with open(config_file, 'w') as f: json.dump(config, f)
            print(f"Đã lưu cấu hình ROI vào {config_file}")
            break
        elif key == ord('q'): break
            
    cv2.destroyAllWindows()
    cap.release()
    return config

# ==================== TOÁN HỌC: QUY TẮC LÀN CHUẨN (KHẢO SÁT DATA) ====================
def calculate_data_driven_lanes(car_boxes, moto_boxes, roi_pts):
    cars_norm = []
    for (x1, y1, x2, y2) in car_boxes:
        nx1 = get_normalized_x(x1, y2, roi_pts)
        nx2 = get_normalized_x(x2, y2, roi_pts)
        cars_norm.append({'left': nx1, 'right': nx2, 'center': (nx1 + nx2)/2})

    motos_norm = []
    for (x1, y1, x2, y2) in moto_boxes:
        nx1 = get_normalized_x(x1, y2, roi_pts)
        nx2 = get_normalized_x(x2, y2, roi_pts)
        motos_norm.append({'left': nx1, 'right': nx2, 'center': (nx1 + nx2)/2})

    boundaries = [0.0]
    lane_labels = []
    car_only_zones = [] # Lưu các khoảng X bị cấm xe máy

    if not cars_norm:
        boundaries = [0.0, 0.4, 0.7, 1.0]
        lane_labels = ["Lane O To (CARS ONLY)", "Lane Tong Hop", "Lane Xe May"]
        car_only_zones.append((0.0, 0.4))
    else:
        # 1. ĐẾM SỐ LƯỢNG LÀN Ô TÔ
        car_centers = [c['center'] for c in cars_norm]
        car_centers.sort()
        clusters = []
        for c in car_centers:
            if not clusters: clusters.append([c])
            else:
                if c - np.mean(clusters[-1]) < 0.20: clusters[-1].append(c)
                else: clusters.append([c])
                
        N_car_lanes = len(clusters)

        # 2. TÌM R_MAX CỦA Ô TÔ VÀ MIDPOINT VỚI XE MÁY ĐỂ CHỐT R_BOUNDARY (TỔNG KHÔNG GIAN)
        last_cluster = clusters[-1]
        clast_rights = [c['right'] for c in cars_norm if c['center'] in last_cluster]
        r_car_max = np.percentile(clast_rights, 90) # Rìa phải xa nhất của ô tô
        
        right_motos = [m for m in motos_norm if m['center'] > r_car_max - 0.05]
        
        if len(right_motos) > 5:
            # Có xe máy đi bên phải -> Ranh giới là Midpoint
            m_left = np.percentile([m['left'] for m in right_motos], 15)
            r_boundary = (r_car_max + m_left) / 2
            r_boundary = max(r_car_max + 0.02, r_boundary) # Đảm bảo hở ra một tí
        else:
            # Cao tốc, không có xe máy -> Ranh giới ép sát mép ô tô
            r_boundary = r_car_max

        # Đảm bảo không gian khối ô tô không nhỏ hơn 20%
        r_boundary = max(0.2, r_boundary)

        # 3. CHIA ĐỀU KHÔNG GIAN VÀ "KHẢO SÁT" ĐỂ DÁN NHÃN
        lane_width = r_boundary / N_car_lanes
        
        for i in range(N_car_lanes):
            b_left = i * lane_width
            b_right = (i + 1) * lane_width
            
            # KHẢO SÁT: Đếm xem có bao nhiêu xe máy lọt vào vùng [b_left, b_right] này
            motos_in_zone = sum(1 for m in motos_norm if b_left <= m['center'] <= b_right)
            
            if motos_in_zone > 5:
                # Nếu có xe máy xuất hiện -> Làn đó bản chất là Làn Tổng Hợp
                label = f"Lane Tong Hop {i+1}" if N_car_lanes > 1 else "Lane Tong Hop"
            else:
                # Nếu hoàn toàn không có xe máy -> Làn đó chính là Làn Ô Tô (Cấm địa)
                label = f"Lane O To {i+1}" if N_car_lanes > 1 else "Lane O To (CARS ONLY)"
                car_only_zones.append((b_left, b_right)) # Đưa vào danh sách bắt lỗi
                
            boundaries.append(b_right)
            lane_labels.append(label)

        # 4. LÀN XE MÁY CÒN LẠI (NẾU CÓ)
        if len(right_motos) > 5:
            boundaries.append(1.0)
            lane_labels.append("Lane Xe May")

    # --- DỰNG POLYGON ---
    top_y, bot_y = roi_pts[0][1], roi_pts[2][1]
    computed_lanes = []
    virtual_lines = []
    
    # Chuẩn hóa để không tràn lề
    for i in range(1, len(boundaries)):
        boundaries[i] = max(boundaries[i-1] + 0.02, boundaries[i])
        if i == len(boundaries) - 1 and "Xe May" in lane_labels[-1]: boundaries[i] = 1.0
        else: boundaries[i] = min(boundaries[i], 1.0)
    
    for i in range(len(boundaries) - 1):
        b_left, b_right = boundaries[i], boundaries[i+1]
        top_l_x = int(roi_pts[0][0] + b_left * (roi_pts[1][0] - roi_pts[0][0]))
        bot_l_x = int(roi_pts[3][0] + b_left * (roi_pts[2][0] - roi_pts[3][0]))
        top_r_x = int(roi_pts[0][0] + b_right * (roi_pts[1][0] - roi_pts[0][0]))
        bot_r_x = int(roi_pts[3][0] + b_right * (roi_pts[2][0] - roi_pts[3][0]))
        
        poly = np.array([[top_l_x, top_y], [top_r_x, top_y], [bot_r_x, bot_y], [bot_l_x, bot_y]], np.int32)
        computed_lanes.append((poly, lane_labels[i]))
        if i > 0: virtual_lines.append((top_l_x, top_y, bot_l_x, bot_y))
            
    return computed_lanes, virtual_lines, car_only_zones

# ==================== MAIN LOOP ====================
def process_video(input_path, output_path="output_all_violations.avi"):
    config_file = f"config_{os.path.splitext(os.path.basename(input_path))[0]}.json"
    if not os.path.exists(config_file):
        config = run_calibration_gui(input_path, config_file)
        if not config: return
    else:
        with open(config_file, 'r') as f: config = json.load(f)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (FRAME_WIDTH, FRAME_HEIGHT))

    roi_polygon = np.array(config["roi_pts"], np.int32)
    roi_top_y = roi_polygon[0][1] 

    frame_idx = 0
    vehicles = {}
    os.makedirs("plate_crops", exist_ok=True)
    is_red_light = True 
    
    car_boxes_learning = []
    moto_boxes_learning = []
    computed_lanes = []
    virtual_lines = []
    car_only_zones = [] # Cập nhật mảng bắt lỗi

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        display = frame.copy()
        frame_idx += 1

        res = coco_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.35)[0]
        boxes, _, clss = extract_boxes(res)
        track_ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else []
        
        if frame_idx <= OBSERVATION_FRAMES:
            cv2.putText(display, f"LEARNING DATA-DRIVEN LANES... {frame_idx}/{OBSERVATION_FRAMES}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 3)
            for i, box in enumerate(boxes):
                cls_id = int(clss[i])
                if cls_id not in VEHICLE_CLASSES: continue
                x1, y1, x2, y2 = map(int, box)
                center_x, bottom_y = (x1 + x2) // 2, y2
                
                if cv2.pointPolygonTest(roi_polygon, (center_x, bottom_y), False) >= 0:
                    if cls_id in CAR_CLASSES: car_boxes_learning.append((x1, y1, x2, y2))
                    else: moto_boxes_learning.append((x1, y1, x2, y2))
                        
            if frame_idx == OBSERVATION_FRAMES:
                computed_lanes, virtual_lines, car_only_zones = calculate_data_driven_lanes(car_boxes_learning, moto_boxes_learning, config["roi_pts"])

        # Vẽ Làn
        overlay = display.copy()
        
        for idx, (lane_poly, label) in enumerate(computed_lanes):
            # Chọn màu tự động theo nhãn dán
            if "Lane O To" in label: color = (0, 100, 255) # Cam đậm cho làn Cấm xe máy
            elif "Lane Xe May" in label: color = (255, 200, 0) # Vàng
            else: color = (0, 255, 100) # Xanh lá cho Làn Tổng Hợp
                
            cv2.fillPoly(overlay, [lane_poly], color)
            M = cv2.moments(lane_poly)
            if M["m00"] != 0:
                cv2.putText(display, label, (int(M["m10"] / M["m00"]) - 70, int(M["m01"] / M["m00"])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
        cv2.addWeighted(overlay, LANE_ALPHA, display, 1 - LANE_ALPHA, 0, display)

        for line in virtual_lines:
            vx1, vy1, vx2, vy2 = line
            for i in range(0, 10):
                cv2.line(display, (int(vx1 + (vx2-vx1)*(i/10.0)), int(vy1 + (vy2-vy1)*(i/10.0))), 
                                  (int(vx1 + (vx2-vx1)*((i+0.5)/10.0)), int(vy1 + (vy2-vy1)*((i+0.5)/10.0))), (255, 255, 255), 3)

        cv2.polylines(display, [roi_polygon], True, (0, 255, 255), 3) 
        cv2.line(display, tuple(roi_polygon[0]), tuple(roi_polygon[1]), (0, 0, 255), 4)

        # Tracking và Bắt lỗi
        for i, box in enumerate(boxes):
            cls_id = int(clss[i])
            if cls_id not in VEHICLE_CLASSES: continue
            tid = track_ids[i] if i < len(track_ids) else -1
            if tid == -1: continue
            
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y, bottom_y = (x1 + x2) // 2, (y1 + y2) // 2, y2
            
            if cv2.pointPolygonTest(roi_polygon, (center_x, bottom_y), False) < 0: continue

            if tid not in vehicles:
                vehicles[tid] = {'path': deque(maxlen=PATH_HISTORY), 'status': 'OK', 
                                 'wrong_lane': False, 'red_light': False, 'wrong_way': False}
            v = vehicles[tid]
            v['path'].append((center_x, bottom_y))

            norm_x = get_normalized_x(center_x, bottom_y, roi_polygon)

            if len(v['path']) > 10 and not v['wrong_way']:
                dy_way = v['path'][-1][1] - v['path'][-10][1]
                if dy_way > 15: 
                    v['wrong_way'] = True

            # BẮT LỖI SAI LÀN BẰNG DANH SÁCH VÙNG CẤM
            if not v['wrong_lane']:
                if cls_id in MOTO_CLASSES:
                    # Kiểm tra xem tâm xe máy có rơi vào BẤT KỲ làn nào được dán nhãn "Lane O To" không
                    for (z_start, z_end) in car_only_zones:
                        if z_start <= norm_x <= z_end:
                            v['wrong_lane'] = True
                            break # Dính 1 vùng là phạt luôn, không cần xét tiếp

            if is_red_light and not v['red_light']:
                if center_y < roi_top_y: 
                    right_turn = False
                    if len(v['path']) > 10: 
                        dx = v['path'][-1][0] - v['path'][-10][0]
                        dy = v['path'][-1][1] - v['path'][-10][1]
                        if dx > 15 and dx > abs(dy) * 1.5: right_turn = True
                    
                    if not right_turn: v['red_light'] = True

            errors = []
            if v['wrong_way']: errors.append("NGUOC CHIEU")
            if v['wrong_lane']: errors.append("SAI LAN")
            if v['red_light']: errors.append("VUOT DEN DO")

            is_violating = len(errors) > 0
            
            if not is_violating:
                if len(v['path']) > 10 and (v['path'][-1][0] - v['path'][-10][0]) > 15 and (v['path'][-1][0] - v['path'][-10][0]) > abs(v['path'][-1][1] - v['path'][-10][1]) * 1.5:
                    v['status'] = 'RE PHAI (OK)'
            else:
                v['status'] = " + ".join(errors)

            color = (0, 0, 255) if is_violating else (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"ID:{tid} | {v['status']}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            for p in v['path']: cv2.circle(display, p, 3, color, -1)

        writer.write(display)

    cap.release()
    writer.release()
    print(f"✅ HOÀN TẤT! Video: {output_path}")

if __name__ == "__main__":
    process_video(VIDEO_NAME)
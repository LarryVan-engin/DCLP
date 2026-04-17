"""
DEMO TRAFFIC VIOLATION - ULTIMATE EDGE-SERVER ARCHITECTURE
- Phân làn Data-Driven: Khảo sát 100 frame đầu.
- AI Đèn Giao Thông: Bắt Đỏ / Vàng.
- Cơ chế Smart Capture + Force Fallback: Lưu nháp bằng chứng gốc, ép buộc gửi Server nếu xe tẩu thoát hoặc quá giờ.
- Xuất CSV & Tra cứu Database chủ xe.
"""

import cv2
import numpy as np
import os
import json
import csv
from datetime import datetime
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
DB_FILE = r"D:\VSCode\DCLP\web_application\database\owners_sample.csv"
LOG_FILE = "./TESTOCR/violations_log.csv"

# MODELS
coco_model = YOLO("yolo12n.pt")
plate_model = YOLO(r"D:\VSCode\DCLP\main_code\runs\detect\model_detect_license_plate.pt")
traffic_light_model = YOLO(r"D:\VSCode\DCLP\main_code\runs\detect\model_detect_traffic_light.pt")
ocr_reader = easyocr.Reader(["en"], gpu=True)

# Tách biệt Class
CAR_CLASSES = {2, 5, 7} # Car, Bus, Truck
MOTO_CLASSES = {3}      # Motorcycle
VEHICLE_CLASSES = CAR_CLASSES.union(MOTO_CLASSES)

# ==================== HELPERS ====================
def nothing(x): pass

def _now_hms():
    return datetime.now().strftime("%H:%M:%S")

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

# ==================== LOAD DATABASE CHỦ XE ====================
def load_owner_db(csv_path):
    db = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_plate = re.sub(r"[^A-Z0-9]", "", row['plate'].upper())
                db[clean_plate] = row
        print(f"✅ Đã tải thành công {len(db)} hồ sơ chủ xe từ cơ sở dữ liệu.")
    else:
        print(f"⚠️ Không tìm thấy file {csv_path}. Hệ thống vẫn chạy nhưng DB chủ xe sẽ để trống.")
    return db

# ==================== GHI LOG VI PHẠM CSV ====================
def log_violation_to_csv(csv_path, record):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', encoding='utf-8-sig', newline='') as f:
        fieldnames = ["plate", "owner", "phone", "class_vehicle", "province", 
                      "registration_date", "id_card", "type", "time", "img", "plate_img"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

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

# ==================== TOÁN HỌC: QUY TẮC LÀN CHUẨN ====================
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
    car_only_zones = []

    if not cars_norm:
        boundaries = [0.0, 0.4, 0.7, 1.0]
        lane_labels = ["Lane O To", "Lane Tong Hop", "Lane Xe May"]
        car_only_zones.append((0.0, 0.4))
    else:
        car_centers = [c['center'] for c in cars_norm]
        car_centers.sort()
        clusters = []
        for c in car_centers:
            if not clusters: clusters.append([c])
            else:
                if c - np.mean(clusters[-1]) < 0.20: clusters[-1].append(c)
                else: clusters.append([c])
                
        N_car_lanes = len(clusters)

        last_cluster = clusters[-1]
        clast_rights = [c['right'] for c in cars_norm if c['center'] in last_cluster]
        r_car_max = np.percentile(clast_rights, 90)
        
        right_motos = [m for m in motos_norm if m['center'] > r_car_max - 0.05]
        
        if len(right_motos) > 5:
            m_left = np.percentile([m['left'] for m in right_motos], 15)
            r_boundary = (r_car_max + m_left) / 2
            r_boundary = max(r_car_max + 0.02, r_boundary) 
        else:
            r_boundary = r_car_max

        r_boundary = max(0.2, r_boundary)
        lane_width = r_boundary / N_car_lanes
        
        for i in range(N_car_lanes):
            b_left = i * lane_width
            b_right = (i + 1) * lane_width
            
            motos_in_zone = sum(1 for m in motos_norm if b_left <= m['center'] <= b_right)
            
            if motos_in_zone > 5:
                label = f"Lane Tong Hop {i+1}" if N_car_lanes > 1 else "Lane Tong Hop"
            else:
                label = f"Lane O To {i+1}" if N_car_lanes > 1 else "Lane O To"
                car_only_zones.append((b_left, b_right)) 
                
            boundaries.append(b_right)
            lane_labels.append(label)

        if len(right_motos) > 5:
            boundaries.append(1.0)
            lane_labels.append("Lane Xe May")

    top_y, bot_y = roi_pts[0][1], roi_pts[2][1]
    computed_lanes = []
    virtual_lines = []
    
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

    # Nạp cơ sở dữ liệu chủ xe
    owner_db = load_owner_db(DB_FILE)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (FRAME_WIDTH, FRAME_HEIGHT))

    roi_polygon = np.array(config["roi_pts"], np.int32)
    roi_top_y = roi_polygon[0][1] 

    frame_idx = 0
    vehicles = {}
    os.makedirs("plate_crops", exist_ok=True)
    
    car_boxes_learning = []
    moto_boxes_learning = []
    computed_lanes = []
    virtual_lines = []
    car_only_zones = [] 

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        display = frame.copy()
        frame_idx += 1

        res = coco_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.35)[0]
        boxes, _, clss = extract_boxes(res)
        track_ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else []
        
        # ==================== PHÁT HIỆN ĐÈN GIAO THÔNG ====================
        top_half_frame = frame[0:FRAME_HEIGHT//2, 0:FRAME_WIDTH]
        res_light = traffic_light_model(top_half_frame, verbose=False)[0]
        boxes_l, confs_l, clss_l = extract_boxes(res_light)
        current_light = "den_xanh" 

        for i in range(len(boxes_l)):
            if confs_l[i] < 0.45: continue
            name = res_light.names[int(clss_l[i])]

            if name == "den_do": current_light = "den_do"
            elif name == "den_vang" and current_light != "den_do": current_light = "den_vang"
        
            xl, yl, xr, yr = map(int, boxes_l[i])
            conf_val = confs_l[i]
            if name == "den_do": color_l = (0, 0, 255)        
            elif name == "den_vang": color_l = (0, 255, 255) 
            else: color_l = (0, 255, 0)                    
            cv2.rectangle(display, (xl, yl), (xr, yr), color_l, 2)
            cv2.putText(display, f"{name.upper()}:{conf_val:.2f}", (xl, max(20, yl - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_l, 2)

        is_red_light = (current_light == "den_do")
        is_yellow_light = (current_light == "den_vang")

        # --- GIAI ĐOẠN 1: HỌC BOUNDING BOX ---
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
            if "Lane O To" in label: color = (0, 100, 255) 
            elif "Lane Xe May" in label: color = (255, 200, 0) 
            else: color = (0, 255, 100) 
            cv2.fillPoly(overlay, [lane_poly], color)
            M = cv2.moments(lane_poly)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.putText(display, label, (cx - 70, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
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
                # Đã thêm các trường lưu Bằng chứng gốc và Đếm frame
                vehicles[tid] = {'path': deque(maxlen=PATH_HISTORY), 'status': 'OK', 
                                 'wrong_lane': False, 'red_light': False, 'yellow_light': False, 'wrong_way': False,
                                 'saved': False, 'violation_frame': 0, 'violation_img': None}
            v = vehicles[tid]
            v['path'].append((center_x, bottom_y))

            norm_x = get_normalized_x(center_x, bottom_y, roi_polygon)

            if len(v['path']) > 10 and not v['wrong_way']:
                if v['path'][-1][1] - v['path'][-10][1] > 15: v['wrong_way'] = True

            # BẮT LỖI SAI LÀN
            if not v['wrong_lane']:
                if cls_id in MOTO_CLASSES:
                    for (z_start, z_end) in car_only_zones:
                        if z_start <= norm_x <= z_end:
                            v['wrong_lane'] = True
                            break

            # BẮT LỖI VƯỢT ĐÈN
            if center_y < roi_top_y: 
                right_turn = False
                if len(v['path']) > 10: 
                    dx = v['path'][-1][0] - v['path'][-10][0]
                    dy = v['path'][-1][1] - v['path'][-10][1]
                    if dx > 15 and dx > abs(dy) * 1.5: right_turn = True
                
                if not right_turn:
                    if not v['red_light'] and not v['yellow_light']:
                        if is_red_light: v['red_light'] = True
                        elif is_yellow_light: v['yellow_light'] = True

            errors = []
            if v['wrong_way']: errors.append("NGUOC CHIEU")
            if v['wrong_lane']: errors.append("SAI LAN")
            if v['red_light']: errors.append("VUOT DEN DO")
            if v['yellow_light']: errors.append("VUOT DEN VANG")

            is_violating = len(errors) > 0
            
            if not is_violating:
                if len(v['path']) > 10 and (v['path'][-1][0] - v['path'][-10][0]) > 15 and (v['path'][-1][0] - v['path'][-10][0]) > abs(v['path'][-1][1] - v['path'][-10][1]) * 1.5:
                    v['status'] = 'RE PHAI (OK)'
            else:
                v['status'] = " + ".join(errors)
                
                # ==================== EDGE LOGIC: SMART CAPTURE + FALLBACK ====================
                if not v.get('saved', False):
                    # --- ĐỊNH NGHĨA VÙNG LẤY NGỮ CẢNH (MỞ RỘNG) ---
                    pad = 150 # Mở rộng thêm 150 pixel mỗi cạnh để lấy toàn cảnh (Vạch kẻ, đèn...)
                    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                    cx2, cy2 = min(FRAME_WIDTH, x2 + pad), min(FRAME_HEIGHT, y2 + pad)
                    
                    # 1. Lưu khoảnh khắc vi phạm đầu tiên làm Bằng chứng gốc (Ảnh rộng)
                    if v.get('violation_frame', 0) == 0:
                        v['violation_frame'] = frame_idx
                        v['violation_img'] = frame[cy1:cy2, cx1:cx2].copy()

                    # --- TÁCH BIỆT 2 LOẠI CROP ---
                    # Ảnh khít: Dùng cho AI đọc biển (chống nhiễu xe bên cạnh)
                    tight_crop = frame[max(0, y1):min(FRAME_HEIGHT, y2), max(0, x1):min(FRAME_WIDTH, x2)]
                    # Ảnh rộng ngữ cảnh: Dùng để lưu biên lai phạt
                    context_crop = frame[cy1:cy2, cx1:cx2] 
                    
                    is_clear_plate = False
                    plate_crop = np.empty((0,0))
                    plate_conf = 0.0
                    plate_area = 0

                    if tight_crop.size > 0:
                        plate_res = plate_model(tight_crop, verbose=False)[0]
                        p_boxes, p_confs, p_clss = extract_boxes(plate_res)
                        
                        if len(p_boxes) > 0:
                            best_idx = np.argmax(p_confs)
                            px1, py1, px2, py2 = map(int, p_boxes[best_idx])
                            plate_area = (px2 - px1) * (py2 - py1)
                            plate_conf = p_confs[best_idx]
                            
                            if plate_conf > 0.60 and plate_area > 600:
                                is_clear_plate = True
                                plate_crop = tight_crop[max(0, py1):min(tight_crop.shape[0], py2), max(0, px1):min(tight_crop.shape[1], px2)]
                    
                    # 2. ĐIỀU KIỆN FALLBACK (ÉP BUỘC GỬI)
                    timeout = (frame_idx - v['violation_frame']) > 60 
                    leaving_frame = (bottom_y >= FRAME_HEIGHT - 20) or (x1 <= 10) or (x2 >= FRAME_WIDTH - 10)
                    force_save = timeout or leaving_frame

                    # 3. THỰC THI (NẾU ĐẠT CHUẨN HOẶC HẾT THỜI GIAN)
                    if is_clear_plate or force_save:
                        error_str = "_".join(errors).replace(" ", "")
                        
                        # --- GỬI LÊN SERVER ---
                        if is_clear_plate and plate_crop.size > 0:
                            raw_plate_text = read_plate(plate_crop)
                            clean_plate_text = re.sub(r"[^A-Z0-9]", "", raw_plate_text.upper())
                            final_img = context_crop # Lưu bức ảnh MỞ RỘNG hiện tại
                        else:
                            clean_plate_text = "UNKNOWN"
                            final_img = v['violation_img'] if v['violation_img'] is not None else context_crop # Lấy Bằng chứng MỞ RỘNG gốc
                        
                        plate_img_path = f"plate_crops/ID{tid}_{error_str}_PLATE_{clean_plate_text}_f{frame_idx}.jpg" if is_clear_plate else ""
                        img_url = f"plate_crops/ID{tid}_{error_str}_CAR_{clean_plate_text}_f{frame_idx}.jpg"
                        
                        # LƯU ẢNH XUỐNG Ổ CỨNG
                        if is_clear_plate and plate_crop.size > 0:
                            cv2.imwrite(plate_img_path, plate_crop)
                        if final_img is not None and final_img.size > 0:
                            cv2.imwrite(img_url, final_img)
                        
                        # Tra cứu DB
                        cv_db = owner_db.get(clean_plate_text, {})
                        
                        record = {
                            "plate": clean_plate_text if clean_plate_text != "UNKNOWN" else "",
                            "owner": cv_db.get("owner", "Không xác định"),
                            "phone": cv_db.get("phone", ""),
                            "class_vehicle": cv_db.get("class_vehicle", ""),
                            "province": cv_db.get("province", ""),
                            "registration_date": cv_db.get("registration_date", ""),
                            "id_card": cv_db.get("id_card", ""),
                            "type": v['status'], 
                            "time": _now_hms(),
                            "img": img_url,
                            "plate_img": plate_img_path
                        }
                        log_violation_to_csv(LOG_FILE, record)
                        
                        if is_clear_plate:
                            print(f"📡 [EDGE -> SERVER] SUCCESS | Xe ID {tid} | Lỗi: {v['status']} | Biển: {clean_plate_text}")
                        else:
                            reason = "Hết thời gian chờ" if timeout else "Xe khuất tầm nhìn"
                            print(f"⚠️ [EDGE -> SERVER] FORCE PUSH ({reason}) | Xe ID {tid} | Lỗi: {v['status']} | Biển: UNKNOWN (Lưu bằng chứng ngữ cảnh gốc)")
                        
                        v['saved'] = True
                # ==============================================================================

            # Vẽ Tracking Box
            color = (0, 0, 255) if is_violating else (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"ID:{tid} | {v['status']}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            for p in v['path']: cv2.circle(display, p, 3, color, -1)

        writer.write(display)

    cap.release()
    writer.release()
    print(f"✅ HOÀN TẤT! Dữ liệu xuất ra file: {LOG_FILE}")

if __name__ == "__main__":
    process_video(VIDEO_NAME)
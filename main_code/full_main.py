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
                      "registration_date", "id_card", "type", "time",
                      "violation_folder", "plate_img"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in fieldnames})

# ==================== SO SÁNH 3 LOGIC PHÁT HIỆN VƯỢT ĐÈN ĐỎ ====================
# Mỗi logic chạy độc lập, tự chụp ảnh đúng vào frame mà CHÍNH NÓ phát hiện xe vượt vạch.
# → 3 ảnh = 3 thời điểm khác nhau, thể hiện rõ logic nào bắt sớm/muộn hơn.

METHOD_COLORS = {
    'TOP_EDGE':    (0,  200, 255),   # cam  — mũi/đầu xe
    'CENTER':      (255,  0, 255),   # tím  — tâm bbox
    'BOTTOM_EDGE': (0,  255, 100),   # xanh — đuôi/bánh xe
}

def save_comparison_snapshot(frame, x1, y1, x2, y2, roi_top_y,
                              tid, method_name, det_y, frame_idx, pad=65):
    """
    Chụp ảnh tại đúng frame mà method_name phát hiện xe vượt vạch đèn đỏ.
    det_y : giá trị y tuyệt đối của điểm kiểm tra (y1, center_y hoặc y2).
    """
    os.makedirs("comparison_test", exist_ok=True)
    color = METHOD_COLORS[method_name]
    h, wf = frame.shape[:2]
    cx = (x1 + x2) // 2

    sx1 = max(0, x1 - pad);  sy1 = max(0, y1 - pad)
    sx2 = min(wf, x2 + pad); sy2 = min(h,  y2 + pad)
    crop = frame[sy1:sy2, sx1:sx2].copy()
    cw, ch = crop.shape[1], crop.shape[0]

    # Toạ độ trong crop
    stopline_c = roi_top_y - sy1
    det_yc     = max(0, min(ch - 1, det_y - sy1))
    det_xc     = max(0, min(cw - 1, cx   - sx1))
    bx1c, by1c = x1 - sx1, y1 - sy1
    bx2c, by2c = x2 - sx1, y2 - sy1

    # Bbox xe (xám mờ)
    cv2.rectangle(crop, (bx1c, by1c), (bx2c, by2c), (200, 200, 200), 1)

    # Vạch dừng (đỏ)
    if 0 <= stopline_c < ch:
        cv2.line(crop, (0, stopline_c), (cw, stopline_c), (0, 0, 220), 2)
        cv2.putText(crop, f"STOP LINE  y={roi_top_y}",
                    (4, max(14, stopline_c - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 40, 220), 1)

    # Đường + điểm detection (màu theo method)
    cv2.line(crop, (0, det_yc), (cw, det_yc), color, 2)
    cv2.circle(crop, (det_xc, det_yc), 9, color, -1)
    cv2.circle(crop, (det_xc, det_yc), 9, (255, 255, 255), 1)

    # Khoảng cách và trạng thái
    dist    = roi_top_y - det_y      # dương → đã vượt
    crossed = det_y < roi_top_y
    status_txt   = ">> DA VUOT VACH <<" if crossed else "CHUA VUOT VACH"
    status_color = (0, 0, 255)       if crossed else (0, 220, 0)

    # Thanh nhãn (nền đen)
    cv2.rectangle(crop, (0, 0), (cw, 62), (20, 20, 20), -1)
    cv2.putText(crop,
                f"[{method_name}]  frame={frame_idx}  det_y={det_y}  stopline={roi_top_y}  dist={dist:+d}px",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)
    cv2.putText(crop, status_txt,
                (6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2)

    path = os.path.join("comparison_test", f"ID{tid}_{method_name}_f{frame_idx}.jpg")
    cv2.imwrite(path, crop)
    print(f"  [COMPARE] {method_name:<14} frame={frame_idx:5d}  "
          f"det_y={det_y:4d}  dist={dist:+4d}px  "
          f"{'VUOT' if crossed else 'CHUA':4s}  -> {path}")


# ==================== GIAI ĐOẠN 1: GUI (CLICK-TO-PLACE) ====================
def run_calibration_gui(video_path, config_file):
    """
    Click 4 điểm theo thứ tự: top-left → top-right → bottom-right → bottom-left để vẽ ROI.
    Cạnh trên (điểm 1-2) tự động là STOP LINE.
    Click lần thứ 5 để đặt đường kẻ Right Turn Zone.
    Phím F/B: lùi/tiến frame. R: reset. S: lưu.
    """
    print("CALIBRATION (Click-to-Place): Click 4 goc ROI, sau do click Right Turn Zone. S=Luu, R=Reset, F/B=doi frame.")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = min(415, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, base_frame = cap.read()
    cap.release()
    if not ret:
        return None
    base_frame = cv2.resize(base_frame, (FRAME_WIDTH, FRAME_HEIGHT))

    roi_pts = []
    right_turn_y = int(FRAME_HEIGHT * 0.7)
    # state: 0-3 = đặt góc ROI, 4 = đặt right turn Y, 5 = hoàn tất
    state = [0]

    STEP_LABELS = [
        "1/5  Click: TOP-LEFT   (goc trai Stop Line)",
        "2/5  Click: TOP-RIGHT  (goc phai Stop Line)",
        "3/5  Click: BOTTOM-RIGHT (goc phai day ROI)",
        "4/5  Click: BOTTOM-LEFT  (goc trai day ROI)",
        "5/5  Click: Right Turn Zone bottom Y",
        "XONG! Nhan 'S' de luu, 'R' de reset",
    ]

    def mouse_cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        s = state[0]
        if s < 4:
            roi_pts.append([x, y])
            state[0] = len(roi_pts)
        elif s == 4:
            param['rty'] = y
            state[0] = 5

    cb_param = {'rty': right_turn_y}
    cv2.namedWindow("CALIBRATION_TOOL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CALIBRATION_TOOL", 1000, 600)
    cv2.setMouseCallback("CALIBRATION_TOOL", mouse_cb, cb_param)

    config = None
    while True:
        display = base_frame.copy()
        rty = cb_param['rty']

        # Vẽ các điểm đã click
        point_labels = ["TL", "TR", "BR", "BL"]
        for i, pt in enumerate(roi_pts):
            cv2.circle(display, tuple(pt), 9, (0, 255, 255), -1)
            cv2.putText(display, point_labels[i], (pt[0] + 10, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Vẽ Stop Line khi có ít nhất 2 điểm
        if len(roi_pts) >= 2:
            cv2.line(display, tuple(roi_pts[0]), tuple(roi_pts[1]), (0, 0, 255), 4)
            cv2.putText(display, "STOP LINE", (roi_pts[0][0] + 10, roi_pts[0][1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Vẽ ROI polygon khi đủ 4 điểm
        if len(roi_pts) == 4:
            pts_arr = np.array(roi_pts, np.int32)
            cv2.polylines(display, [pts_arr], True, (0, 255, 255), 2)

        # Vẽ Right Turn Zone
        if state[0] >= 4 and len(roi_pts) == 4:
            top_y = roi_pts[0][1]
            ov = display.copy()
            cv2.rectangle(ov, (0, top_y), (FRAME_WIDTH, rty), (0, 200, 100), -1)
            cv2.addWeighted(ov, 0.15, display, 0.85, 0, display)
            cv2.line(display, (0, rty), (FRAME_WIDTH, rty), (0, 255, 0), 3)
            cv2.putText(display, "RIGHT TURN ZONE END", (20, rty - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Hướng dẫn
        label = STEP_LABELS[min(state[0], 5)]
        cv2.rectangle(display, (0, 0), (FRAME_WIDTH, 90), (0, 0, 0), -1)
        cv2.putText(display, label, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if state[0] == 5 else (0, 255, 255), 2)
        cv2.putText(display, "S=Luu  R=Reset  Q=Thoat  F=Frame+10  B=Frame-10",
                    (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("CALIBRATION_TOOL", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s') and state[0] == 5:
            config = {"roi_pts": roi_pts, "right_turn_zone_bottom_y": int(rty)}
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Da luu cau hinh vao {config_file}")
            break
        elif key == ord('r'):
            roi_pts.clear()
            cb_param['rty'] = int(FRAME_HEIGHT * 0.7)
            state[0] = 0
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
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
    roi_bottom_y = config.get("right_turn_zone_bottom_y", FRAME_HEIGHT)  # Mặc định: mép dưới frame

    frame_idx = 0
    vehicles = {}
    total_violation_count = 0
    # So sánh 3 logic tuần tự — state machine:
    #   stage 0 → chờ TOP_EDGE    (y1 < roi_top_y)
    #   stage 1 → chờ CENTER      (center_y < roi_top_y)
    #   stage 2 → chờ BOTTOM_EDGE (y2 < roi_top_y)
    #   stage 3 → xong
    # Khoá vào xe đầu tiên có y1 vượt vạch khi đèn đỏ.
    # Các stage sau không yêu cầu đèn đỏ (chỉ cần đúng xe, đúng thứ tự).
    CMP_STAGES = [
        ('TOP_EDGE',    lambda _y1, _cy, _y2: _y1 < roi_top_y,  lambda _y1, _cy, _y2: _y1),
        ('CENTER',      lambda _y1, _cy, _y2: _cy < roi_top_y,  lambda _y1, _cy, _y2: _cy),
        ('BOTTOM_EDGE', lambda _y1, _cy, _y2: _y2 < roi_top_y,  lambda _y1, _cy, _y2: _y2),
    ]
    cmp_vehicle_id = None
    cmp_stage      = 0      # index vào CMP_STAGES
    os.makedirs("violations", exist_ok=True)
    
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
        top_half_frame = frame[0:FRAME_HEIGHT//2, FRAME_WIDTH//2:FRAME_WIDTH]
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
            # Offset X vì crop từ FRAME_WIDTH//2 sang phải
            xl += FRAME_WIDTH // 2
            xr += FRAME_WIDTH // 2
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
        
        # Vẽ vùng rẽ phải (Right Turn Zone)
        overlay_rtz = display.copy()
        cv2.rectangle(overlay_rtz, (0, roi_top_y), (FRAME_WIDTH, roi_bottom_y), (0, 200, 100), -1)
        cv2.addWeighted(overlay_rtz, 0.15, display, 0.85, 0, display)
        cv2.line(display, (0, roi_bottom_y), (FRAME_WIDTH, roi_bottom_y), (0, 255, 0), 3)
        cv2.putText(display, "RIGHT TURN ZONE", (20, roi_bottom_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Tracking và Bắt lỗi
        for i, box in enumerate(boxes):
            cls_id = int(clss[i])
            if cls_id not in VEHICLE_CLASSES: continue
            tid = track_ids[i] if i < len(track_ids) else -1
            if tid == -1: continue
            
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y, bottom_y = (x1 + x2) // 2, (y1 + y2) // 2, y2
            
            norm_x = get_normalized_x(center_x, bottom_y, roi_polygon)
            right_turn_lane_min = max([end for (_start, end) in car_only_zones] + [0.65])
            in_roi = cv2.pointPolygonTest(roi_polygon, (center_x, bottom_y), False) >= 0
            in_right_turn_zone = roi_top_y <= center_y < roi_bottom_y and norm_x >= right_turn_lane_min
            # Tiếp tục theo dõi xe đã được ghi nhận mà chưa lưu xong (đã vượt vạch, đang xét rẽ phải)
            past_stopline_active = (tid in vehicles) and (center_y < roi_top_y)
            if not (in_roi or in_right_turn_zone or past_stopline_active): continue

            if tid not in vehicles:
                # Đã thêm các trường lưu Bằng chứng gốc và Đếm frame
                vehicles[tid] = {'path': deque(maxlen=PATH_HISTORY), 'status': 'OK',
                                 'wrong_lane': False, 'red_light': False, 'yellow_light': False, 'wrong_way': False,
                                 'right_turn': False, 'cls_id': cls_id,
                                 'was_right_lane': False,        # True khi xe máy từng ở trong in_right_turn_zone
                                 'saved': False, 'violation_frame': 0,
                                 'violation_full_frame': None,
                                 'violation_img': None}
            v = vehicles[tid]
            v['path'].append((center_x, bottom_y))

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

            # ========== SO SÁNH 3 LOGIC VƯỢT ĐÈN ĐỎ — state machine tuần tự ==========
            # Stage 0: khoá xe + chờ TOP_EDGE   (yêu cầu đèn đỏ)
            # Stage 1: chờ CENTER                (chỉ cần đúng xe)
            # Stage 2: chờ BOTTOM_EDGE           (chỉ cần đúng xe)
            if cmp_stage < len(CMP_STAGES):
                method_name, cond_fn, det_fn = CMP_STAGES[cmp_stage]

                # Khoá xe ở stage 0: phải đang đèn đỏ và TOP_EDGE vừa vượt
                if cmp_vehicle_id is None and cmp_stage == 0 and is_red_light and y1 < roi_top_y:
                    cmp_vehicle_id = tid
                    print(f"\n[COMPARE] Locked ID={tid} | stopline_y={roi_top_y} | frame={frame_idx}")

                # Chỉ xử lý đúng xe đã khoá
                if tid == cmp_vehicle_id:
                    if cond_fn(y1, center_y, y2):
                        save_comparison_snapshot(
                            frame, x1, y1, x2, y2, roi_top_y,
                            tid, method_name, det_fn(y1, center_y, y2), frame_idx
                        )
                        cmp_stage += 1   # chuyển sang stage tiếp theo
            # ===========================================================================

            # ==================== LOGIC RẼ PHẢI (CHỈ CHO XE MÁY) ====================
            right_turn_lane_min = max([end for (_start, end) in car_only_zones] + [0.65])
            in_right_turn_zone = roi_top_y <= center_y < roi_bottom_y and norm_x >= right_turn_lane_min
            # Ghi nhận xe máy đang ở đúng làn rẽ phải (trước vạch)
            if v['cls_id'] in MOTO_CLASSES and in_right_turn_zone:
                v['was_right_lane'] = True
            # Chỉ theo dõi sau vạch nếu xe máy đã được xác nhận xuất phát từ làn rẽ phải
            # → loại xe từ làn khác lọt qua past_stopline_active gây xóa nhầm lỗi
            monitoring_post_line = (center_y < roi_top_y) and (v['cls_id'] in MOTO_CLASSES) and v.get('was_right_lane', False)

            if v['cls_id'] in MOTO_CLASSES and (in_right_turn_zone or monitoring_post_line) and len(v['path']) > 10:
                dx = v['path'][-1][0] - v['path'][-10][0]
                dy = v['path'][-1][1] - v['path'][-10][1]
                currently_right_turn = dx > 15 and dx > abs(dy) * 1.5

                if currently_right_turn and not v['right_turn']:
                    v['right_turn'] = True
                    # Xe máy xác nhận rẽ phải sau khi qua vạch → xóa lỗi vượt đèn nếu chưa lưu
                    if monitoring_post_line and v['red_light'] and not v.get('saved', False):
                        v['red_light'] = False
                elif v['right_turn'] and not currently_right_turn and dy > dx * 0.5:
                    v['right_turn'] = False
            elif v['cls_id'] in MOTO_CLASSES and not in_right_turn_zone and not monitoring_post_line:
                # Chỉ reset khi xe không ở cả hai vùng (đã rời khỏi khu vực theo dõi)
                v['right_turn'] = False
            
            if not is_violating:
                # Chỉ thêm "RE PHAI (OK)" nếu đang rẽ phải và là xe máy
                if v['cls_id'] in MOTO_CLASSES and v['right_turn']:
                    v['status'] = 'RE PHAI (OK)'
            else:
                if v['cls_id'] in MOTO_CLASSES and v['right_turn']:
                    errors.append('RE PHAI')
                v['status'] = " + ".join(errors)
                
                # ==================== EDGE LOGIC: SMART CAPTURE + FALLBACK ====================
                if not v.get('saved', False):
                    # --- ĐỊNH NGHĨA VÙNG LẤY NGỮ CẢNH (MỞ RỘNG) ---
                    pad = 150 # Mở rộng thêm 150 pixel mỗi cạnh để lấy toàn cảnh (Vạch kẻ, đèn...)
                    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                    cx2, cy2 = min(FRAME_WIDTH, x2 + pad), min(FRAME_HEIGHT, y2 + pad)
                    
                    # 1. Lưu khoảnh khắc vi phạm đầu tiên
                    if v.get('violation_frame', 0) == 0:
                        v['violation_frame'] = frame_idx
                        v['violation_full_frame'] = frame.copy()          # full frame A
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
                        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

                        if is_clear_plate and plate_crop.size > 0:
                            raw_plate_text = read_plate(plate_crop)
                            clean_plate_text = re.sub(r"[^A-Z0-9]", "", raw_plate_text.upper())
                        else:
                            clean_plate_text = "UNKNOWN"

                        # --- TẠO THƯ MỤC VI PHẠM ---
                        # Tên thư mục: ID{tid}_{timestamp}_{biển}_{lỗi}
                        # → dễ liên kết với DB, dễ tìm kiếm theo ID hoặc thời điểm
                        folder_name = f"ID{tid}_{ts_str}_{clean_plate_text}_{error_str}"
                        vfolder = os.path.join("violations", folder_name)
                        os.makedirs(vfolder, exist_ok=True)

                        # --- LƯU 3 ẢNH ---
                        # Ảnh 1: full frame tại thời điểm vi phạm đầu tiên (thấy đèn giao thông)
                        full_a_path = os.path.join(vfolder, f"full_A_{error_str}.jpg")
                        src_full_a = v['violation_full_frame'] if v['violation_full_frame'] is not None else frame
                        cv2.imwrite(full_a_path, src_full_a)

                        # Ảnh 2: full frame hiện tại khi lưu (thấy đèn + xe ở vị trí rõ hơn)
                        full_b_path = os.path.join(vfolder, f"full_B_{error_str}.jpg")
                        cv2.imwrite(full_b_path, frame)

                        # Ảnh 3: smart crop (bounding box + 65px, đủ nhỏ để rõ xe, đủ rộng để thấy biển)
                        SC_PAD = 65
                        scx1 = max(0, x1 - SC_PAD);  scy1 = max(0, y1 - SC_PAD)
                        scx2 = min(FRAME_WIDTH, x2 + SC_PAD); scy2 = min(FRAME_HEIGHT, y2 + SC_PAD)
                        smart_crop_img = frame[scy1:scy2, scx1:scx2]
                        crop_path = os.path.join(vfolder, f"smart_crop_{error_str}.jpg")
                        if smart_crop_img.size > 0:
                            cv2.imwrite(crop_path, smart_crop_img)

                        # Ảnh biển số riêng (nếu nhận diện được)
                        plate_img_path = ""
                        if is_clear_plate and plate_crop.size > 0:
                            plate_img_path = os.path.join(vfolder, f"plate_{clean_plate_text}.jpg")
                            cv2.imwrite(plate_img_path, plate_crop)

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
                            "violation_folder": vfolder,
                            "plate_img": plate_img_path,
                        }
                        log_violation_to_csv(LOG_FILE, record)
                        total_violation_count += 1

                        if is_clear_plate:
                            print(f"[SAVED] ID {tid} | {v['status']} | Bien: {clean_plate_text} | {vfolder}")
                        else:
                            reason = "Timeout" if timeout else "Xe thoat frame"
                            print(f"[FORCE] ID {tid} | {v['status']} | {reason} | Bien: UNKNOWN | {vfolder}")

                        v['saved'] = True
                # ==============================================================================

            # Vẽ Tracking Box
            color = (0, 0, 255) if is_violating else (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"ID:{tid} | {v['status']}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Vẽ tracking dots:
            #   cyan  = xe máy đang theo dõi rẽ phải (trước vạch)
            #   vàng  = xe đã vượt vạch, đang chờ xác nhận rẽ (sau vạch)
            #   đỏ/xanh = màu vi phạm / OK bình thường
            if v['cls_id'] in MOTO_CLASSES and center_y < roi_top_y:
                dot_color, dot_radius = (0, 200, 255), 5   # vàng-cam: theo dõi sau vạch
            elif v['cls_id'] in MOTO_CLASSES and (v['right_turn'] or in_right_turn_zone):
                dot_color, dot_radius = (0, 255, 255), 4   # cyan: tiếp cận vùng rẽ phải
            else:
                dot_color, dot_radius = color, 3
            for p in v['path']:
                cv2.circle(display, p, dot_radius, dot_color, -1)

        writer.write(display)

    cap.release()
    writer.release()

    total_vehicles = len(vehicles)
    print("=" * 55)
    print(f"  TONG KET XU LY VIDEO")
    print(f"  Tong phuong tien da theo doi : {total_vehicles}")
    print(f"  Tong phuong tien vi pham     : {total_violation_count}")
    print(f"  Ti le vi pham                : {total_violation_count/max(total_vehicles,1)*100:.1f}%")
    print(f"  Log CSV                      : {LOG_FILE}")
    print(f"  Thu muc bang chung           : violations/")
    print("=" * 55)

if __name__ == "__main__":
    process_video(VIDEO_NAME)

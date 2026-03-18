"""
DEMO TRAFFIC VIOLATION - CALIBRATION SCRUBBER & HYBRID LANE DETECTION
- Tool Setup: Kéo thanh trượt để tua Video và chọn Frame đẹp nhất để Calib.
- AI (YOLO): Nhận diện xe và tạo mặt nạ để loại bỏ nhiễu xe khỏi hình ảnh.
- Toán học (OpenCV): Nới lỏng ngưỡng nhị phân và tinh chỉnh minLineLength/maxLineGap để bắt vạch đứt.
- Tối ưu hiệu năng: Chỉ theo dõi và xử lý các phương tiện nằm trong vùng ROI.
- Phân loại vạch đường: Dựa trên chiều dài để phân biệt vạch liền (đường xanh dương) và vạch đứt (đường xanh nhạt).
- Logic vi phạm: Cán qua vạch dừng khi đèn đỏ, nhưng miễn trừ nếu có dấu hiệu rẽ phải.
- OCR: Nhận diện biển số từ các crop được phát hiện, lưu ảnh crop để kiểm tra sau.
Lưu ý: Cần chạy giai đoạn hiệu chuẩn trước để có thông số chính xác cho từng video, đặc biệt là ngưỡng nhị phân để bắt vạch đứt.
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
LANE_ALPHA = 0.20
PATH_HISTORY = 30           # History dài để phân tích rẽ phải
PLATE_CONF = 0.35
# TÊN FILE CONFIG LINH ĐỘNG THEO TÊN VIDEO (VD: KDT.mp4 -> config_KDT.json)
VIDEO_NAME = r"E:\Video\train\video_test.mp4" 
CONFIG_FILE = f"config_{os.path.splitext(os.path.basename(VIDEO_NAME))[0]}.json"

# MODELS
coco_model = YOLO("yolo12n.pt")
plate_model = YOLO(r"D:\VSCode\DCLP\main_code\runs\detect\model_detect_license_plate.pt")
ocr_reader = easyocr.Reader(["en"], gpu=True)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

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

def create_vehicle_mask(frame, boxes):
    """Sử dụng YOLO để tạo mặt nạ cho các phương tiện, giúp loại bỏ nhiễu"""
    mask = np.zeros_like(frame[:,:,0])
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

# ==================== GIAI ĐOẠN 1: SETUP GUI VỚI SCRUBBER ====================
def run_calibration_gui(video_path, config_file):
    print("🛠️ KHỞI ĐỘNG GIAI ĐOẠN HIỆU CHUẨN (CALIBRATION)... BẮT VẠCH ĐỨT...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cv2.namedWindow("CALIBRATION_TOOL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CALIBRATION_TOOL", 1200, 800)
    
    # Thanh trượt tua Video
    cv2.createTrackbar("Frame", "CALIBRATION_TOOL", 513, total_frames - 1, nothing)
    
    # Thanh trượt cấu hình ROI dời trái/phải
    cv2.createTrackbar("ROI_Center_X", "CALIBRATION_TOOL", FRAME_WIDTH//2, FRAME_WIDTH, nothing)
    cv2.createTrackbar("Stop_Line_Y", "CALIBRATION_TOOL", 450, FRAME_HEIGHT, nothing)
    cv2.createTrackbar("ROI_Top_Y", "CALIBRATION_TOOL", int(FRAME_HEIGHT*0.3), FRAME_HEIGHT, nothing)
    cv2.createTrackbar("ROI_Top_W", "CALIBRATION_TOOL", 300, FRAME_WIDTH, nothing)
    cv2.createTrackbar("ROI_Bot_W", "CALIBRATION_TOOL", 800, FRAME_WIDTH, nothing)
    
    # --- SỬA ĐỔI CHÍ MẠNG: GIẢM NGƯỠNG ĐỂ BẮT VẠCH ĐỨT MỜ ---
    # Thay vì chỉnh Canny mờ mịt, ta chỉnh ngưỡng nhị phân để làm nổi vạch đường
    cv2.createTrackbar("Binary_Threshold", "CALIBRATION_TOOL", 120, 255, nothing) # Mặc định thấp hơn để bắt vạch đứt
    # Tinh chỉnh minLineLength và maxLineGap để bắt vạch ngắn và nối chúng
    cv2.createTrackbar("MinLineLength", "CALIBRATION_TOOL", 20, 200, nothing) # Rất thấp để bắt vạch đứt ngắn
    cv2.createTrackbar("MaxLineGap", "CALIBRATION_TOOL", 50, 200, nothing)    # Cao để nối các đoạn đứt quãng

    print("👉 Kéo thanh 'Frame' đến đoạn đường vắng/rõ vạch nhất.")
    print("👉 Kéo 'Binary_Threshold' xuống thấp dần cho đến khi vạch đứt hiện lên trắng tinh trên nền đen.")
    print("👉 Bấm 'S' để Save. Bấm 'Q' để Thoát.")

    config = {}
    last_frame_idx = -1
    frame = None

    while True:
        current_frame_idx = cv2.getTrackbarPos("Frame", "CALIBRATION_TOOL")
        
        # Chỉ đọc lại frame nếu thanh trượt thay đổi
        if current_frame_idx != last_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if ret: 
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            last_frame_idx = current_frame_idx
            
        if frame is None: break
        display = frame.copy()
        
        # --- LẤY THÔNG SỐ TỪ YOLO ĐỂ LÀM SẠCH NHIỄU XE ---
        res = coco_model(frame, verbose=False, conf=0.35)[0]
        boxes, _, clss = extract_boxes(res)
        vehicle_boxes = [boxes[i] for i in range(len(boxes)) if int(clss[i]) in VEHICLE_CLASSES]
        
        vehicle_mask = create_vehicle_mask(frame, vehicle_boxes)
        # ---------------------------------------------------------------------

        # Lấy thông số từ thanh trượt
        center_x = cv2.getTrackbarPos("ROI_Center_X", "CALIBRATION_TOOL")
        stop_y = cv2.getTrackbarPos("Stop_Line_Y", "CALIBRATION_TOOL")
        roi_top_y = cv2.getTrackbarPos("ROI_Top_Y", "CALIBRATION_TOOL")
        roi_top_w = cv2.getTrackbarPos("ROI_Top_W", "CALIBRATION_TOOL")
        roi_bot_w = cv2.getTrackbarPos("ROI_Bot_W", "CALIBRATION_TOOL")
        bin_thresh = cv2.getTrackbarPos("Binary_Threshold", "CALIBRATION_TOOL")
        min_line_len = cv2.getTrackbarPos("MinLineLength", "CALIBRATION_TOOL")
        max_line_gap = cv2.getTrackbarPos("MaxLineGap", "CALIBRATION_TOOL")
        
        pts = np.array([
            [center_x - roi_top_w//2, roi_top_y],
            [center_x + roi_top_w//2, roi_top_y],
            [center_x + roi_bot_w//2, FRAME_HEIGHT],
            [center_x - roi_bot_w//2, FRAME_HEIGHT]
        ], np.int32)
        
        # --- PREVIEW VẠCH ĐƯỜNG (CÓ LÀM SẠCH NHIỄU XE TỪ AI) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Loại bỏ nhiễu xe bằng mặt nạ YOLO
        gray = cv2.bitwise_and(gray, cv2.bitwise_not(vehicle_mask))
        
        # 2. Tạo Mask ROI hình thang tự động
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [pts], 255)
        masked_gray = cv2.bitwise_and(gray, mask)

        # 3. Ép độ tương phản mạnh để làm nổi vạch mờ (Chuẩn thế kỷ 21)
        _, binary = cv2.threshold(masked_gray, bin_thresh, 255, cv2.THRESH_BINARY)
        
        # 4. HoughLinesP với tham số được tinh chỉnh để bắt vạch đứt ngắn
        edges = cv2.Canny(binary, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=min_line_len, maxLineGap=max_line_gap)

        # Vẽ các vạch đường tìm được lên màn hình bằng màu HỒNG
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # LỌC GÓC: Chỉ lấy vạch dốc dọc
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 70 < angle < 110: 
                    cv2.line(display, (x1, y1), (x2, y2), (255, 0, 255), 3)
        # ------------------------------------------------

        # Vẽ ROI và Stop Line
        cv2.polylines(display, [pts], True, (0, 255, 255), 2)
        cv2.line(display, (0, stop_y), (FRAME_WIDTH, stop_y), (0, 0, 255), 3)
        cv2.putText(display, f"Frame:{current_frame_idx} | 'A':Auto-ROI | 'S':Save", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Ghép ảnh Binary vào góc trên cùng bên phải để debug
        small_binary = cv2.resize(binary, (320, 180))
        small_binary_bgr = cv2.cvtColor(small_binary, cv2.COLOR_GRAY2BGR)
        display[0:180, FRAME_WIDTH-320:FRAME_WIDTH] = small_binary_bgr
        cv2.putText(display, "Strict Thresholding Preview", (FRAME_WIDTH-310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("CALIBRATION_TOOL", display)
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('s'):
            config = {
                "stop_line_y": stop_y,
                "roi_pts": pts.tolist(),
                "bin_thresh": bin_thresh,
                "min_line_len": min_line_len,
                "max_line_gap": max_line_gap
            }
            with open(config_file, 'w') as f:
                json.dump(config, f)
            print(f"✅ Đã lưu cấu hình vào {config_file}")
            break
        elif key == ord('q'):
            break
            
    cv2.destroyAllWindows()
    cap.release()
    return config

# ==================== MAIN LOOP (VIDEO PROCESSING) ====================
def process_video(input_path, output_path="output_demo_final.avi"):
    # Tên file config linh động theo tên video (VD: KDT.mp4 -> config_KDT.json)
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    config_file = f"config_{video_name}.json"
    
    if not os.path.exists(config_file):
        config = run_calibration_gui(input_path, config_file)
        if not config: return
    else:
        with open(config_file, 'r') as f:
            config = json.load(f)
            print(f"✅ Đã load cấu hình từ {config_file}")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (FRAME_WIDTH, FRAME_HEIGHT))

    roi_polygon = np.array(config["roi_pts"], np.int32)
    stop_y = config["stop_line_y"]
    bin_thresh = config["bin_thresh"]
    min_line_len = config["min_line_len"]
    max_line_gap = config["max_line_gap"]

    frame_idx = 0
    vehicles = {}
    os.makedirs("plate_crops", exist_ok=True)
    
    # Cờ mô phỏng đèn đỏ (Thực tế bạn sẽ lấy từ traffic_light_model)
    is_red_light = True 

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        display = frame.copy()
        frame_idx += 1

        res = coco_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.35)[0]
        boxes, _, clss = extract_boxes(res)
        track_ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else []
        
        vehicle_boxes = [boxes[i] for i in range(len(boxes)) if int(clss[i]) in VEHICLE_CLASSES]
        vehicle_mask = create_vehicle_mask(frame, vehicle_boxes)

        # --- GIAI ĐOẠN 1: NHẬN DIỆN LÀN ĐƯỜNG (CÓ LÀM SẠCH NHIỄU XE TỪ AI) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, cv2.bitwise_not(vehicle_mask)) # Loại bỏ nhiễu xe
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [roi_polygon], 255)
        masked_gray = cv2.bitwise_and(gray, mask)
        _, binary = cv2.threshold(masked_gray, bin_thresh, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(binary, 50, 150)
        
        # HoughLinesP với tham số được tinh chỉnh từ Config
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=min_line_len, maxLineGap=max_line_gap)

        # Vẽ vạch (Solid = Xanh dương, Dashed = Xanh nhạt)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 70 < angle < 110:
                    length = np.hypot(x2 - x1, y2 - y1)
                    # Phân loại dựa trên chiều dài: ngắn -> nét đứt, dài -> nét liền
                    color = (255, 255, 0) if length < 100 else (255, 0, 0)
                    cv2.line(display, (x1, y1), (x2, y2), color, 4)
        # ---------------------------------------------------------------------

        # Vẽ ROI và Stop Line để quan sát
        cv2.polylines(display, [roi_polygon], True, (0, 255, 255), 2)
        cv2.line(display, (0, stop_y), (FRAME_WIDTH, stop_y), (0, 0, 255), 3)

        # --- GIAI ĐOẠN 2: THEO DÕI XE & BẮT VI PHẠM (Chỉ xử lý xe trong ROI) ---
        detected_plates = []
        if frame_idx % 5 == 0:
            p_res = plate_model(frame, verbose=False)[0]
            pb, pc, _ = extract_boxes(p_res)
            detected_plates = [tuple(map(int, b)) for j, b in enumerate(pb) if pc[j] >= PLATE_CONF]

        for i, box in enumerate(boxes):
            if int(clss[i]) not in VEHICLE_CLASSES: continue
            tid = track_ids[i] if i < len(track_ids) else -1
            if tid == -1: continue
            
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2 # Lấy tâm xe xét "cán quá nửa" (Sửa lỗi chí mạng trước)
            bottom_y = y2 
            
            # TỐI ƯU EDGE: Chỉ xử lý xe nằm TRONG vùng ROI (Để Jetson Nano thở)
            if cv2.pointPolygonTest(roi_polygon, (center_x, bottom_y), False) < 0:
                continue

            if tid not in vehicles:
                vehicles[tid] = {'path': deque(maxlen=PATH_HISTORY), 'plate': '', 'violation': False, 'status': 'OK'}
            v = vehicles[tid]
            v['path'].append((center_x, bottom_y))

            # LOGIC XỬ LÝ VI PHẠM & RẼ PHẢI
            if is_red_light and not v['violation']:
                # Xe cán qua vạch dừng: Chạy xa camera (Y giảm), center_y nhỏ hơn stop_y mới phạt
                if center_y < stop_y:
                    v['violation'] = True
                    v['status'] = 'VIOLATION'
                    
                    # NGOẠI LỆ RẼ PHẢI: Phân tích quỹ đạo (ít nhất 10 frame gần đây)
                    if len(v['path']) > 10:
                        start_pt = v['path'][-10]
                        end_pt = v['path'][-1]
                        dx = end_pt[0] - start_pt[0]
                        dy = end_pt[1] - start_pt[1]
                        
                        # Đi sang phải (dx > 0) và khoảng cách ngang gấp 1.5 lần khoảng cách dọc
                        if dx > 15 and dx > abs(dy) * 1.5:
                            v['status'] = 'TURNING RIGHT (OK)'
                            v['violation'] = False # Miễn trừ vi phạm

            # Hiển thị Tracking, Plate & Trạng thái
            if not v['plate'] and detected_plates:
                for px1, py1, px2, py2 in detected_plates:
                    if px1 > x1 and py1 > y1 and px2 < x2 and py2 < y2:
                        crop = frame[py1:py2, px1:px2]
                        plate_text = read_plate(crop)
                        cv2.imwrite(f"plate_crops/plate_{tid}_{frame_idx}.jpg", crop)
                        if len(plate_text) >= 4: v['plate'] = plate_text
                        break

            color = (0, 0, 255) if v['violation'] else (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID{tid} {v['plate'] or 'Reading...'} | {v['status']}"
            cv2.putText(display, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            for p in v['path']:
                cv2.circle(display, p, 3, color, -1)

        writer.write(display)

    cap.release()
    writer.release()
    print(f"✅ HOÀN TẤT! Video: {output_path}")

if __name__ == "__main__":
    process_video(VIDEO_NAME)

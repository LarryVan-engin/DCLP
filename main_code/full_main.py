"""
*******************************************************************************************************************
General Information
********************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTEGRATION
File:          MAIN_TRACK_VIDEO.py
Description:   Detect, track multiple vehicle types, and recognize license plates using YOLOv12 and DeepSORT.

Author:        LARRY PHONG TRUC
Email:         vanphongtruc1808@gmail.com
Created:       05/10/2025
Last Update:   30/10/2025
Version:       1.5 (Multi-class DeepSORT tracking)
*******************************************************************************************************************
"""

#######################################################################################################################
# Imports
#######################################################################################################################
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from module_utils import get_car, read_license_plate, write_csv


#######################################################################################################################
# Constants and Configuration
#######################################################################################################################
COCO_MODEL_PATH = 'yolo12n.pt'
LICENSE_MODEL_PATH = r'D:\VSCode\DCLP\main_code\runs\detect\model_detect_license_plate.pt'
VIDEO_PATH = r'D:\VSCode\DCLP\big_dataset\Video\train\\7194238501489.mp4'
SAVE_CSV_PATH = r'D:\VSCode\DCLP\main_code\result\result_log.csv'


# Vehicle class IDs in COCO dataset (person=0, bicycle=1, car=2, motorbike=3, bus=5, truck=7, ...)
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}


#######################################################################################################################
# Helper Functions
#######################################################################################################################
def process_license_plate(frame, x1, y1, x2, y2):
    """Cắt ảnh biển số và trả về ảnh BGR (không threshold mạnh)."""
    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
    return plate_crop  # Trả về ảnh gốc hoặc chỉ grayscale nhẹ


#######################################################################################################################
# Main Function
#######################################################################################################################
def main_function():
    """Chạy quá trình detect → track → đọc biển số → lưu kết quả ra CSV."""
    results = {}

    # --------------------------------------------------------------------------------------------
    # Khởi tạo DeepSORT cho từng class phương tiện
    # --------------------------------------------------------------------------------------------
    trackers = {
        cls_name: DeepSort(
            max_age=30,
            n_init=2,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            embedder="mobilenet",
        )
        for cls_name in VEHICLE_CLASSES.values()
    }

    # Load YOLO models
    print("[INFO] Loading YOLO models...")
    coco_model = YOLO(COCO_MODEL_PATH)
    license_plate_detector = YOLO(LICENSE_MODEL_PATH)

    # Load video
    print(f"[INFO] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_nmr = -1
    ret = True

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()

        if not ret or frame_nmr >= 500:  # Giới hạn test
            break

        results[frame_nmr] = {}

        # --------------------------------------------------------------------------------------------
        # STEP 1: Detect Vehicles
        # --------------------------------------------------------------------------------------------
        detections = coco_model(frame)[0]
        dets_by_class = {cls_name: [] for cls_name in VEHICLE_CLASSES.values()}

        for *xyxy, conf, cls in detections.boxes.data.tolist():
            x1, y1, x2, y2 = map(float, xyxy)
            class_id = int(cls)

            if class_id in VEHICLE_CLASSES:
                cls_name = VEHICLE_CLASSES[class_id]
                dets_by_class[cls_name].append(([x1, y1, x2, y2], conf, cls_name))

        # --------------------------------------------------------------------------------------------
        # STEP 2: Track Vehicles (DeepSORT riêng cho từng class)
        # --------------------------------------------------------------------------------------------
        track_ids_all = []

        for cls_name, detections_ in dets_by_class.items():
            tracks = trackers[cls_name].update_tracks(detections_, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                track_id = track.track_id
                track_ids_all.append([x1, y1, x2, y2, track_id, cls_name])

                # Vẽ khung & ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name}-{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # --------------------------------------------------------------------------------------------
        # STEP 3: Detect License Plates
        # --------------------------------------------------------------------------------------------
        license_plates = license_plate_detector(frame)[0]

        for lp in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = map(float, lp)

            # Gán biển số vào xe tương ứng
            vehicle = get_car(lp, track_ids_all)  # lp đã đúng format
            xcar1, ycar1, xcar2, ycar2, vehicle_id, cls_name = vehicle
            if vehicle_id == -1:
                continue

            # ----------------------------------------------------------------------------------------
            # STEP 4: Đọc biển số
            # ----------------------------------------------------------------------------------------
            plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

            h, w = plate_crop.shape[:2]
            if h > 0 and w > 0:
                plate_resized = cv2.resize(plate_crop, (200, 60))  # Kích thước chuẩn cho OCR
                license_plate_text, license_plate_text_score = read_license_plate(plate_resized)
            else:
                license_plate_text, license_plate_text_score = None, 0.0

                
            # Không cần threshold mạnh → EasyOCR tự xử lý tốt hơn
            # Nếu cần tăng độ tương phản, chỉ cần grayscale hoặc nhẹ
            license_plate_text, license_plate_text_score = read_license_plate(plate_crop)

            if license_plate_text:
                print(f"[DEBUG] Frame {frame_nmr}: ID {vehicle_id} → {license_plate_text} ({license_plate_text_score:.2f})")


            if license_plate_text:
                results[frame_nmr][vehicle_id] = {
                    'vehicle': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score,
                    },
                }
                # Vẽ biển số lên frame để kiểm tra
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Hiển thị frame
        cv2.imshow("Tracking Multi-Class", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC để dừng
            break

        print(f"[INFO] Processed frame {frame_nmr}")

    # --------------------------------------------------------------------------------------------
    # STEP 5: Save Results
    # --------------------------------------------------------------------------------------------
    write_csv(results, SAVE_CSV_PATH)
    print(f"[INFO] Results saved to {SAVE_CSV_PATH}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Process completed successfully.")


#######################################################################################################################
# Main Execution
#######################################################################################################################
if __name__ == "__main__":
    main_function()
    

# End of File

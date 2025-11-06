"""
*******************************************************************************************************************
General Information
********************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTERGRATION
File:          PREDICT_TEST.py
Description:   Test YOLO prediction after training with filtered vehicle classes and overlay plate boxes.

Author:        LARRY PHONG TRUC
Email:         vanphongtruc1808@gmail.com
Created:       05/10/2025
Last Update:   29/10/2025
Version:       1.3
*******************************************************************************************************************
"""

#######################################################################################################################
# Imports
#######################################################################################################################
import cv2
import os
from ultralytics import YOLO
from PIL import Image


#######################################################################################################################
# Constants and Configuration
#######################################################################################################################
YOLO_PATH = r"D:\VSCode\DCLP\main_code\runs\detect\model_detect_license_plate.pt"
IMAGE_TEST_PATH = r"D:\VSCode\DCLP\big_dataset\test\xelam.png"
SAVE_PATH = r"D:\VSCode\DCLP\main_code\result\plate_detect"

# Load models
coco_model = YOLO('yolo12n.pt')
plate_model = YOLO(YOLO_PATH)

# Lọc class cho COCO model
CLASS_MAPPING = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}
SELECTED_VEHICLE_CLASSES = list(CLASS_MAPPING.keys())


#######################################################################################################################
# Helper Functions
#######################################################################################################################
def filter_boxes_by_class(result, allowed_classes):
    """
    Lọc các bounding boxes theo class_id được phép.
    """
    boxes = result.boxes
    filtered_indices = [i for i, box in enumerate(boxes) if int(box.cls[0]) in allowed_classes]

    if not filtered_indices:
        result.boxes = boxes[:0]
        return result

    result.boxes = boxes[filtered_indices]
    return result


def relabel_boxes(result, class_map):
    """
    Ghi đè lại tên class trong result.names theo class_map tùy chỉnh.
    """
    for cls_id, name in class_map.items():
        result.names[cls_id] = name
    return result


def draw_boxes_on_image(image, boxes, names, color=(0, 255, 0)):
    """
    Vẽ các bounding box và nhãn lên ảnh, có khung nền label đẹp hơn.
    """
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"

        # Vẽ bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Tính kích thước text
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        
        # Vẽ nền label (đậm hơn hoặc cùng màu box)
        cv2.rectangle(image, (x1, y1 - text_h - 8),
                      (x1 + text_w + 2, y1), color, -1)
        
        # Vẽ chữ trắng
        cv2.putText(image, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image



#######################################################################################################################
# Main Function
#######################################################################################################################
def main_function():
    # Đọc ảnh gốc
    img = cv2.imread(IMAGE_TEST_PATH)
    img_display = img.copy()

    # Dự đoán
    vehicle_result = coco_model(img, verbose=True)[0]
    plate_result = plate_model(img, verbose=True)[0]

    # Lọc và đổi tên class cho model COCO
    vehicle_result = filter_boxes_by_class(vehicle_result, SELECTED_VEHICLE_CLASSES)
    vehicle_result = relabel_boxes(vehicle_result, CLASS_MAPPING)

    # Vẽ các box của xe (xanh dương)
    img_display = draw_boxes_on_image(img_display, vehicle_result.boxes, vehicle_result.names, color=(255, 0, 0))

    # Vẽ các box của biển số (đỏ)
    img_display = draw_boxes_on_image(img_display, plate_result.boxes, plate_result.names, color=(0, 0, 255))

    # Hiển thị và lưu ảnh
    im_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im_rgb)
    im.show()

    os.makedirs(SAVE_PATH, exist_ok=True)
    file_name = os.path.basename(IMAGE_TEST_PATH)
    save_path = os.path.join(SAVE_PATH, f"result_{file_name}")
    im.save(save_path)
    print(f"[INFO] Saved combined detection result at: {save_path}")


#######################################################################################################################
# Main Execution
#######################################################################################################################
if __name__ == "__main__":
    main_function()

# End of File

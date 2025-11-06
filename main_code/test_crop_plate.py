"""
*******************************************************************************************************************
General Information
********************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTERGRATION
File:          CROP_PLATE.py
Description:   This file is used to test model YOLO after training to crop plate.

Author:        LARRY PHONG TRUC
Email:         vanphongtruc1808@gmail.com
Created:       26/10/2025
Last Update:   26/10
Version:       1.0

Python:        3.10.11
Dependencies:  - Model YOLOv12 best.pt
               - Test image/video

Copyright:     (c) 2025 IOE INNOVATION Team
License:       [LICENSE_TYPE]

Notes:         - Just for test, may use for module
*******************************************************************************************************************
"""

#######################################################################################################################
# Imports
#######################################################################################################################
# Standard library imports
from ultralytics import YOLO
import cv2
import os


#######################################################################################################################
# Constants and Configuration
#######################################################################################################################
YOLO_PATH = r"runs\detect\train\weights\best.pt"
IMAGE_PATH = r"D:\VSCode\DCLP\big_dataset\test\image.png"
SAVE_PATH = r"D:\VSCode\DCLP\main_code\result\crop_plate"


#######################################################################################################################
# Main Classes
#######################################################################################################################
def main_function():
    # Load model đã train nhận diện biển số (đường dẫn file weights .pt)
    model = YOLO(YOLO_PATH)

    # Đọc ảnh cần detect
    image = cv2.imread(IMAGE_PATH)

    # Chạy detect trên ảnh
    results = model(image)

    # Lấy bbox kết quả đầu tiên (nếu phát hiện được)
    if results and results[0].boxes:
        # results[0].boxes.xyxy trả về list bbox dạng [x1, y1, x2, y2]
        bbox = results[0].boxes.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = bbox.astype(int)

        # Crop vùng biển số theo bbox
        cropped_plate = image[y1:y2, x1:x2]

        # Hiển thị ảnh crop vùng biển số
        cv2.imshow("Cropped License Plate", cropped_plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Lưu ảnh crop
        cv2.imwrite(os.path.join(SAVE_PATH, "cropped_plate.jpg"), cropped_plate)
    else:
        print("Không phát hiện biển số trên ảnh")


#######################################################################################################################
# Main Execution
#######################################################################################################################
if __name__ == "__main__":
    main_function()

# End of File
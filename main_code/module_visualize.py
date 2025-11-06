"""
*******************************************************************************************************************
General Information
********************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTEGRATION
File:          module_visualize.py
Description:   Generate annotated video showing vehicles and license plates.
               Includes bordered boxes, cropped license plate previews, 
               and text overlays of recognized plate numbers.

Author:        LARRY PHONG TRUC (formatted by ChatGPT)
Email:         vanphongtruc1808@gmail.com
Created:       05/10/2025
Last Update:   01/11/2025
Version:       1.6

Python:        3.10.11
*******************************************************************************************************************
"""

#######################################################################################################################
# Imports
#######################################################################################################################
import cv2
import numpy as np
import pandas as pd
import os

#######################################################################################################################
# Constants and Configuration
#######################################################################################################################
CSV_PATH = r"D:\VSCode\DCLP\main_code\result\test_interpolated.csv"
VIDEO_PATH = r"D:\VSCode\DCLP\big_dataset\Video\train\\7194238501489.mp4"
OUTPUT_VIDEO_PATH = r"D:\VSCode\DCLP\main_code\result\video\out.mp4"

#######################################################################################################################
# Helper Functions
#######################################################################################################################
def draw_border(img, top_left, bottom_right, color=(0, 255, 0),
                thickness=5, line_length_x=200, line_length_y=200):
    """
    Vẽ viền góc vuông bo tròn quanh bounding box.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-left corner
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

#######################################################################################################################
# Main Function
#######################################################################################################################
def main_function():
    """
    Tạo video có vẽ khung xe, biển số và hiển thị text biển số OCR tương ứng.
    """

    # Đọc file kết quả CSV
    results = pd.read_csv(CSV_PATH)

    # Mở video gốc
    cap = cv2.VideoCapture(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    # Tạo dict lưu biển số có độ tin cậy cao nhất cho mỗi xe
    license_plate = {}
    for vehicle_id in np.unique(results['vehicle_id']):
        subset = results[results['vehicle_id'] == vehicle_id]
        max_score = np.amax(subset['license_number_score'])
        best_row = subset[subset['license_number_score'] == max_score].iloc[0]

        license_plate[vehicle_id] = {
            'license_crop': None,
            'license_plate_number': best_row['license_number']
        }

        # Lấy frame ứng với biển số tốt nhất
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_row['frame_nmr'])
        ret, frame = cap.read()
        if not ret:
            continue

        # Lấy toạ độ biển số từ 4 cột riêng
        x1, y1, x2, y2 = (
            int(best_row['license_x1']),
            int(best_row['license_y1']),
            int(best_row['license_x2']),
            int(best_row['license_y2'])
        )

        license_crop = frame[y1:y2, x1:x2, :]
        if license_crop.size == 0:
            continue
        ratio = 400 / (y2 - y1 + 1e-6)
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * ratio), 400))
        license_plate[vehicle_id]['license_crop'] = license_crop

    # Reset video về đầu
    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Duyệt từng frame và vẽ kết quả
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if not ret:
            break

        df_frame = results[results['frame_nmr'] == frame_nmr]
        for _, row in df_frame.iterrows():
            vehicle_id = row['vehicle_id']

            # --- Vẽ khung xe ---
            v_x1, v_y1, v_x2, v_y2 = (
                int(row['vehicle_x1']),
                int(row['vehicle_y1']),
                int(row['vehicle_x2']),
                int(row['vehicle_y2'])
            )
            draw_border(frame, (v_x1, v_y1), (v_x2, v_y2),
                        (0, 255, 0), 25, line_length_x=200, line_length_y=200)

            # --- Vẽ khung biển số ---
            x1, y1, x2, y2 = (
                int(row['license_x1']),
                int(row['license_y1']),
                int(row['license_x2']),
                int(row['license_y2'])
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 12)

            # --- Chèn ảnh biển số và text ---
            license_crop = license_plate.get(vehicle_id, {}).get('license_crop', None)
            if license_crop is None:
                continue

            H, W, _ = license_crop.shape
            try:
                # Vẽ ảnh biển số phía trên xe
                frame[v_y1 - H - 100:v_y1 - 100,
                      int((v_x2 + v_x1 - W) / 2):int((v_x2 + v_x1 + W) / 2), :] = license_crop

                # Vẽ nền trắng cho text
                frame[v_y1 - H - 400:v_y1 - H - 100,
                      int((v_x2 + v_x1 - W) / 2):int((v_x2 + v_x1 + W) / 2), :] = (255, 255, 255)

                # Ghi biển số
                text = str(license_plate[vehicle_id]['license_plate_number'])
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
                cv2.putText(frame, text,
                            (int((v_x2 + v_x1 - text_w) / 2),
                             int(v_y1 - H - 250 + (text_h / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
            except Exception:
                continue

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

    # Giải phóng tài nguyên
    out.release()
    cap.release()
    print(f"[INFO] Video with annotations saved at: {OUTPUT_VIDEO_PATH}")

#######################################################################################################################
# Main Execution
#######################################################################################################################
if __name__ == "__main__":
    main_function()

#######################################################################################################################
# End of File
#######################################################################################################################

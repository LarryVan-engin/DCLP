"""
*******************************************************************************************************************
General Information
********************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTEGRATION
File:          add_missing_data.py
Description:   Interpolates missing bounding box data between frames where vehicles or license plates
               were not detected, producing smoother visualization and continuous tracking.

Author:        LARRY PHONG TRUC (formatted by ChatGPT)
Email:         vanphongtruc1808@gmail.com
Created:       05/10/2025
Last Update:   28/10/2025
Version:       1.1

Python:        3.10.11
*******************************************************************************************************************
"""

#######################################################################################################################
# Imports
#######################################################################################################################
import csv
import numpy as np
from scipy.interpolate import interp1d


#######################################################################################################################
# Constants and Configuration
#######################################################################################################################
INPUT_CSV_PATH = r"D:\VSCode\DCLP\main_code\result\result_log.csv"
OUTPUT_CSV_PATH = r"D:\VSCode\DCLP\main_code\result\test_interpolated.csv"

CSV_HEADER = [
    'frame_nmr',
    'vehicle_id',
    'vehicle_x1', 'vehicle_y1', 'vehicle_x2', 'vehicle_y2',
    'license_x1','license_y1','license_x2','license_y2',
    'license_plate_bbox_score',
    'license_number',
    'license_number_score'
]


#######################################################################################################################
# Helper Functions
#######################################################################################################################
def interpolate_bounding_boxes(data):
    """
    Thực hiện nội suy (interpolation) bounding boxes bị thiếu giữa các frame
    cho mỗi xe, để đảm bảo chuyển động mượt hơn trong video.

    Args:
        data (list[dict]): Dữ liệu đọc từ file CSV, chứa thông tin bounding boxes.

    Returns:
        list[dict]: Dữ liệu sau khi đã nội suy và điền vào các frame bị thiếu.
    """

    # Chuyển dữ liệu sang dạng numpy để dễ xử lý
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    vehicle_ids = np.array([int(float(row['vehicle_id'])) for row in data])
    vehicle_bboxes = np.array([
        [float(row['vehicle_x1']), float(row['vehicle_y1']),
         float(row['vehicle_x2']), float(row['vehicle_y2'])]
         for row in data
    ])
    plate_bboxes = np.array([
        [float(row['license_x1']), float(row['license_y1']),
         float(row['license_x2']), float(row['license_y2'])]
         for row in data
    ])

    interpolated_data = []
    unique_vehicle_ids = np.unique(vehicle_ids)

    # Duyệt từng xe riêng biệt
    for vehicle_id in unique_vehicle_ids:
        # Lấy frame có car_id tương ứng
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['vehicle_id'])) == int(vehicle_id)]
        print(f"[INFO] Processing Vehicle ID {vehicle_id}: Frames {frame_numbers_}")

        vehicle_mask = vehicle_ids == vehicle_id
        vehicle_frame_numbers = frame_numbers[vehicle_mask]

        vehicle_bboxes_interpolated = []
        plate_bboxes_interpolated = []

        first_frame_number = vehicle_frame_numbers[0]
        last_frame_number = vehicle_frame_numbers[-1]

        # Lặp qua từng cặp frame liên tiếp
        for i in range(len(vehicle_bboxes[vehicle_mask])):
            frame_number = vehicle_frame_numbers[i]
            vehicle_bbox = vehicle_bboxes[vehicle_mask][i]
            plate_bbox = plate_bboxes[vehicle_mask][i]

            if i > 0:
                prev_frame_number = vehicle_frame_numbers[i - 1]
                prev_vehicle_bbox = vehicle_bboxes_interpolated[-1]
                prev_plate_bbox = plate_bboxes_interpolated[-1]

                # Nếu giữa 2 frame có khoảng trống (mất detection)
                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)

                    # Nội suy tuyến tính cho car bbox
                    interp_func = interp1d(x, np.vstack((prev_vehicle_bbox, vehicle_bbox)), axis=0, kind='linear')
                    interpolated_vehicle_bboxes = interp_func(x_new)

                    # Nội suy tuyến tính cho plate bbox
                    interp_func = interp1d(x, np.vstack((prev_plate_bbox, plate_bbox)), axis=0, kind='linear')
                    interpolated_plate_bboxes = interp_func(x_new)

                    # Thêm kết quả (bỏ frame đầu vì đã có sẵn)
                    vehicle_bboxes_interpolated.extend(interpolated_vehicle_bboxes[1:])
                    plate_bboxes_interpolated.extend(interpolated_plate_bboxes[1:])

            vehicle_bboxes_interpolated.append(vehicle_bbox)
            plate_bboxes_interpolated.append(plate_bbox)

        # Gộp kết quả lại thành danh sách dict
        for i in range(len(vehicle_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {
                'frame_nmr': str(frame_number),
                'vehicle_id': str(vehicle_id),
                'vehicle_x1': vehicle_bboxes_interpolated[i][0],
                'vehicle_y1': vehicle_bboxes_interpolated[i][1],
                'vehicle_x2': vehicle_bboxes_interpolated[i][2],
                'vehicle_y2': vehicle_bboxes_interpolated[i][3],
                'license_x1': plate_bboxes_interpolated[i][0],
                'license_y1': plate_bboxes_interpolated[i][1],
                'license_x2': plate_bboxes_interpolated[i][2],
                'license_y2': plate_bboxes_interpolated[i][3],
            }

            if str(frame_number) not in frame_numbers_:
                # Nếu là frame nội suy → điền giá trị mặc định
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Nếu là frame gốc → giữ nguyên dữ liệu thật
                original_row = [
                    p for p in data
                    if int(p['frame_nmr']) == frame_number and int(float(p['vehicle_id'])) == int(vehicle_id)
                ][0]

                row['license_plate_bbox_score'] = original_row.get('license_plate_bbox_score', '0')
                row['license_number'] = original_row.get('license_number', '0')
                row['license_number_score'] = original_row.get('license_number_score', '0')

            interpolated_data.append(row)

    return interpolated_data


#######################################################################################################################
# Main Function
#######################################################################################################################
def main_function():
    """
    Hàm chính: đọc dữ liệu CSV, nội suy bounding boxes, và ghi kết quả ra file mới.
    """
    print("[INFO] Loading input CSV file...")
    with open(INPUT_CSV_PATH, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    print(f"[INFO] Loaded {len(data)} records from {INPUT_CSV_PATH}")
    print("[INFO] Performing interpolation...")

    interpolated_data = interpolate_bounding_boxes(data)

    print(f"[INFO] Writing {len(interpolated_data)} interpolated records to {OUTPUT_CSV_PATH}")
    with open(OUTPUT_CSV_PATH, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(interpolated_data)

    print("[INFO] Interpolation completed successfully.")


#######################################################################################################################
# Main Execution
#######################################################################################################################
if __name__ == "__main__":
    main_function()

#######################################################################################################################
# End of File
#######################################################################################################################

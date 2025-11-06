"""
*******************************************************************************************************************
General Information
********************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTERGRATION
File:          UTILS.py
Description:   Utility functions for license plate reading, vehicle association, and result export.
               Integrated support for OCR recognition and YOLO-based detection pipeline.

Author:        LARRY PHONG TRUC
Email:         vanphongtruc1808@gmail.com
Created:       05/10/2025
Last Update:   05/11/2025
Version:       1.3

Python:        3.10.11
Dependencies:  - easyocr
               - numpy
               - YOLOv12 models for vehicle/plate detection

Copyright:     (c) 2025 IOE INNOVATION Team
License:       [LICENSE_TYPE]

Notes:         - Used by main modules for detection/tracking pipeline.
*******************************************************************************************************************
"""

#######################################################################################################################
# Imports
#######################################################################################################################
import string
import easyocr
import cv2
import numpy as np

#######################################################################################################################
# OCR Initialization and Global Dictionaries
#######################################################################################################################
# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion (OCR normalization)
DICT_CHAR_TO_INT = {
    'O': '0',
    'I': '1',
    'J': '3',
    'A': '4',
    'G': '6',
    'S': '5'
}

DICT_INT_TO_CHAR = {
    '0': 'O',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '5': 'S'
}


#######################################################################################################################
# CSV Writing Utility
#######################################################################################################################
def write_csv(results, output_path):
    """
    Write detection and recognition results to a CSV file.

    Args:
        results (dict): Nested dictionary containing detection results.
        output_path (str): Destination file path for CSV output.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(
            'frame_nmr,vehicle_id,'
            'vehicle_x1,vehicle_y1,vehicle_x2,vehicle_y2,'
            'license_x1,license_y1,license_x2,license_y2,'
            'license_plate_bbox_score,license_number,license_number_score\n'
        )

        for frame_nmr in results.keys():
            for vehicle_id in results[frame_nmr].keys():
                data = results[frame_nmr][vehicle_id]

                # Check for valid entries
                if (
                    'vehicle' in data and
                    'license_plate' in data and
                    'text' in data['license_plate']
                ):
                    vx1, vy1, vx2, vy2 = data['vehicle']['bbox']
                    lx1, ly1, lx2, ly2 = data['license_plate']['bbox']

                    bbox_score = data['license_plate']['bbox_score']
                    text = data['license_plate']['text']
                    text_score = data['license_plate']['text_score']

                    f.write(
                        '{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                            frame_nmr, vehicle_id,
                            vx1, vy1, vx2, vy2,
                            lx1, ly1, lx2, ly2,
                            bbox_score, text, text_score
                        )
                    )


    print(f"[INFO] CSV results successfully written to: {output_path}")


#######################################################################################################################
# License Plate Format Validation and Normalization
#######################################################################################################################
def license_complies_format(text):
    """
    Validate license plate format.

    Args:
        text (str): Detected license plate string.

    Returns:
        bool: True if format is valid, else False.
    """
    if len(text) != 7:
        return False

    if (
        (text[0] in string.ascii_uppercase or text[0] in DICT_INT_TO_CHAR) and
        (text[1] in string.ascii_uppercase or text[1] in DICT_INT_TO_CHAR) and
        (text[2] in '0123456789' or text[2] in DICT_CHAR_TO_INT) and
        (text[3] in '0123456789' or text[3] in DICT_CHAR_TO_INT) and
        (text[4] in string.ascii_uppercase or text[4] in DICT_INT_TO_CHAR) and
        (text[5] in string.ascii_uppercase or text[5] in DICT_INT_TO_CHAR) and
        (text[6] in string.ascii_uppercase or text[6] in DICT_INT_TO_CHAR)
    ):
        return True

    return False


def format_license(text):
    """
    Normalize license plate text using mapping dictionaries.

    Args:
        text (str): Raw detected license string.

    Returns:
        str: Normalized license plate string.
    """
    formatted_text = ''
    mapping = {
        0: DICT_INT_TO_CHAR, 1: DICT_INT_TO_CHAR, 4: DICT_INT_TO_CHAR,
        5: DICT_INT_TO_CHAR, 6: DICT_INT_TO_CHAR, 2: DICT_CHAR_TO_INT,
        3: DICT_CHAR_TO_INT
    }

    for j in range(7):
        formatted_text += mapping[j].get(text[j], text[j])

    return formatted_text

def process_license_plate(frame, x1, y1, x2, y2, scale_factor=1.5):
    """
    Cắt, tiền xử lý và tăng độ nét ảnh biển số xe.
    
    Args:
        frame (numpy.ndarray): Ảnh gốc (frame video).
        x1, y1, x2, y2 (int): Tọa độ Bounding Box của biển số xe.
        scale_factor (float): Hệ số để crop rộng hơn một chút (margin).

    Returns:
        numpy.ndarray: Ảnh biển số xe đã được xử lý để cải thiện OCR.
    """
    
    # Tính toán tọa độ với margin (để lấy được toàn bộ biển số, đề phòng lỗi cắt)
    h, w, _ = frame.shape
    
    # Mở rộng bounding box
    margin_w = int((x2 - x1) * (scale_factor - 1) / 2)
    margin_h = int((y2 - y1) * (scale_factor - 1) / 2)

    nx1 = max(0, x1 - margin_w)
    ny1 = max(0, y1 - margin_h)
    nx2 = min(w, x2 + margin_w)
    ny2 = min(h, y2 + margin_h)

    # Cắt ảnh
    plate_crop = frame[ny1:ny2, nx1:nx2]
    
    if plate_crop.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8) # Trả về ảnh rỗng nếu cắt lỗi
        
    # Chuyển sang ảnh xám (Grayscale)
    gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # Cân bằng histogram (tùy chọn)
    # final_plate = cv2.equalizeHist(gray_plate)
    
    # Áp dụng ngưỡng (Binary Thresholding) để làm nổi bật ký tự
    # Đây là một phương pháp phổ biến nhưng có thể cần tinh chỉnh tùy thuộc vào chất lượng ảnh
    _, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Có thể cần sử dụng thêm các phép toán hình thái học (Morphological operations) như Erode/Dilate
    
    return thresh # Trả về ảnh đã được tiền xử lý


#######################################################################################################################
# License Plate Reading (OCR)
#######################################################################################################################
def read_license_plate(license_plate_crop):
    """
    Read license plate number from cropped plate image.

    Args:
        license_plate_crop (numpy.ndarray): Cropped license plate image.

    Returns:
        tuple: (formatted_license_text, confidence_score) or (None, None)
    """
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        _, text, score = detection
        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


#######################################################################################################################
# Vehicle Association Utility
#######################################################################################################################
def get_car(license_plate, vehicle_track_ids):
    """
    Match a detected license plate to a corresponding tracked vehicle.

    Args:
        license_plate (tuple): (x1, y1, x2, y2, score, class_id)
        vehicle_track_ids (list): List of tracked vehicles [x1, y1, x2, y2, id].

    Returns:
        tuple: (x1, y1, x2, y2, car_id) if found, else (-1, -1, -1, -1, -1)
    """
    x1, y1, x2, y2= license_plate

    for vehicle in vehicle_track_ids:

        xcar1, ycar1, xcar2, ycar2, car_id, cls_name = vehicle

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle

    return -1, -1, -1, -1, -1, -1


# End of File

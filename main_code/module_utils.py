"""
*******************************************************************************************************************
General Information
********************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTEGRATION
File:          UTILS.py
Description:   Utility functions for license plate reading, vehicle association, and result export.
               Integrated support for OCR recognition and YOLO-based detection pipeline.

Author:        LARRY PHONG TRUC
Email:         vanphongtruc1808@gmail.com
Created:       05/10/2025
Last Update:   06/11/2025
Version:       1.3 (Vietnam License Plate Optimized)

Python:        3.10.11
Dependencies:  - easyocr
               - numpy
               - YOLOv12 models for vehicle/plate detection

Copyright:     (c) 2025 IOE INNOVATION Team
License:       [LICENSE_TYPE]

Notes:         - Improved regex for Vietnamese license plates
               - Added filtering for OCR noise and best-score selection
*******************************************************************************************************************
"""

#######################################################################################################################
# Imports
#######################################################################################################################
import re
import easyocr

#######################################################################################################################
# OCR Initialization and Global Dictionaries
#######################################################################################################################
reader = easyocr.Reader(['en'], gpu=True)

# Mapping for typical OCR misreads
DICT_CHAR_TO_INT = {
    'O': '0',
    'I': '1',
    'J': '3',
    'A': '4',
    'G': '6',
    'S': '5',
    'B': '8',
    'Z': '2',
    'T': '7'
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

                if 'vehicle' in data and 'license_plate' in data:
                    vx1, vy1, vx2, vy2 = data['vehicle']['bbox']
                    lx1, ly1, lx2, ly2 = data['license_plate']['bbox']

                    bbox_score = data['license_plate']['bbox_score']
                    text = data['license_plate'].get('text', 'N/A')
                    text_score = data['license_plate'].get('text_score', 0.0)

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
    Kiểm tra xem chuỗi biển số có tuân theo chuẩn Việt Nam hay không.
    Hỗ trợ:
        - Xe máy: 63-B9.68550, 51-A1.5678
        - Ô tô:   30E-12345, 50D-9999
        - Không dấu: 51H12345, 63B968550
    """
    text = text.upper().replace(' ', '').replace(':', '').replace('_', '').replace(',', '')
    for k, v in DICT_CHAR_TO_INT.items():
        text = text.replace(k, v)
    for k, v in DICT_INT_TO_CHAR.items():
        text = text.replace(k, v)

    patterns = [
        r"^\d{2}[A-Z]-\d{4,5}$",      # ô tô: 30E-12345 hoặc 50D-9999
        r"^\d{2}[A-Z]\d-\d{4,5}$",    # xe máy: 63B9-68550 hoặc 51A1-5678
        r"^\d{2}[A-Z]\d\.\d{4,5}$",   # xe máy có dấu chấm
        r"^\d{2}[A-Z]-\d{4}$",        # biển ngắn
        r"^\d{2}[A-Z]\d{4,5}$",       # ô tô không dấu
        r"^\d{2}[A-Z]\d\d{4,5}$"      # xe máy không dấu
    ]
    return any(re.match(p, text) for p in patterns)

def format_license(text):
    """
    Chuẩn hóa chuỗi biển số Việt Nam:
      - Sửa ký tự OCR sai
      - Thêm dấu '-' hoặc '.' đúng vị trí nếu thiếu
      - Viết hoa ký tự
    """
    text = text.upper().replace(' ', '').replace(':', '').replace('_', '')
    text = re.sub(r'[^A-Z0-9\-\.]', '', text)

    for k, v in DICT_CHAR_TO_INT.items():
        text = text.replace(k, v)

    if re.match(r"^\d{2}[A-Z]\d\d{4,5}$", text):
        text = text[:2] + '-' + text[2:4] + '.' + text[4:]
    elif re.match(r"^\d{2}[A-Z]\d{4,5}$", text):
        text = text[:3] + '-' + text[3:]
    elif not ('-' in text or '.' in text):
        text = text[:3] + '-' + text[3:]

    return text

#######################################################################################################################
# License Plate Reading (OCR)
#######################################################################################################################
def read_license_plate(license_plate_crop):
    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, 0.0

    try:
        texts = reader.readtext(license_plate_crop, paragraph=True, detail=0)
    except:
        return None, 0.0

    best_text = None
    for text in texts:
        if not isinstance(text, str):
            continue
        clean_text = re.sub(r'[^A-Z0-9\-\.]', '', text.upper().replace(' ', ''))
        if len(clean_text) >= 6:
            formatted = format_license(clean_text)
            if license_complies_format(formatted):
                if best_text is None or len(formatted) > len(best_text):
                    best_text = formatted

    return best_text, 0.85


#######################################################################################################################
# Vehicle Association Utility
#######################################################################################################################
def get_car(license_plate, vehicle_track_ids):
    """
    Match a detected license plate to a corresponding tracked vehicle.
    """
    x1, y1, x2, y2, _, _ = license_plate
    margin = 20  # cho phép sai lệch nhỏ

    for vehicle in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id, cls_name = vehicle
        if (x1 > xcar1 - margin and y1 > ycar1 - margin and
            x2 < xcar2 + margin and y2 < ycar2 + margin):
            return vehicle

    return -1, -1, -1, -1, -1, -1

# End of File

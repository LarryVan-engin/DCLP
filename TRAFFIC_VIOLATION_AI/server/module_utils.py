"""
********************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTEGRATION
File:          module_utils.py
Description:   Utility module for Vietnamese license plate OCR, normalization,
               vehicle association, and result export (FINAL VERSION).

Author:        LARRY PHONG TRUC
Updated by:    ChatGPT (VN ANPR refactor)
Last Update:   2026-01-07
Python:        3.10+
********************************************************************************************************************
"""

# =====================================================================
# IMPORTS
# =====================================================================
import cv2
import re
import csv
import numpy as np
import easyocr

# =====================================================================
# OCR INITIALIZATION
# =====================================================================
reader = easyocr.Reader(["en"], gpu=True)

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."

# =====================================================================
# CHARACTER FIX TABLES (SEPARATED & CONTEXT-AWARE)
# =====================================================================
CHAR_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6"
}

DIGIT_TO_CHAR = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B"
}

# =====================================================================
# BASIC TEXT UTILS
# =====================================================================
def strip_symbols(text: str) -> str:
    return re.sub(r"[.\-\s]", "", text)

def post_process_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9\.\-]", "", text.upper())

def fix_char_to_digit(text: str) -> str:
    return "".join(CHAR_TO_DIGIT.get(c, c) for c in text)

def fix_digit_to_char(text: str) -> str:
    return "".join(DIGIT_TO_CHAR.get(c, c) for c in text)

# =====================================================================
# IMAGE PREPROCESSING FOR OCR
# =====================================================================
def preprocess_variants(bgr_img):
    """
    Generate multiple grayscale/binary variants for robust OCR.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

    variants = []
    for b in [11, 15]:
        variants.append(
            cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, b, 2
            )
        )

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))

    return variants

# =====================================================================
# OCR LOW-LEVEL
# =====================================================================
def ocr_single(img_bw):
    """
    OCR a single grayscale/binary image.
    Returns (text, confidence).
    """
    res = reader.readtext(img_bw, allowlist=ALLOWLIST, detail=1)
    if not res:
        return "", 0.0

    text = "".join(r[1] for r in res)
    conf = float(np.mean([r[2] for r in res]))
    return post_process_text(text), conf

# =====================================================================
# FORMAT HELPERS
# =====================================================================
def format_motor_2line(top4: str, bottom_digits: str) -> str:
    prefix = top4[:2]
    series = top4[2:]
    if len(bottom_digits) >= 5:
        num = f"{bottom_digits[:3]}.{bottom_digits[3:5]}"
    else:
        num = bottom_digits
    return f"{prefix} - {series} {num}"

def format_car_2line(top3: str, bottom_digits: str) -> str:
    if len(bottom_digits) >= 5:
        num = f"{bottom_digits[:3]}.{bottom_digits[3:5]}"
    else:
        num = bottom_digits
    return f"{top3} {num}"

def format_car_1line(clean_digits: str) -> str:
    prefix = clean_digits[:3]
    rest = clean_digits[3:]
    if len(rest) >= 5:
        rest = f"{rest[:3]}.{rest[3:5]}"
    return f"{prefix}-{rest}"

# =====================================================================
# OCR TWO-LINE (BEST SCORE)
# =====================================================================
def ocr_two_line_bestscore(crop_bgr, plate_type: str):
    """
    OCR for Vietnamese 2-line plates using best-score selection.
    plate_type: 'motor' or 'car'
    """
    h = crop_bgr.shape[0]
    split = int(h * 0.48)

    top = crop_bgr[:split, :]
    bottom = crop_bgr[split:, :]

    best = {
        "score": -1e9,
        "top": "",
        "bottom": "",
        "ok": False
    }

    for bw_top in preprocess_variants(top):
        t_raw, t_conf = ocr_single(bw_top)
        if not t_raw:
            continue
        t = strip_symbols(t_raw)

        for bw_bot in preprocess_variants(bottom):
            b_raw, b_conf = ocr_single(bw_bot)
            if not b_raw:
                continue
            b = strip_symbols(b_raw)

            if plate_type == "motor":
                if len(t) < 4:
                    continue

                t_fixed = (
                    fix_char_to_digit(t[:2]) +
                    fix_digit_to_char(t[2]) +
                    fix_char_to_digit(t[3])
                )
                b_fixed = fix_char_to_digit(b)

                if not re.fullmatch(r"\d{2}[A-Z]\d", t_fixed):
                    continue
                if not re.fullmatch(r"\d{3,5}", b_fixed):
                    continue

                bonus = 3.0 if len(b_fixed) == 5 else 0.5
                score = (t_conf + b_conf) * 10.0 + bonus

                if score > best["score"]:
                    best.update(score=score, top=t_fixed, bottom=b_fixed, ok=True)

            else:  # car 2-line
                if len(t) < 3:
                    continue

                t_fixed = fix_char_to_digit(t[:2]) + fix_digit_to_char(t[2])
                b_fixed = fix_char_to_digit(b)

                if not re.fullmatch(r"\d{2}[A-Z]", t_fixed):
                    continue
                if not re.fullmatch(r"\d{3,5}", b_fixed):
                    continue

                bonus = 2.5 if len(b_fixed) == 5 else 0.5
                score = (t_conf + b_conf) * 10.0 + bonus

                if score > best["score"]:
                    best.update(score=score, top=t_fixed, bottom=b_fixed, ok=True)

    if not best["ok"]:
        return "", False

    if plate_type == "motor":
        return format_motor_2line(best["top"], best["bottom"]), True
    else:
        return format_car_2line(best["top"], best["bottom"]), True

# =====================================================================
# MAIN OCR ENTRY (USED BY API)
# =====================================================================
def read_license_plate_vn(frame, x1, y1, x2, y2, pad_ratio=0.18):
    """
    Main OCR function for Vietnamese license plates.
    Returns (text, confidence_flag).
    """
    H, W = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)

    crop = frame[
        max(0, y1 - py):min(H, y2 + py),
        max(0, x1 - px):min(W, x2 + px)
    ]

    if crop.size == 0:
        return None, False

    h, w = crop.shape[:2]
    ratio = w / max(h, 1)

    # ================= TWO LINE =================
    if ratio < 1.6 and (h / w) > 0.55:
        plate_type = "motor" if ratio < 1.35 else "car"
        text, ok = ocr_two_line_bestscore(crop, plate_type)
        if text:
            return text, ok

    # ================= ONE LINE CAR =================
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

    raw, conf = ocr_single(gray)
    clean = fix_char_to_digit(strip_symbols(raw))

    if 6 <= len(clean) <= 8:
        return format_car_1line(clean), True

    return None, False

# =====================================================================
# VEHICLE ASSOCIATION
# =====================================================================
def get_car(license_bbox, vehicle_tracks):
    """
    Match license plate bbox to vehicle bbox.
    """
    x1, y1, x2, y2 = license_bbox

    for vx1, vy1, vx2, vy2, vid, cls_name in vehicle_tracks:
        if x1 > vx1 and y1 > vy1 and x2 < vx2 and y2 < vy2:
            return vx1, vy1, vx2, vy2, vid, cls_name

    return -1, -1, -1, -1, -1, -1

# =====================================================================
# CSV EXPORT
# =====================================================================
def write_csv(results, output_path):
    """
    Export results dictionary to CSV.
    """
    header = [
        "frame_nmr", "vehicle_id",
        "vehicle_x1", "vehicle_y1", "vehicle_x2", "vehicle_y2",
        "license_x1", "license_y1", "license_x2", "license_y2",
        "license_plate_bbox_score",
        "license_number", "license_number_score"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for frame_nmr, vehicles in results.items():
            for vehicle_id, data in vehicles.items():
                if "vehicle" not in data or "license_plate" not in data:
                    continue

                lp = data["license_plate"]
                if "text" not in lp:
                    continue

                writer.writerow([
                    frame_nmr, vehicle_id,
                    *data["vehicle"]["bbox"],
                    *lp["bbox"],
                    lp.get("bbox_score", 0.0),
                    lp["text"],
                    lp.get("text_score", 0.0)
                ])

    print(f"[INFO] CSV written to: {output_path}")

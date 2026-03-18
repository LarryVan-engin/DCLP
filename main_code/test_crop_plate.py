"""
*******************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTERGRATION
File:          PREDICT_TEST_OCR_VN_FIXED_FINAL2.py
Description:   Vehicle + Plate Detection + OCR VN (Car & Motorbike) – FINAL STABLE VERSION (2-line best-score)
*******************************************************************************************************************
"""

import cv2
import os
import re
import numpy as np
from ultralytics import YOLO
from PIL import Image
import easyocr

# =====================================================================
# CONFIG
# =====================================================================
YOLO_PATH = r"D:\VSCode\DCLP\main_code\runs\detect\model_detect_license_plate.pt"
IMAGE_TEST_PATH = r"D:\VSCode\DCLP\big_dataset\test\xemay.png"
SAVE_PATH = r"D:\VSCode\DCLP\main_code\result\plate_detect"

PLATE_PAD_PERCENT = 0.18

plate_model = YOLO(YOLO_PATH)
ocr_reader = easyocr.Reader(["en"], gpu=True)

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."

# =====================================================================
# CHAR FIX TABLES (SEPARATED)
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
# UTILS
# =====================================================================
def strip_symbols(t: str) -> str:
    return re.sub(r"[.\-\s]", "", t)

def post_process_text(t: str) -> str:
    return re.sub(r"[^A-Z0-9\.\-]", "", t.upper())

def fix_char_to_digit(t: str) -> str:
    return "".join(CHAR_TO_DIGIT.get(c, c) for c in t)

def fix_digit_to_char(t: str) -> str:
    return "".join(DIGIT_TO_CHAR.get(c, c) for c in t)

def preprocess_variants(img_bgr):
    """
    Tạo vài biến thể ảnh để OCR ổn định.
    Lưu ý: upscale nhẹ giúp OCR top/bot dễ hơn.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

    outs = []
    for b in [11, 15]:
        outs.append(cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, b, 2
        ))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outs.append(otsu)
    outs.append(cv2.bitwise_not(otsu))
    return outs

# =====================================================================
# OCR CORE
# =====================================================================
def ocr_single(img_bw):
    """
    OCR 1 ảnh (đã là gray/bw) -> (text, conf)
    """
    res = ocr_reader.readtext(img_bw, allowlist=ALLOWLIST, detail=1)
    if not res:
        return "", 0.0
    text = "".join(r[1] for r in res)
    conf = float(np.mean([r[2] for r in res])) if res else 0.0
    return post_process_text(text), conf

def format_motor_2line(t4: str, b_digits: str) -> str:
    # t4: 4 ký tự: AA + X + Y
    prefix = t4[:2]
    series = t4[2:]  # XY
    if len(b_digits) >= 5:
        num = f"{b_digits[:3]}.{b_digits[3:5]}"
    else:
        num = b_digits
    return f"{prefix} - {series} {num}"

def format_car_2line(t3: str, b_digits: str) -> str:
    if len(b_digits) >= 5:
        num = f"{b_digits[:3]}.{b_digits[3:5]}"
    else:
        num = b_digits
    return f"{t3} {num}"

def ocr_two_line_plate_bestscore(crop_bgr, plate_type: str):
    """
    plate_type: "motor" hoặc "car"
    - Motor 2-line:
        top: đúng 4 ký tự  (\\d{2}[A-Z]\\d) sau khi fix theo vị trí
        bot: chỉ số, ưu tiên 5 digits (14883)
    - Car 2-line:
        top: \\d{2}[A-Z]
        bot: chỉ số, ưu tiên 5 digits (00228)
    """
    h = crop_bgr.shape[0]
    split = int(h * 0.48)

    top_bgr = crop_bgr[:split, :]
    bot_bgr = crop_bgr[split:, :]

    best = {
        "score": -1e9,
        "top": "",
        "bot": "",
        "ok": False
    }

    top_vars = preprocess_variants(top_bgr)
    bot_vars = preprocess_variants(bot_bgr)

    for bw_top in top_vars:
        t_raw, t_conf = ocr_single(bw_top)
        if not t_raw:
            continue
        t = strip_symbols(t_raw)

        for bw_bot in bot_vars:
            b_raw, b_conf = ocr_single(bw_bot)
            if not b_raw:
                continue
            b = strip_symbols(b_raw)

            # ===== Apply context fixes =====
            if plate_type == "motor":
                # cần đủ 4 ký tự top
                if len(t) < 4:
                    continue

                # FIX theo vị trí:
                # - 2 ký tự đầu phải là số
                # - ký tự thứ 3 phải là CHỮ
                # - ký tự thứ 4 phải là SỐ
                t_fixed = fix_char_to_digit(t[:2]) + fix_digit_to_char(t[2]) + fix_char_to_digit(t[3])

                # bot: chỉ số -> ép chữ->số
                b_fixed = fix_char_to_digit(b)

                # validate cứng
                if not re.fullmatch(r"\d{2}[A-Z]\d", t_fixed):
                    continue
                if not re.fullmatch(r"\d{3,5}", b_fixed):
                    continue

                # ===== scoring =====
                # ưu tiên bot 5 chữ số (14883) hơn bot 4 chữ số (1811)
                len_bonus = 3.0 if len(b_fixed) == 5 else (0.5 if len(b_fixed) == 4 else 0.0)
                # ưu tiên top đúng 4 ký tự (đã fixed) -> bonus
                top_bonus = 1.0
                score = (t_conf + b_conf) * 10.0 + len_bonus + top_bonus

                if score > best["score"]:
                    best.update(score=score, top=t_fixed, bot=b_fixed, ok=True)

            else:
                # car 2-line: top cần 3 ký tự
                if len(t) < 3:
                    continue

                t_fixed = fix_char_to_digit(t[:2]) + fix_digit_to_char(t[2])
                b_fixed = fix_char_to_digit(b)

                if not re.fullmatch(r"\d{2}[A-Z]", t_fixed):
                    continue
                if not re.fullmatch(r"\d{3,5}", b_fixed):
                    continue

                len_bonus = 2.5 if len(b_fixed) == 5 else (0.5 if len(b_fixed) == 4 else 0.0)
                score = (t_conf + b_conf) * 10.0 + len_bonus

                if score > best["score"]:
                    best.update(score=score, top=t_fixed, bot=b_fixed, ok=True)

    if not best["ok"]:
        return "", False

    if plate_type == "motor":
        return format_motor_2line(best["top"], best["bot"]), True
    else:
        return format_car_2line(best["top"], best["bot"]), True

def ocr_plate(img_bgr, box):
    """
    - Tự nhận diện 2 dòng bằng hình học (tương đối ổn).
    - 2 dòng: dùng best-score (khắc phục nhận nhầm 50 thay vì 59, và 1811 thay vì 14883).
    - 1 dòng ô tô: giữ logic cũ.
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * PLATE_PAD_PERCENT), int(bh * PLATE_PAD_PERCENT)

    crop = img_bgr[
        max(0, y1 - py):min(H, y2 + py),
        max(0, x1 - px):min(W, x2 + px)
    ]
    if crop.size == 0:
        return "", False

    h, w = crop.shape[:2]
    ratio = w / max(h, 1)

    # =============================
    # 2 DÒNG (vuông-ish)
    # =============================
    if ratio < 1.6 and (h / w) > 0.55:
        plate_type = "motor" if ratio < 1.35 else "car"
        text, ok = ocr_two_line_plate_bestscore(crop, plate_type)
        if text:
            return text, ok
        # fallback xuống 1 dòng nếu fail

    # =============================
    # 1 DÒNG Ô TÔ
    # =============================
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

    raw, conf = ocr_single(gray)
    clean = strip_symbols(raw)
    clean = fix_char_to_digit(clean)

    if 6 <= len(clean) <= 8:
        p = clean[:3]
        n = clean[3:]
        n = f"{n[:3]}.{n[3:5]}" if len(n) >= 5 else n
        return f"{p}-{n}", True

    return "", False

# =====================================================================
# DRAW
# =====================================================================
def draw_plate(img, box, text):
    if not text:
        return
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
    ty = y1 - 10 if y1 > 40 else y2 + th + 10

    cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 6, ty), (0, 200, 0), -1)
    cv2.putText(
        img, text, (x1 + 3, ty - 5),
        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )

# =====================================================================
# MAIN
# =====================================================================
def main_function():
    img = cv2.imread(IMAGE_TEST_PATH)
    if img is None:
        print("[ERROR] Cannot read image:", IMAGE_TEST_PATH)
        return

    vis = img.copy()
    plate_res = plate_model(img)[0]

    for box in plate_res.boxes:
        text, _ = ocr_plate(img, box)
        draw_plate(vis, box, text)

    im = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    im.show()

    os.makedirs(SAVE_PATH, exist_ok=True)
    save_path = os.path.join(SAVE_PATH, "result_final2.png")
    im.save(save_path)
    print("[INFO] Saved:", save_path)

if __name__ == "__main__":
    main_function()

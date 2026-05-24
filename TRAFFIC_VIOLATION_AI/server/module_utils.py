"""
********************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTEGRATION
File:          module_utils.py
Description:   Utility module for Vietnamese license plate OCR, normalization,
               vehicle association, and result export (FINAL VERSION).

Author:        LARRY PHONG TRUC
Updated by:    
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

# ─── Bảng chuyển đổi mở rộng CHỈ dùng cho trường BẮT BUỘC là chữ số ───────
# (hàng số dưới của biển 2 dòng — KHÔNG dùng cho vị trí ký tự series)
# Lý do tách riêng: "U" là ký tự series hợp lệ (VD: 51-U1 12345),
# nếu thêm U→4 vào CHAR_TO_DIGIT toàn cục sẽ phá vỡ nhận dạng đó.
_EXTRA_DIGIT_FIXES = {
    "U": "4",   # "4" stencil dễ bị EasyOCR đọc nhầm thành "U"
    "A": "4",   # "4" góc trên tam giác trông giống "A" khi ảnh mờ
    "J": "1",   # "J" không đuôi trông giống "1"
    "T": "7",   # "T" nằm ngang có thể nhầm "7"
    "Y": "4",   # "Y" thỉnh thoảng nhầm "4" ở góc nhìn nghiêng
}

# ─── Bảng chuyển đổi riêng cho VỊ TRÍ MÃ TỈNH (2 ký tự đầu hàng trên) ──────
# Khi crop riêng hàng trên của biển 2-dòng, ảnh nhỏ → EasyOCR dễ đọc nhầm:
#   "8" → "A"  (top-loop của stencil "8" trông giống tam giác)
#   "8" → "M"  (ký tự "81" merge ở low-res, hoặc "8" với serif)
# Quan trọng: "A"→"8" ở đây, NGƯỢC với "A"→"4" trong _EXTRA_DIGIT_FIXES.
# Lý do: hàng số DƯỚI (trình tự xe) thường nhầm "4"→"A",
#         hàng TRÊN (mã tỉnh) thường nhầm "8"→"A" do crop nhỏ ít ngữ cảnh hơn.
_PROVINCE_FIXES = {
    **CHAR_TO_DIGIT,   # O→0, Q→0, D→0, I→1, L→1, Z→2, S→5, B→8, G→6
    "A": "8",          # "8" stencil top-loop → "A" khi crop nhỏ/tối
    "M": "8",          # "81" merge hoặc "8" low-res → "M"
    "U": "4",          # từ _EXTRA_DIGIT_FIXES
    "J": "1",          # từ _EXTRA_DIGIT_FIXES
    "T": "7",          # từ _EXTRA_DIGIT_FIXES
    "Y": "4",          # từ _EXTRA_DIGIT_FIXES
    "W": "9",          # "9" stencil → "W" ở low-res
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

def force_to_digits(text: str) -> str:
    """
    Chuyển đổi tất cả ký tự sang chữ số — dùng cho trường CHỈ ĐƯỢC PHÉP là số.
    Áp dụng cả CHAR_TO_DIGIT lẫn _EXTRA_DIGIT_FIXES.
    KHÔNG dùng cho vị trí ký tự series (letter) của biển số.
    """
    _extended = {**CHAR_TO_DIGIT, **_EXTRA_DIGIT_FIXES}
    return "".join(_extended.get(c, c) for c in text)

def fix_province_digits(text: str) -> str:
    """
    Fix chuyên biệt cho 2 ký tự mã tỉnh (đầu hàng trên biển 2-dòng).
    Ưu tiên A→8 (ngược với A→4 trong _EXTRA_DIGIT_FIXES) vì khi OCR crop riêng
    hàng trên có ít ngữ cảnh hơn, "8" stencil hay bị đọc thành "A" hoặc "M".
    """
    return "".join(_PROVINCE_FIXES.get(c, c) for c in text)

def _motor_crossval(text_2line: str, stripped_1l: str) -> str:
    """
    Cross-validate sequence number của biển xe máy 2-dòng bằng kết quả OCR 1-dòng.

    Vấn đề cần giải quyết:
      · Crop hàng dưới riêng lẻ → ít ngữ cảnh → "4" dễ bị đọc nhầm thành "6"
      · OCR 1-dòng toàn biển → có ngữ cảnh → sequence đáng tin hơn
        (dù hay đọc "4" → "U", nhưng force_to_digits đã xử lý)

    Ví dụ:
      text_2line = "81-H1 2206"   (last digit sai: 4→6)
      stripped_1l = "81H220U"     → force_to_digits → sequence "2204"
      → trả về "81-H1 2204"

    Trả về "" nếu không cross-validate được (giữ nguyên 2-line result).
    """
    if len(stripped_1l) < 6:
        return ""

    p2_1l   = fix_province_digits(stripped_1l[:2])
    sl_1l   = fix_digit_to_char(stripped_1l[2])
    seq_1l  = force_to_digits(stripped_1l[3:8])   # tối đa 5 ký tự còn lại
    cand_1l = p2_1l + sl_1l + seq_1l              # 2+1+4 hoặc 2+1+5 ký tự

    # Province + series letter từ 2-line ("81-H1 2206" → strip → "81H12206"[:3] = "81H")
    stripped_2l = strip_symbols(text_2line)
    psl_2l = stripped_2l[:3] if len(stripped_2l) >= 3 else ""

    # Hai bên phải đồng ý về province + series letter
    if not psl_2l or cand_1l[:3] != psl_2l:
        return ""

    if re.fullmatch(r"\d{2}[A-Z]\d{5}", cand_1l):
        # 1-line đọc đủ 8 ký tự (có cả series digit + sequence) → dùng trực tiếp
        return format_motor_2line(cand_1l[:4], cand_1l[4:])

    if re.fullmatch(r"\d{2}[A-Z]\d{4}", cand_1l) and len(stripped_2l) >= 4:
        # 1-line thiếu series digit (OCR bỏ qua) → lấy series digit từ 2-line
        # VD: cand_1l="81H2204" (7 chars), stripped_2l="81H12206", sd="1"
        sd_2l    = stripped_2l[3]                          # series digit từ 2-line
        combined = psl_2l + sd_2l + cand_1l[3:]           # "81H"+"1"+"2204" = "81H12204"
        if re.fullmatch(r"\d{2}[A-Z]\d{5}", combined):
            return format_motor_2line(combined[:4], combined[4:])

    return ""

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
# FORMAT HELPERS (Cập nhật thêm Safety Check)
# =====================================================================
def format_motor_2line(top4: str, bottom_digits: str) -> str:
    if len(top4) < 3: return f"{top4} - {bottom_digits}" # Tránh lỗi index
    prefix = top4[:2]
    series = top4[2:]
    if len(bottom_digits) >= 5:
        num = f"{bottom_digits[:3]}.{bottom_digits[3:5]}"
    else:
        num = bottom_digits
    return f"{prefix}-{series} {num}"

def format_car_2line(top3: str, bottom_digits: str) -> str:
    if len(bottom_digits) >= 5:
        num = f"{bottom_digits[:3]}.{bottom_digits[3:5]}"
    else:
        num = bottom_digits
    return f"{top3} {num}"

def format_car_1line(clean_digits: str) -> str:
    # Check an toàn trước khi cắt chuỗi
    if len(clean_digits) < 4: return clean_digits
    
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
                    fix_province_digits(t[:2]) +  # 2 chữ số tỉnh — bảng province-specific
                    fix_digit_to_char(t[2]) +     # 1 ký tự series (giữ letter)
                    force_to_digits(t[3])         # 1 chữ số series — force để bắt L→1…
                )
                # Hàng dưới PHẢI là số → dùng force_to_digits để bắt cả U→4, A→4 …
                b_fixed = force_to_digits(b)

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

                t_fixed = fix_province_digits(t[:2]) + fix_digit_to_char(t[2])
                # Hàng dưới PHẢI là số → dùng force_to_digits
                b_fixed = force_to_digits(b)

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

    # Ghi nhớ loại biển để dùng cho fallback 1-dòng bên dưới
    # Motor: tỷ lệ gần vuông (w/h < 1.35) — Car: nằm ngang hơn (1.35 ≤ w/h < 1.6)
    is_motor = ratio < 1.35

    # ================= TWO LINE =================
    if ratio < 1.6 and (h / w) > 0.55:
        plate_type = "motor" if is_motor else "car"
        text_2l, ok_2l = ocr_two_line_bestscore(crop, plate_type)

        if text_2l:
            if is_motor:
                # Cross-validate sequence number bằng 1-line OCR (toàn biển có context đầy đủ)
                # → tránh nhầm 4↔6 do crop hàng dưới riêng lẻ ít ngữ cảnh
                gray_cv = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray_cv = cv2.resize(gray_cv, None, fx=1.8, fy=1.8,
                                     interpolation=cv2.INTER_CUBIC)
                raw_cv, _ = ocr_single(gray_cv)
                corrected = _motor_crossval(text_2l, strip_symbols(raw_cv))
                if corrected:
                    return corrected, True
            return text_2l, ok_2l

    # ================= ONE LINE FALLBACK =================
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

    raw, conf = ocr_single(gray)
    stripped = strip_symbols(raw)

    # ── Fallback xe máy 1 dòng ──────────────────────────────────────────────
    # Khi 2-dòng fail, biển xe máy đôi khi được merge thành chuỗi 7-8 ký tự.
    # Định dạng đầy đủ (8 ký tự): \d{2}[A-Z]\d{5}  VD: "81H12204" → "81-H1 2204"
    # Định dạng thiếu series digit (7 ký tự): \d{2}[A-Z]\d{4}  VD: "81H2204" (1 bị OCR drop)
    if is_motor and len(stripped) >= 7:
        prefix_2 = fix_province_digits(stripped[:2])  # 2 số tỉnh — dùng province fix
        series_l = fix_digit_to_char(stripped[2])      # ký tự series (giữ chữ)
        rest_raw = force_to_digits(stripped[3:8])      # tối đa 5 ký tự còn lại
        candidate = prefix_2 + series_l + rest_raw

        if re.fullmatch(r"\d{2}[A-Z]\d{5}", candidate):
            # Đủ 8 ký tự → định dạng chuẩn
            return format_motor_2line(candidate[:4], candidate[4:]), True

        if re.fullmatch(r"\d{2}[A-Z]\d{4}", candidate):
            # 7 ký tự: OCR bị mất series digit → định dạng không có series digit
            # VD: "81H2204" → "81-H 2204" (thông tin bị mất nhưng tốt hơn sai format)
            prov = candidate[:2]
            sl   = candidate[2]
            num  = candidate[3:]
            return f"{prov}-{sl} {num}", True

    # ── Fallback xe hơi/xe tải 1 dòng ──────────────────────────────────────
    clean = fix_char_to_digit(stripped)
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

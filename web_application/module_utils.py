"""
********************************************************************************************************************
Project:      Traffic Violation Detection
File:         module_utils.py
Description:  OCR utility – DEMO MODE (all license plates, priority single-line)
********************************************************************************************************************
"""

import cv2
import re
import numpy as np
import easyocr

# =====================================================================
# CONFIG
# =====================================================================
DEMO_MODE = True          # 🔥 BẬT DEMO OCR
PLATE_PAD_PERCENT = 0.18
MIN_LEN_DEMO = 4
MAX_LEN_DEMO = 12

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."

# =====================================================================
# OCR INIT
# =====================================================================
reader = easyocr.Reader(["en"], gpu=True)

# =====================================================================
# BASIC UTILS
# =====================================================================
def _strip_symbols(t: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", t.upper())

def _remove_red_channel(img):
    b, g, r = cv2.split(img)
    r = cv2.min(r, g)
    return cv2.merge([b, g, r])

def _preprocess_variants(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(3.0, (8, 8))
    gray = clahe.apply(gray)

    outs = []
    for b in [11, 15]:
        outs.append(
            cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, b, 2
            )
        )

    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outs.append(otsu)
    outs.append(cv2.bitwise_not(otsu))

    return outs

# =====================================================================
# OCR CORE – DEMO
# =====================================================================
def read_license_plate_vn(frame, x1, y1, x2, y2):
    """
    OCR license plate from frame using bbox.
    DEMO MODE: accept all plates, priority single-line.
    """

    H, W = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * PLATE_PAD_PERCENT), int(bh * PLATE_PAD_PERCENT)

    crop = frame[
        max(0, y1 - py):min(H, y2 + py),
        max(0, x1 - px):min(W, x2 + px)
    ]

    if crop.size == 0:
        return "", False

    crop = _remove_red_channel(crop)

    best_text = ""
    best_score = -1

    # ==========================
    # PRIORITY: SINGLE LINE OCR
    # ==========================
    for bw_img in _preprocess_variants(crop):
        res = reader.readtext(
            bw_img,
            allowlist=ALLOWLIST,
            detail=1,
            paragraph=False
        )

        if not res:
            continue

        raw = "".join([r[1] for r in res])
        conf = float(np.mean([r[2] for r in res]))

        clean = _strip_symbols(raw)

        if not (MIN_LEN_DEMO <= len(clean) <= MAX_LEN_DEMO):
            continue

        score = conf * 10 + len(clean)

        if score > best_score:
            best_score = score
            best_text = clean

    if not best_text:
        return "", False

    # ==========================
    # FORMAT (DEMO – SIMPLE)
    # ==========================
    if len(best_text) >= 6:
        return best_text, True

    return best_text, False

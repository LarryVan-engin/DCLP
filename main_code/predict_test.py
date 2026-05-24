"""
********************************************************************************************************************
Project:       Traffic Violation Detection — Edge Inference Benchmark
File:          predict_test.py
Description:   Đo tốc độ inference của 2 model AI (traffic light + YOLO12n vehicle) trong 2 kịch bản:

               Kịch bản 1 — FULL FRAME :
                   Cả 2 model chạy trên toàn bộ khung ảnh (imgsz=640).

               Kịch bản 2 — CROPPED INPUT :
                   • Traffic light → nửa trên-phải  frame[0:h/2, w/2:w]
                     (khớp đúng với parse_light_status() trong main_edge.py)
                   • Vehicle      → bounding-box của polygon ROI
                     (fallback: nửa dưới frame nếu không có ROI hợp lệ)

               Kết quả:   console report + ảnh annotated + benchmark_report.json

Author:        LARRY PHONG TRUC
Email:         vanphongtruc1808@gmail.com
Created:       24/05/2026
Version:       2.0
********************************************************************************************************************
"""

# [FIX JETSON NANO] Load libgomp trước khi import cv2/PyTorch — tránh lỗi TLS block
import ctypes, os, sys
try:
    ctypes.CDLL("/usr/lib/aarch64-linux-gnu/libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass  # Không có trên Windows — bỏ qua

import cv2
import json
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ============================================================
#  CONFIG — chỉnh tại đây trước khi chạy
# ============================================================

# Đường dẫn ảnh test (có thể override bằng CLI: python predict_test.py path/to/img.jpg)
IMAGE_PATH = r"E:\Video\train\Demo_image.png"

# Đường dẫn model
VEHICLE_MODEL_PATH = r"D:\VSCode\DCLP\TRAFFIC_VIOLATION_AI\edge\models\yolo12n.pt"
LIGHT_MODEL_PATH   = r"D:\VSCode\DCLP\TRAFFIC_VIOLATION_AI\edge\models\model_detect_traffic_light.pt"

# ROI tham chiếu cho camera thực (1920×1080).
# Chỉ dùng khi CUSTOM_ROI_PTS = None (auto-scale theo kích thước ảnh test).
DEFAULT_ROI_1920 = np.array([
    [ 604,  674],   # Top-Left
    [1464,  659],   # Top-Right
    [1894, 1058],   # Bottom-Right
    [ 244, 1071],   # Bottom-Left
], dtype=np.float32)

# ── ROI tuỳ chỉnh cho ảnh test ─────────────────────────────────────────────
# Toạ độ pixel TUYỆT ĐỐI — không bị scale, dùng thẳng cho ảnh test.
# Đặt = None để fallback về DEFAULT_ROI_1920 tự scale.
#
# Cho ảnh 2048×2048 (Demo_image.png):
#   • Vạch kẻ đường bộ hành (stop line) nằm ở  y ≈ 720–800
#   • Xe phương tiện hiện diện từ               y ≈ 760 → 2020
#   • Trapezoid bao vùng giao thông dưới vạch dừng:
#       Top  : x = 390–1660  @ y = 760   (~37 % H)
#       Bot  : x = 0–2048    @ y = 2020  (~99 % H)
CUSTOM_ROI_PTS = np.array([
    [ 390,  760],   # Top-Left   (~19 % W, ~37 % H)
    [1660,  760],   # Top-Right  (~81 % W, ~37 % H)
    [2048, 2020],   # Bottom-Right
    [   0, 2020],   # Bottom-Left
], dtype=np.int32)
# CUSTOM_ROI_PTS = None   # ← bỏ comment để dùng auto-scale từ DEFAULT_ROI_1920

# ── Vùng phát hiện đèn giao thông (tỷ lệ 0.0–1.0 so với frame) ─────────────
# Deployment default (khớp parse_light_status trong main_edge.py):
#   LIGHT_Y2 = 0.50,  LIGHT_X1 = 0.50  →  nửa trên-phải
#
# Tinh chỉnh cho ảnh 2048×2048 (đèn thực nằm ở y ≈ 130–390, x ≈ 850–1600):
#   LIGHT_Y2 = 0.25  →  crop đến y = 512   (bao gọn đèn ở y ≤ 390)
#   LIGHT_X1 = 1/3   →  crop từ x = 682   (bao gọn đèn trái ở x ≈ 850)
LIGHT_Y1_RATIO = 0.00    # đỉnh
LIGHT_Y2_RATIO = 0.25    # đáy crop  (1/4 chiều cao)
LIGHT_X1_RATIO = 1 / 3   # cạnh trái (1/3 chiều rộng)
LIGHT_X2_RATIO = 1.00    # cạnh phải (đến hết)

# Ngưỡng confidence (giống edge_config.py)
CONF_VEHICLE = 0.35
CONF_LIGHT   = 0.45

# Kích thước input YOLO
IMGSZ = 640

# Số vòng benchmark
N_WARMUP = 5    # Warm-up: kết quả bị loại, chỉ để CUDA cache sẵn
N_RUNS   = 30   # Vòng đo chính thức

# Thư mục lưu ảnh kết quả
OUTPUT_DIR = Path(__file__).parent / "benchmark_output"

# ============================================================
#  Vehicle class IDs (COCO)
# ============================================================
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ============================================================
#  UTILITY FUNCTIONS
# ============================================================

def scale_roi(roi_pts: np.ndarray, w: int, h: int,
              ref_w: int = 1920, ref_h: int = 1080) -> np.ndarray:
    """Scale toạ độ ROI từ độ phân giải tham chiếu sang kích thước thực."""
    return (roi_pts * [w / ref_w, h / ref_h]).astype(np.int32)


def roi_bbox_crop(frame: np.ndarray, roi_pts: np.ndarray):
    """
    Crop vùng bounding-box bao quanh polygon ROI.
    Trả về (crop, offset_x, offset_y) để remap lại toạ độ về frame gốc.
    """
    x1 = max(0, int(roi_pts[:, 0].min()))
    y1 = max(0, int(roi_pts[:, 1].min()))
    x2 = min(frame.shape[1], int(roi_pts[:, 0].max()))
    y2 = min(frame.shape[0], int(roi_pts[:, 1].max()))
    return frame[y1:y2, x1:x2].copy(), x1, y1


def compute_stats(times: list) -> dict:
    """Tính thống kê benchmark từ danh sách thời gian (ms)."""
    if not times:
        return {"avg_ms": 0, "min_ms": 0, "max_ms": 0,
                "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "fps": 0}
    arr = np.array(times, dtype=np.float64)
    avg = float(arr.mean())
    return {
        "avg_ms": round(avg, 2),
        "min_ms": round(float(arr.min()),  2),
        "max_ms": round(float(arr.max()),  2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "fps":    round(1000.0 / avg, 1) if avg > 0 else 0,
    }


# ============================================================
#  SCENARIO RUNNERS
# ============================================================

def run_scenario1(model_v, model_l, frame: np.ndarray) -> dict:
    """
    Kịch bản 1 — FULL FRAME
    ─────────────────────────────────────────────────────────
    Cả 2 model nhận toàn bộ khung ảnh không crop.
    Input vehicle model : W × H   (ảnh gốc)
    Input light model   : W × H   (ảnh gốc)
    """
    h, w = frame.shape[:2]
    print(f"  Input size  → cả 2 model: {w}×{h}")

    # Warm-up
    for _ in range(N_WARMUP):
        model_v(frame, imgsz=IMGSZ, conf=CONF_VEHICLE, verbose=False)
        model_l(frame, imgsz=IMGSZ, conf=CONF_LIGHT,   verbose=False)

    t_v_list, t_l_list, t_tot_list = [], [], []

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        model_v(frame, imgsz=IMGSZ, conf=CONF_VEHICLE, verbose=False)
        t1 = time.perf_counter()
        model_l(frame, imgsz=IMGSZ, conf=CONF_LIGHT,   verbose=False)
        t2 = time.perf_counter()
        t_v_list.append((t1 - t0) * 1e3)
        t_l_list.append((t2 - t1) * 1e3)
        t_tot_list.append((t2 - t0) * 1e3)

    # Chạy thêm 1 lần để lấy kết quả boxes cho visualization
    r_v = model_v(frame, imgsz=IMGSZ, conf=CONF_VEHICLE, verbose=False)[0]
    r_l = model_l(frame, imgsz=IMGSZ, conf=CONF_LIGHT,   verbose=False)[0]

    return {
        "input_vehicle": f"{w}×{h}",
        "input_light":   f"{w}×{h}",
        "vehicle": compute_stats(t_v_list),
        "light":   compute_stats(t_l_list),
        "total":   compute_stats(t_tot_list),
        # Kết quả detections để vẽ ảnh
        "_results": (r_v, r_l, {"lx": 0, "ly": 0, "vx": 0, "vy": 0}),
    }


def run_scenario2(model_v, model_l, frame: np.ndarray, roi_pts: np.ndarray) -> dict:
    """
    Kịch bản 2 — CROPPED INPUT
    ─────────────────────────────────────────────────────────
    Traffic light : frame[0 : h//2,  w//2 : w]  (nửa trên-phải)
    Vehicle       : bounding-box của ROI polygon  (fallback: nửa dưới)

    Offsets được lưu lại để remap toạ độ box về frame gốc khi vẽ.
    """
    h, w = frame.shape[:2]

    # ---------- Crop traffic light (theo tỷ lệ cấu hình) ----------
    lx_off = int(w * LIGHT_X1_RATIO)
    lx_end = int(w * LIGHT_X2_RATIO)
    ly_off = int(h * LIGHT_Y1_RATIO)
    ly_end = int(h * LIGHT_Y2_RATIO)
    light_crop = frame[ly_off:ly_end, lx_off:lx_end].copy()

    # ---------- Crop vehicle (ROI hoặc bottom half) ----------
    use_roi = roi_pts is not None and len(roi_pts) == 4
    if use_roi:
        v_crop, vx_off, vy_off = roi_bbox_crop(frame, roi_pts)
        if v_crop.size == 0:
            use_roi = False
    if not use_roi:
        vy_off = h // 2
        vx_off = 0
        v_crop = frame[vy_off:h, :].copy()

    zone_desc = (f"y={ly_off}:{ly_end} x={lx_off}:{lx_end} "
                 f"({LIGHT_X1_RATIO:.0%}–{LIGHT_X2_RATIO:.0%} W, "
                 f"top {LIGHT_Y2_RATIO:.0%} H)")
    print(f"  Input vehicle : {v_crop.shape[1]}×{v_crop.shape[0]}"
          f"  ({'ROI bbox' if use_roi else 'bottom half'})")
    print(f"  Input light   : {light_crop.shape[1]}×{light_crop.shape[0]}"
          f"  ({zone_desc})")

    # Warm-up
    for _ in range(N_WARMUP):
        model_v(v_crop,     imgsz=IMGSZ, conf=CONF_VEHICLE, verbose=False)
        model_l(light_crop, imgsz=IMGSZ, conf=CONF_LIGHT,   verbose=False)

    t_v_list, t_l_list, t_tot_list = [], [], []

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        model_v(v_crop,     imgsz=IMGSZ, conf=CONF_VEHICLE, verbose=False)
        t1 = time.perf_counter()
        model_l(light_crop, imgsz=IMGSZ, conf=CONF_LIGHT,   verbose=False)
        t2 = time.perf_counter()
        t_v_list.append((t1 - t0) * 1e3)
        t_l_list.append((t2 - t1) * 1e3)
        t_tot_list.append((t2 - t0) * 1e3)

    # Chạy thêm 1 lần để lấy boxes
    r_v = model_v(v_crop,     imgsz=IMGSZ, conf=CONF_VEHICLE, verbose=False)[0]
    r_l = model_l(light_crop, imgsz=IMGSZ, conf=CONF_LIGHT,   verbose=False)[0]

    return {
        "input_vehicle": f"{v_crop.shape[1]}×{v_crop.shape[0]}",
        "input_light":   f"{light_crop.shape[1]}×{light_crop.shape[0]}",
        "use_roi":       use_roi,
        "vehicle": compute_stats(t_v_list),
        "light":   compute_stats(t_l_list),
        "total":   compute_stats(t_tot_list),
        "_results": (r_v, r_l, {
            "lx": lx_off, "ly": ly_off,
            "lx_end": lx_end, "ly_end": ly_end,   # biên phải/dưới của light zone
            "vx": vx_off, "vy": vy_off,
        }),
    }


# ============================================================
#  VISUALIZATION
# ============================================================

def _draw_box(img, x1, y1, x2, y2, label, box_color, text_color=(255, 255, 255)):
    """Vẽ 1 bounding box với label có nền tô màu."""
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, text_color, 1, cv2.LINE_AA)


def draw_scenario1(frame: np.ndarray, s1: dict) -> np.ndarray:
    """
    Vẽ kết quả Kịch bản 1:
      - Xe     : box xanh lá (green)
      - Đèn   : box đỏ cam (orange-red)
    """
    vis = frame.copy()
    r_v, r_l, off = s1["_results"]

    # --- Xe ---
    if r_v.boxes is not None and len(r_v.boxes):
        for box in r_v.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            name   = VEHICLE_CLASSES.get(cls_id, f"cls{cls_id}")
            _draw_box(vis, x1, y1, x2, y2, f"{name} {conf:.2f}", (0, 200, 50))

    # --- Đèn ---
    if r_l.boxes is not None and len(r_l.boxes):
        for box in r_l.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            name   = r_l.names.get(cls_id, "light")
            _draw_box(vis, x1, y1, x2, y2, f"{name} {conf:.2f}", (0, 80, 255))

    return vis


def draw_scenario2(frame: np.ndarray, s2: dict, roi_pts: np.ndarray) -> np.ndarray:
    """
    Vẽ kết quả Kịch bản 2:
      - ROI polygon      : viền vàng
      - Vùng đèn crop    : hình chữ nhật tím nhạt
      - Xe               : box xanh lá (tọa độ đã remap)
      - Đèn              : box đỏ cam  (tọa độ đã remap)
    """
    vis  = frame.copy()
    h, w = vis.shape[:2]
    r_v, r_l, off = s2["_results"]
    vx, vy = off["vx"], off["vy"]
    lx, ly = off["lx"], off["ly"]

    # Vùng đèn — lấy biên thực tế từ offsets (không hardcode h//2, w)
    lx_end = off.get("lx_end", w)
    ly_end = off.get("ly_end", h // 2)
    overlay = vis.copy()
    cv2.rectangle(overlay, (lx, ly), (lx_end, ly_end), (180, 80, 255), -1)
    cv2.addWeighted(overlay, 0.12, vis, 0.88, 0, vis)
    cv2.rectangle(vis, (lx, ly), (lx_end, ly_end), (180, 80, 255), 2)
    zone_label = (f"LIGHT ZONE  {lx}:{lx_end} x {ly}:{ly_end}"
                  f"  ({lx_end - lx}x{ly_end - ly}px)")
    cv2.putText(vis, zone_label, (lx + 6, ly + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 120, 255), 2)

    # ROI polygon
    if roi_pts is not None and len(roi_pts) == 4:
        cv2.polylines(vis, [roi_pts.reshape(-1, 1, 2)], True, (0, 240, 240), 2)
        cv2.putText(vis, "VEHICLE ROI", (roi_pts[0][0] + 6, roi_pts[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 240), 2)

    # --- Xe (remap từ crop về frame) ---
    if r_v.boxes is not None and len(r_v.boxes):
        for box in r_v.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 += vx; x2 += vx
            y1 += vy; y2 += vy
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            name   = VEHICLE_CLASSES.get(cls_id, f"cls{cls_id}")
            _draw_box(vis, x1, y1, x2, y2, f"{name} {conf:.2f}", (0, 200, 50))

    # --- Đèn (remap từ crop về frame) ---
    if r_l.boxes is not None and len(r_l.boxes):
        for box in r_l.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 += lx; x2 += lx
            y1 += ly; y2 += ly
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            name   = r_l.names.get(cls_id, "light")
            _draw_box(vis, x1, y1, x2, y2, f"{name} {conf:.2f}", (0, 80, 255))

    return vis


def add_header(img: np.ndarray, title: str, total: dict) -> np.ndarray:
    """Thêm banner trên và dải timing dưới ảnh."""
    h, w = img.shape[:2]

    # --- Banner trên ---
    cv2.rectangle(img, (0, 0), (w, 36), (25, 25, 25), -1)
    cv2.putText(img, title, (10, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Dải timing dưới ---
    cv2.rectangle(img, (0, h - 44), (w, h), (20, 20, 20), -1)
    fps_str = f"~{total['fps']:.1f} FPS"
    timing  = (f"Avg: {total['avg_ms']:.1f} ms  |  "
               f"P50: {total['p50_ms']:.1f} ms  |  "
               f"P95: {total['p95_ms']:.1f} ms  |  "
               f"Min: {total['min_ms']:.1f} ms  |  "
               f"Max: {total['max_ms']:.1f} ms  |  "
               f"{fps_str}")
    cv2.putText(img, timing, (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 180), 1, cv2.LINE_AA)
    return img


def make_side_by_side(vis1: np.ndarray, vis2: np.ndarray,
                      target_h: int = 540) -> np.ndarray:
    """Ghép 2 ảnh cạnh nhau, căn chỉnh chiều cao."""
    def _resize(img):
        s = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), target_h),
                          interpolation=cv2.INTER_AREA)

    r1 = _resize(vis1)
    r2 = _resize(vis2)

    # Thêm đường phân cách 4px
    sep = np.full((target_h, 4, 3), (80, 80, 80), dtype=np.uint8)
    return np.hstack([r1, sep, r2])


# ============================================================
#  REPORT
# ============================================================

def print_report(s1: dict, s2: dict):
    """In bảng so sánh ra console."""
    SEP = "=" * 72

    def _row(label, d, indent=4):
        pad = " " * indent
        return (f"{pad}{label:<26s}"
                f"  {d['avg_ms']:>7.1f}"
                f"  {d['min_ms']:>7.1f}"
                f"  {d['max_ms']:>7.1f}"
                f"  {d['p50_ms']:>7.1f}"
                f"  {d['p95_ms']:>7.1f}"
                f"  {d['fps']:>7.1f}")

    header = (f"  {'Scenario / Model':<26s}"
              f"  {'Avg(ms)':>7}  {'Min(ms)':>7}  {'Max(ms)':>7}"
              f"  {'P50(ms)':>7}  {'P95(ms)':>7}  {'FPS':>7}")

    print("\n" + SEP)
    print("  [BENCHMARK] 🚀  INFERENCE SPEED — EDGE KIT")
    print(SEP)
    print(header)
    print("-" * 72)

    print("  📌 Kịch bản 1 — FULL FRAME")
    print(f"     Input : {s1['input_vehicle']} (cả 2 model)")
    print(_row("  Vehicle  (YOLO12n)",  s1["vehicle"]))
    print(_row("  Traffic Light",       s1["light"]))
    print(_row("  ── TỔNG / FRAME",     s1["total"]))
    print("-" * 72)

    roi_note = "ROI bbox" if s2["use_roi"] else "bottom half (fallback)"
    print("  📌 Kịch bản 2 — CROPPED INPUT")
    print(f"     Input vehicle : {s2['input_vehicle']}  ({roi_note})")
    print(f"     Input light   : {s2['input_light']}  (top-right quadrant)")
    print(_row("  Vehicle  (YOLO12n)",  s2["vehicle"]))
    print(_row("  Traffic Light",       s2["light"]))
    print(_row("  ── TỔNG / FRAME",     s2["total"]))
    print("-" * 72)

    a1 = s1["total"]["avg_ms"]
    a2 = s2["total"]["avg_ms"]
    if a1 > 0 and a2 > 0:
        speedup = a1 / a2
        saved   = a1 - a2
        print(f"  ⚡ Speedup (S2 / S1)  : {speedup:.2f}×  "
              f"(tiết kiệm {saved:+.1f} ms/frame, "
              f"+{s2['total']['fps'] - s1['total']['fps']:.1f} FPS)")
    print(SEP + "\n")


# ============================================================
#  MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Cho phép override đường dẫn ảnh qua CLI
    img_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH

    frame = cv2.imread(img_path)
    if frame is None:
        sys.exit(f"[ERROR] Không đọc được ảnh: {img_path}\n"
                 f"        Hãy đặt IMAGE_PATH đúng trong CONFIG section.")

    h, w = frame.shape[:2]
    print(f"\n[INFO] Ảnh test  : {img_path}  ({w}×{h})")
    print(f"[INFO] Benchmark : {N_WARMUP} warm-up + {N_RUNS} runs  |  imgsz={IMGSZ}\n")

    # Dùng ROI tuỳ chỉnh (toạ độ tuyệt đối) hoặc auto-scale từ DEFAULT_ROI_1920
    if CUSTOM_ROI_PTS is not None:
        roi_pts = CUSTOM_ROI_PTS
        print(f"[INFO] ROI      : CUSTOM_ROI_PTS  "
              f"(top y={roi_pts[0,1]}, x={roi_pts[0,0]}–{roi_pts[1,0]})")
    else:
        roi_pts = scale_roi(DEFAULT_ROI_1920, w, h)
        print(f"[INFO] ROI      : auto-scaled từ 1920x1080  "
              f"(top y={roi_pts[0,1]}, x={roi_pts[0,0]}–{roi_pts[1,0]})")

    # Nạp model
    print("[INFO] Loading models ...")
    model_v = YOLO(VEHICLE_MODEL_PATH)
    model_l = YOLO(LIGHT_MODEL_PATH)
    print("[INFO] Models loaded.\n")

    # ── Kịch bản 1 ──────────────────────────────────────────
    print(f"[1/2] Kịch bản 1 — FULL FRAME ...")
    s1 = run_scenario1(model_v, model_l, frame)
    print(f"  ✅  Avg total: {s1['total']['avg_ms']:.1f} ms  "
          f"(~{s1['total']['fps']:.1f} FPS)\n")

    # ── Kịch bản 2 ──────────────────────────────────────────
    print(f"[2/2] Kịch bản 2 — CROPPED INPUT ...")
    s2 = run_scenario2(model_v, model_l, frame, roi_pts)
    print(f"  ✅  Avg total: {s2['total']['avg_ms']:.1f} ms  "
          f"(~{s2['total']['fps']:.1f} FPS)\n")

    # ── Report ──────────────────────────────────────────────
    print_report(s1, s2)

    # ── Vẽ ảnh ──────────────────────────────────────────────
    vis1 = draw_scenario1(frame, s1)
    vis1 = add_header(vis1,
                      "Scenario 1 — Full Frame (Vehicle + Light: full resolution)",
                      s1["total"])

    vis2 = draw_scenario2(frame, s2, roi_pts)
    vis2 = add_header(vis2,
                      "Scenario 2 — Cropped Input (Vehicle: ROI bbox  |  Light: top-right)",
                      s2["total"])

    compare = make_side_by_side(vis1, vis2)

    p1 = OUTPUT_DIR / "scenario1_full_frame.jpg"
    p2 = OUTPUT_DIR / "scenario2_cropped.jpg"
    pc = OUTPUT_DIR / "comparison.jpg"
    cv2.imwrite(str(p1), vis1,   [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(p2), vis2,   [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(pc), compare,[cv2.IMWRITE_JPEG_QUALITY, 92])

    # ── Lưu JSON ────────────────────────────────────────────
    speedup = round(s1["total"]["avg_ms"] / s2["total"]["avg_ms"], 3) \
              if s2["total"]["avg_ms"] > 0 else None
    report = {
        "image":           img_path,
        "image_size":      f"{w}×{h}",
        "n_warmup":        N_WARMUP,
        "n_runs":          N_RUNS,
        "imgsz":           IMGSZ,
        "scenario1_full_frame": {
            "input_vehicle": s1["input_vehicle"],
            "input_light":   s1["input_light"],
            "vehicle":       s1["vehicle"],
            "traffic_light": s1["light"],
            "total":         s1["total"],
        },
        "scenario2_cropped": {
            "input_vehicle": s2["input_vehicle"],
            "input_light":   s2["input_light"],
            "use_roi":       s2["use_roi"],
            "vehicle":       s2["vehicle"],
            "traffic_light": s2["light"],
            "total":         s2["total"],
        },
        "speedup_x":        speedup,
        "saved_ms_per_frame": round(s1["total"]["avg_ms"] - s2["total"]["avg_ms"], 2),
    }
    json_path = OUTPUT_DIR / "benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("[INFO] Đã lưu kết quả:")
    print(f"  → {p1}")
    print(f"  → {p2}")
    print(f"  → {pc}")
    print(f"  → {json_path}\n")


########################################################################################################################
# Main Execution
########################################################################################################################
if __name__ == "__main__":
    main()

# End of File

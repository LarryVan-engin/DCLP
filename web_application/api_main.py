"""
********************************************************************************************************************
Project:      Traffic Violation Detection (Pro Version - Flexible ROI)
File:         api_main.py
Description:  Uses User-Defined Light Zones instead of Split Screen logic.
********************************************************************************************************************
"""

import asyncio, base64, os, time, uuid, traceback
from datetime import datetime
from typing import Dict, Any, Generator, List, Optional, Tuple

import cv2, torch, numpy as np, pandas as pd
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import re

# Local imports
from module_utils import read_license_plate_vn

# CONFIG
DB_PATH = "database/owners_sample.csv"
VIOLATION_DIR = "violations"
UPLOADS_DIR = "uploads"
INDEX_HTML_PATH = "index.html"
TRACKER_YAML = "bytetrack.yaml"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
PLATE_PENDING = "Reading..."

TRAFFIC_LIGHT_EVERY_N_FRAMES = 3
PLATE_EVERY_N_FRAMES = 5
OCR_EVERY_N_FRAMES = 5  
OCR_MAX_FRAMES = 8             
MIN_PLATE_LENGTH = 4
VIOLATION_COOLDOWN_SEC = 2.0

# STATE
os.makedirs(VIOLATION_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

cap: Optional[cv2.VideoCapture] = None
pause_processing = False
use_traffic_light = True

violations: List[Dict[str, Any]] = []
current_vehicles: Dict[int, Dict[str, Any]] = {}
history_vehicles: Dict[int, Dict[str, Any]] = {}
prev_positions: Dict[int, Tuple[int, int]] = {}
prev_inside: Dict[int, bool] = {}
plate_cache: Dict[int, Dict[str, Any]] = {}
violation_last_ts: Dict[Tuple[int, str], float] = {}
frame_idx = 0

shared_data = {
    "stats": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
    "violations": [],
    "lights": {"left": "green", "straight": "green"}, # Default green safe
    "fps": 0.0,
}

# Cấu trúc Zones mới:
# zones["lines"] = [ {"points": [[x1,y1],[x2,y2]], "label": "left"}, ... ]
# zones["light_zones"] = [ {"points": [[x,y]...], "label": "straight"}, ... ]
zones = {"lines": [], "polygons": [], "light_zones": []}

display_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

print("[INFO] Loading models...")
coco_model = YOLO("yolo12n.pt")
plate_model = YOLO(r"D:\VSCode\DCLP\main_code\runs\detect\model_detect_license_plate.pt")
traffic_light_model = YOLO(r"D:\VSCode\DCLP\main_code\runs\detect\traffic_light\weights\best.pt")

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# DB Load
try:
    owner_db = pd.read_csv(DB_PATH)
    vehicle_info = owner_db.set_index("plate").to_dict("index") if "plate" in owner_db.columns else {}
except: vehicle_info = {}

# HELPERS
def normalize_plate_db(s: str) -> str:
    """
    Chuẩn hoá biển số để so khớp:
    - Uppercase
    - Bỏ dấu -, .
    """
    if not s:
        return ""
    return re.sub(r"[^A-Z0-9]", "", s.upper())

# =========================
# LOAD CSV DATABASE
# =========================
try:
    owner_db = pd.read_csv(DB_PATH)

    # Tạo dict tra cứu nhanh theo biển số đã chuẩn hoá
    vehicle_info = {}
    for _, row in owner_db.iterrows():
        key = normalize_plate_db(row["plate"])
        vehicle_info[key] = {
            "plate": row["plate"],
            "owner": row["owner"],
            "phone": row["phone"],
            "class_vehicle": row["class_vehicle"],
            "province": row["province"],
            "registration_date": row["registration_date"],
            "id_card": row["id_card"]
        }

    print(f"[INFO] Loaded {len(vehicle_info)} vehicle records from CSV")

except Exception as e:
    print("[ERROR] Cannot load vehicle database:", e)
    vehicle_info = {}

def _norm_plate(s): return "".join(ch for ch in str(s).upper() if ch.isalnum()) if s else ""

def crop_and_encode(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = img[max(0,y1):min(img.shape[0],y2), max(0,x1):min(img.shape[1],x2)]
    if crop.size == 0: return None
    ok, buf = cv2.imencode(".jpg", crop)
    return base64.b64encode(buf).decode() if ok else None

def _extract_boxes(res):
    if res is None or res.boxes is None: return np.empty((0,4)), np.empty((0,)), np.empty((0,))
    return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()

def check_line_crossing(prev, curr, line_pts):
    def ccw(a, b, c): return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    p1, p2, q1, q2 = prev, curr, line_pts[0], line_pts[1]
    return (ccw(p1,q1,q2) != ccw(p2,q1,q2)) and (ccw(p1,p2,q1) != ccw(p1,p2,q2))

def _should_log_violation(track_id, vtype):
    now = time.time()
    key = (track_id, vtype)
    if now - violation_last_ts.get(key, 0.0) < VIOLATION_COOLDOWN_SEC: return False
    violation_last_ts[key] = now
    return True

def _now_hms(): return datetime.now().strftime("%H:%M:%S")

# MAIN LOOP
def gen_frames():
    global frame_idx, display_frame, plate_cache, violations, current_vehicles, history_vehicles, prev_positions, prev_inside, violation_last_ts, zones
    last_ts = time.time()
    # Cache đèn để giữ trạng thái khi không detect được frame này
    cached_lights = {"left": "green", "straight": "green"} 

    while True:
        if cap is None or not cap.isOpened() or pause_processing:
            if display_frame is not None:
                ok, buf = cv2.imencode(".jpg", display_frame)
                if ok: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        display_frame = frame.copy()


        
        frame_idx += 1
        
        now = time.time()
        shared_data["fps"] = 1.0 / max(1e-6, now - last_ts)
        last_ts = now

        # ====================================================
        # 1. LOGIC ĐÈN TÍN HIỆU (THEO VÙNG VẼ - ROI)
        # ====================================================
        if frame_idx % TRAFFIC_LIGHT_EVERY_N_FRAMES == 0:
            res = traffic_light_model(frame, verbose=False)[0]
            boxes, confs, clss = _extract_boxes(res)
            
            # Reset tạm thời, nếu không thấy đèn trong vùng nào thì giữ nguyên cache
            # Nhưng nếu thấy đèn trong vùng thì cập nhật
            
            for i, box in enumerate(boxes):
                if confs[i] < 0.4: continue # Confidence threshold
                x1, y1, x2, y2 = map(int, box)
                center_light = ((x1+x2)//2, (y1+y2)//2)
                
                cls_name = res.names[int(clss[i])] # red, green, yellow
                
                # Check xem đèn này nằm trong vùng nào (Left hay Straight)
                assigned_direction = None
                
                for lz in zones.get("light_zones", []):
                    pts = np.array(lz["points"], np.int32)
                    # PointPolygonTest: >0 inside, =0 on edge, <0 outside
                    if cv2.pointPolygonTest(pts, center_light, False) >= 0:
                        assigned_direction = lz["label"] # 'left' hoặc 'straight'
                        break
                
                # Nếu đèn nằm trong vùng đã vẽ -> Cập nhật trạng thái
                if assigned_direction:
                    cached_lights[assigned_direction] = cls_name
                    # Vẽ box đèn để debug
                    color = (0,0,255) if cls_name=="red" else (0,255,0)

                    cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(display_frame, f"{assigned_direction}:{cls_name}", (x1, y1-5), 0, 0.5, color, 1)

        shared_data["lights"] = cached_lights

        # ====================================================
        # 2. TRACKING & VIOLATIONS
        # ====================================================
        res = coco_model.track(frame, persist=True, tracker=TRACKER_YAML, verbose=False, conf=0.35)[0]
        boxes, confs, clss = _extract_boxes(res)
        track_ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else []

        shared_data["stats"] = {k:0 for k in shared_data["stats"]}
        seen_ids = set()

        # ===============================
        # PLATE DETECT (ONCE PER FRAME)
        # ===============================
        detected_plates = []
        if frame_idx % PLATE_EVERY_N_FRAMES == 0:
            pres = plate_model(frame, verbose=False)[0]
            pb, pc, _ = _extract_boxes(pres)


            for j, pbox in enumerate(pb):
                if pc[j] < 0.35:
                    continue
                px1, py1, px2, py2 = map(int, pbox)
                detected_plates.append((px1, py1, px2, py2))

        for i, box in enumerate(boxes):
            if int(clss[i]) not in VEHICLE_CLASSES: 
                continue

            track_id = track_ids[i] if i < len(track_ids) else -1
            if track_id == -1: 
                continue
            
            seen_ids.add(track_id)
            cls_name = VEHICLE_CLASSES[int(clss[i])]
            shared_data["stats"][cls_name] += 1
            x1, y1, x2, y2 = map(int, box)
            center = ((x1+x2)//2, (y1+y2)//2)


            # ===============================
            # INIT PLATE CACHE FOR VEHICLE
            # ===============================
            if track_id not in plate_cache:
                plate_cache[track_id] = {
                    "bbox": None,
                    "ocr_results": [],
                    "final_plate": None,
                    "stable": False,
                    "last_seen": frame_idx,
                    "best_vehicle_img": None,
                    "best_vehicle_area": 0,
                    "best_frame_idx": None,
                    "plate_vehicle_img": None,
                    "plate_img_taken": False
                }

            # ===============================
            # MATCH PLATE → VEHICLE (DISTANCE)
            # ===============================
            def plate_inside_vehicle(px1,py1,px2,py2, x1,y1,x2,y2):
                return px1 > x1 and py1 > y1 and px2 < x2 and py2 < y2

            best_plate = None
            for (px1, py1, px2, py2) in detected_plates:
                if plate_inside_vehicle(px1,py1,px2,py2, x1,y1,x2,y2):
                    best_plate = (px1,py1,px2,py2)
                    break

            if best_plate:
                plate_cache[track_id]["bbox"] = best_plate
                plate_cache[track_id]["last_seen"] = frame_idx

            # ===============================
            # OCR PROCESSING (SEPARATE)
            # ===============================
            p_data = plate_cache.get(track_id)
            if p_data["bbox"] and not p_data["stable"] and frame_idx % OCR_EVERY_N_FRAMES == 0:
                px1, py1, px2, py2 = p_data["bbox"]

                if px2 - px1 >= 20 and py2 - py1 >= 10:

                    txt, ok = read_license_plate_vn(frame, px1, py1, px2, py2)

                    if txt:
                        txt_norm = _norm_plate(txt)
                        if len(txt_norm) >= MIN_PLATE_LENGTH:
                            p_data["ocr_results"].append(txt_norm)

                        if len(p_data["ocr_results"]) >= OCR_MAX_FRAMES:
                            from collections import Counter
                            # Khi OCR đã đủ frame và vote xong
                            final = Counter(p_data["ocr_results"]).most_common(1)[0][0]
                            final_norm = normalize_plate_db(final)

                            p_data["final_plate"] = final
                            p_data["stable"] = True

                            # =========================
                            # TRUY XUẤT DATABASE
                            # =========================
                            info = vehicle_info.get(final_norm)

                            if info:
                                current_vehicles[track_id].update({
                                    "plate": info["plate"],                # giữ format gốc trong DB
                                    "owner": info["owner"],
                                    "phone": info["phone"],
                                    "class_vehicle": info["class_vehicle"],
                                    "province": info["province"],
                                    "registration_date": info["registration_date"],
                                    "id_card": info["id_card"]
                                })
                            else:
                                current_vehicles[track_id].update({
                                    "plate": final,
                                    "owner": "Không xác định",
                                    "phone": "",
                                    "class_vehicle": "",
                                    "province": "",
                                    "registration_date": "",
                                    "id_card": ""
                                })

                            history_vehicles[track_id] = dict(current_vehicles[track_id])
            
            v_area = (x2 - x1) * (y2 - y1)
            cache = plate_cache[track_id]

            if v_area > cache["best_vehicle_area"]:
                # ENCODE IMAGE
                img_b64 = crop_and_encode(frame, [x1, y1, x2, y2])
                if img_b64:
                    cache["best_vehicle_img"] = f"data:image/jpeg;base64,{img_b64}"
                    cache["best_vehicle_area"] = v_area
            
            if best_plate and not cache["plate_img_taken"]:
                img_b64 = crop_and_encode(frame, cache["bbox"])
                if img_b64:
                    cache["plate_vehicle_img"] = f"data:image/jpeg;base64,{img_b64}"
                    cache["plate_img_taken"] = True

            img_url = (
                cache.get("best_vehicle_img") or 
                cache.get("plate_vehicle_img") or 
                ""
                )

            # VẼ BBOX VEHICLE + ID
            cv2.rectangle(display_frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(display_frame, f"{cls_name}-{track_id}", (x1,y1-10), 0, 0.6, (0,255,0), 2)
            
            # VẼ BBOX PLATE + KẾT QUẢ OCR
            if p_data["bbox"]:
                px1, py1, px2, py2 = p_data["bbox"]
                cv2.rectangle(display_frame, (px1,py1), (px2,py2), (255,255,0), 2)
                if p_data["final_plate"]:
                    cv2.putText(display_frame, p_data["final_plate"], (px1, py2+20), 0, 0.7, (255,255,0), 2)


            # UPDATE DATA
            if track_id not in current_vehicles: 
                current_vehicles[track_id] = {
                    "plate": PLATE_PENDING,
                    "img": None
                    }
            
            best_img = cache.get("best_vehicle_img")
            current_vehicles[track_id].update({
                "img": img_url, 
                "plate": cache["final_plate"] or PLATE_PENDING,
                "type": cls_name, 
                "time": _now_hms(), 
                "last_seen": time.time()
                })
            
            if best_plate and best_plate != p_data["bbox"]:
                p_data["ocr_results"].clear()
                p_data["stable"] = False


            history_vehicles[track_id] = dict(current_vehicles[track_id])
            

            # --------------------------------------------------------
            # VIOLATION CHECK (NÂNG CẤP)
            # --------------------------------------------------------
            if track_id in prev_positions:
                prev_pt = prev_positions[track_id]
                
                # Check LINE Crossing
                for line_obj in zones.get("lines", []):
                    line_pts = line_obj["points"]
                    line_label = line_obj.get("label", "straight") # 'left' or 'straight'
                    
                    if check_line_crossing(prev_pt, center, line_pts):
                        # Xác định trạng thái đèn dựa trên Label của Line
                        light_status = shared_data["lights"].get(line_label, "green")
                        
                        is_violation = False
                        # Logic phạt: Nếu đèn là Đỏ (hoặc Vàng tùy bạn) thì phạt
                        if use_traffic_light and light_status == "red":
                            is_violation = True
                        elif not use_traffic_light:
                            is_violation = True # Chế độ test không cần đèn
                            
                        if is_violation and _should_log_violation(track_id, "line"):
                            # Save violation image
                            today = datetime.now().strftime("%Y-%m-%d")
                            sdir = os.path.join(VIOLATION_DIR, today)
                            os.makedirs(sdir, exist_ok=True)
                            fname = f"{datetime.now().strftime('%H-%M-%S')}_{track_id}.jpg"
                            fpath = os.path.join(sdir, fname)
                            try: 
                                cv2.imwrite(fpath, frame[y1:y2, x1:x2])
                            except: 
                                pass

                            cv = current_vehicles.get(track_id, {})
                            pc = plate_cache.get(track_id, {})

                            violations.append({
                                "id": track_id,

                                # ===== BIỂN SỐ =====
                                "plate": cv.get("plate", ""),

                                # ===== THÔNG TIN DB =====
                                "owner": cv.get("owner", "Không xác định"),
                                "phone": cv.get("phone", ""),
                                "class_vehicle": cv.get("class_vehicle", ""),
                                "province": cv.get("province", ""),
                                "registration_date": cv.get("registration_date", ""),
                                "id_card": cv.get("id_card", ""),

                                # ===== VI PHẠM =====
                                "type": f"VƯỢT ĐÈN ĐỎ ({line_label.upper()})",
                                "time": _now_hms(),

                                # ===== HÌNH ẢNH =====
                                "img": img_url,
                                "plate_img": pc.get("plate_vehicle_img", "")
                            })

                # =========================
                # 2. POLYGON VIOLATION
                # =========================
                for poly in zones.get("polygons", []):
                    pts = np.array(poly, np.int32)

                    inside = cv2.pointPolygonTest(pts, center, False) >= 0

                    key = (track_id, id(poly))
                    was_inside = prev_inside.get(key, False)

                    # Chỉ phạt khi vừa đi vào
                    if inside and not was_inside:
                        if _should_log_violation(track_id, "polygon"):
                            cv = current_vehicles.get(track_id, {})
                            pc = plate_cache.get(track_id, {})

                            violations.append({
                                "id": track_id,
                                "plate": cv.get("plate", ""),

                                "owner": cv.get("owner", "Không xác định"),
                                "phone": cv.get("phone", ""),
                                "class_vehicle": cv.get("class_vehicle", ""),
                                "province": cv.get("province", ""),
                                "registration_date": cv.get("registration_date", ""),
                                "id_card": cv.get("id_card", ""),

                                "type": "VÀO VÙNG CẤM",
                                "time": _now_hms(),

                                "img": img_url,
                                "plate_img": pc.get("plate_vehicle_img", "")
                            })


                    prev_inside[key] = inside

            prev_positions[track_id] = center


        now=time.time()
        
        ids_to_remove = []
        for tid, info in current_vehicles.items():
                # Nếu ID không nằm trong frame này (seen_ids) VÀ đã mất tích > 5s
                if tid not in seen_ids and (now - info.get("last_seen", 0) > 5.0):
                    ids_to_remove.append(tid)
        for tid in ids_to_remove:
                current_vehicles.pop(tid, None)
                plate_cache.pop(tid, None)
                prev_positions.pop(tid, None)
                prev_inside.pop(tid, None)
        
        # DRAW ZONES ON FRAME
        for l in zones["lines"]: 
            p = l["points"] 
            c = (0,0,255) if l.get("label") == "left" else (255,0,0) # Đỏ cho trái, Xanh cho thẳng 
            cv2.line(display_frame, tuple(map(int,p[0])), tuple(map(int,p[1])), c, 3) 
            
        for lz in zones.get("light_zones", []): 
            pts = np.array(lz["points"], np.int32) 
            c = (0,255,255) if lz.get("label") == "left" else (255,255,0) 
            cv2.polylines(display_frame, [pts], True, c, 2)

            
        for p in zones["polygons"]:
            pts = np.array(p, np.int32)
            
            cv2.polylines(display_frame, [pts], True, (255,0,255), 3)


        # ENCODE FRAME
        try:
            ok, buf = cv2.imencode(".jpg", display_frame)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        except Exception as e:
            print("--------------------------------------------------")
            print(f"[ERROR] Encoding frame: {e}")
            traceback.print_exc()
            print("--------------------------------------------------")
            
        time.sleep(0.01)

# ENDPOINTS
@app.get("/")
async def index(): return FileResponse(INDEX_HTML_PATH)
@app.get("/stream")
async def stream(): return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    file_path = "static/favicon.ico"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("")

@app.post("/api/set_option")
async def set_option(data: Dict[str, Any]):
    global use_traffic_light
    use_traffic_light = bool(data.get("use_traffic_light", True))
    print(f"[OPTION] Traffic Light Enforcement: {use_traffic_light}")
    return {"status": "ok", "use_traffic_light": use_traffic_light}

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    global cap, frame_idx, violations, current_vehicles, prev_positions, history_vehicles, plate_cache
    
    try:
        # 1. Tạo tên file duy nhất để tránh trùng
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "mp4"
        filename = f"{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(UPLOADS_DIR, filename)

        # 2. Lưu file xuống ổ cứng
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 3. Giải phóng camera cũ (nếu có) và load video mới
        if cap is not None:
            cap.release()
        
        cap = cv2.VideoCapture(file_path)
        
        # 4. Reset toàn bộ trạng thái AI để bắt đầu video mới sạch sẽ
        violations.clear()
        current_vehicles.clear()
        history_vehicles.clear()
        prev_positions.clear()
        prev_inside.clear()
        plate_cache.clear()
        frame_idx = 0
        
        print(f"[INFO] Upload thành công: {file_path}")
        return {"status": "ok", "message": f"Đã tải lên và đang xử lý: {file.filename}"}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/api/zones")
async def update_zones(data: Dict[str, Any]): 
    global zones; 
    zones = {
        "lines": data.get("lines", []),
        "polygons": data.get("polygons", []),
        "light_zones": data.get("light_zones", [])
    } 
    return {"status": "ok"}

@app.post("/api/pause")
async def toggle_pause(data: Dict[str, Any]): global pause_processing; pause_processing = data.get("pause", False); return {"status": "ok"}

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            # Gửi thêm biến "is_paused"
            await websocket.send_json({
                "vehicles": dict(list(history_vehicles.items())[-20:]),
                "violations": violations[-10:],
                "stats": shared_data["stats"],
                "lights": shared_data["lights"],
                "fps": shared_data.get("fps", 0),
                "is_paused": pause_processing  # <--- THÊM DÒNG NÀY
            })
            await asyncio.sleep(0.1)
        except WebSocketDisconnect: # Bắt lỗi ngắt kết nối đúng cách
            print("[INFO] Client disconnected")
            break
        except Exception as e:
            print(f"[ERROR] WS: {e}")
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
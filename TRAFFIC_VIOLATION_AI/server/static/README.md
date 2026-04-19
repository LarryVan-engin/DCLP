1. TỔNG QUAN HỆ THỐNG
    Hệ thống được thiết kế theo kiến trúc Hybrid Edge-Server, mục tiêu cốt lõi là chuyển toàn bộ AI nặng sang Edge (Jetson Nano), Server chỉ giữ lại các phần nhẹ và giao diện quản trị.
    Phân vai rõ ràng:

    Edge (Jetson Nano):
        Chạy inference AI nặng và logic vi phạm thời gian thực.
        Models: YOLOv12n (vehicle detection + ByteTrack), traffic_light_model.
        Không chạy Plate Detection.
        Mode Real-time: Detect xe, đèn giao thông, stream live MJPEG + heartbeat stats.
        Mode Video: Xử lý đầy đủ violation logic (vượt đèn đỏ/vàng, sai làn, ngược chiều, vào vùng cấm…), sử dụng data-driven lane detection, smart capture + force fallback, crop ảnh xe (vehicle crop) và gửi về Server qua WebSocket.

    Server (FastAPI Dashboard):
        Nhận video upload, cho phép vẽ zones linh hoạt (Konva.js).
        Gửi video + zones đến Edge.
        Nhận violation packet từ Edge → chạy plate_model + read_license_plate_vn (module_utils) + tra cứu database CSV.
        Lưu trữ, hiển thị realtime, export CSV.


    Lợi ích chính của kiến trúc này:

        Edge xử lý nhanh, giảm latency và băng thông.
        Server tập trung vào OCR chính xác, nghiệp vụ và giao diện người dùng.
        Dễ scale nhiều camera Jetson.


2. LUỒNG XỬ LÝ CHI TIẾT (End-to-End)
'''
    text========================================================================================
                        LUỒNG XỬ LÝ SERVER - EDGE (Giai đoạn 4)
    ========================================================================================

    [ USER ] 
    ↓ Upload video hoặc chọn camera Jetson

    ────────────────────────────────────────────────────────────────────────────────────────
                            SERVER (FastAPI + Dashboard)
    ────────────────────────────────────────────────────────────────────────────────────────
    1. Nhận video upload → Lưu tạm vào /uploads
        ↓
    2. User vẽ zones linh hoạt (light_zone, violation lines, forbidden polygons) qua Konva.js
        ↓
    3. Nhấn "Gửi xử lý Edge"
        ↓
    Gửi package qua REST API:
    → Video file + zones JSON + mode ("video" hoặc "realtime")
    → Đến Jetson Nano tương ứng
        ↓
    ────────────────────────────────────────────────────────────────────────────────────────
                            EDGE DEVICE (JETSON NANO)
    ────────────────────────────────────────────────────────────────────────────────────────
    4. Nhận package từ Server
        ↓
    5. Chạy inference (TensorRT):
        ├── YOLOv12n Vehicle Detection + ByteTrack
        ├── Traffic Light Detection (theo light_zones)
        ├── Data-driven Lane Detection (quan sát 100 frame đầu)
        └── Path tracking & Violation Engine
        ↓
    6. Violation Logic (chỉ mode=video):
        ├── Vượt đèn đỏ / đèn vàng
        ├── Sai làn / Lấn làn
        ├── Đi ngược chiều
        ├── Vào vùng cấm
        └── Vượt vạch dừng
        ↓
    Nếu phát hiện vi phạm:
        ├── Smart Capture: crop vehicle (context + tight)
        ├── Force Fallback (timeout hoặc xe rời khung hình)
        ├── Gói packet: JSON metadata + vehicle_crop_base64
        └── Gửi realtime về Server qua WebSocket
        ↓
    7. Mode Real-time:
    → Stream MJPEG live + heartbeat (fps, stats xe, trạng thái đèn)
    ────────────────────────────────────────────────────────────────────────────────────────
                            SERVER (Dashboard)
    ────────────────────────────────────────────────────────────────────────────────────────
    8. Nhận packet vi phạm qua WebSocket
        ↓
    9. Xử lý nhẹ trên Server:
        ├── Chạy plate_model detect biển số trên vehicle_crop
        ├── EasyOCR + read_license_plate_vn() (module_utils)
        ├── Tra cứu owners_sample.csv
        ├── Lưu violation hoàn chỉnh (ảnh xe + ảnh plate) vào folder + DB
        └── Push realtime lên giao diện Dashboard
        ↓
    10. Hiển thị:
        ├── Live stream / Video processing
        ├── Danh sách xe đang theo dõi
        ├── Danh sách vi phạm (modal chi tiết ảnh xe + plate)
        ├── Thống kê, Export CSV
        └── Quản lý zones & Jetson cameras
    ========================================================================================
'''
    Hai Chế độ chính trên Dashboard:
        Tab Video Processing:

        Upload video → Vẽ zones → Gửi xử lý Edge.
        Edge xử lý violation + crop ảnh xe → gửi packet về.
        Server chạy Plate Detection + OCR + DB lookup → lưu & hiển thị.

        Tab Real-time Monitoring:

        Chọn Jetson → Xem live MJPEG stream.
        Nhận heartbeat stats (fps, đèn, số lượng xe…).


3. Kết nối Server - Edge

    Gửi task: REST API (video file + zones JSON + mode).
    Nhận kết quả: WebSocket (violation packets + vehicle_crop_base64).
    Live stream: MJPEG từ Edge.
    Heartbeat & Status: Edge gửi định kỳ qua WebSocket.


4. Mô tả mẫu dữ liệu
    Violation Packet từ Edge → Server:
'''
    JSON{
    "camera_id": "JETSON_01",
    "mode": "video",
    "timestamp": "2026-04-18T12:30:45+07:00",
    "track_id": 245,
    "violation_type": "VUOT DEN DO + SAI LAN",
    "lane": 2,
    "direction": "straight",
    "confidence": 0.94,
    "vehicle_crop_base64": "base64_string_cua_anh_xe_da_crop"
    }
'''
5. Các bước cần thực hiện (Giai đoạn 4)

    Hoàn thiện: Phân tích hệ thống chi tiết.
    Thiết kế cấu trúc folder project mới cho Edge-Server.
    Chỉnh sửa api_main.py thành Server hỗ trợ Edge (thêm endpoint nhận violation, tách plate processing).
    Viết Edge Service (dựa trên full_main.py, rút gọn, chỉ crop vehicle).
    Cập nhật giao diện frontend (thêm nút "Gửi xử lý Edge", chọn camera, trạng thái kết nối).
    Test end-to-end và triển khai trên Jetson Nano.

Cấu trúc tổng thể dự án

'''
    TRAFFIC_VIOLATION_AI/
    ├── edge/                          # Code chạy trên Jetson Nano
    │   ├── main_edge.py               # Service chính trên Edge (inference + violation logic)
    │   ├── edge_config.py             # Cấu hình Edge (camera rtsp, mode, ip server...)
    │   ├── models/                    # Models tối ưu TensorRT (sẽ copy từ server)
    │   │   ├── yolo12n.engine
    │   │   └── traffic_light.engine
    │   ├── utils/
    │   │   ├── lane_detection.py      # Data-driven lane + tracking
    │   │   ├── violation_engine.py    # Logic phát hiện vi phạm
    │   │   └── capture_utils.py       # Smart crop + force fallback
    │   └── requirements_edge.txt
    │
    ├── server/                        # Code FastAPI Dashboard
    │   ├── api_main.py                # FastAPI server (sẽ chỉnh sửa)
    │   ├── module_utils.py            # OCR + Plate processing (chạy trên server)
    │   ├── database/
    │   │   └── owners_sample.csv
    │   ├── violations/                # Thư mục lưu vi phạm
    │   ├── uploads/                   # Video user upload
    │   ├── static/
    │   │   ├── style.css
    │   │   ├── app.js
    │   │   └── favicon.ico
    │   ├── templates/
    │   │   └── index.html
    │   └── requirements_server.txt
    │
    ├── shared/                        # Code chung giữa Edge và Server
    │   ├── zones_utils.py             # Xử lý zones JSON
    │   └── schemas.py                 # Pydantic models cho packet violation
    │
    ├── config/                        # File config chung
    │   ├── cameras.json               # Danh sách Jetson cameras
    │   └── default_zones.json
    │
    ├── docs/                          # Tài liệu
    │   └── Phân_tich_he_thong_G4.md
    │
    ├── scripts/                       # Script hỗ trợ
    │   ├── export_tensorrt.py         # Export model sang TensorRT cho Jetson
    │   ├── deploy_edge.sh
    │   └── backup_db.sh
    │
    ├── README.md
    ├── .gitignore
    └── run_server.bat / run_edge.sh

'''

Giải thích chi tiết cấu trúc
'''
====================================================================================
Thư mục  | Mục đích chính                                        | File quan trọng
edge/    | Toàn bộ code chạy trên Jetson Nano                    | main_edge.py
server/  | Dashboard FastAPI + Plate OCR + DB + UI               | api_main.py
shared/  | "Code và schema dùng chung (violation packet, zones)" | schemas.py
config/  | Quản lý cấu hình camera và zones                      | cameras.json
scripts/ |"Script triển khai, export model, backup               |export_tensorrt.
====================================================================================
'''

Cấu trúc file quan trọng sẽ được tạo/chỉnh sửa

1. Edge side (edge/main_edge.py)

    Nhận video + zones từ Server qua REST
    Chạy 2 mode: realtime và video
    Chỉ crop vehicle_crop khi có vi phạm
    Gửi packet về Server qua WebSocket

2. Server side (server/api_main.py)

    Giao diện vẽ zones hay ROI sẽ được thu gọn lại, thao tác đơn giản như CalibreGUI đã thiết kế khi chạy demo, hoạt động tương tự như trên full_main demo nhưng thích nghi với dashboard để nhìn hiện đại và chuyên nghiệp hơn
    Thêm endpoint nhận violation packet từ Edge
    Thêm logic Plate Detection + OCR khi nhận crop từ Edge

3. Shared schemas

    Định nghĩa chuẩn format packet gửi từ Edge → Server
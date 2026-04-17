PHÂN TÍCH HỆ THỐNG CAMERA AI (EDGE AI) – THEO DÕI & PHẠT NGUỘI PHƯƠNG TIỆN GIAO THÔNG
(Phiên bản Pro - Flexible ROI)
Trạng thái hiện tại: Đang thực hiện Giai đoạn 4

1. TỔNG QUAN HỆ THỐNG
    Hệ thống được thiết kế theo kiến trúc Hybrid Edge-Server, nhằm tối ưu hiệu suất và khả năng mở rộng cho việc xử lý phạt nguội giao thông.
    Mục tiêu chính:
    Tách biệt hoàn toàn phần AI nặng (detection, tracking, violation logic) sang Edge (Jetson Nano). Server chỉ thực hiện các phần nhẹ: nhận kết quả, OCR cuối cùng, tra cứu database, lưu trữ và giao diện dashboard.
    Phân vai rõ ràng:

    Edge (Jetson Nano):
    Chạy inference AI nặng (YOLOv12n + ByteTrack + Traffic Light model + Plate Detection).
    Mode Real-time: Chỉ detect xe + đèn giao thông, stream live MJPEG về Server.
    Mode Video: Xử lý đầy đủ violation logic (vượt đèn đỏ/vàng, đi ngược chiều, lấn làn, vào vùng cấm…), trích xuất ảnh xe + ảnh plate crop, gửi kết quả về Server.

    Server (Dashboard):
    Nhận video upload từ người dùng, cho phép vẽ zones (light_zone, line, polygon) qua giao diện Konva.js, gửi gói video + zones sang Edge, nhận kết quả vi phạm từ Edge, chạy plate_model + EasyOCR + tra cứu owners_sample.csv, lưu trữ và hiển thị kết quả.

Lợi ích:

    Edge xử lý realtime, giảm tải và latency.
    Server tập trung vào UI/UX và nghiệp vụ.
    Dễ scale nhiều camera Jetson.


2. LUỒNG XỬ LÝ CHI TIẾT (End-to-End)

Dưới đây là sơ đồ luồng xử lý chi tiết:

'''

                        LUỒNG XỬ LÝ SERVER - EDGE (Giai đoạn 4)
    ========================================================================================

    [ USER ] 
    ↓ Upload video hoặc chọn camera thực tế

    ────────────────────────────────────────────────────────────────────────────────────────
                            SERVER (FastAPI + Dashboard)
    ────────────────────────────────────────────────────────────────────────────────────────
    1. Nhận video upload → Lưu tạm vào folder uploads
        ↓
    2. User vẽ zones (light_zone, violation lines, forbidden polygons) trên giao diện web (Konva)
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
    5. Chạy inference full (TensorRT):
        ├── YOLOv12n Vehicle + Tracking (ByteTrack)
        ├── Traffic Light Detection (theo light_zones)
        ├── Plate Detection
        └── Tự động xác định làn đường & hướng di chuyển
        ↓
    6. Violation Engine (chỉ khi mode=video):
        ├── Vượt đèn đỏ/vàng
        ├── Đi ngược chiều
        ├── Lấn làn
        ├── Vào vùng cấm
        └── Vượt vạch dừng
        ↓
    Nếu phát hiện vi phạm:
        ├── Crop ảnh xe + ảnh plate
        ├── Gói thành packet (JSON metadata + 2 ảnh base64)
        └── Gửi ngay về Server qua WebSocket
        ↓
    7. (Mode Real-time): 
    → Chỉ stream MJPEG live + heartbeat (fps, stats xe, trạng thái đèn)
    ────────────────────────────────────────────────────────────────────────────────────────
                            SERVER (Dashboard)
    ────────────────────────────────────────────────────────────────────────────────────────
    8. Nhận packet vi phạm qua WebSocket
        ↓
    9. Xử lý nhẹ trên Server:
        ├── Chạy plate_model + EasyOCR trên plate_crop (nếu cần tinh chỉnh)
        ├── Tra cứu thông tin chủ xe từ owners_sample.csv
        ├── Lưu violation hoàn chỉnh vào folder + Database
        └── Push realtime lên giao diện Dashboard
        ↓
    10. Hiển thị:
        ├── Danh sách xe đang theo dõi
        ├── Danh sách vi phạm (có modal chi tiết ảnh + thông tin)
        ├── Thống kê, Export CSV
        └── Live stream từ Edge (nếu realtime)
    ========================================================================================

'''
Hai Tab chính trên Dashboard:
Tab Video Processing:

    User upload video → Server lưu tạm.
    User vẽ zones qua Konva.
    Nhấn “Gửi xử lý Edge” → Server gửi video + zones + mode=video qua REST đến Edge.
    Edge xử lý full violation → gửi từng violation packet (JSON + 2 ảnh base64) qua WebSocket về Server.
    Server nhận → chạy plate detection + EasyOCR + tra DB → lưu và hiển thị.

Tab Real-time Monitoring:

Chọn Jetson → hiển thị live stream MJPEG từ Edge.
Nhận heartbeat stats (fps, lights, count xe…).


3. Kết nối Server - Edge

Chính: REST API (gửi video + zones) + WebSocket (nhận violation realtime).
Real-time stream: MJPEG stream từ Edge.
Heartbeat: Edge gửi định kỳ stats qua WebSocket.
Tương lai: Có thể bổ sung MQTT cho quản lý nhiều Edge.


4. Mô tả mẫu dữ liệu
    Packet Violation từ Edge → Server:
    JSON{
        "mode": "video",
        "timestamp": "2026-04-18T00:30:45+07:00",
        "track_id": 156,
        "violation_type": "VƯỢT ĐÈN ĐỎ (straight)",
        "vehicle_crop_base64": "...",
        "plate_crop_base64": "...",
        "lane": 2,
        "confidence": 0.93
    }
  "camera_id": "JETSON_01",
  
5. Các bước cần thực hiện (Giai đoạn 4)

    Hoàn thiện REST API trên Server để nhận và forward video + zones đến Edge.
    Triển khai WebSocket receiver trên Server để nhận violation packet từ Edge.
    Phát triển service trên Jetson Nano (nhận package, xử lý 2 mode: realtime & video).
    Tích hợp gửi base64 ảnh + metadata từ Edge về Server.
    Hoàn thiện logic plate final OCR + DB lookup trên Server.
    Test end-to-end với video mô phỏng.
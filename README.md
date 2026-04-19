PHÂN TÍCH HỆ THỐNG CAMERA AI (EDGE AI) – THEO DÕI & PHẠT NGUỘI PHƯƠNG TIỆN GIAO THÔNG
(Phiên bản Pro - Hybrid Edge-Server với MQTT Realtime - Flow Tối Ưu)
Trạng thái hiện tại: Đang thực hiện Giai đoạn 4 (đã tích hợp MQTT làm giao thức chính)

1. TỔNG QUAN HỆ THỐNG
    Hệ thống được thiết kế theo kiến trúc Hybrid Edge-Server, sử dụng MQTT làm giao thức chính cho realtime communication, kết hợp REST cho một số tác vụ dữ liệu lớn.
    
    Mục tiêu cốt lõi:

        Video được lưu sẵn và xử lý hoàn toàn trên Jetson Nano (Edge).
        Dashboard chỉ đóng vai trò điều khiển, giám sát realtime và lưu trữ kết quả.
        Tối ưu tốc độ, giảm tải hệ thống, dễ kết nối ngoại vi và dễ scale nhiều camera.

Phân vai rõ ràng:

    Edge (Jetson Nano):
        Lưu trữ và xử lý toàn bộ video local.
        Chạy AI nặng: YOLOv12n + ByteTrack, Traffic Light Detection, Data-driven Lane Detection, Violation Engine, Smart Capture.
        Không chạy Plate Detection.
        Publish stream MJPEG, heartbeat, violation packet và gói kết quả cuối cùng qua MQTT.

    Server (FastAPI Dashboard):
        Subscribe MQTT từ Edge để nhận dữ liệu realtime.
        Tab Realtime Monitoring: Xem stream camera từ Edge, thống kê lượng phương tiện và trạng thái đèn giao thông.
        Tab Video Processing: Chọn video có sẵn trên Edge để xử lý phạt nguội.
        Nhận violation packet → chạy Plate Detection + OCR (module_utils).
        Nhận gói kết quả cuối → lưu trữ + đồng bộ lên Cloud MongoDB Atlas.



2. LUỒNG XỬ LÝ CHI TIẾT (End-to-End với MQTT)
    '''
    text========================================================================================
                        LUỒNG XỬ LÝ SERVER - EDGE VỚI MQTT (Tối ưu 2026)
    ========================================================================================

    [ USER trên Dashboard ]
    ↓ Chọn Jetson Nano → Chọn chế độ (Realtime Monitoring hoặc Video Processing)

    ────────────────────────────────────────────────────────────────────────────────────────
                            MQTT BROKER (HiveMQ Cloud / Mosquitto)
    ────────────────────────────────────────────────────────────────────────────────────────
    Các Topic chính:
    - control/{camera_id}/command     → Server publish lệnh điều khiển
    - status/{camera_id}/heartbeat    → Edge publish stats + đèn + fps
    - stream/{camera_id}/mjpeg        → Edge publish stream frame realtime
    - violation/{camera_id}           → Edge publish ViolationPacket
    - complete/{camera_id}            → Edge publish gói kết quả video xong

    ────────────────────────────────────────────────────────────────────────────────────────
                            TAB REALTIME MONITORING
    ────────────────────────────────────────────────────────────────────────────────────────
    1. Dashboard publish lệnh kết nối camera
        ↓ MQTT control/{camera_id}/command
    2. Edge:
    - Bắt đầu stream từ camera gắn trên Edge (hoặc video mô phỏng)
    - Chạy model phát hiện phương tiện → thống kê realtime (car, motorcycle, bus, truck)
    - Phát hiện đèn giao thông → cập nhật trạng thái
    - Publish stream MJPEG + heartbeat liên tục
        ↓
    3. Dashboard subscribe và hiển thị:
    - Luồng camera realtime
    - Thống kê lượng phương tiện theo thời gian thực
    - Trạng thái đèn giao thông
    - Xử lý trực tiếp (không lưu vi phạm)

    ────────────────────────────────────────────────────────────────────────────────────────
                            TAB VIDEO PROCESSING
    ────────────────────────────────────────────────────────────────────────────────────────
    4. User chọn video có sẵn trên Edge + cấu hình zones
        ↓ MQTT control/{camera_id}/command
    5. Edge xử lý video:
    - Load video từ thư mục local
    - Chạy đầy đủ violation logic + smart capture + force fallback
    - Publish stream quá trình xử lý realtime
    - Khi có vi phạm → publish ViolationPacket (chứa vehicle_crop_base64)
    - Khi xử lý xong toàn bộ video → publish gói kết quả hoàn chỉnh
        ↓
    6. Server (subscribe MQTT):
    - Nhận violation packet → chạy plate_model + read_license_plate_vn() + tra cứu DB
    - Lưu violation + đẩy lên Cloud MongoDB Atlas ngay lập tức
    - Nhận gói complete → lưu video annotated + toàn bộ metadata vào MongoDB
    ========================================================================================
    '''

3. Kiến trúc Kết nối & MQTT Design
    MQTT là giao thức chính (ưu tiên cho tốc độ, giảm load và dễ kết nối ngoại vi):

        Control: control/{camera_id}/command — Lệnh từ Server sang Edge (start/stop, chọn video, cập nhật zones…)
        Status: status/{camera_id}/heartbeat — Heartbeat + thống kê xe + đèn + fps
        Stream: stream/{camera_id}/mjpeg — Stream frame realtime (annotated)
        Violation: violation/{camera_id} — Violation packet realtime
        Complete: complete/{camera_id} — Gói kết quả cuối cùng sau khi xử lý video

REST chỉ dùng cho:

    Upload gói kết quả video lớn (nếu MQTT không phù hợp)
    Một số API config ban đầu hoặc fallback khi MQTT tạm mất

Cloud MongoDB Atlas: Server tự động insert violation và metadata ngay khi nhận từ Edge qua MQTT.

Lợi ích khi dùng MQTT làm chính:

    Tốc độ realtime cao, latency thấp.
    Giảm đáng kể tải CPU/network trên Jetson Nano.
    Reconnect tự động, QoS đảm bảo không mất gói tin vi phạm.
    Dễ scale thêm nhiều camera sau này.
    Tiết kiệm băng thông so với REST polling.


4. Mô tả mẫu dữ liệu
    ViolationPacket (Edge → Server qua MQTT):
        JSON{
        "camera_id": "JETSON_01",
        "mode": "video",
        "timestamp": "2026-04-18T16:45:12+07:00",
        "track_id": 234,
        "violation_type": "VUOT DEN DO + SAI LAN",
        "lane": 3,
        "direction": "straight",
        "confidence": 0.94,
        "vehicle_crop_base64": "base64_string..."
        }
    Heartbeat từ Edge:
        JSON{
        "camera_id": "JETSON_01",
        "stats": {"car": 18, "motorcycle": 35, "bus": 4, "truck": 6},
        "lights": {"left": "green", "straight": "red"},
        "fps": 27.3,
        "active_video": "video_test_01.mp4"
        }

5. Các bước cần thực hiện (Giai đoạn 4)

    Hoàn thiện phân tích hệ thống với MQTT (đã xong).
    Cài đặt MQTT Broker (HiveMQ Cloud khuyến nghị).
    Viết edge/main_edge.py với MQTT Client (paho-mqtt) – hỗ trợ cả 2 tab.
    Chỉnh sửa server/api_main.py để tích hợp MQTT Subscriber + MongoDB Atlas.
    Cập nhật frontend (thêm chọn camera, nút điều khiển, hiển thị 2 Tab rõ ràng).
    Test end-to-end: Realtime Monitoring, Video Processing, Violation realtime, Gói hoàn thành.
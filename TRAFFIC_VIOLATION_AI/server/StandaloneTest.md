# BẢN KẾ HOẠCH KIỂM THỬ ĐỘC LẬP (STANDALONE TEST PLAN)

**Dự án:** AI Traffic Monitoring & Violation Detection System  
**Giai đoạn:** Kiểm thử Module trước khi Tích hợp (Pre-Integration Testing)  
**Thực hiện:** Larry Phong Trực & Phan Thành Sang  
**Đơn vị:** HCMUT AI Research  
**Ngày lập:** 20/04/2026

---

## PHẦN 1: KIỂM THỬ ĐỘC LẬP EDGE NODE (JETSON NANO)

**Mục tiêu:** Đảm bảo bộ não AI trên Edge có thể đọc video, phân làn, phát hiện vi phạm và đóng gói dữ liệu (Base64) chuẩn xác mà **không cần Server**.

**Môi trường Setup:**
- Chạy `main_edge.py` với một video test lưu sẵn trên Jetson Nano.
- Sử dụng MQTT Broker cục bộ (Mosquitto) hoặc HiveMQ Cloud, nhưng **tắt hoàn toàn Server**.
- Hardcode zones_config tạm thời trong code để mô phỏng vùng ROI.

### Test Cases

| Mã        | Kịch bản Test                          | Cách thực hiện                                      | Kết quả mong đợi (Expected Result)                                      | Trạng thái |
|-----------|----------------------------------------|-----------------------------------------------------|--------------------------------------------------------------------------|------------|
| UNIT-E01  | Luồng đọc Video & Tracking            | Chạy AI với video ngã tư, bật `verbose=True`       | Load thành công `yolo12n.pt`, console in tọa độ boxes và track_ids liên tục, không crash | ⏳        |
| UNIT-E02  | Thuật toán Data-Driven Lane (Auto-ROI)| Quan sát log trong 100 frame đầu                   | Sau 100 frame in ra `[LANE DETECTION] ✅ Đã học xong!` và tọa độ `car_only_zones` hợp lý | ⏳        |
| UNIT-E03  | Violation Engine (Giả lập đèn)         | Gán cứng `current_light = "red"`, cho xe vượt đèn  | Console in ra log: `[EDGE] Đã gửi vi phạm: ID X ...`                   | ⏳        |
| UNIT-E04  | Smart Crop & Mã hóa Base64             | Thêm lệnh `cv2.imwrite` sau hàm crop               | Ảnh lưu rõ nét, có padding, chuỗi Base64 không rỗng                     | ⏳        |
| UNIT-E05  | Hiệu năng độc lập (Raw Performance)    | Tắt gửi MQTT, chỉ chạy inference thuần             | FPS ≥ 25-30, không Memory Leak sau 30 phút chạy                         | ⏳        |

---

## PHẦN 2: KIỂM THỬ ĐỘC LẬP SERVER NODE & DASHBOARD

**Mục tiêu:** Đảm bảo Server có thể bắt MQTT, chạy OCR, lưu Database và hiển thị Dashboard mượt mà mà **không cần Jetson Nano phải online**.

**Môi trường Setup:**
- Chạy `api_main.py` trên PC/Laptop và mở Dashboard tại `http://localhost:8000`.
- Sử dụng **MQTT Explorer** hoặc file `mock_publisher.py` để bơm dữ liệu giả vào Broker.

### Test Cases

| Mã        | Kịch bản Test                              | Cách thực hiện                                              | Kết quả mong đợi (Expected Result)                                      | Trạng thái |
|-----------|--------------------------------------------|-------------------------------------------------------------|--------------------------------------------------------------------------|------------|
| UNIT-S01  | Kết nối Dashboard & WebSockets            | Truy cập `http://localhost:8000`                           | Giao diện tải hoàn chỉnh, hiển thị "MQTT: Connected", không lỗi console | ⏳        |
| UNIT-S02  | Nhận Heartbeat giả lập (Real-time UI)     | Gửi JSON heartbeat qua MQTT Explorer                       | Dashboard cập nhật số lượng xe, FPS và trạng thái đèn ngay lập tức      | ⏳        |
| UNIT-S03  | Xử lý luồng Vi phạm (OCR Core)            | Gửi ViolationPacket chứa Base64 ảnh xe rõ biển số         | Server xử lý OCR thành công, lưu ảnh vào thư mục `violations/`          | ⏳        |
| UNIT-S04  | Map Database & Lưu MongoDB                | Gửi vi phạm với biển số có trong `owners_sample.csv`      | Tạo document mới trên MongoDB Atlas với đầy đủ thông tin chủ xe         | ⏳        |
| UNIT-S05  | Render Modal Vi Phạm trên UI              | Gửi vi phạm → Click vào thẻ vi phạm trên Dashboard         | Modal hiển thị đúng thông tin chủ xe và ảnh bằng chứng rõ nét           | ⏳        |

---

## PHẦN 3: GHI CHÚ & HƯỚNG DẪN THỰC HIỆN

- Các test case này được thiết kế để kiểm tra **độc lập từng module** trước khi tích hợp hoàn chỉnh Edge-Server.
- Ưu tiên chạy UNIT-E01 → UNIT-E05 trước, sau đó mới sang phần Server.
- Sử dụng công cụ: **MQTT Explorer**, **Postman**, **Chrome DevTools (F12)**.
- Ghi chép kết quả thực tế, FPS, lỗi (nếu có) và ảnh chụp màn hình vào báo cáo kiểm thử.

---

**Người soạn:** Larry Phong Trực  
**Ngày:** 20/04/2026

---


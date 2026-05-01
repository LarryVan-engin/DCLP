# 🧪 AUTOMATION TEST SUITE - COMPLETE GUIDE

**Tài liệu:** Hướng dẫn chạy Automation Test Suite (Toàn diện)  
**Dành cho:** Research Team (HCMUT)  
**Ngày cập nhật:** 02/05/2026  
**Phiên bản:** Final Release

---

## 📌 Tổng Quan Hệ Thống Test

Hệ thống kiểm thử đã được nâng cấp và gộp lại thành **2 file chính** để tiện lợi nhất cho quá trình CI/CD và triển khai:

```
TRAFFIC_VIOLATION_AI/docs/
├── test_automation_full_suite.py      # Chạy mô phỏng TOÀN BỘ 28 test cases trên máy Host
├── test_automation_real.py            # Chạy benchmark và test trực tiếp trên phần cứng
├── Testplan.md                        # Kế hoạch kiểm thử chính thức
└── TEST_AUTOMATION_COMPLETE_GUIDE.md  # Hướng dẫn này
```

Tổng cộng có **28 Test Cases** được chia thành 5 nhóm:
1. **Standalone (Độc lập):** Kiểm thử tính năng cốt lõi của Edge và Server.
2. **Integration (Tích hợp):** Kiểm thử luồng dữ liệu E2E (End-to-End).
3. **Non-Functional (Phi chức năng):** Chịu tải (Stress test), mạng, tài nguyên.
4. **AI Evaluation (Đánh giá AI):** Benchmark tốc độ Inference của YOLO và OCR.
5. **Real Deployment (Thực tế):** Test kết nối phần cứng, mạng và MongoDB Cloud.

---

## 🏗️ 1. Chạy Mô Phỏng Toàn Hệ Thống (Full Suite)

File `test_automation_full_suite.py` sẽ tự động giả lập toàn bộ các module (MQTT, WebSockets, Camera) để đánh giá nhanh logic của toàn dự án mà không cần cắm điện thiết bị Jetson.

### ⚡ Lệnh chạy:
```bash
cd d:\VSCode\DCLP
python TRAFFIC_VIOLATION_AI\docs\test_automation_full_suite.py
```

### 📋 Nhóm Test "AI Evaluation" (Mới Cập Nhật)
Nhóm test này đặc biệt quan trọng để đánh giá độ trễ của các mô hình Deep Learning:
- **AI-01: Vehicle Detection:** Tốc độ quét xe (yolo12n.pt)
- **AI-02: Traffic Light:** Tốc độ quét đèn đỏ
- **AI-03: License Plate Detection:** Tốc độ tìm khung biển số (model_detect_license_plate.pt)
- **AI-04: OCR Reading:** Tốc độ đọc chữ bằng EasyOCR
- **AI-05: Violation Engine Stress Test:** Đánh giá tốc độ tính toán thuật toán bắt lỗi hình học

**Báo cáo sẽ được lưu tại:** `TRAFFIC_VIOLATION_AI/docs/test_report_full_YYYYMMDD_HHMMSS.txt`

---

## 🚀 2. Chạy Triển Khai Thực Tế (Real Deployment)

File `test_automation_real.py` được thiết kế để kết nối trực tiếp với Jetson Nano và Cloud DB để test kết nối vật lý thực sự. Nó hỗ trợ chạy cục bộ trên Jetson hoặc kết nối từ xa.

### ⚡ Các chế độ chạy:

**1. Chạy trực tiếp trên Jetson Nano (Khuyên dùng):**
Mở terminal trên Jetson và chạy:
```bash
python3 docs/test_automation_real.py --local
```

**2. Chạy từ Laptop/PC thông qua SSH tới Jetson:**
```bash
python docs\test_automation_real.py --ssh 192.168.1.100 --user jetson --password "mật_khẩu"
```

**3. Chỉ Test luồng MQTT Server:**
```bash
python docs\test_automation_real.py --mqtt 127.0.0.1
```

### 📋 Các Test Cases Thực Tế (RealDeploy)
- **RD-01:** Ping & MQTT Connectivity
- **RD-02:** Camera Stream FPS
- **RD-03:** Đo độ trễ E2E thực tế (Hardware)
- **RD-04:** Check RAM/CPU/Nhiệt độ vật lý của Jetson
- **RD-05:** Ping MongoDB Atlas Cloud trực tiếp

**Báo cáo sẽ được lưu tại:** `TRAFFIC_VIOLATION_AI/docs/test_report_real_YYYYMMDD_HHMMSS.txt`

---

## ✅ Tiêu Chí Thành Công (Acceptance Criteria)

Để Đồ án Tốt Nghiệp được đánh giá là hoàn hảo, hệ thống cần đạt:
- ✅ **Pass Rate:** Đạt 28/28 Test Cases (100%).
- ✅ **E2E Latency:** Dưới 1.5s từ lúc xe vi phạm đến lúc hiển thị Dashboard.
- ✅ **AI Inference:** Tổng thời gian quét xe + đèn + biển số + OCR < 100ms trên Server PC và < 250ms trên Jetson.
- ✅ **Resource:** Nhiệt độ Jetson duy trì dưới 80°C.

---

## 🐛 Khắc Phục Lỗi Cơ Bản (Troubleshooting)

1. **Lỗi `ModuleNotFoundError: No module named 'ultralytics'` hoặc `easyocr`**
   ```bash
   pip install ultralytics easyocr
   ```

2. **Test AI-04 (OCR) báo SKIP:**
   - Nguyên nhân: Bạn chưa cài `easyocr` hoặc máy tính thiếu thư viện C++ build tools.
   - Khắc phục: Hệ thống đã tự fallback sang SKIP thay vì FAIL để không làm dừng bộ test. Cài đặt lại thư viện để test chạy đủ.

3. **Lỗi RD-05 (MongoDB) báo FAIL:**
   - Cập nhật lại IP của bạn vào danh sách Whitelist trên giao diện MongoDB Atlas.

---

## 📞 Thông Tin
**Dự án:** Traffic Violation Detection System Using AI  
**Tác giả:** Larry Phong Truc & Phan Thành Sang  
**Hướng dẫn:** ThS. Đinh Quốc Hùng

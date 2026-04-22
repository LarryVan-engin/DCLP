# 🧪 AUTOMATION TEST SUITE - STANDALONE & INTEGRATION TESTING

**Tài liệu:** Hướng dẫn chạy Automation Test Suite (Standalone + Integration)  
**Dành cho:** Research Team (HCMUT)
**Ngày cập nhật:** 22/04/2026  
**Phiên bản:** Final  

---

## 📌 Tổng Quan Hệ Thống Test

Hệ thống kiểm thử gồm **2 file chính**:

```
TRAFFIC_VIOLATION_AI/docs/
├── test_automation_standalone.py      # 9 test cases độc lập (S-ED-01 to S-SV-04)
├── test_automation_integration.py     # 10 test cases tích hợp (INT-01 to NF-04)
├── Testplan.md                        # Test plan chính thức
└── TEST_AUTOMATION_GUIDE.md           # Hướng dẫn Standalone (cũ)
```

---

## 🏗️ File 1: test_automation_standalone.py

### 📋 Test Cases

#### Edge Node Tests (S-ED-01 đến S-ED-05)

| Test ID | Tên | Mô Tả | Mục Đích |
|---------|-----|-------|----------|
| **S-ED-01** | Khởi tạo & Tracking | Mock YOLO model, load & track 2 vehicles | Đảm bảo YOLO hoạt động |
| **S-ED-02** | Auto-ROI Learning | Gom cụm car/motorcycle boxes trong 10 frames | RAM được giải phóng sau learning |
| **S-ED-03** | Memory Leak Check | Xử lý 100 vòng violations, check RAM usage | Memory < 1MB, ổn định |
| **S-ED-04** | Smart Crop & Encoding | Crop bounding box, encode Base64 | Ảnh sắc nét, Base64 không rỗng |
| **S-ED-05** | Violation Logic | Mô phỏng trajectory rẽ phải vs đi thẳng | Rẽ phải KHÔNG bắt, đi thẳng BẮT |

#### Server Node Tests (S-SV-01 đến S-SV-04)

| Test ID | Tên | Mô Tả | Mục Đích |
|---------|-----|-------|----------|
| **S-SV-01** | OCR Processing | Mock EasyOCR, clean plate text "TH6788" | OCR chính xác, fix ký tự |
| **S-SV-02** | CSV Database Lookup | Map plate với owners_sample.csv | Lấy được owner info đúng |
| **S-SV-03** | MongoDB Storage | Tạo violation document, serialize JSON | Lưu đầy đủ, không lỗi ObjectId |
| **S-SV-04** | WebSocket Comm. | Mock WebSocket, broadcast heartbeat | UI cập nhật realtime |

### ⚡ Chạy Standalone Tests

```bash
cd d:\VSCode\DCLP
python TRAFFIC_VIOLATION_AI\docs\test_automation_standalone.py
```

**Expected Output:**
```
🚀 BẮT ĐẦU CHẠY AUTOMATION TEST SUITE - STANDALONE TESTING
═════════════════════════════════════════════════════════

✅ S-ED-01 PASS (0.002s)
✅ S-ED-02 PASS (0.015s)
✅ S-ED-03 PASS (0.008s)
✅ S-ED-04 PASS (0.012s)
✅ S-ED-05 PASS (0.010s)
✅ S-SV-01 PASS (0.005s)
✅ S-SV-02 PASS (0.018s)
✅ S-SV-03 PASS (0.008s)
✅ S-SV-04 PASS (0.009s)

📊 TÓM TẮT: 9/9 test cases PASS (100.0%)
```

**Báo cáo:** `test_report_YYYYMMDD_HHMMSS.txt`

---

## 🏗️ File 2: test_automation_integration.py

### 📋 Test Cases

#### Integration Tests (INT-01 đến INT-06)

Kiểm thử **End-to-End flow** từ Edge → MQTT → Server → WebSocket → UI

| Test ID | Tên | Flow | Kỳ Vọng |
|---------|-----|------|---------|
| **INT-01** | Tinh chỉnh Auto-ROI | UI gửi config → MQTT → Edge cập nhật → Broadcast | Status "Đã cập nhật zones" |
| **INT-02** | Bắt lỗi Vượt Đèn Đỏ | Edge phát hiện → Server OCR/DB → UI alert | **E2E Latency ≤ 1.5s** ✓ |
| **INT-03** | Bắt lỗi Combo | 2 violations (Sai làn + Vượt đèn) cùng track_id | Combine → Modal đầy đủ |
| **INT-04** | Chế độ Đường Cấm | Bật mode → Edge phát hiện → Alert | Severity "high" |
| **INT-05** | Đo Latency E2E | 10 measurements liên tục | Avg ≤ 1.5s, Max ≤ 2.0s |
| **INT-06** | Synthetic vs Real Video | Process 100 frames mỗi loại | 0 errors, stable FPS |

#### Non-Functional Tests (NF-01 đến NF-04)

Kiểm thử **hiệu năng & độ tin cậy** của hệ thống

| Test ID | Tên | Scenario | Kỳ Vọng |
|---------|-----|----------|---------|
| **NF-01** | Stress Test | 50 vehicles/frame × 100 frames = 5000 vehicles | **Không crash**, violations ≥ 500 |
| **NF-02** | Network Drop | Disconnect → Reconnect → Resend | Buffered messages được gửi lại ✓ |
| **NF-03** | Data Quality | OCR failure (lóa, ký tự rác, biển số rỗng) | Reject, **không lưu garbage** |
| **NF-04** | Resource Usage | 10 phút simulation (~5000 frames) | CPU < 80%, GPU < 90%, RAM < 85%, **Temp < 80°C** |

### ⚡ Chạy Integration Tests

```bash
cd d:\VSCode\DCLP
python TRAFFIC_VIOLATION_AI\docs\test_automation_integration.py
```

**Expected Output:**
```
📍 CHẠY INTEGRATION TESTS (INT-01 đến INT-06)

✅ INT-01 PASS (0.105s)
✅ INT-02 PASS (0.115s) [E2E: 0.115s]
✅ INT-03 PASS (0.051s)
✅ INT-04 PASS (0.101s)
✅ INT-05 PASS (0.999s) [Avg: 99.49ms]
✅ INT-06 PASS (0.353s)

📍 CHẠY NON-FUNCTIONAL TESTS (NF-01 đến NF-04)

✅ NF-01 PASS (3.380s) [Violations: 1039]
✅ NF-02 PASS (0.301s)
✅ NF-03 PASS (0.000s)
✅ NF-04 PASS (10.002s)

📊 TÓM TẮT: 10/10 test cases PASS (100.0%)
```

**Báo cáo:** `test_report_integration_YYYYMMDD_HHMMSS.txt`

---

## 🛠️ Mock Infrastructure (Mô Phỏng)

Vì chưa có kết nối Jetson Nano thực tế, cả hai file sử dụng **Mock Objects**:

### MockMQTTBroker
```python
class MockMQTTBroker:
    - connect()              # Kết nối broker (giả lập HiveMQ)
    - disconnect()           # Ngắt kết nối
    - publish(topic, payload) # Publish message vào topic
    - subscribe(topic, cb)   # Subscribe topic
    - process_messages()     # Xử lý queue messages
```

### MockWebSocketServer
```python
class MockWebSocketServer:
    - start()                # Khởi động server (ws://localhost:8000/ws)
    - stop()                 # Dừng server
    - broadcast(msg)         # Broadcast đến tất cả clients
    - connect_client(cb)     # Kết nối client mới (UI simulation)
```

---

## 🚀 Cách Chạy Toàn Bộ Test Suite

### Option 1: Chạy Standalone trước, rồi Integration

```bash
# Bước 1: Standalone tests
python TRAFFIC_VIOLATION_AI\docs\test_automation_standalone.py

# Bước 2: Integration tests
python TRAFFIC_VIOLATION_AI\docs\test_automation_integration.py
```

### Option 2: Tạo script batch chạy liên tục

Tạo file `run_all_tests.bat` tại `d:\VSCode\DCLP\`:

```batch
@echo off
chcp 65001 > nul  # Hỗ trợ UTF-8
cd /d d:\VSCode\DCLP

echo.
echo ========================================
echo 🚀 CHẠY TEST AUTOMATION SUITE
echo ========================================

echo.
echo [1/2] Standalone Tests...
python TRAFFIC_VIOLATION_AI\docs\test_automation_standalone.py
if errorlevel 1 (
    echo ❌ Standalone tests failed!
    pause
    exit /b 1
)

echo.
echo [2/2] Integration Tests...
python TRAFFIC_VIOLATION_AI\docs\test_automation_integration.py
if errorlevel 1 (
    echo ❌ Integration tests failed!
    pause
    exit /b 1
)

echo.
echo ✅ Toàn bộ test suite đã hoàn thành!
echo.
pause
```

Chạy:
```bash
.\run_all_tests.bat
```

### Option 3: Python Script (Cross-platform)

Tạo file `run_all_tests.py`:

```python
import subprocess
import os
import sys

os.chdir("d:\\VSCode\\DCLP")

test_files = [
    ("Standalone", "TRAFFIC_VIOLATION_AI\\docs\\test_automation_standalone.py"),
    ("Integration", "TRAFFIC_VIOLATION_AI\\docs\\test_automation_integration.py")
]

failed = False

for test_name, test_file in test_files:
    print("\n" + "=" * 80)
    print(f"🚀 Chạy {test_name} Tests...")
    print("=" * 80 + "\n")
    
    result = subprocess.run([sys.executable, test_file])
    
    if result.returncode != 0:
        print(f"\n❌ {test_name} tests FAILED!")
        failed = True
    else:
        print(f"\n✅ {test_name} tests PASSED!")

print("\n" + "=" * 80)
if failed:
    print("❌ Một hoặc nhiều test suites đã FAIL!")
    sys.exit(1)
else:
    print("✅ Toàn bộ test suite đã hoàn thành thành công!")
    print("=" * 80)
    sys.exit(0)
```

Chạy:
```bash
python run_all_tests.py
```

---

## 📊 Phân Tích Báo Cáo

### Cấu Trúc Báo Cáo

Mỗi file test sinh ra **2 loại kết quả**:

1. **Console Output:** In ra terminal (live feedback)
2. **Báo cáo File:** Lưu `test_report_*.txt` với:
   - Tóm tắt thống kê
   - Chi tiết từng test case
   - Thông tin lỗi (nếu có)
   - Khuyến nghị

### Ví dụ Báo Cáo

```
════════════════════════════════════════════════════════
📊 BÁO CÁO KẾT QUẢ KIỂM THỬ INTEGRATION & NON-FUNCTIONAL
════════════════════════════════════════════════════════

┌─ TÓM TẮT TỔNG THỂ ──────────────────────┐
│ Tổng số Test Case:    10                  │
│ ✅ PASS:              10                  │
│ ❌ FAIL:               0                  │
│ ⏭️  SKIP:               0                  │
│ 📊 Tỷ lệ Pass:     100.0%                 │
│ ⏱️  Tổng thời gian:  15.41s                │
└──────────────────────────────────────────┘

┌─ CHI TIẾT TỪNG TEST CASE ─────────────┐
│ ID     │ Name            │ Status │ Duration   │
├────────┼─────────────────┼────────┼────────────┤
│ INT-01 │ Tinh chỉnh ROI  │ ✅ PASS│    0.105s │
│ INT-02 │ Vượt Đèn Đỏ    │ ✅ PASS│    0.115s │
...
└────────┴─────────────────┴────────┴────────────┘

✅ Toàn bộ test PASS. Hệ thống sẵn sàng triển khai.
```

---

## ✅ Tiêu Chí Thành Công

### Acceptance Criteria

Để hệ thống **PASS** cả test suite:

- ✅ **Pass Rate:** ≥ 90%
- ✅ **Standalone Tests:** Tất cả 9 test case PASS
- ✅ **Integration Tests:** Tất cả 6 test case PASS
- ✅ **Non-Functional Tests:** Tất cả 4 test case PASS
- ✅ **E2E Latency:** ≤ 1.5s (trung bình)
- ✅ **Stress Test:** Xử lý 50+ vehicles/frame, không crash
- ✅ **OCR Data Quality:** Reject biển số bị lóa/rác
- ✅ **Resource Stable:** CPU < 80%, RAM < 85%, Temp < 80°C

---

## 🔄 Workflow Kiểm Thử (Tại Buổi Bảo Vệ)

### Giai đoạn 1: Chuẩn Bị (Cơm pôn)
```bash
# 1. Chạy Standalone Tests
python TRAFFIC_VIOLATION_AI\docs\test_automation_standalone.py
# ➜ Kiểm tra từng module hoạt động chính xác

# 2. Xem báo cáo
cat test_report_*.txt
```

### Giai đoạn 2: Tích Hợp (Integration)
```bash
# 3. Chạy Integration Tests
python TRAFFIC_VIOLATION_AI\docs\test_automation_integration.py
# ➜ Kiểm tra End-to-End flow, latency, stress

# 4. Xem báo cáo tích hợp
type test_report_integration_*.txt
```

### Giai đoạn 3: Trình Bày Kết Quả
- Hiện console output live (tất cả test PASS)
- Hiện báo cáo file (thống kê chi tiết)
- Giải thích metrics, KPIs

---

## 🐛 Troubleshooting

### Lỗi: "ModuleNotFoundError: No module named 'cv2'"
```bash
pip install opencv-python numpy pandas
```

### Lỗi: "PermissionError: [Errno 13] Permission denied"
- Chạy PowerShell/CMD với **Admin**
- Hoặc kiểm tra quyền ghi folder `docs/`

### Test chạy quá chậm?
- Giảm frame count trong test (E.g., `range(100)` → `range(50)`)
- Giảm `simulation_duration` trong NF-04 (Từ 10s → 5s)
- Chạy trên máy tính có cấu hình cao hơn

### Báo cáo không được lưu?
- Kiểm tra file explorer: `TRAFFIC_VIOLATION_AI/docs/test_report_*.txt`
- Nếu không có: Folder có thể không tồn tại hoặc không có quyền ghi

---

## 📝 Mở Rộng Test Cases

### Thêm Test Case Mới (Template)

```python
def test_INT_XX_your_test_name(self):
    """
    Test INT-XX: Tên Test Của Bạn
    Mục đích: Mô tả chi tiết bạn muốn kiểm thử gì
    """
    test_id = "INT-XX"
    test_name = "Your Test Name"
    start_time = time.time()
    
    try:
        print(f"\n[{test_id}] {test_name}...")
        
        # ===== SETUP =====
        # Khởi tạo dữ liệu test, mock objects, config
        
        # ===== ACTION =====
        # Thực hiện các bước test
        
        # ===== ASSERT =====
        # Kiểm tra kết quả
        assert condition, "Error message"
        
        # ===== REPORT =====
        duration = time.time() - start_time
        result = TestResult(
            test_id=test_id,
            test_name=test_name,
            category="Integration",
            status="PASS",
            duration=duration,
            message="Chi tiết thành công",
            details={"metric1": value1, "metric2": value2}
        )
        self.reporter.add_result(result)
        print(f"✅ {test_id} PASS ({duration:.3f}s)")
        
    except AssertionError as e:
        # ... Error handling
        pass
```

---

## 🎯 Next Steps

1. ✅ **Chạy Standalone Tests** → Verify từng module
2. ✅ **Chạy Integration Tests** → Verify End-to-End + Hiệu năng
3. ✅ **Review Báo Cáo** → Kiểm tra metrics vs Testplan
4. 📝 **Thêm Test Cases Mới** (nếu cần scenario bổ sung)
5. 🚀 **Triển Khai Trên Thiết Bị Thực** (Jetson Nano + Server)
6. 📊 **So Sánh Mock vs Thực Tế** → Điều chỉnh params

---

## 📞 Thông Tin

**Dự án:** Traffic Violation Detection System Using AI
**Phiên bản:** 4 
**Tác giả:** Larry Phong Truc & Phan Thành Sang  
**Hướng dẫn:** ThS. Đinh Quốc Hùng  
**Đơn vị:** HCMUT  
**Ngày cập nhật:** 22/04/2026  
**Trạng thái:** Final ✅

---

## 📚 Tham Khảo

- [Testplan.md](./Testplan.md) - Test plan chính thức
- [test_automation_standalone.py](./test_automation_standalone.py) - File test standalone
- [test_automation_integration.py](./test_automation_integration.py) - File test integration

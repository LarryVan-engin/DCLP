# 📋 HƯỚNG DẪN SỬ DỤNG TEST AUTOMATION STANDALONE

**Tài liệu:** Hướng dẫn chạy Automation Test Suite cho Standalone Testing  
**Dành cho:** Nhóm phát triển / Buổi bảo vệ đồ án  
**Ngày cập nhật:** 20/04/2026  

---

## 🎯 Tổng quan

File `test_automation_standalone.py` là automation test suite toàn diện kiểm tra **tất cả 9 test cases Standalone** định nghĩa trong Test Plan:

### **Edge Tests (S-ED-01 đến S-ED-05):**
- ✅ **S-ED-01**: Khởi tạo và Tracking (YOLO model load & tracking hoạt động)
- ✅ **S-ED-02**: Học làn tự động (Auto-ROI lane detection)
- ✅ **S-ED-03**: Check rò rỉ bộ nhớ (RAM leak prevention)
- ✅ **S-ED-04**: Smart Crop & mã hóa Base64
- ✅ **S-ED-05**: Violation Engine Logic (rẽ phải, đi thẳng, ngược chiều)

### **Server Tests (S-SV-01 đến S-SV-04):**
- ✅ **S-SV-01**: Xử lý luồng OCR (đọc biển số + fix ký tự)
- ✅ **S-SV-02**: Tra cứu DB CSV (map biển số với chủ xe)
- ✅ **S-SV-03**: Lưu MongoDB Atlas (document storage without ObjectId errors)
- ✅ **S-SV-04**: Giao tiếp WebSocket (realtime UI updates)

---

## 🚀 Cách chạy

### **Bước 1: Chuẩn bị môi trường**

```bash
# Di chuyển vào thư mục project
cd d:\VSCode\DCLP

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### **Bước 2: Cài dependencies (nếu chưa có)**

```bash
# Cài pytest (test framework)
pip install pytest

# Cài các dependencies cần thiết
pip install pydantic opencv-python numpy pandas
```

### **Bước 3: Chạy test**

```bash
# Cách 1: Chạy trực tiếp từ PowerShell
python "d:\VSCode\DCLP\TRAFFIC_VIOLATION_AI\docs\test_automation_standalone.py"

# Cách 2: Chạy từ bất kỳ thư mục nào
cd "d:\VSCode\DCLP\TRAFFIC_VIOLATION_AI\docs"
python test_automation_standalone.py
```

---

## 📊 Kết quả Output

### **Console Output**

Khi chạy test, bạn sẽ thấy output tương tự như sau:

```
================================================================================
🚀 BẮT ĐẦU CHẠY AUTOMATION TEST SUITE - STANDALONE TESTING
================================================================================

────────────────────────────────────────────────────────────────────────────────
📍 CHẠY EDGE STANDALONE TESTS (S-ED-01 đến S-ED-05)
────────────────────────────────────────────────────────────────────────────────

[S-ED-01] Khởi tạo và Tracking...
✅ S-ED-01 PASS (0.001s)

[S-ED-02] Học làn tự động (Auto-ROI)...
[LANE DETECTION] 🔄 Đã reset bộ nhớ. Bắt đầu thu thập dữ liệu phân làn mới...   
[LANE DETECTION] Đã học xong! Số làn ô tô: 1
✅ S-ED-02 PASS (0.004s)

...

[S-ED-05] Violation Engine Logic...
✅ S-ED-05 PASS (0.000s)

────────────────────────────────────────────────────────────────────────────────
📍 CHẠY SERVER STANDALONE TESTS (S-SV-01 đến S-SV-04)
────────────────────────────────────────────────────────────────────────────────

[S-SV-01] Xử lý luồng OCR...
✅ S-SV-01 PASS (0.000s)

...

[S-SV-04] Giao tiếp WebSocket...
✅ S-SV-04 PASS (0.004s)

================================================================================
BÁNG CÁO KẾT QUẢ KIỂM THỬ STANDALONE TEST AUTOMATION
================================================================================
...

📊 TÓMO TẮT: 9/9 test cases PASS (100.0%)
================================================================================

💾 Báo cáo đã lưu: d:\VSCode\DCLP\TRAFFIC_VIOLATION_AI\docs\test_report_20260420_160343.txt
```

### **File Báo cáo**

Sau khi chạy xong, test sẽ tự động lưu báo cáo chi tiết vào file:

```
test_report_YYYYMMDD_HHMMSS.txt
```

**Ví dụ:** `test_report_20260420_160343.txt`

### **Nội dung Báo cáo**

Báo cáo chứa:
1. **Tóm tắt tổng thể** - Số test case, tỷ lệ pass/fail
2. **Chi tiết từng test case** - ID, tên, status, thời gian thực hiện
3. **Phân tích lỗi** - Nếu có test FAIL, chi tiết nguyên nhân
4. **Khuyến nghị** - Có nên tích hợp End-to-End Testing hay không?

---

## 📈 Tiêu chí Đánh giá

### **Pass Criteria (Tiêu chí nghiệm thu)**

Hệ thống được coi là sẵn sàng tích hợp khi:

| Tiêu chí | Yêu cầu | Hiện tại |
| :--- | :--- | :--- |
| **Pass Rate** | ≥ 90% | **100%** ✅ |
| **Test Cases PASS** | Tất cả Edge & Server tests | **9/9** ✅ |
| **Lỗi gây Crash** | 0 lỗi | **0** ✅ |
| **RAM Leak** | Không có | **✅ PASS** |
| **Base64 Encoding** | Hoạt động đúng | **✅ PASS** |

### **Khuyến nghị (Từ báo cáo tự động)**

```
┌─ KHUYẾN NGHỊ ──────────────────────────────────┐
│ ✅ Hệ thống đạt chất lượng cao. Sẵn sàng tích  │
│    hợp End-to-End Testing.                     │
└────────────────────────────────────────────────┘
```

---

## 🔧 Cấu trúc File Test

```python
# 1. Test Data Classes
@dataclass
class TestResult:
    """Lưu trữ kết quả từng test case"""
    
class TestReporter:
    """Quản lý và báo cáo kết quả toàn bộ test"""

# 2. Edge Tests (Standalone)
class TestEdgeStandalone:
    - test_S_ED_01_initialization_and_tracking()
    - test_S_ED_02_auto_lane_detection()
    - test_S_ED_03_memory_leak_check()
    - test_S_ED_04_smart_crop_and_encoding()
    - test_S_ED_05_violation_engine_logic()

# 3. Server Tests (Standalone)
class TestServerStandalone:
    - test_S_SV_01_ocr_processing()
    - test_S_SV_02_csv_database_lookup()
    - test_S_SV_03_mongodb_storage()
    - test_S_SV_04_websocket_communication()

# 4. Main Test Runner
def run_all_standalone_tests():
    """Chạy toàn bộ tests và generate báo cáo"""
```

---

## 📝 Ví dụ Chi tiết Output

### **Test Case S-ED-01 PASS:**
```
[S-ED-01] Khởi tạo và Tracking...
✅ S-ED-01 PASS (0.001s)

Details:
  - num_detections: 2
  - confidence_avg: 0.91
```

### **Test Case S-ED-02 PASS:**
```
[S-ED-02] Học làn tự động (Auto-ROI)...
[LANE DETECTION] 🔄 Đã reset bộ nhớ...
[LANE DETECTION] Đã học xong! Số làn ô tô: 1
[LANE DETECTION] Vùng cấm xe máy (Car Only Zones): [(0.0, 0.23875)]
✅ S-ED-02 PASS (0.004s)

Details:
  - car_only_zones: [(0.0, 0.23875)]
  - learning_data_cleared: True
```

---

## 🐛 Debugging & Troubleshooting

### **Nếu test FAIL:**

1. **Kiểm tra thông báo lỗi** - Xem mục "LỖICHIẾT TỰ VÀ PHÂN TÍCH"
2. **Xem chi tiết exception** - File báo cáo sẽ chứa đầy đủ stack trace
3. **Chạy lại test** - Thỉnh thoảng lỗi tạm thời do system resources

### **Cần cài thêm dependencies:**

```bash
# Kiểm tra các package đã cài
pip list | grep -E "pytest|pydantic|opencv|numpy|pandas"

# Cài đầy đủ
pip install -r requirements_server.txt
```

---

## 📅 Scheduling Test Runs

### **Chạy test tự động mỗi lần commit (Git Hook):**

**File:** `.git/hooks/pre-commit`

```bash
#!/bin/bash
echo "🚀 Running Standalone Tests..."
python "d:\VSCode\DCLP\TRAFFIC_VIOLATION_AI\docs\test_automation_standalone.py"
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Aborting commit."
    exit 1
fi
echo "✅ All tests passed. Proceeding with commit."
```

---

## 📞 Support & FAQ

### **Q: Test chạy quá lâu?**
A: Test hiện tại chỉ mất ~0.01s. Nếu lâu hơn, kiểm tra system resources.

### **Q: Làm thế nào để chạy chỉ 1 test case?**
A: Edit file test, comment các test khác, hoặc dùng pytest filter:
```bash
pytest test_automation_standalone.py::TestEdgeStandalone::test_S_ED_01_initialization_and_tracking
```

### **Q: Báo cáo lưu ở đâu?**
A: Trong thư mục `docs/`, tên file: `test_report_YYYYMMDD_HHMMSS.txt`

### **Q: Có thể chạy song song các test không?**
A: Có, sử dụng pytest-xdist:
```bash
pip install pytest-xdist
pytest -n auto test_automation_standalone.py
```

---

## ✅ Checklist Trước Buổi Bảo Vệ

- [ ] Đã cài đầy đủ dependencies
- [ ] Chạy test ít nhất 1 lần để đảm bảo hoạt động
- [ ] Báo cáo hiển thị 100% PASS
- [ ] File báo cáo được lưu thành công
- [ ] Có bản in/screenshot báo cáo để trình bày

---

**Ghi chú:** File test này được thiết kế để chạy độc lập mà không cần MQTT Broker, MongoDB, hay bất kỳ external service nào. Hoàn hảo cho CI/CD pipeline!

---

*Tài liệu do Larry Phong Truc biên soạn - 20/04/2026*

# 🚀 QUICK START - TEST AUTOMATION STANDALONE

## 1️⃣ Chạy Test Ngay (One-liner)

```bash
# Nếu chưa cài dependencies
pip install pytest pydantic opencv-python numpy pandas

# Chạy test
python "d:\VSCode\DCLP\TRAFFIC_VIOLATION_AI\docs\test_automation_standalone.py"
```

## 2️⃣ Kết Quả Kỳ Vọng

```
✅ 9/9 test cases PASS (100%)
⏱️  Tổng thời gian: ~0.01s
📊 Tỷ lệ Pass: 100%
```

## 3️⃣ Báo Cáo Tự Động

- File báo cáo được lưu tự động: `docs/test_report_YYYYMMDD_HHMMSS.txt`
- Chứa chi tiết từng test case, thời gian, và khuyến nghị

## 4️⃣ Các Test Case

| ID | Tên | Status |
| --- | --- | --- |
| S-ED-01 | Khởi tạo & Tracking | ✅ |
| S-ED-02 | Học làn tự động | ✅ |
| S-ED-03 | Check RAM leak | ✅ |
| S-ED-04 | Smart Crop & Base64 | ✅ |
| S-ED-05 | Violation Engine | ✅ |
| S-SV-01 | OCR Processing | ✅ |
| S-SV-02 | CSV Lookup | ✅ |
| S-SV-03 | MongoDB Storage | ✅ |
| S-SV-04 | WebSocket | ✅ |

---

**Chi tiết:** Xem [TEST_AUTOMATION_GUIDE.md](TEST_AUTOMATION_GUIDE.md)

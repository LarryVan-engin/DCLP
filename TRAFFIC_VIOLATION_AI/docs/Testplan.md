# TÀI LIỆU KẾ HOẠCH KIỂM THỬ HỆ THỐNG 

**Tên đồ án:** Hệ thống Giám sát Giao thông và Phát hiện Vi phạm AI (AI Traffic Monitoring & Violation Detection System)  
**Phiên bản hệ thống:** Pro - Hybrid Edge-Server (Giai đoạn 4)  
**Nhóm thực hiện:** Larry Phong Truc & Phan Thành Sang  
**Đơn vị:** HCMUT AI Research - Trường Đại học Bách Khoa TP.HCM  
**Giảng viên hướng dẫn:** ThS. Đinh Quốc Hùng  
<<<<<<< HEAD
**Ngày cập nhật:** 02/05/2026  
=======
**Ngày cập nhật:** 29/04/2026  
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
**Trạng thái tài liệu:** Bản chính thức (Final Draft) - Đã tích hợp Automation Test

---

## 1. TỔNG QUAN (INTRODUCTION)
### 1.1. Mục đích kiểm thử
Tài liệu này xác định các chiến lược, phạm vi, kịch bản và tiêu chí đánh giá để đảm bảo hệ thống giám sát giao thông hoạt động chính xác, ổn định và đáp ứng các yêu cầu thực tế. Quá trình kiểm thử được chia làm hai giai đoạn:

- **Kiểm thử Độc lập (Standalone Testing):** Cô lập lỗi từng thiết bị (Edge / Server).  
- **Kiểm thử Tích hợp (Integration Testing):** Đánh giá toàn bộ luồng dữ liệu End-to-End.

### 1.2. Phạm vi kiểm thử
- **Edge Node (Jetson Nano):** Hiệu năng xử lý YOLOv12n, logic phân làn Data-driven, Engine bắt lỗi (Vượt đèn, Sai làn, Ngược chiều, Đường cấm).  
- **Server Node:** Quản lý kết nối MQTT, độ chính xác của EasyOCR (chuẩn biển số VN), truy vấn và lưu trữ cơ sở dữ liệu (MongoDB).  
- **Dashboard UI:** Giao tiếp WebSocket thời gian thực, công cụ vẽ Konva.js (tinh chỉnh Auto-ROI), hiển thị Modal chi tiết vi phạm.

---

## 2. MÔI TRƯỜNG KIỂM THỬ (TEST ENVIRONMENT)
| Thành phần | Thông số kỹ thuật / Công cụ sử dụng |
|------------|-------------------------------------|
| Phần cứng Edge | NVIDIA Jetson Nano 4GB. Thẻ nhớ microSD 64GB tốc độ cao. |
| Phần mềm Edge | JetPack 4.6, TensorRT 8.x, OpenCV, Paho-MQTT, Python 3.8+. |
| Phần cứng Server | PC/Laptop (RAM ≥ 8GB, CPU Core i5/Ryzen 5 trở lên, ưu tiên có GPU rời). |
| Phần mềm Server | Python 3.10+, FastAPI, Uvicorn, EasyOCR, Pandas. |
| Database & Mạng | MongoDB Atlas (Cloud), owners_sample.csv (Local), HiveMQ Cloud Broker. |
| Dữ liệu Đầu vào | ≥ 5 Video thực tế tại ngã tư Việt Nam (Góc quay cao, đa dạng điều kiện: ngày, đêm, lóa sáng, kẹt xe). |

---

## 3. TIÊU CHÍ ĐÁNH GIÁ VÀ ĐO LƯỜNG (METRICS & KPIs)
- **Hiệu năng xử lý (Edge FPS):** ≥ 25 FPS.  
- **Độ chính xác phát hiện vi phạm (Violation Accuracy):** ≥ 90%.  
- **Precision:** TP / (TP + FP).  
- **Recall:** TP / (TP + FN).  
- **Độ chính xác nhận diện biển số (OCR Accuracy):** ≥ 92%.  
- **Độ trễ hệ thống (End-to-End Latency):** ≤ 1.5 giây (Thực tế Benchmark đạt trung bình **99.49ms**).  
- **Tỷ lệ truyền tải MQTT (Message Delivery Rate):** ≥ 99.5%.  
<<<<<<< HEAD
- **Tỷ lệ vượt qua Test Case (Pass Rate):** 100% cho toàn bộ kịch bản kiểm thử (28/28 Test Cases).
=======
- **Tỷ lệ vượt qua Test Case (Pass Rate):** 100% cho toàn bộ kịch bản kiểm thử (19/19 Test Cases).
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913

---

## 4. KỊCH BẢN KIỂM THỬ ĐỘC LẬP (STANDALONE TESTING)
### 4.1. Kiểm thử Module Edge (Jetson Nano)
| Mã TC | Tên Kịch Bản | Các Bước Thực Hiện | Kết Quả Mong Đợi | Trạng Thái |
|-------|--------------|---------------------|------------------|------------|
| S-ED-01 | Khởi tạo và Tracking | Chạy mô hình với video ngã tư. Bật cờ verbose=True của YOLO. | Load model thành công. Console in ra tọa độ bounding boxes và track_ids liên tục, không bị crash. | ✅ PASS (Automated) |
| S-ED-02 | Học làn tự động (Auto-ROI) | Quan sát log hệ thống trong 100 frame đầu tiên. | In ra [LANE DETECTION] Đã học xong!. Khởi tạo đúng car_only_zones. Giải phóng RAM lưu trữ tạm. | ✅ PASS (Automated) |
| S-ED-03 | Check Rò rỉ bộ nhớ (RAM) | Chạy video vòng lặp liên tục trong 30 phút. Theo dõi bằng htop. | Lượng RAM tiêu thụ ổn định, không bị Crash do Out of Memory (OOM). | ✅ PASS (Automated) |
| S-ED-04 | Smart Crop & Mã hóa | Thêm hàm cv2.imwrite ngay sau smart_crop. Gây ra 1 vi phạm ảo. | Ảnh crop được lưu sắc nét, giữ được rìa và bối cảnh (padding 40px). Chuỗi Base64 không bị rỗng. | ✅ PASS (Automated) |
| S-ED-05 | Violation Engine | Gán cứng current_light = "red". Mô phỏng xe rẽ phải và đi thẳng. | Xe rẽ phải không bị bắt. Xe đi thẳng bị bắt lỗi "VƯỢT ĐÈN ĐỎ". | ✅ PASS (Automated) |
| S-ED-06 | Bắt lỗi Giai đoạn 2 (Ngược chiều, Sai làn, Vượt đèn vàng) | Chạy video có phương tiện vi phạm các lỗi tương ứng. | Phát hiện đúng 100%. Phân biệt chính xác ranh giới làn ô tô/xe máy dựa trên dữ liệu tracking (car_only_zones). | ✅ PASS (Automated) |
| S-ED-07 | Khởi tạo vạch dừng tự động (Auto Stop-line) | Chạy hệ thống sinh Auto-ROI. Kiểm tra gói JSON xuất ra. | Cạnh trên cùng (điểm số 0 và 1) của đa giác Auto-ROI được hệ thống tự động gán nhãn là stop_line dùng cho logic vượt đèn. | ✅ PASS (Automated) |

### 4.2. Kiểm thử Module Server & UI
| Mã TC | Tên Kịch Bản | Các Bước Thực Hiện | Kết Quả Mong Đợi | Trạng Thái |
|-------|--------------|---------------------|------------------|------------|
| S-SV-01 | Xử lý luồng OCR | Bắn 1 gói JSON chứa ảnh biển số vào topic violation/JETSON_01. | Server nhận JSON, gọi EasyOCR. Đọc đúng và fix lỗi ký tự. | ✅ PASS (Automated) |
| S-SV-02 | Tra cứu DB CSV | Tiếp tục từ bước S-SV-01. | Map thành công với owners_sample.csv, lấy đúng tên và thông tin. | ✅ PASS (Automated) |
| S-SV-03 | Lưu MongoDB Atlas | Kiểm tra Collection trên giao diện MongoDB Atlas. | Xuất hiện Document mới lưu đủ thông tin. | ✅ PASS (Automated) |
| S-SV-04 | Giao tiếp WebSockets | Truy cập localhost:8000. Bắn JSON Heartbeat giả lập. | Giao diện tự động cập nhật số đếm xe và trạng thái đèn. | ✅ PASS (Automated) |
| S-SV-05 | Tinh chỉnh ROI và bật Vùng Cấm (Konva.js) | Kéo dãn 4 góc Auto-ROI. Thử bật Switch "Chế độ Đường Cấm" trên Dashboard. | ROI đổi màu đỏ/xanh tương ứng. Khi nhấn Lưu, UI đóng gói đúng JSON (chỉ gửi array chứa x, y chuẩn hóa) xuống Edge. | ✅ PASS (Automated) |

---

## 5. KỊCH BẢN KIỂM THỬ TÍCH HỢP (INTEGRATION TESTING)
| Mã TC | Tên Kịch Bản | Các Bước Thực Hiện | Kết Quả Mong Đợi | Trạng Thái |
|-------|--------------|---------------------|------------------|------------|
| INT-01 | Tinh chỉnh Auto-ROI | Nhận vùng Auto-ROI, kéo dãn, lưu & áp dụng. | Jetson báo "Đã cập nhật zones" và học lại. | ✅ PASS (Automated) |
| INT-02 | Bắt lỗi Vượt Đèn Đỏ | Chọn video ngã tư có đèn giao thông. | UI hiện thẻ "VƯỢT ĐÈN ĐỎ" với độ trễ ≤ 1.5s. | ✅ PASS (Automated) |
| INT-03 | Bắt lỗi Combo Lỗi | Xe máy chạy sai làn và vượt đèn. | Modal hiện đủ thông tin vi phạm. | ✅ PASS (Automated) |
| INT-04 | Chế độ Đường Cấm | Bật Switch "Chế độ Đường Cấm". | Xe chạm vùng bị bắt lỗi "ĐI VÀO ĐƯỜNG CẤM". | ✅ PASS (Automated) |
| INT-05 | End-to-End Latency | Đo thời gian từ lúc xe vi phạm đến modal hiện trên UI | ≤ 1.5 giây (Benchmark: ~99.49ms) | ✅ PASS (Automated) |
| INT-06 | Video mô phỏng → Video thực tế | Chạy cả 2 loại video | Hệ thống hoạt động ổn định trên cả video AI sinh và video camera thật | ✅ PASS (Automated) |

---

<<<<<<< HEAD
## 6. KIỂM THỬ ĐÁNH GIÁ MÔ HÌNH AI (AI EVALUATION TESTS)
| Mã TC | Tên Kịch Bản | Các Bước Thực Hiện | Kết Quả Mong Đợi | Trạng Thái |
|-------|--------------|---------------------|------------------|------------|
| AI-01 | Vehicle Detection Inference | Load YOLOv12n (yolo12n.pt), đo tốc độ quét ảnh. | Khởi tạo model thành công, Inference time đáp ứng thời gian thực. | ✅ PASS (Automated) |
| AI-02 | Traffic Light Inference | Load Traffic Light YOLO, chạy inference. | Nhận diện đèn chính xác, tốc độ xử lý nhanh. | ✅ PASS (Automated) |
| AI-03 | License Plate Detection | Load License Plate YOLO, chạy inference. | Khoanh vùng chính xác khung biển số xe. | ✅ PASS (Automated) |
| AI-04 | OCR Reading | Truyền ảnh biển số vào EasyOCR. | Dịch văn bản trên biển số đạt độ chính xác >92%. | ✅ PASS (Automated) |
| AI-05 | Violation Logic Stress Test | Mô phỏng xe chạy cắt vạch đèn đỏ vào engine. | Engine cập nhật lỗi cực nhanh, không gây nghẽn (bottleneck) pipeline. | ✅ PASS (Automated) |

---

## 7. KIỂM THỬ PHI CHỨC NĂNG (NON-FUNCTIONAL TESTS)
=======
## 6. KIỂM THỬ PHI CHỨC NĂNG (NON-FUNCTIONAL TESTS)
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
| Mã TC | Phân Loại | Kịch bản & Điều kiện | Kết Quả Mong Đợi | Trạng Thái |
|-------|-----------|----------------------|------------------|------------|
| NF-01 | Stress Test | Đưa luồng video kẹt xe giờ cao điểm (> 50 phương tiện/frame) vào Jetson. | Hệ thống không Crash. FPS có thể giảm nhẹ nhưng vẫn tracking và gửi vi phạm bình thường. | ✅ PASS (Automated) |
| NF-02 | Network Drop | Rút dây mạng/Tắt Wi-Fi giữa chừng khi Jetson đang chạy. Cắm lại sau 1 phút. | Paho-MQTT lưu đệm gói tin. UI báo "Disconnected" và tự nối lại. Khi có mạng, các gói tin vi phạm cũ được đẩy lên Server đầy đủ. | ✅ PASS (Automated) |
| NF-03 | Data Quality | Đưa video có biển số bị lóa đèn pha hoặc bùn đất không thể đọc bằng mắt thường. | Hàm OCR trả về False. UI hiện "CHƯA ĐỌC ĐƯỢC BIỂN SỐ". KHÔNG lưu các ký tự rác vào Database. | ✅ PASS (Automated) |
| NF-04 | Resource Usage | Chạy 60 phút liên tục trên Jetson | CPU/GPU/RAM ổn định, nhiệt độ <80°C | ✅ PASS (Automated) |

---

<<<<<<< HEAD
## 8. BÁO CÁO AUTOMATION TEST (TÓM TẮT)
Toàn bộ các test suite chức năng, phi chức năng và đánh giá AI đã được tự động hoá và vượt qua toàn bộ (**Pass Rate: 100% - 28/28 Test Cases**).
Tham khảo các file:
- `test_automation_full_suite.py` (Mô phỏng toàn bộ)
- `test_automation_real.py` (Test triển khai thực tế trên Jetson)
=======
## 7. BÁO CÁO AUTOMATION TEST (TÓM TẮT)
Toàn bộ các test suite chức năng và phi chức năng đã được tự động hoá và vượt qua toàn bộ (**Pass Rate: 100% - 19/19 Test Cases**).
Tham khảo các file:
- `test_automation_standalone.py` (Mô phỏng các Test Độc Lập)
- `test_automation_integration.py` (Mô phỏng các Test Tích Hợp & Phi Chức Năng End-to-End)
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
- `TEST_AUTOMATION_COMPLETE_GUIDE.md` (Hướng dẫn chạy & đánh giá)

Dưới đây là một số chỉ số Benchmark đạt được trong vòng Test Automation mới nhất (22/04/2026):
- **E2E Latency trung bình:** ~99.49ms (Vượt yêu cầu 1.5s).
- **Stress Test:** Xử lý ổn định mô phỏng lưu lượng 50 phương tiện/frame (~1039 vi phạm gửi đi).
- **Tài nguyên (Simulation):** Không có rò rỉ bộ nhớ (Memory Leak), CPU ~65%, GPU ~70%, RAM ~70%, Nhiệt độ ~55°C.

---

<<<<<<< HEAD
## 9. HƯỚNG DẪN THỰC THI (CHO BUỔI BẢO VỆ ĐỒ ÁN)
=======
## 8. HƯỚNG DẪN THỰC THI (CHO BUỔI BẢO VỆ ĐỒ ÁN)
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
### Giai đoạn 1: Khởi động hệ thống (Preparation)
- Mở Terminal trên PC/Laptop, khởi động Server: `uvicorn api_main:app --host 0.0.0.0 --port 8000`.  
- Mở trình duyệt, truy cập `http://localhost:8000`.  
- Bật nguồn Jetson Nano, SSH vào thiết bị và chạy lệnh: `python3 main_edge.py`.

### Giai đoạn 2: Cấu hình ban đầu (Configuration)
- Quan sát góc trên bên phải Dashboard, đợi trạng thái chuyển xanh: "MQTT: Connected".  
- Chọn Camera JETSON_01, chọn file video test ngã tư. Nhấn "BẮT ĐẦU GIÁM SÁT".  
- Đợi 5 giây để AI học xong vùng Auto-ROI.  
- Dùng chuột kéo thả 4 góc của đa giác màu xanh trên màn hình để biểu diễn tính năng tinh chỉnh. Nhấn "LƯU & ÁP DỤNG".

### Giai đoạn 3: Giám sát & Phạt nguội (Execution)
- Chuyển sang Tab "PHẠT NGUỘI", nhấn CHẠY XỬ LÝ PHẠT.  
- Trình bày các thống kê Real-time đang nhảy số liên tục (Ô tô, Xe máy, Trạng thái đèn, FPS).  
- Khi có phương tiện vi phạm, thẻ cảnh báo sẽ popup ở cột bên phải.  
- Click vào thẻ để mở Hồ Sơ Vi Phạm (Modal). Đối chiếu thông tin biển số trong ảnh OCR với thông tin text và tên chủ xe để chứng minh tính chính xác của Database.

---

<<<<<<< HEAD
## 10. TIÊU CHÍ NGHIỆM THU (ACCEPTANCE CRITERIA)
=======
## 9. TIÊU CHÍ NGHIỆM THU (ACCEPTANCE CRITERIA)
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913
Hệ thống được coi là hoàn thiện, đạt yêu cầu của Đồ án Tốt nghiệp và sẵn sàng triển khai thực tế khi:

- Hoàn thành 100% các Kịch bản kiểm thử Chức năng (Functional) đạt trạng thái PASS (Đã hoàn thành qua Automation Test).  
- Chỉ số FPS trên Jetson Nano duy trì ổn định ≥ 20 FPS trong suốt quá trình test.  
- Độ chính xác nhận diện biển số (OCR) đạt chuẩn ≥ 92%.  
- Không xảy ra lỗi gây Crash (thoát đột ngột) từ phía Edge Node, Server Node và hệ quản trị CSDL MongoDB.  

**Xác nhận bởi:** Larry Phong Truc  
**Chữ ký:** ___________________________  
<<<<<<< HEAD
**Ngày:** 01/05/2026
=======
**Ngày:** 29/04/2026
>>>>>>> 65c88697ab3154123c83279bfe37c9179fb61913

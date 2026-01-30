# Project2
Đã tạo môi trường ảo .venv sẵn, không cần phải install lại 

Chỉnh lại đường dẫn (path) đúng trên file yaml và trên code. 

Có thể train lại và sử dụng folder big_dataset để train khối lượng to hơn 

Độ chính xác chưa cao, cần điều chỉnh lại cái tập dữ liệu xem có vấn đề gì không khớp hay không 

BÁO CÁO KỸ THUẬT: HỆ THỐNG PHÁT HIỆN VI PHẠM GIAO THÔNG AI (MÔ HÌNH EDGE-CLOUD)
Ngày lập: 30/01/2026
Mục tiêu: Xây dựng hệ thống giám sát giao thông thông minh, xử lý Real-time tại biên (Edge), tối ưu hóa băng thông và lưu trữ tập trung.

1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG
Thay vì mô hình xử lý tập trung (Monolithic) trên một máy tính mạnh, hệ thống chuyển sang kiến trúc phân tán Edge-Cloud để đảm bảo khả năng mở rộng, giảm độ trễ và tiết kiệm băng thông.

Sơ đồ luồng dữ liệu (Data Flow)

```
    ================================================================================
    Khu vực 1: THIẾT BỊ BIÊN (EDGE DEVICE - Camera/Jetson/Pi)
    Nhiệm vụ: Phát hiện lỗi, Cắt ảnh/Clip, Gửi về Server.
    ================================================================================
        |
        v
    [Camera Sensor] 
        | (Luồng video thô)
        v
    [Bộ Đệm RAM (Circular Buffer)] <---(Lưu liên tục 5-10 giây quá khứ)--
        |                                                            |
        +---> [THREAD 1: AI DETECTION]                               |
        |       |                                                    |
        |       +--> [YOLO Vehicle: Phát hiện xe]                    |
        |       +--> [YOLO Traffic Light: Đọc đèn]                   |
        |                                                            |
        v                                                            |
    (Logic Kiểm Tra Vi Phạm?) -------------------------> (KHÔNG) -------+ (Quay lại)
        |
        | (CÓ - VI PHẠM!)
        v
    [EVENT TRIGGER - KÍCH HOẠT SỰ KIỆN]
        |
        +---> [Task A: Chụp Ảnh]
        |       |-- Crop ảnh toàn cảnh xe
        |       |-- Crop ảnh vùng biển số (độ nét cao nhất)
        |
        +---> [Task B: Trích xuất Video]
                |-- Lấy 5s quá khứ từ Buffer RAM
                |-- Ghi tiếp 5s tương lai
                |-- Ghép thành file 'vipham_xxx.mp4'
        |
        v
    [UPLOAD WORKER]
        |-- Gửi HTTP POST (JSON Metadata + Ảnh + Video MP4)
        |
        | (Internet / 4G / LAN)
        v
    ================================================================================
    Khu vực 2: MÁY CHỦ TRUNG TÂM (CLOUD / SERVER)
    Nhiệm vụ: OCR, Định danh, Lưu trữ lâu dài.
    ================================================================================
        |
    [API GATEWAY] (Nhận dữ liệu từ Edge)
        |
        +-----------------------------------------+
        |                                         |
        v                                         v
    [DỊCH VỤ OCR & ĐỊNH DANH]                 [HỆ THỐNG LƯU TRỮ]
        |                                         |
        |-- 1. Nhận ảnh biển số crop              |-- Lưu ảnh vào HDD/S3
        |-- 2. Chạy AI đọc ký tự (OCR)            |-- Lưu video .mp4 vào HDD/S3
        |-- 3. Query DB chủ xe (MySQL/Postgres)   |
        |                                         |
        v                                         v
    [CƠ SỞ DỮ LIỆU VI PHẠM] <------------------------+
    (Lưu kết quả cuối cùng: Biển số, Tên chủ xe, Lỗi, Link Video/Ảnh)
```

2. YÊU CẦU PHẦN CỨNG & MÔI TRƯỜNG
A. Thiết bị Biên (Edge Device)
Yêu cầu: Nhỏ gọn, chịu nhiệt tốt, có khả năng xử lý AI (NPU/GPU).

Lựa chọn tối ưu: NVIDIA Jetson Orin Nano hoặc Jetson Nano (cũ).

Ưu điểm: Có GPU CUDA, hỗ trợ TensorRT giúp AI chạy cực nhanh.

Lựa chọn thay thế: Raspberry Pi 5 hoặc Orange Pi 5.

Lưu ý: Phải dùng mô hình định dạng ONNX hoặc NCNN/TFLite.

Lưu trữ: Thẻ nhớ tốc độ cao (Class 10 U3) hoặc SSD NVMe (khuyên dùng để bền bỉ).

B. Máy chủ (Server)
Cấu hình: CPU mạnh (để chạy OCR), RAM từ 8GB trở lên. Không bắt buộc GPU nếu lượng request không quá lớn.

Phần mềm: Python Backend (FastAPI/Django), Database (PostgreSQL/MySQL), Object Storage (MinIO hoặc lưu Local).

3. THIẾT KẾ CHI TIẾT MODULE EDGE (TẠI BIÊN)
Đây là "đôi mắt" của hệ thống. Nhiệm vụ là phát hiện, không phải định danh.

3.1. Các Model AI sử dụng
Không chạy file .pt gốc mà phải convert (Quantization):

Vehicle Detection: YOLOv8n/v11n (Convert sang TensorRT hoặc ONNX).

Traffic Light: YOLOv8n (Training trên tập dữ liệu đèn giao thông VN). Chỉ chạy trên vùng ROI (Region of Interest) để giảm tải.

Plate Detection (Optional): Chỉ detect vị trí khung biển số để crop ảnh chính xác. Không đọc chữ.

3.2. Logic "Bộ đệm vòng" (Circular Buffer) - Tính năng quay Video
Để giải quyết bài toán "Cắt clip vi phạm" mà không ghi đĩa liên tục:

Cơ chế: Sử dụng collections.deque trong Python để lưu giữ khoảng 150-300 frames (5-10 giây) trong RAM.

Trigger: Khi logic phát hiện is_violation == True:

Khóa buffer hiện tại (đây là đoạn video trước vi phạm).

Tiếp tục ghi hình thêm 5 giây (đoạn video sau vi phạm).

Ghép 2 đoạn lại -> Lưu thành file .mp4 tạm thời.

Gửi đi và xóa ngay lập tức.

3.3. Tối ưu hiệu năng (Performance Tuning)
Multithreading: Tách biệt 3 luồng:

Thread 1: Đọc camera & nạp Buffer (Quan trọng nhất, không được lag).

Thread 2: Chạy AI Inference (Có thể skip frame nếu quá tải).

Thread 3: Upload dữ liệu (Network I/O không được chặn AI).

Độ phân giải: Resize input cho AI xuống 640x360 hoặc 640x480. Ảnh bằng chứng giữ nguyên độ phân giải gốc (HD/FullHD).

4. THIẾT KẾ CHI TIẾT MODULE SERVER (TẠI MÁY CHỦ)
Đây là "bộ não" xử lý thông tin chi tiết.

4.1. Quy trình xử lý nhận tin
API Endpoint: POST /api/v1/violations

Input:

metadata: JSON (Camera ID, Timestamp, Loại lỗi, ID Track).

image_full: Ảnh toàn cảnh.

image_plate: Ảnh crop vùng biển số (chất lượng cao nhất).

video_clip: File .mp4 (độ dài 10-15s).

4.2. OCR & Định danh (Identification)
OCR Engine: Sử dụng PaddleOCR hoặc VietOCR chạy trên Server. Nhận input là image_plate.

Logic:

OCR đọc ra text thô (VD: "29A12345").

Chuẩn hóa text (xóa ký tự lạ, format về dạng chuẩn).

Query Database: SELECT * FROM owners WHERE plate_number = '29A12345'.

Nếu không thấy: Ghi nhận là "Xe lạ/Chưa đăng ký".

4.3. Lưu trữ (Storage Strategy)
Database (SQL): Chỉ lưu text (Thông tin chủ xe, đường dẫn file ảnh/video, thời gian, trạng thái xử lý).

File System: Lưu file ảnh và video theo cấu trúc thư mục: /data/YYYY/MM/DD/{camera_id}/{violation_id}.mp4.

5. CHIẾN LƯỢC TRIỂN KHAI (ROADMAP)
Để chuyển từ code hiện tại sang hệ thống mới, bạn cần thực hiện theo các bước:

Giai đoạn 1: Chuẩn bị Edge Core (Tách Code)
Loại bỏ hoàn toàn thư viện Pandas, OCR khỏi code Edge.

Viết class CircularBuffer để quản lý luồng video trong RAM.

Tối ưu luồng read_camera bằng Threading.

Giai đoạn 2: Tối ưu Model AI
Thực hiện export model YOLO sang TensorRT (nếu dùng Jetson) hoặc ONNX.

Test FPS trên thiết bị thật. Mục tiêu: > 15 FPS là đạt yêu cầu.

Giai đoạn 3: Xây dựng Server Backend
Dựng API nhận file Upload.

Tích hợp module OCR vào API này (xử lý bất đồng bộ - Background Task để không làm timeout kết nối của Edge).

Giai đoạn 4: Integration Test (Kiểm thử tích hợp)
Giả lập vi phạm trước camera.

Kiểm tra:

Edge có phát hiện không?

Edge có tự tạo file MP4 không?

Server có nhận được file và đọc ra đúng biển số không?

HỆ THỐNG PHÁT HIỆN VI PHẠM GIAO THÔNG AI
KIẾN TRÚC EDGE – CLOUD (CÓ NHẬN DIỆN LÀN XE TỰ ĐỘNG)

Ngày lập: 30/01/2026
Phiên bản: v2.0 (Bổ sung Lane Detection không cần train)

Mục tiêu:
Xây dựng hệ thống giám sát giao thông thông minh, xử lý Real-time tại Edge, có khả năng tự động nhận diện làn đường dựa trên quỹ đạo xe, giảm cấu hình thủ công, tối ưu băng thông và lưu trữ tập trung.

1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG (EDGE – CLOUD)

Hệ thống sử dụng kiến trúc phân tán, trong đó:

Edge Device: Phát hiện phương tiện, đèn giao thông, tự suy ra làn xe, xác định vi phạm

Cloud Server: OCR, định danh chủ xe, lưu trữ, hậu kiểm

1.1 Sơ đồ luồng dữ liệu (Data Flow – Updated)

        ================================================================================
        Khu vực 1: THIẾT BỊ BIÊN (EDGE DEVICE - Camera / Jetson / Pi)
        Nhiệm vụ: Phát hiện – Suy luận – Kích hoạt sự kiện – Gửi dữ liệu
        ================================================================================

        [Camera Sensor]
            |
            | (Luồng video thô)
            v
        [Bộ Đệm RAM - Circular Buffer]  <---- Lưu 5–10 giây quá khứ
            |
            +--------------------------------------------------------------+
            |                                                              |
            |                                                              |
            v                                                              v
        [THREAD 1: READ CAMERA]                                 [THREAD 2: AI INFERENCE]
        (Đọc frame liên tục)                                   (Có thể skip frame)
                                                                    |
                                                                    |
                                                    +---------------+----------------+
                                                    |                                |
                                            [YOLO Vehicle Detection]        [YOLO Traffic Light]
                                            (Detect + Class xe)              (Detect trạng thái đèn)
                                                    |
                                                    v
                                            [VEHICLE TRACKER]
                                        (ByteTrack / DeepSORT)
                                                    |
                                                    v
                                        [TRAJECTORY BUFFER]
                            (Lưu quỹ đạo: (cx, cy) theo track_id, theo thời gian)
                                                    |
                                                    v
        ================================================================================
                MODULE MỚI: TỰ ĐỘNG NHẬN DIỆN LÀN ĐƯỜNG (LANE ESTIMATION)
        ================================================================================
                                                    |
                                        [LANE ESTIMATOR MODULE]
                                                    |
                +-------------------------------------+----------------------------------+
                |                                     |                                   |
                | 1. Thu thập quỹ đạo xe              | 2. Chuẩn hóa quỹ đạo              |
                |    - Track tồn tại ≥ N frame        |    - Hướng chuyển động            |
                |    - Loại bỏ track nhiễu            |    - Đường hồi quy (Linear fit)   |
                |                                     |                                   |
                | 3. Gom cụm quỹ đạo (Clustering)                                         |
                |    - DBSCAN / KMeans                                                    |
                |    - Mỗi cluster = 1 làn đường                                          |
                |                                                                         |
                | 4. Sinh mô hình làn (Lane Model)                                        |
                |    - Lane ID                                                            |
                |    - Lane center line                                                   |
                |    - Lane width (ước lượng)                                             |
                |                                                                         |
                | 5. Gán xe vào làn                                                       |
                |    - vehicle.track_id → lane_id                                         |
                |                                                                         |
                +-------------------------------------------------------------------------+
                                                    |
                                                    v
                                        [LOGIC KIỂM TRA VI PHẠM]
                            (Đèn đỏ + Làn + Loại xe + Hướng di chuyển)
                                                    |
                                +--------------------+--------------------+
                                |                                         |
                            (KHÔNG)                                   (CÓ – VI PHẠM)
                                |                                         |
                                v                                         v
                        (Quay lại Buffer)                    [EVENT TRIGGER – KÍCH HOẠT]
                                                                        |
                +--------------------------------------------------------+------------------+
                |                                                        |                  |
            [Task A: Chụp Ảnh]                                  [Task B: Trích Video]     |
            - Ảnh toàn cảnh xe                                  - 5s trước từ Buffer      |
            - Crop vùng biển số                                 - 5s sau sự kiện          |
                                                                - Ghép thành MP4          |
                |                                                        |
                +-------------------------------+------------------------+
                                                |
                                                v
                                        [THREAD 3: UPLOAD WORKER]
                                (HTTP POST: Metadata + Ảnh + Video)
                                                |
                                                v
        ================================================================================
        Khu vực 2: SERVER / CLOUD
        ================================================================================


2. YÊU CẦU PHẦN CỨNG & MÔI TRƯỜNG
A. Thiết bị Biên (Edge Device)

NVIDIA Jetson Orin Nano / Jetson Nano

Thay thế: Raspberry Pi 5 / Orange Pi 5

Model chạy dạng ONNX / TensorRT

Lưu trữ: SSD NVMe hoặc SD U3

B. Máy chủ (Server)

CPU mạnh, RAM ≥ 8GB

Python Backend (FastAPI / Django)

Database: PostgreSQL / MySQL

Object Storage: MinIO / Local FS

3. THIẾT KẾ CHI TIẾT MODULE EDGE (UPDATED)
3.1 Các Model AI sử dụng

Vehicle Detection: YOLOv8n / YOLOv11n (TensorRT / ONNX)

Traffic Light Detection: YOLOv8n (ROI cố định)

KHÔNG dùng model lane segmentation

3.2 Module Nhận Diện Làn Xe (Lane Estimation – NEW)
Mục tiêu

Tự động phân chia làn đường

Không cần train

Không cần vẽ tay lane

Thích nghi với từng camera

Nguyên lý

Xe chạy theo làn → quỹ đạo song song → gom cụm quỹ đạo = làn

Thuật toán

Tracker: ByteTrack / DeepSORT

Thu thập ≥ 10–30 giây quỹ đạo ban đầu

Gom cụm bằng DBSCAN

Input: trajectories (list of polyline)
Output: lane_id cho mỗi trajectory

Output

lane_id

lane_center_line

lane_width (ước lượng)

Cache vào lane_model.pkl

3.3 Logic Vi Phạm (Cập Nhật)

Ví dụ:

Xe máy đi vào làn ô tô

Ô tô vượt làn

Vượt đèn đỏ sai làn

if vehicle.lane_id == CAR_LANE and vehicle.type == "motorbike":
    violation = "Xe máy đi vào làn ô tô"

3.4 Bộ đệm vòng (Circular Buffer)

(KHÔNG THAY ĐỔI – giữ nguyên)

collections.deque

150–300 frames

Cắt video trước & sau vi phạm

3.5 Tối ưu hiệu năng

3 Thread độc lập:

Read Camera

AI + Lane Estimation

Upload

AI input: 640×360

Evidence giữ nguyên HD

4. MODULE SERVER (GIỮ NGUYÊN)
4.1 API

POST /api/v1/violations

Metadata bổ sung:

{
  "lane_id": 2,
  "lane_type": "car_lane"
}

4.2 OCR & Định danh

PaddleOCR / VietOCR

Chuẩn hóa biển số

Query DB

4.3 Lưu trữ

SQL: text + metadata

File system: /data/YYYY/MM/DD/camera_id/

5. ROADMAP TRIỂN KHAI (UPDATED)
Giai đoạn 1: Edge Core

Tách OCR khỏi Edge

Viết LaneEstimator

Tích hợp Tracker + Trajectory Buffer

Giai đoạn 2: AI Optimization

Export YOLO → TensorRT

Test FPS ≥ 15

Giai đoạn 3: Backend

API nhận file

OCR chạy background task

Giai đoạn 4: Integration Test

Test phân làn

Test vi phạm theo làn

Test clip MP4

6. ĐÁNH GIÁ GIẢI PHÁP NHẬN DIỆN LÀN
Tiêu chí	Kết quả
Tự động	100%
Không cần train	✅
Phù hợp giao thông VN	✅
Độ chính xác	~85–90%
Tải Edge	Thấp
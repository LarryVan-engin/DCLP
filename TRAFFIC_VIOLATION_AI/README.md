TRAFFIC_VIOLATION_AI/
│
├── config/                        # Thư mục cấu hình chung (mở rộng sau này)
│   ├── cameras.json               # Lưu danh sách IP/ID của các camera
│   └── default_zones.json         # Lưu các mẫu zone mặc định
│
├── docs/                          
│   └── Phân_tich_he_thong_G4.md   # Bản phân tích kiến trúc MQTT chúng ta đã chốt
│
├── edge/                          # [TRẠM BIÊN - JETSON NANO]
│   ├── edge_config.py             # ⚙️ File cấu hình trung tâm (MQTT Broker, Thresholds, Camera ID).
│   ├── main_edge.py               # 🚀 Trái tim của Edge: Chạy MQTT Client, bắt hình YOLO, gọi Engine kiểm tra lỗi và Publish dữ liệu.
│   ├── models/                    # Thư mục Model nhẹ, tối ưu cho Edge
│   │   ├── yolo12n.pt             # Model bắt xe (sẽ convert sang TensorRT .engine)
│   │   ├── model_detect_traffic_light.pt # Model bắt đèn giao thông
│   │   └── bytetrack.yaml         # File cấu hình tracker
│   ├── utils/                     # Các module AI Logic chuyên biệt
│   │   ├── capture_utils.py       # Chứa hàm `smart_crop` (cắt mở rộng viền) và mã hóa Base64 siêu tốc.
│   │   ├── lane_detection.py      # Thuật toán Data-Driven tự động học và chia làn đường sau 100 frame.
│   │   └── violation_engine.py    # Động cơ kiểm tra combo lỗi: Vượt đèn (có chờ rẽ phải), Đi ngược chiều, Sai làn.
│   ├── videos/                    # Thư mục chứa các file video test nằm sẵn trên Jetson (Zero-Upload).
│   └── requirements_edge.txt
│
├── scripts/                       # Các tool hỗ trợ triển khai (DevOps)
│   ├── export_tensorrt.py         # Script convert YOLO .pt sang .engine
│   ├── deploy_edge.sh             # Script đẩy code tự động lên Jetson
│   └── backup_db.sh               
│
├── server/                        # [MÁY CHỦ TRUNG TÂM - DASHBOARD]
│   ├── api_main.py                # 🌐 Trái tim của Server: FastAPI + Subscribe MQTT + Đẩy dữ liệu vào MongoDB Atlas + WebSocket lên UI.
│   ├── module_utils.py            # 🧠 Module chứa EasyOCR và logic bắt format biển số VN (Bạn đã cung cấp).
│   ├── database/
│   │   └── owners_sample.csv      # File CSV cơ sở dữ liệu chủ xe (để tra cứu offline).
│   ├── models/                    # Thư mục Model nặng (chạy trên Server)
│   │   └── model_detect_license_plate.pt # Model bắt Bounding Box của biển số.
│   ├── static/                    # Frontend Assets (Chuẩn bị viết)
│   │   ├── style.css              
│   │   ├── app.js                 
│   │   └── favicon.ico
│   ├── templates/                 # Frontend UI (Chuẩn bị viết)
│   │   └── index.html             
│   ├── violations/                # Folder lưu dự phòng ảnh vi phạm (local fallback nếu rớt mạng).
│   ├── uploads/                   # Thư mục dự phòng (không còn dùng để upload video nữa do đã qua luồng Zero-Upload).
│   └── requirements_server.txt
│
├── shared/                        # [GIAO THỨC CHUNG EDGE & SERVER]
│   ├── schemas.py                 # 📝 Cấu trúc Pydantic quy định format chuẩn của tin nhắn MQTT (ViolationPacket, Heartbeat, ControlCommand...).
│   └── zones_utils.py             # 📐 Bộ công cụ Toán học/Hình học: Tính Vector ngược chiều, xét giao điểm đè vạch, điểm nằm trong đa giác.
│
├── README.md                      # Hướng dẫn cài đặt project
└── .gitignore                     # Bỏ qua các file rác, __pycache__, thư mục ảo...
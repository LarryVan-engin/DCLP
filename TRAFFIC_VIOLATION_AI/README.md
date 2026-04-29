# 🚦 Traffic Violation Detection (Pro Version - MQTT Hybrid)

## 📁 Cấu trúc thư mục dự án

```text
TRAFFIC_VIOLATION_AI/
│
├── config/                        # Thư mục cấu hình chung
│   ├── cameras.json               # Lưu danh sách IP/ID của các camera
│   └── default_zones.json         # Lưu các mẫu zone mặc định
│
├── docs/                          
│   └── Phân_tich_he_thong_G4.md   # Bản phân tích kiến trúc MQTT
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
│   ├── module_utils.py            # 🧠 Module chứa EasyOCR và logic bắt format biển số VN
│   ├── database/
│   │   └── owners_sample.csv      # File CSV cơ sở dữ liệu chủ xe (để tra cứu offline).
│   ├── models/                    # Thư mục Model nặng (chạy trên Server)
│   │   └── model_detect_license_plate.pt # Model bắt Bounding Box của biển số.
│   ├── static/                    # Frontend Assets
│   │   ├── style.css              
│   │   ├── app.js                 
│   │   └── favicon.ico
│   ├── templates/                 # Frontend UI
│   │   └── index.html             
│   ├── violations/                # Folder lưu dự phòng ảnh vi phạm (local fallback nếu rớt mạng).
│   ├── uploads/                   # Thư mục dự phòng (không còn dùng để upload video).
│   └── requirements_server.txt
│
├── shared/                        # [GIAO THỨC CHUNG EDGE & SERVER]
│   ├── schemas.py                 # 📝 Cấu trúc Pydantic quy định format chuẩn của tin nhắn MQTT (ViolationPacket, Heartbeat, ControlCommand...).
│   └── zones_utils.py             # 📐 Bộ công cụ Toán học/Hình học: Tính Vector ngược chiều, xét giao điểm đè vạch, điểm nằm trong đa giác.
│
├── README.md                      # Hướng dẫn cài đặt project
└── .gitignore                     # Bỏ qua các file rác, __pycache__, thư mục ảo...
```

---

## 💻 Hướng dẫn truy cập và điều khiển Jetson Nano

Để truy cập và điều khiển Jetson Nano từ Laptop của bạn một cách mượt mà nhất cho đồ án này, dưới đây là 3 cách từ cơ bản đến "Pro":

### CÁCH 1: Kết nối trực tiếp qua cáp Micro-USB (Dễ nhất, không cần mạng)
Khi bạn mới mua Jetson Nano về hoặc đem lên trường bảo vệ mà không có Wi-Fi, đây là cứu cánh tuyệt vời nhất. Khi cắm cáp Micro-USB vào máy tính, Jetson Nano sẽ tự động tạo ra một mạng LAN ảo (RNDIS).

**Các bước thực hiện:**
1. Cắm cáp nguồn (Jack DC) cho Jetson Nano để khởi động.
2. Dùng cáp Micro-USB kết nối cổng Micro-USB của Jetson Nano vào cổng USB của Laptop.
3. Chờ khoảng 1-2 phút để Jetson Nano khởi động xong.
4. Mở Terminal (trên Mac/Linux) hoặc Command Prompt/PowerShell (trên Windows) và gõ lệnh:
   ```bash
   ssh <tên_user_của_jetson>@192.168.xx.x
   # Ví dụ: ssh larry@192.168.55.1
   ```
5. Nhập mật khẩu của Jetson Nano. Vậy là bạn đã vào được Terminal của Jetson!

### CÁCH 2: SSH qua mạng LAN / Wi-Fi (Dùng khi chạy thực tế)
Khi bạn đã cắm dây mạng LAN từ Router vào Jetson Nano (hoặc gắn USB Wi-Fi cho nó), cả Laptop và Jetson Nano phải dùng chung một mạng.

**Các bước thực hiện:**
1. Cắm dây LAN từ Router vào Jetson Nano.
2. Tìm địa chỉ IP của Jetson Nano. Có 2 cách:
   - **Cách lười:** Vào trình duyệt gõ địa chỉ IP của Router (thường là `192.168.1.1`), vào phần DHCP Client List tìm xem thiết bị nào tên "Jetson" hoặc "Ubuntu" rồi lấy IP.
   - **Cách trực tiếp:** Nếu có màn hình kết nối với Jetson Nano, mở Terminal và gõ:
     ```bash
     systemctl status ssh # Kiểm tra trạng thái ssh
     ```
     Nếu chưa mở:
     ```bash
     sudo systemctl enable ssh
     sudo systemctl start ssh
     ```
     Lấy địa chỉ IP của Jetson bằng:
     ```bash
     ifconfig
     # hoặc
     ip addr show wlan0
     ```
   - **Cách Ping:** Mở Terminal/CMD trên Laptop gõ lệnh:
     ```bash
     ping <tên_host_của_jetson>.local 
     # Ví dụ: ping jetson-nano.local
     ```
3. SSH vào thiết bị bằng IP vừa tìm được:
   ```bash
   ssh <tên_user_của_jetson>@<địa_chỉ_IP>
   # Ví dụ: ssh larry@192.168.1.15
   ```

### 🌟 CÁCH 3: Dùng VS Code Remote - SSH (CỰC KỲ KHUYÊN DÙNG)
Bạn không thể nào sửa code file `main_edge.py` bằng trình soạn thảo nano đen trắng trên Terminal được, sẽ rất cực. Hãy dùng chính VS Code trên Laptop của bạn để sửa code nằm trong Jetson!

**Các bước thực hiện:**
1. Mở **VS Code** trên Laptop.
2. Vào mục **Extensions**, tìm và cài đặt extension: **Remote - SSH** (của Microsoft).
3. Bấm vào biểu tượng màu xanh lá `><` ở góc dưới cùng bên trái.
4. Chọn **Connect to Host...** -> **Add New SSH Host...**
5. Gõ lệnh SSH: `ssh <user>@<IP>`.
6. Chọn hệ điều hành đích là Linux, nhập mật khẩu.
7. Chọn **Open Folder** và mở thư mục chứa code Đồ án.

👉 **Mẹo:** Bây giờ bạn có thể sửa code trực tiếp bằng giao diện VS Code, và mở Terminal ngay trong VS Code để chạy lệnh `python3 main_edge.py`.

### ⚠️ (Tùy chọn) CÁCH 4: Remote Desktop (Cần giao diện đồ họa)
Bạn có thể cài phần mềm NoMachine hoặc dùng VNC có sẵn của Ubuntu.
Tuy nhiên, **KHÔNG KHUYÊN DÙNG** vì xuất giao diện đồ họa qua mạng tốn rất nhiều RAM và CPU của Jetson Nano, khiến FPS của model YOLO bị tụt giảm. Đồ án của chúng ta đã có Dashboard Web, chỉ cần dùng SSH để chạy ngầm là tối ưu nhất.

---

## 🛠 Hướng dẫn triển khai hệ thống (Deploy)

### 1. Chuẩn bị môi trường trên Jetson Nano

```bash
# Cập nhật hệ thống
sudo apt-get update
sudo apt-get upgrade -y

# Cài đặt pip và các thư viện biên dịch cần thiết
sudo apt-get install -y python3-pip python3-dev libjpeg-dev liblapack-dev libblas-dev gfortran

# Nâng cấp pip và cài đặt công cụ theo dõi
python3 -m pip install --upgrade pip
sudo apt update
sudo apt install htop -y
sudo -H pip3 install -U jetson-stats
sudo systemctl restart jetson_stats.service
sudo reboot
```

### 2. Cài đặt PyTorch và Torchvision cho Jetson Nano

**Cài đặt PyTorch (Bản 1.10.0 cho JetPack 4.6):**
```bash
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip3 install Cython
pip3 install "Pillow==8.4.0"
pip3 install numpy torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

**Biên dịch và cài đặt Torchvision:**
```bash
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user
cd ..
```

### 3. Cài đặt Ultralytics (YOLO) và MQTT

```bash
# Cài đặt các thư viện phụ trợ
pip3 install matplotlib pyyaml tqdm scipy seaborn pandas pydantic paho-mqtt

# Kiểm tra GPU có hoạt động không
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 4. Truyền tải code lên Jetson Nano

Trên Terminal của **Laptop**, trỏ tới thư mục chứa source code và đẩy lên Jetson bằng lệnh `scp`:

```bash
# Chuyển toàn bộ thư mục edge sang Jetson Nano
scp -r ./edge <tên_user_jetson>@<IP_của_Jetson>:/home/<tên_user_jetson>/

# Tạo thư mục mới trên Jetson và chuyển file cụ thể
ssh <tên_user_jetson>@<IP_của_Jetson> "mkdir -p /home/<tên_user_jetson>/edge/<tên_thư_mục_mới>"
scp <tên_file> <tên_user_jetson>@<IP_của_Jetson>:/home/<tên_user_jetson>/edge/<tên_thư_mục_mới>
```

---

## 🚀 Chạy hệ thống với Docker trên Jetson Nano

**BƯỚC 1: Di chuyển vào thư mục chứa code**
Trên Terminal của Jetson Nano, đi tới thư mục chứa file `main_edge.py`:
```bash
cd /home/<user>/Traffic_Project/edge
```

**BƯỚC 2: Tải và Chạy Docker (Đã thêm Volume & Network)**
Chạy lệnh sau để khởi động môi trường GPU cách ly:
```bash
t=ultralytics/ultralytics:latest-jetson-jetpack4
sudo docker pull $t

# Bật vùng cách ly GPU, mount thư mục hiện tại vào /workspace
sudo docker run -it --ipc=host --network host --runtime=nvidia -v $(pwd):/workspace ultralytics/ultralytics:latest-jetson-jetpack4 /bin/bash

# Lùi lại thư mục gốc và vào workspace
ls
cd workspace
```

**BƯỚC 3: Cài đặt bổ sung và khởi chạy AI**
Sau khi vào Docker, cài đặt thêm các thư viện cần thiết và chạy ứng dụng:
```bash
# Cài duy nhất MQTT & Pydantic
pip install --upgrade pip setuptools wheel packaging
pip install "paho-mqtt<2.0.0"
pip install pydantic

export PYTHONPATH=/ultralytics:$PYTHONPATH

# Khởi động hệ thống Edge (Jetson Nano)
python3 main_edge.py
```

**Khởi động Server (chạy trên máy chủ trung tâm):**
```bash
uvicorn api_main:app --host 0.0.0.0 --port 8000
```

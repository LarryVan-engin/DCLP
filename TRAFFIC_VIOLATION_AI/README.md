# 🚦 Traffic Violation Detection (Pro Version - MQTT Hybrid)

---

## 📁 Cấu trúc thư mục dự án

```text
TRAFFIC_VIOLATION_AI/
│
├── config/                        # Thư mục cấu hình chung
│   ├── cameras.json               # Danh sách IP/ID các camera
│   └── default_zones.json         # Mẫu zone mặc định
│
├── docs/
│   └── Phân_tich_he_thong_G4.md   # Bản phân tích kiến trúc MQTT
│
├── edge/                          # [TRẠM BIÊN - JETSON NANO]
│   ├── edge_config.py             # ⚙️ Cấu hình trung tâm: MQTT Broker, Thresholds, Camera ID
│   ├── main_edge.py               # 🚀 MQTT Client + YOLO inference + Publish vi phạm
│   ├── models/                    # Model nhẹ tối ưu cho Edge
│   │   ├── yolo12n.engine         # Model phát hiện xe (TensorRT)
│   │   ├── model_detect_traffic_light.engine  # Model phát hiện đèn (TensorRT)
│   │   └── bytetrack.yaml         # Cấu hình tracker
│   ├── utils/
│   │   ├── capture_utils.py       # smart_crop + mã hóa Base64
│   │   ├── lane_detection.py      # Tự học phân làn sau 100 frame
│   │   └── violation_engine.py    # Kiểm tra: Vượt đèn, Ngược chiều, Sai làn
│   ├── shared/
│   │   ├── schemas.py             # Pydantic schemas cho MQTT packets
│   │   └── zones_utils.py         # Toán học hình học: ROI, perspective
│   ├── videos/                    # Video test nằm sẵn trên Jetson
│   └── models/requirements_edge.txt
│
├── server/                        # [MÁY CHỦ TRUNG TÂM]
│   ├── api_main.py                # 🌐 FastAPI + MQTT Subscribe + MongoDB + WebSocket
│   ├── module_utils.py            # 🧠 EasyOCR + logic biển số VN
│   ├── database/
│   │   └── owners_sample.csv      # CSDL chủ xe tra cứu offline
│   ├── models/
│   │   └── model_detect_license_plate.pt
│   ├── static/                    # Frontend: style.css, app.js, favicon.ico
│   ├── templates/
│   │   └── index.html             # Dashboard UI
│   ├── violations/                # Thư mục lưu ảnh vi phạm
│   └── requirements_server.txt
│
├── README.md
└── .gitignore
```

---

## 🏗 Kiến trúc hệ thống

```
[Jetson Nano]                    [Server]                  [Browser]
  Camera                           MQTT Broker
  YOLO Inference    →  MQTT  →    FastAPI (api_main.py)  →  WebSocket  →  Dashboard
  main_edge.py         topics:     OCR Biển số
                       violation/  MongoDB Atlas
                       heartbeat/  
                       stream/     
```

**Yêu cầu:** Jetson Nano và Server phải **cùng mạng LAN**, Server có IP tĩnh.

---

## ⚙️ Cấu hình trước khi chạy

Mở `edge/edge_config.py`, cập nhật 2 thông số quan trọng:

```python
CAMERA_ID  = "JETSON_01"          # Tên duy nhất cho mỗi Jetson
MQTT_BROKER = "192.168.1.x"       # IP của máy Server trong mạng LAN
```

---

## 🖥 PHẦN 1: Cài đặt Server

### Bước 1 — Cài dependencies

```bash
cd server/
pip install -r requirements_server.txt
```

### Bước 2 — Cài và khởi động MQTT Broker (Mosquitto)

```bash
sudo apt-get install -y mosquitto mosquitto-clients
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# Kiểm tra
sudo systemctl status mosquitto
```

### Bước 3 — Chạy Server

```bash
cd server/
uvicorn api_main:app --host 0.0.0.0 --port 8000
```

Mở Dashboard tại: `http://<IP_server>:8000`

---

## 🤖 PHẦN 2: Cài đặt Edge (Jetson Nano)

### 2.1 — Kết nối vào Jetson Nano

**Cách 1: SSH qua LAN (thường dùng)**
```bash
ssh larry@<IP_jetson>
# Ví dụ: ssh larry@192.168.1.8
```

> Lần đầu SSH sau khi reflash JetPack bị lỗi "host key changed":
> ```bash
> ssh-keygen -R <IP_jetson>   # xóa key cũ rồi SSH lại
> ```

**Cách 2: Micro-USB (không cần mạng, khi bảo vệ đồ án)**
1. Cắm cáp Micro-USB từ Jetson → Laptop
2. Chờ 1-2 phút để khởi động
3. `ssh larry@192.168.55.1`

**Cách 3: VS Code Remote-SSH (khuyên dùng để sửa code)**
1. Cài extension **Remote - SSH** trong VS Code
2. `><` góc dưới trái → Connect to Host → nhập `ssh larry@<IP>`
3. Open Folder → chọn thư mục edge

---

### 2.2 — Chuyển code lên Jetson

```bash
# Từ máy tính, chuyển toàn bộ thư mục edge
scp -r ./edge larry@<IP_jetson>:/home/larry/

# Chuyển video test
scp video_test.mp4 larry@<IP_jetson>:/home/larry/edge/videos/
```

---

### 2.3 — Cài Docker và pull image

```bash
# Cài Docker (nếu chưa có)
sudo apt-get install -y docker.io
sudo systemctl enable docker

# Pull image Ultralytics cho Jetson JetPack 4
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack4
```

---

### 2.4 — Build OpenCV với GStreamer *(Chỉ làm 1 lần)*

> **Tại sao cần?** `opencv-python` từ pip không có GStreamer → NVDEC không hoạt động.  
> Sau khi build: CPU giảm ~30-40% tải decode video → YOLO inference nhanh hơn.  
> **Sau khi build xong phải `docker commit` ngay** để không mất công.

#### Kiểm tra điều kiện trước khi build

```bash
# Bên trong container
python3.8 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer
# → NO: cần build  |  YES: đã xong, bỏ qua phần này

ls -la /dev/nvhost-nvdec 2>/dev/null && echo "✅ Device OK" || echo "❌ Thiếu --device flag"
gst-inspect-1.0 nvv4l2decoder 2>/dev/null | head -2 && echo "✅ Plugin OK" || echo "❌ Plugin thiếu"
```

#### B0 — Bật screen trên HOST (bắt buộc để tránh mất build khi SSH đứt)

```bash
# Trên HOST larry@jetson — TRƯỚC KHI chạy docker
sudo apt-get install -y screen    # nếu chưa có
screen -S opencv_build
```

| Thao tác | Phím tắt |
|---|---|
| Detach (thoát nhưng giữ session) | `Ctrl+A` rồi `D` |
| Reattach (vào lại) | `screen -r opencv_build` |

#### B1 — Chạy Docker với NVDEC device

```bash
cd /home/larry/edge
sudo docker run -it --ipc=host --network host --runtime=nvidia \
    --device=/dev/nvhost-nvdec \
    -v $(pwd):/workspace \
    ultralytics/ultralytics:latest-jetson-jetpack4 /bin/bash
```

#### B2 — Cài GStreamer dev headers

```bash
apt-get update && apt-get install -y \
    cmake \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad \
    libgstreamer-plugins-bad1.0-dev
```

#### B3 — Clone source

```bash
cd /tmp
git clone https://github.com/opencv/opencv.git --branch 4.8.0 --depth 1
git clone https://github.com/opencv/opencv_contrib.git --branch 4.8.0 --depth 1
```

#### B4 — CMake

```bash
cd /tmp/opencv && mkdir build && cd build

PY38=/usr/bin/python3.8
PY38_INC=$(python3.8 -c "import sysconfig; print(sysconfig.get_path('include'))")
PY38_LIB=$(find /usr -name "libpython3.8*.so*" | grep -v config | head -1)
PY38_PKG=$(python3.8 -c "import site; print(site.getsitepackages()[0])")
PY38_NPY=$(python3.8 -c "import numpy; print(numpy.get_include())")

pip3 uninstall opencv-python -y

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_GSTREAMER=ON \
      -D WITH_CUDA=OFF \
      -D WITH_CUBLAS=OFF \
      -D WITH_CUFFT=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
      -D PYTHON3_EXECUTABLE=$PY38 \
      -D PYTHON3_LIBRARY=$PY38_LIB \
      -D PYTHON3_INCLUDE_DIR=$PY38_INC \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=$PY38_NPY \
      -D PYTHON3_PACKAGES_PATH=$PY38_PKG \
      -D PYTHON_DEFAULT_EXECUTABLE=$PY38 \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      ..
```

> Verify output cmake — phải thấy:
> ```
> GStreamer:    YES (1.14.5)
> Interpreter: /usr/bin/python3.8
> ```
> Nếu `CUDA_npp*: NOTFOUND` → bình thường, dùng `WITH_CUDA=OFF` là đúng.

#### B5 — Build (~2-3 tiếng, có thể tắt SSH)

```bash
nohup make -j4 > /tmp/build.log 2>&1 &
trap '' HUP
tail -f /tmp/build.log
```

- Tắt SSH an toàn: `Ctrl+C` → `Ctrl+A D` (detach screen) → đóng SSH
- Sáng hôm sau: SSH lại → `screen -r opencv_build` → kiểm tra log
- Build thành công: dòng cuối log là `[100%] Built target opencv_python3`

#### B6 — Cài và verify

```bash
make install && ldconfig

python3.8 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer
# Kỳ vọng: GStreamer:    YES (1.14.5)
```

#### B7 — Test NVDEC

```bash
python3.8 - << 'EOF'
import cv2
pipeline = (
    "filesrc location=/workspace/videos/video_test.mp4 ! "
    "qtdemux ! h264parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print("✅ NVDEC OK!" if cap.isOpened() else "❌ NVDEC fail")
cap.release()
EOF
```

#### B8 — Docker commit ⚠️ Làm ngay, không để quên

```bash
# Từ HOST (terminal mới)
sudo docker ps
sudo docker commit <container_id> ultralytics-gstreamer:jetson

# Dọn source build bên trong container (~3GB)
rm -rf /tmp/opencv /tmp/opencv_contrib && apt-get clean

# Commit lần cuối sau khi dọn
sudo docker commit <container_id> ultralytics-gstreamer:jetson
sudo docker images | grep ultralytics-gstreamer
```

---

### 2.5 — Chạy Edge hàng ngày

> Từ lần này trở đi dùng image `ultralytics-gstreamer:jetson` đã có GStreamer sẵn.

**Bước 1 — Vào Docker:**
```bash
cd /home/larry/edge
sudo docker run -it --ipc=host --network host --runtime=nvidia \
    --device=/dev/nvhost-nvdec \
    -v $(pwd):/workspace \
    ultralytics-gstreamer:jetson /bin/bash
```

**Bước 2 — Cài packages (lần đầu hoặc sau khi tạo container mới):**
```bash
pip install --upgrade pip setuptools wheel packaging
pip install "paho-mqtt<2.0.0"
pip install pydantic numpy==1.23.5
pip3 install "lap>=0.5.12"
```

> Sau khi cài xong nhớ commit lại: `sudo docker commit <id> ultralytics-gstreamer:jetson`

**Bước 3 — Chạy Edge:**
```bash
cd /workspace
export PYTHONPATH=/ultralytics:$PYTHONPATH
python3.8 main_edge.py
```

---

## 🔁 Quy trình vận hành hàng ngày

```
1. Bật Server:    uvicorn api_main:app --host 0.0.0.0 --port 8000
2. SSH Jetson:    ssh larry@<IP>
3. Vào Docker:    sudo docker run ... ultralytics-gstreamer:jetson /bin/bash
4. Chạy Edge:     cd /workspace && python3.8 main_edge.py
5. Mở Dashboard:  http://<IP_server>:8000
```

---

## 🆘 Troubleshooting

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `socket.timeout` khi chạy main_edge.py | MQTT Broker chưa chạy hoặc sai IP | Kiểm tra `MQTT_BROKER` trong `edge_config.py`, khởi động Mosquitto trên server |
| `GStreamer: NO` sau build | cmake chạy sai Python | Kiểm tra `install path` trong cmake output phải là `python3.8` |
| `Resource not found` khi test NVDEC | File video không tồn tại | Copy video vào `/workspace/videos/` |
| `❌ Device chưa map` | Thiếu `--device=/dev/nvhost-nvdec` | Thêm flag vào lệnh `docker run` |
| Container chết khi SSH đóng | Không dùng `screen` trước `docker run` | Dùng `screen` trên HOST trước, build lại |
| SQUASHFS error khi boot | SD card corrupt (mất điện khi build) | Reflash JetPack, làm lại từ mục 2.3 |
| `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED` | Reflash JetPack tạo SSH key mới | `ssh-keygen -R <IP_jetson>` rồi SSH lại |
| `cmake: command not found` | Container chưa cài cmake | `apt-get install -y cmake` |
| `lap` tự download lúc chạy | Chưa cài sẵn | `pip3 install "lap>=0.5.12"` trước khi chạy |
| `numpy==1.23.5 not found` | Version mismatch | `pip3 install numpy==1.23.5` |

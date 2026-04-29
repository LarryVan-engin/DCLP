#!/bin/bash
# ================================================
# DEPLOY EDGE SERVICE - JETSON NANO
# Traffic Violation AI System - Edge Deployment
# Author: LARRY PHONG TRUC
# Version: 1.3 - 2026
# ================================================

# ================================================
# Hướng dẫn sử dụng:
# 1. Đảm bảo bạn đã kết nối với Jetson Nano qua SSH hoặc terminal
# 2. Chạy script này để tự động cài đặt dependencies, thiết lập virtual
# Script này sẽ tự động:

# Cài đặt dependencies
# Tạo virtual environment
# Cài thư viện Python
# Copy models
# Tạo systemd service chạy ngầm main_edge.py
# Khởi động và enable auto-start khi boot Jetson Nano
# cd scripts
# chmod +x deploy_edge.sh
# ./deploy_edge.sh
# ================================================

set -e

echo "========================================"
echo "🚀 DEPLOYING EDGE SERVICE ON JETSON NANO"
echo "========================================"

# ====================== CONFIG ======================
PROJECT_DIR="/home/$(whoami)/TRAFFIC_VIOLATION_AI_PRO"
EDGE_DIR="${PROJECT_DIR}/edge"
VENV_DIR="${EDGE_DIR}/venv"
SERVICE_NAME="traffic-edge.service"
LOG_DIR="${PROJECT_DIR}/logs"

# Tạo thư mục cần thiết
mkdir -p "${LOG_DIR}"
mkdir -p "${EDGE_DIR}/models"

echo "Project directory: ${PROJECT_DIR}"

# ====================== 1. CÀI ĐẶT DEPENDENCIES ======================
echo "Cài đặt các gói hệ thống cần thiết..."
sudo apt update
sudo apt install -y python3-pip python3-venv libatlas-base-dev ffmpeg mosquitto mosquitto-clients

# ====================== 2. TẠO VIRTUAL ENVIRONMENT ======================
echo "Tạo Virtual Environment..."
cd "${EDGE_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "Đã tạo virtual environment mới"
else
    echo "Virtual environment đã tồn tại"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# ====================== 3. CÀI ĐẶT PYTHON PACKAGES ======================
echo "Cài đặt các thư viện Python..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics paho-mqtt opencv-python-headless numpy pandas pydantic python-dotenv

echo "Đã cài đặt dependencies thành công"

# ====================== 4. COPY MODELS ======================
echo "Copy models vào Edge..."
cp -r "${PROJECT_DIR}/edge/models/"* "${EDGE_DIR}/models/" 2>/dev/null || echo "Không tìm thấy models, vui lòng copy thủ công"

# ====================== 5. TẠO SYSTEMD SERVICE ======================
echo "Tạo systemd service để chạy tự động..."

sudo tee /etc/systemd/system/${SERVICE_NAME} > /dev/null << EOF
[Unit]
Description=Traffic AI Edge Service - Jetson Nano
After=network.target mosquitto.service

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${EDGE_DIR}
ExecStart=${VENV_DIR}/bin/python ${EDGE_DIR}/main_edge.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
StandardOutput=append:${LOG_DIR}/edge_service.log
StandardError=append:${LOG_DIR}/edge_service_error.log

[Install]
WantedBy=multi-user.target
EOF

# ====================== 6. KHỞI ĐỘNG SERVICE ======================
echo "Khởi động service..."
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl restart ${SERVICE_NAME}

# ====================== 7. KIỂM TRA TRẠNG THÁI ======================
echo "Kiểm tra trạng thái service..."
sleep 2
sudo systemctl status ${SERVICE_NAME} --no-pager -l

echo ""
echo "========================================"
echo "🎉 DEPLOY EDGE SERVICE HOÀN TẤT!"
echo "========================================"
echo "Service name : ${SERVICE_NAME}"
echo "Log file     : ${LOG_DIR}/edge_service.log"
echo "Auto restart : Enabled"
echo ""
echo "Các lệnh hữu ích:"
echo "  sudo systemctl status traffic-edge.service"
echo "  sudo systemctl restart traffic-edge.service"
echo "  sudo systemctl stop traffic-edge.service"
echo "  tail -f ${LOG_DIR}/edge_service.log"
echo "========================================"

exit 0
@echo off
chcp 65001 > nul
title AI Traffic Monitor Server - Dashboard (Pro Version)

:: ================================================
:: AI TRAFFIC MONITOR SERVER LAUNCHER
:: Best Version - Professional & User Friendly
:: ================================================

echo.
echo ================================================
echo    AI TRAFFIC MONITOR SYSTEM - SERVER
echo    Hybrid Edge-Server Architecture (2026)
echo ================================================
echo.

cd /d "%~dp0server"

:: ====================== KIỂM TRA & TẠO VIRTUAL ENV ======================
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual Environment chưa tồn tại. Đang tạo mới...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Không thể tạo virtual environment. Kiểm tra Python đã cài chưa.
        pause
        exit /b 1
    )
    echo [OK] Đã tạo Virtual Environment thành công.
)

:: Activate venv
call venv\Scripts\activate.bat

echo [OK] Virtual Environment đã kích hoạt.

:: ====================== CÀI ĐẶT / CẬP NHẬT DEPENDENCIES ======================
echo [INFO] Đang kiểm tra và cài đặt dependencies...
pip install --quiet --upgrade pip
pip install --quiet fastapi uvicorn python-multipart ultralytics opencv-python pandas paho-mqtt pymongo python-dotenv

if %errorlevel% equ 0 (
    echo [OK] Dependencies đã sẵn sàng.
) else (
    echo [WARNING] Có lỗi khi cài dependencies. Tiếp tục chạy...
)

echo.
echo ================================================
echo               SERVER ĐANG KHỞI ĐỘNG
echo ================================================
echo.

:: ====================== MENU CHỌN ======================
echo Chọn chế độ chạy:
echo.
echo   1. Chạy Server bình thường (Port 8000)
echo   2. Chạy Server với Reload (Debug Mode)
echo   3. Chạy Server Port 8080
echo   4. Thoát
echo.

set /p choice="Nhập lựa chọn (1-4): "

if "%choice%"=="1" (
    echo [RUN] Khởi động Server (Production Mode)...
    python api_main.py
) else if "%choice%"=="2" (
    echo [RUN] Khởi động Server với Reload (Debug Mode)...
    uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload
) else if "%choice%"=="3" (
    echo [RUN] Khởi động Server trên Port 8080...
    uvicorn api_main:app --host 0.0.0.0 --port 8080
) else if "%choice%"=="4" (
    echo Thoát chương trình.
    exit /b 0
) else (
    echo Lựa chọn không hợp lệ. Chạy mặc định Port 8000...
    python api_main.py
)

echo.
echo Server đã dừng.
echo Nhấn bất kỳ phím nào để thoát...
pause > nul
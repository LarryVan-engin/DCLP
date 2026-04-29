@echo off
setlocal enabledelayedexpansion
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

:: ====================== DI CHUYỂN VỀ THƯ MỤC DCLP ======================
cd /d "%~dp0.."
echo [INFO] Thư mục hiện tại: %cd%

:: ====================== KIỂM TRA & KÍ HOẠT VIRTUAL ENV ======================
echo [INFO] Kiểm tra Virtual Environment...
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual Environment không tìm thấy tại .venv
    echo [INFO] Đang tạo mới...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Không thể tạo virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Đã tạo Virtual Environment.
)

:: Activate venv
call .venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo [ERROR] Không thể kích hoạt Virtual Environment.
    pause
    exit /b 1
)
echo [OK] Virtual Environment đã kích hoạt.

:: ====================== CÀI ĐẶT / CẬP NHẬT DEPENDENCIES ======================
echo.
echo [INFO] Đang cập nhật pip...
python -m pip install --upgrade pip --quiet

echo [INFO] Đang cài đặt thư viện cần thiết...
pip install --no-cache-dir --quiet ^
    fastapi ^
    uvicorn ^
    python-multipart ^
    ultralytics ^
    opencv-python ^
    pandas ^
    paho-mqtt ^
    pymongo ^
    python-dotenv ^
    Pillow ^
    numpy ^
    pydantic

if !errorlevel! equ 0 (
    echo [OK] Tất cả dependencies đã được cài đặt.
) else (
    echo [WARNING] Có lỗi khi cài dependencies. Tiếp tục chạy...
)

:: ====================== DI CHUYỂN VÀO THƯ MỤC SERVER ======================
cd /d "%~dp0TRAFFIC_VIOLATION_AI\server"
echo [INFO] Thư mục server: %cd%

echo.
echo ================================================
echo               SERVER ĐANG KHỞI ĐỘNG
echo ================================================
echo.

:: ====================== MENU CHỌN ======================
:menu
echo Chọn chế độ chạy:
echo.
echo   1. Chạy Server bình thường (Port 8000)
echo   2. Chạy Server với Reload (Debug Mode)
echo   3. Chạy Server Port 8080
echo   4. Kiểm tra Dependencies
echo   5. Thoát
echo.

set /p choice="Nhập lựa chọn (1-5): "

if "!choice:~0,1!"=="1" (
    echo [RUN] Khởi động Server (Production Mode)...
    python api_main.py
) else if "!choice:~0,1!"=="2" (
    echo [RUN] Khởi động Server với Reload (Debug Mode)...
    uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload
) else if "!choice:~0,1!"=="3" (
    echo [RUN] Khởi động Server trên Port 8080...
    uvicorn api_main:app --host 0.0.0.0 --port 8080
) else if "!choice:~0,1!"=="4" (
    echo [RUN] Kiểm tra các thư viện cài đặt...
    pip list
    pause
    goto menu
) else if "!choice:~0,1!"=="5" (
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
endlocal
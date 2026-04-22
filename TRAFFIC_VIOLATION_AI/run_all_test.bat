@echo off
chcp 65001 > nul  # Hỗ trợ UTF-8
cd /d d:\VSCode\DCLP

echo.
echo ========================================
echo 🚀 START TO RUN AUTOMATION TEST SUITE
echo ========================================

echo.
echo [1/2] Standalone Tests...
python TRAFFIC_VIOLATION_AI\docs\test_automation_standalone.py
if errorlevel 1 (
    echo ❌ Standalone tests failed!
    pause
    exit /b 1
)

echo.
echo [2/2] Integration Tests...
python TRAFFIC_VIOLATION_AI\docs\test_automation_integration.py
if errorlevel 1 (
    echo ❌ Integration tests failed!
    pause
    exit /b 1
)

echo.
echo ✅ All of test suites are done perfectly!
echo.
pause
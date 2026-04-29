@echo off
chcp 65001 > nul
cd /d d:\VSCode\DCLP

:MENU
echo.
echo ========================================
echo 🚀 TRAFFIC VIOLATION AI - TEST RUNNER
echo ========================================
echo 1. Run Simulated Tests (Standalone + Integration)
echo 2. Run Real Deployment Tests (Jetson Nano)
echo 3. Run All Tests
echo 4. Exit
echo.
set /p choice=Select an option (1-4): 

if "%choice%"=="1" goto SIMULATED
if "%choice%"=="2" goto REAL
if "%choice%"=="3" goto ALL
if "%choice%"=="4" goto EXIT
goto MENU

:SIMULATED
echo.
echo [1/2] Standalone Tests...
python TRAFFIC_VIOLATION_AI\docs\test_automation_standalone.py
if errorlevel 1 goto ERROR

echo.
echo [2/2] Integration Tests...
python TRAFFIC_VIOLATION_AI\docs\test_automation_integration.py
if errorlevel 1 goto ERROR

echo.
echo ✅ Simulated test suites are done perfectly!
pause
goto MENU

:REAL
echo.
echo [1/1] Real Deployment Tests...
python TRAFFIC_VIOLATION_AI\docs\test_automation_real_deploy.py
if errorlevel 1 goto ERROR

echo.
echo ✅ Real Deployment test suite is done perfectly!
pause
goto MENU

:ALL
echo.
echo [1/3] Standalone Tests...
python TRAFFIC_VIOLATION_AI\docs\test_automation_standalone.py
if errorlevel 1 goto ERROR

echo.
echo [2/3] Integration Tests...
python TRAFFIC_VIOLATION_AI\docs\test_automation_integration.py
if errorlevel 1 goto ERROR

echo.
echo [3/3] Real Deployment Tests...
python TRAFFIC_VIOLATION_AI\docs\test_automation_real_deploy.py
if errorlevel 1 goto ERROR

echo.
echo ✅ All test suites are done perfectly!
pause
goto MENU

:ERROR
echo ❌ Tests failed!
pause
goto MENU

:EXIT
exit /b 0

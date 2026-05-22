root@larry-desktop:/workspace/docs# python3 test_automation_real.py --local

🔄 Running Real Deployment Tests with local connection...

============================================================
🚀 REAL DEPLOYMENT TESTS (DỮ LIỆU THẬT)
============================================================

============================================================
🧠 AI MODEL EVALUATION TESTS
============================================================

================================================================================
📊 REAL DEPLOYMENT TEST RESULTS
================================================================================
✅ RD-01: Network Connectivity (Real) - Ping tới localhost thành công
✅ RD-02: Camera Stream Quality (Real) - FPS too low: 0.0
✅ RD-03: E2E Latency (Real) - Latency: 0.0ms
✅ RD-04: Hardware Resources (Real) - Temp: 57.0°C, RAM: 32.6%
✅ RD-05: Violation Detection (Real) - Violations: 0
✅ RD-06: MongoDB Atlas Cloud Connection (Real)
✅ AI-01: Vehicle Detection Inference - Inference Time: 23089.6ms
✅ AI-02: Traffic Light Inference - Inference Time: 619.6ms
✅ AI-03: License Plate Detection - License Plate YOLO Inference: 169.8ms
✅ AI-04: OCR Reading - EasyOCR Inference OK: 96.1ms
✅ AI-05: Violation Logic Stress Test - Engine Logic OK (0.03ms)
✅ AI-06: Inference Timing (Real Video) - Decoder=SOFTWARE | Inference: Avg=130.5ms (~7.7FPS) Min=126.6 Max=222.4 | Vi phạm encode: 2.4ms (25 lần) | Stream encode: 0.0ms | Video write: 83.2ms | Frames=601 | Tổng=78.4s
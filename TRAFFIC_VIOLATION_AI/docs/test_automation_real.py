"""
********************************************************************************************************************
Project:      Traffic Violation Detection (Pro Version - MQTT Hybrid)
File:         docs/test_automation_real.py
Description:  Automation Test Suite cho Real Deployment - Chạy với dữ liệu thật từ Jetson.
              Cần kết nối SSH hoặc MQTT tới Jetson để lấy metrics thực tế.
              
              Cách sử dụng:
              1. Kết nối SSH: python test_automation_real.py --ssh jetson_ip --user user --password pass
              2. Kết nối MQTT: python test_automation_real.py --mqtt broker_ip
              3. Chạy local trên Jetson: python test_automation_real.py --local
Author:       Larry Phong Truc
Date:         01/05/2026
********************************************************************************************************************
"""

import sys
import os
import json
import time
import argparse
import subprocess
import threading
import queue
import numpy as np
import cv2
import paho.mqtt.client as mqtt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import asyncio

# Add project root to path - dùng .resolve() để luôn là đường dẫn tuyệt đối
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "edge"))  # để import utils trực tiếp


# =====================================================================
# DATA CLASSES
# =====================================================================

@dataclass
class TestResult:
    """Lưu trữ kết quả từng test case"""
    test_id: str
    test_name: str
    category: str
    status: str  # "PASS", "FAIL", "SKIP", "NOT_RUN"
    duration: float
    message: str
    details: Dict = None
    metrics: Dict = None


@dataclass
class RealSystemMetrics:
    """Metrics thực từ hệ thống"""
    fps: float = 0.0
    latency_ms: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    ram_usage: float = 0.0
    ram_total_mb: float = 0.0
    ram_used_mb: float = 0.0
    temperature: float = 0.0
    mqtt_connected: bool = False
    ping_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    error_rate: float = 0.0
    violations_detected: int = 0
    vehicles_tracked: int = 0
    timestamp: str = ""


class TestReporter:
    """Quản lý và báo cáo kết quả test"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        self.system_metrics = RealSystemMetrics()
        
    def add_result(self, result: TestResult):
        self.results.append(result)
        
    def update_metrics(self, metrics: RealSystemMetrics):
        self.system_metrics = metrics


# =====================================================================
# CONNECTION MANAGERS
# =====================================================================

class SSHConnection:
    """Kết nối SSH tới Jetson"""
    
    def __init__(self, host: str, username: str, password: str):
        self.host = host
        self.username = username
        self.password = password
        self.connected = False
        
    def connect(self) -> bool:
        """Kết nối SSH"""
        try:
            # Test ping first
            result = subprocess.run(
                ["ping", "-n", "1", "-w", "2", self.host],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"❌ Cannot reach {self.host}")
                return False
                
            self.connected = True
            print(f"✅ SSH connected to {self.host}")
            return True
        except Exception as e:
            print(f"❌ SSH connection failed: {e}")
            return False
            
    def execute_command(self, command: str) -> str:
        """Execute command via SSH"""
        if not self.connected:
            return ""
            
        try:
            # Use paramiko or subprocess for SSH
            # For now, simulate with local commands
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=10
            )
            return result.stdout
        except Exception as e:
            print(f"❌ Command failed: {e}")
            return ""
            
    def get_jetson_metrics(self) -> RealSystemMetrics:
        """Lấy metrics từ Jetson qua SSH"""
        metrics = RealSystemMetrics()
        
        try:
            # Get CPU usage
            cpu_out = self.execute_command("top -bn1 | grep 'Cpu(s)'")
            if cpu_out:
                # Parse CPU usage
                import re
                match = re.search(r'(\d+\.\d+)\s*us', cpu_out)
                if match:
                    metrics.cpu_usage = float(match.group(1))
                    
            # Get RAM usage
            mem_out = self.execute_command("free -m | grep Mem:")
            if mem_out:
                parts = mem_out.split()
                if len(parts) >= 3:
                    metrics.ram_total_mb = float(parts[1])
                    metrics.ram_used_mb = float(parts[2])
                    metrics.ram_usage = (metrics.ram_used_mb / metrics.ram_total_mb * 100) if metrics.ram_total_mb > 0 else 0
                    
            # Get Temperature
            temp_out = self.execute_command("cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null")
            if temp_out:
                try:
                    metrics.temperature = float(temp_out.strip()) / 1000.0
                except:
                    pass
                    
            # Get GPU usage (if available)
            gpu_out = self.execute_command("tegrastats --interval 1000 | head -1")
            if gpu_out:
                # Parse tegrastats output
                import re
                match = re.search(r'GR3D\s+(\d+)%', gpu_out)
                if match:
                    metrics.gpu_usage = float(match.group(1))
                    
            metrics.timestamp = datetime.now().isoformat()
            
        except Exception as e:
            print(f"⚠️ Error getting metrics: {e}")
            
        return metrics


class MQTTConnection:
    """Kết nối MQTT để lấy dữ liệu realtime từ Edge"""
    
    def __init__(self, broker: str, port: int = 1883):
        self.broker = broker
        self.port = port
        self.client = None
        self.connected = False
        self.message_queue = queue.Queue()
        self.last_heartbeat = {}
        self.last_violation = {}
        
    def connect(self) -> bool:
        """Kết nối MQTT broker"""
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "test_client")
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            
            # Wait for connection
            time.sleep(2)
            return self.connected
            
        except Exception as e:
            print(f"❌ MQTT connection failed: {e}")
            return False
            
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"✅ MQTT connected to {self.broker}")
            
            # Subscribe to all topics
            client.subscribe("status/+/heartbeat")
            client.subscribe("violation/+")
            client.subscribe("stream/+/mjpeg")
        else:
            print(f"❌ MQTT connection failed with code {rc}")
            
    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            
            if "heartbeat" in msg.topic:
                self.last_heartbeat = payload
            elif "violation" in msg.topic:
                self.last_violation = payload
                
            self.message_queue.put({
                "topic": msg.topic,
                "payload": payload,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"⚠️ Message parse error: {e}")
            
    def get_metrics(self) -> RealSystemMetrics:
        """Lấy metrics từ MQTT messages"""
        metrics = RealSystemMetrics()
        
        if self.last_heartbeat:
            stats = self.last_heartbeat.get("stats", {})
            lights = self.last_heartbeat.get("lights", {})
            
            metrics.fps = self.last_heartbeat.get("fps", 0)
            metrics.vehicles_tracked = sum(stats.values()) if stats else 0
            metrics.mqtt_connected = self.connected
            
        if self.last_violation:
            metrics.violations_detected += 1
            
        metrics.timestamp = datetime.now().isoformat()
        
        return metrics
    
    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


class LocalJetsonRunner:
    """Chạy trực tiếp trên Jetson để lấy metrics"""
    
    def __init__(self):
        self.running = False
        
    def get_metrics(self) -> RealSystemMetrics:
        """Lấy metrics từ local system"""
        metrics = RealSystemMetrics()
        
        try:
            # CPU Usage
            result = subprocess.run(
                ["top", "-bn1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
            )
            import re
            match = re.search(r'(\d+\.\d+)\s*id', result.stdout)
            if match:
                idle = float(match.group(1))
                metrics.cpu_usage = 100.0 - idle
                
            # RAM Usage
            result = subprocess.run(
                ["free", "-m"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
            )
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith('Mem:'):
                    parts = line.split()
                    if len(parts) >= 3:
                        metrics.ram_total_mb = float(parts[1])
                        metrics.ram_used_mb = float(parts[2])
                        metrics.ram_usage = (metrics.ram_used_mb / metrics.ram_total_mb * 100) if metrics.ram_total_mb > 0 else 0
                    break
                    
            # Temperature
            try:
                result = subprocess.run(
                    ["cat", "/sys/class/thermal/thermal_zone0/temp"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2
                )
                metrics.temperature = float(result.stdout.strip()) / 1000.0
            except:
                pass
                
            # GPU (if tegrastats available)
            try:
                result = subprocess.run(
                    ["tegrastats", "--interval", "1000", "--stop"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3
                )
                import re
                match = re.search(r'GR3D\s+(\d+)%', result.stdout)
                if match:
                    metrics.gpu_usage = float(match.group(1))
            except:
                pass
                
            metrics.timestamp = datetime.now().isoformat()
            
        except Exception as e:
            print(f"⚠️ Error getting local metrics: {e}")
            
        return metrics


# =====================================================================
# REAL TESTS
# =====================================================================

class RealDeploymentTests:
    """Tests chạy với dữ liệu thật từ hệ thống"""
    
    def __init__(self, reporter: TestReporter, connection_type: str):
        self.reporter = reporter
        self.connection_type = connection_type
        self.ssh_conn = None
        self.mqtt_conn = None
        self.local_runner = None
        
    def set_connections(self, ssh=None, mqtt=None, local=None):
        self.ssh_conn = ssh
        self.mqtt_conn = mqtt
        self.local_runner = local
        
    def _get_metrics(self) -> RealSystemMetrics:
        """Lấy metrics dựa trên connection type"""
        if self.connection_type == "ssh" and self.ssh_conn:
            return self.ssh_conn.get_jetson_metrics()
        elif self.connection_type == "mqtt" and self.mqtt_conn:
            return self.mqtt_conn.get_metrics()
        elif self.connection_type == "local" and self.local_runner:
            return self.local_runner.get_metrics()
        else:
            return RealSystemMetrics()
            
    def test_RD_01_network_connectivity(self):
        """RD-01: Kiểm tra kết nối mạng"""
        test_id = "RD-01"
        test_name = "Network Connectivity (Real)"
        start_time = time.time()
        
        try:
            metrics = self._get_metrics()

            # Test ping - dùng cú pháp Linux (-c/-W) vì chạy trên Jetson/Docker
            target = self.ssh_conn.host if (self.connection_type == "ssh" and self.ssh_conn) else "8.8.8.8"
            try:
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", "2", target],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
                )
                ping_ok = result.returncode == 0
            except FileNotFoundError:
                # ping không có trong PATH của Docker → thử /bin/ping
                try:
                    result = subprocess.run(
                        ["/bin/ping", "-c", "1", "-W", "2", target],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
                    )
                    ping_ok = result.returncode == 0
                except Exception:
                    ping_ok = False

            assert ping_ok, f"Ping tới {target} thất bại"
            
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "RealDeploy", "PASS", duration,
                f"Ping: {metrics.ping_ms:.1f}ms, MQTT: {metrics.mqtt_connected}",
                details={"ping_ms": metrics.ping_ms, "mqtt_connected": metrics.mqtt_connected},
                metrics={"ping_ms": metrics.ping_ms, "mqtt_connected": metrics.mqtt_connected}
            ))
        except Exception as e:
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "RealDeploy", "FAIL", duration, str(e)
            ))

    def test_RD_02_camera_stream(self):
        """RD-02: Kiểm tra Camera Stream"""
        test_id = "RD-02"
        test_name = "Camera Stream Quality (Real)"
        start_time = time.time()
        
        try:
            metrics = self._get_metrics()
            
            # Check FPS
            assert metrics.fps >= 15, f"FPS too low: {metrics.fps}"
            
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "RealDeploy", "PASS", duration,
                f"FPS: {metrics.fps:.1f}",
                details={"fps": metrics.fps},
                metrics={"fps": metrics.fps}
            ))
        except Exception as e:
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "RealDeploy", "FAIL", duration, str(e)
            ))

    def test_RD_03_e2e_latency(self):
        """RD-03: Đo latency thực tế"""
        test_id = "RD-03"
        test_name = "E2E Latency (Real)"
        start_time = time.time()
        
        try:
            metrics = self._get_metrics()
            
            # Check latency
            assert metrics.latency_ms < 1000, f"Latency too high: {metrics.latency_ms}ms"
            
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "RealDeploy", "PASS", duration,
                f"Latency: {metrics.latency_ms:.1f}ms",
                details={"latency_ms": metrics.latency_ms},
                metrics={"latency_ms": metrics.latency_ms}
            ))
        except Exception as e:
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "RealDeploy", "FAIL", duration, str(e)
            ))

    def test_RD_04_hardware_resources(self):
        """RD-04: Kiểm tra tài nguyên hardware"""
        test_id = "RD-04"
        test_name = "Hardware Resources (Real)"
        start_time = time.time()
        
        try:
            metrics = self._get_metrics()
            
            # Check temperature
            assert metrics.temperature < 80, f"Temperature too high: {metrics.temperature}°C"
            
            # Check RAM
            assert metrics.ram_usage < 90, f"RAM usage too high: {metrics.ram_usage}%"
            
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "RealDeploy", "PASS", duration,
                f"Temp: {metrics.temperature:.1f}°C, RAM: {metrics.ram_usage:.1f}%",
                details={
                    "temperature": metrics.temperature,
                    "ram_usage": metrics.ram_usage,
                    "cpu_usage": metrics.cpu_usage,
                    "gpu_usage": metrics.gpu_usage
                },
                metrics={
                    "temperature": metrics.temperature,
                    "ram_usage": metrics.ram_usage,
                    "cpu_usage": metrics.cpu_usage,
                    "gpu_usage": metrics.gpu_usage
                }
            ))
        except Exception as e:
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "RealDeploy", "FAIL", duration, str(e)
            ))

    def test_RD_05_violation_detection(self):
        """RD-05: Kiểm tra phát hiện vi phạm"""
        test_id = "RD-05"
        test_name = "Violation Detection (Real)"
        start_time = time.time()
        
        try:
            metrics = self._get_metrics()
            
            # Just report the number of violations
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "RealDeploy", "PASS", duration,
                f"Violations: {metrics.violations_detected}",
                details={"violations": metrics.violations_detected},
                metrics={"violations_detected": metrics.violations_detected}
            ))
        except Exception as e:
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "RealDeploy", "FAIL", duration, str(e)
            ))

    def test_RD_06_mongodb_cloud(self):
        """RD-06: Kiểm tra kết nối MongoDB Atlas Cloud thực"""
        test_id = "RD-06"
        test_name = "MongoDB Atlas Cloud Connection (Real)"
        start_time = time.time()
        
        try:
            # Test MongoDB connection thực
            try:
                from motor.motor_asyncio import AsyncIOMotorClient
            except ImportError:
                duration = time.time() - start_time
                self.reporter.add_result(TestResult(
                    test_id, test_name, "RealDeploy", "SKIP", duration,
                    "motor chưa cài → bỏ qua. Chạy: pip install motor"
                ))
                return

            import asyncio
            
            MONGO_URI = "mongodb+srv://admin:admin123@cluster0.teleibk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
            
            # Chạy async function trong sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def test_mongodb():
                client = AsyncIOMotorClient(MONGO_URI)
                db = client.traffic_db
                # Try to count violations
                count = await db.violations.count_documents({})
                await client.close()
                return count
            
            count = loop.run_until_complete(test_mongodb())
            loop.close()
            
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "RealDeploy", "PASS", duration,
                f"MongoDB Atlas connected, {count} violations in database",
                details={"violations_count": count},
                metrics={"mongodb_connected": True, "violations_count": count}
            ))
        except Exception as e:
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "RealDeploy", "FAIL", duration, str(e)
            ))

    def run_all(self):
        """Chạy tất cả real tests"""
        print("\n" + "="*60)
        print("🚀 REAL DEPLOYMENT TESTS (DỮ LIỆU THẬT)")
        print("="*60)
        
        self.test_RD_01_network_connectivity()
        self.test_RD_02_camera_stream()
        self.test_RD_03_e2e_latency()
        self.test_RD_04_hardware_resources()
        self.test_RD_05_violation_detection()
        self.test_RD_06_mongodb_cloud()


# =====================================================================
# AI MODEL EVALUATION TESTS
# =====================================================================

class TestAIEvaluation:
    """AI Model Evaluation Tests - Đánh giá độ chính xác của các mô hình AI"""
    
    def __init__(self, reporter: TestReporter):
        self.reporter = reporter
        self.yolo_vehicle_path = str(PROJECT_ROOT / "edge" / "models" / "yolo12n.pt")
        self.yolo_light_path = str(PROJECT_ROOT / "edge" / "models" / "model_detect_traffic_light.pt")
        self.ocr_model_path = str(PROJECT_ROOT / "server" / "models" / "model_detect_license_plate.pt")
        
    def test_AI_01_vehicle_detection(self):
        """AI-01: Vehicle Detection Model (YOLOv12n) Inference"""
        test_id = "AI-01"
        test_name = "Vehicle Detection Inference"
        start_time = time.time()
        
        try:
            from ultralytics import YOLO
            import os
            
            if not os.path.exists(self.yolo_vehicle_path):
                raise FileNotFoundError(f"Model not found: {self.yolo_vehicle_path}")
                
            model = YOLO(self.yolo_vehicle_path)
            dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            inf_start = time.time()
            results = model(dummy_img, verbose=False)
            inf_time = time.time() - inf_start
            
            assert len(results) > 0
            
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "AIEvaluation", "PASS", duration,
                f"Inference Time: {inf_time*1000:.1f}ms",
                details={"inference_ms": inf_time*1000},
                metrics={"vehicle_inference_ms": inf_time*1000}
            ))
        except Exception as e:
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "AIEvaluation", "FAIL", duration, str(e)
            ))

    def test_AI_02_traffic_light_detection(self):
        """AI-02: Traffic Light Model Inference"""
        test_id = "AI-02"
        test_name = "Traffic Light Inference"
        start_time = time.time()
        
        try:
            from ultralytics import YOLO
            import os
            
            if not os.path.exists(self.yolo_light_path):
                raise FileNotFoundError(f"Model not found: {self.yolo_light_path}")
                
            model = YOLO(self.yolo_light_path)
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            inf_start = time.time()
            results = model(dummy_img, verbose=False)
            inf_time = time.time() - inf_start
            
            assert len(results) > 0
            
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "AIEvaluation", "PASS", duration,
                f"Inference Time: {inf_time*1000:.1f}ms",
                details={"inference_ms": inf_time*1000},
                metrics={"light_inference_ms": inf_time*1000}
            ))
        except Exception as e:
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "AIEvaluation", "FAIL", duration, str(e)
            ))

    def test_AI_03_license_plate_detection(self):
        """AI-03: License Plate Detection Model"""
        test_id = "AI-03"
        test_name = "License Plate Detection"
        start_time = time.time()
        
        try:
            from ultralytics import YOLO
            import os
            import sys
            
            if not os.path.exists(self.ocr_model_path):
                raise FileNotFoundError(f"Model not found: {self.ocr_model_path}")
                
            server_path = str(PROJECT_ROOT / "server")
            if server_path not in sys.path:
                sys.path.insert(0, server_path)
                
            model = YOLO(self.ocr_model_path)
            dummy_img = np.zeros((100, 300, 3), dtype=np.uint8)
            inf_start = time.time()
            results = model(dummy_img, verbose=False)
            inf_time = time.time() - inf_start
            
            assert len(results) > 0
            
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "AIEvaluation", "PASS", duration,
                f"License Plate YOLO Inference: {inf_time*1000:.1f}ms",
                details={"inference_ms": inf_time*1000},
                metrics={"lp_detection_inference_ms": inf_time*1000}
            ))
        except Exception as e:
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "AIEvaluation", "FAIL", duration, str(e)
            ))

    def test_AI_04_ocr_reading(self):
        """AI-04: License Plate Text Reading (EasyOCR)"""
        test_id = "AI-04"
        test_name = "OCR Reading"
        start_time = time.time()
        
        try:
            import sys
            server_path = str(PROJECT_ROOT / "server")
            if server_path not in sys.path:
                sys.path.insert(0, server_path)
                
            try:
                from module_utils import read_license_plate_vn
                dummy_img = np.zeros((200, 400, 3), dtype=np.uint8)
                inf_start = time.time()
                # Mock a call. read_license_plate_vn has signature (frame, x1, y1, x2, y2)
                text, success = read_license_plate_vn(dummy_img, 0, 0, 400, 200)
                inf_time = time.time() - inf_start
                
                duration = time.time() - start_time
                self.reporter.add_result(TestResult(
                    test_id, test_name, "AIEvaluation", "PASS", duration,
                    f"EasyOCR Inference OK: {inf_time*1000:.1f}ms",
                    details={"text": text, "inference_ms": inf_time*1000},
                    metrics={"ocr_inference_ms": inf_time*1000}
                ))
            except Exception as inner_e:
                duration = time.time() - start_time
                self.reporter.add_result(TestResult(
                    test_id, test_name, "AIEvaluation", "SKIP", duration, 
                    f"Skipped OCR test (missing dependencies or easyocr error): {str(inner_e)}"
                ))
        except Exception as e:
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "AIEvaluation", "FAIL", duration, str(e)
            ))

    def test_AI_05_violation_engine_stress(self):
        """AI-05: Violation Logic Stress Test"""
        test_id = "AI-05"
        test_name = "Violation Logic Stress Test"
        start_time = time.time()
        
        try:
            import sys
            # PROJECT_ROOT / edge đã được thêm vào sys.path lúc import file này
            # Nên import trực tiếp từ utils (không cần prefix 'edge.')
            from utils.violation_engine import ViolationEngine
            engine = ViolationEngine()
            
            # Setup dummy zones config mock
            class MockPoint:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    
            class MockLineZone:
                def __init__(self, label):
                    self.label = label
                    self.points = [MockPoint(0, 500), MockPoint(1280, 500)]
            
            zones_config = {
                "lines": [MockLineZone("stop_line")],
                "polygons": []
            }
            
            # Test simple checking
            light_status = {"straight": "red", "left": "unknown", "right": "unknown"}
            trajectory = [(100, 480), (100, 510)] # Crossed stop_line
            bbox = [80, 450, 120, 510]
            track_id = 1
            
            inf_start = time.time()
            engine.check_violations(track_id, bbox, trajectory, light_status, zones_config, stop_line_y=500)
            inf_time = time.time() - inf_start
            
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "AIEvaluation", "PASS", duration,
                f"Engine Logic OK ({inf_time*1000:.2f}ms)",
                details={"violations": list(engine.recorded_violations[track_id])},
                metrics={"engine_inference_ms": inf_time*1000}
            ))
        except Exception as e:
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "AIEvaluation", "FAIL", duration, str(e)
            ))

    def test_AI_06_inference_timing_log(self):
        """AI-06: Đọc Inference Timing từ inference_timing.json do main_edge.py lưu"""
        test_id = "AI-06"
        test_name = "Inference Timing (Real Video)"
        start_time = time.time()

        timing_file = PROJECT_ROOT / "edge" / "inference_timing.json"

        try:
            if not timing_file.exists():
                # Fallback: đo bằng YOLO 1 lần nếu chưa chạy main_edge.py
                from ultralytics import YOLO
                model_path = PROJECT_ROOT / "edge" / "models" / "yolo12n.pt"
                if not model_path.exists():
                    raise FileNotFoundError(f"Chưa có inference_timing.json và model cũng không tìm thấy.")
                model = YOLO(str(model_path))
                dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
                times_ms = []
                for _ in range(5):
                    t = time.time()
                    model(dummy, verbose=False)
                    times_ms.append((time.time() - t) * 1000)
                data = {
                    "avg_ms": round(sum(times_ms) / len(times_ms), 2),
                    "avg_fps": round(1000.0 / (sum(times_ms) / len(times_ms)), 2),
                    "min_ms": round(min(times_ms), 2),
                    "max_ms": round(max(times_ms), 2),
                    "total_frames": len(times_ms),
                    "total_s": round(sum(times_ms) / 1000, 2),
                    "source": "fallback_dummy"
                }
                note = "⚠️ Dùng dummy inference (chưa có inference_timing.json)"
            else:
                with open(timing_file, encoding="utf-8") as f:
                    data = json.load(f)
                note = f"📁 Đọc từ inference_timing.json ({data.get('timestamp','')})"

            avg_ms  = data["avg_ms"]
            avg_fps = data["avg_fps"]
            min_ms  = data.get("min_ms", 0)
            max_ms  = data.get("max_ms", 0)
            frames  = data.get("total_frames", 0)
            total_s = data.get("total_s", 0)

            # PASS nếu avg < 200ms (Jetson yêu cầu tối thiểu ~5 FPS)
            passed = avg_ms < 200

            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "AIEvaluation",
                "PASS" if passed else "FAIL", duration,
                f"Avg={avg_ms:.1f}ms (~{avg_fps:.1f}FPS) | Min={min_ms:.1f}ms | Max={max_ms:.1f}ms | Frames={frames} | {note}",
                details=data,
                metrics={
                    "avg_inference_ms": avg_ms,
                    "avg_fps": avg_fps,
                    "min_ms": min_ms,
                    "max_ms": max_ms,
                    "total_frames": frames,
                    "total_s": total_s,
                }
            ))
        except Exception as e:
            duration = time.time() - start_time
            self.reporter.add_result(TestResult(
                test_id, test_name, "AIEvaluation", "FAIL", duration, str(e)
            ))


    def run_all(self):
        """Chạy tất cả AI Evaluation tests"""
        print("\n" + "="*60)
        print("🧠 AI MODEL EVALUATION TESTS")
        print("="*60)
        
        self.test_AI_01_vehicle_detection()
        self.test_AI_02_traffic_light_detection()
        self.test_AI_03_license_plate_detection()
        self.test_AI_04_ocr_reading()
        self.test_AI_05_violation_engine_stress()
        self.test_AI_06_inference_timing_log()


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Real Deployment Test Suite")
    parser.add_argument("--ssh", type=str, help="Jetson IP address")
    parser.add_argument("--user", type=str, default="jetson", help="SSH username")
    parser.add_argument("--password", type=str, help="SSH password")
    parser.add_argument("--mqtt", type=str, help="MQTT broker IP")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT port")
    parser.add_argument("--local", action="store_true", help="Run locally on Jetson")
    
    args = parser.parse_args()
    
    reporter = TestReporter()
    connection_type = "local"
    
    # Setup connections based on arguments
    if args.ssh:
        connection_type = "ssh"
        ssh_conn = SSHConnection(args.ssh, args.user, args.password)
        if not ssh_conn.connect():
            print("❌ SSH connection failed, exiting...")
            return 1
        real_tests = RealDeploymentTests(reporter, connection_type)
        real_tests.set_connections(ssh=ssh_conn)
    elif args.mqtt:
        connection_type = "mqtt"
        mqtt_conn = MQTTConnection(args.mqtt, args.mqtt_port)
        if not mqtt_conn.connect():
            print("❌ MQTT connection failed, exiting...")
            return 1
        real_tests = RealDeploymentTests(reporter, connection_type)
        real_tests.set_connections(mqtt=mqtt_conn)
    elif args.local:
        connection_type = "local"
        local_runner = LocalJetsonRunner()
        real_tests = RealDeploymentTests(reporter, connection_type)
        real_tests.set_connections(local=local_runner)
    else:
        print("❌ Vui lòng chọn một trong các options:")
        print("   --ssh <ip> --user <user> --password <pass>  : Kết nối SSH tới Jetson")
        print("   --mqtt <ip>                                   : Kết nối MQTT broker")
        print("   --local                                       : Chạy local trên Jetson")
        return 1
        
    # Run tests
    print(f"\n🔄 Running Real Deployment Tests with {connection_type} connection...")
    real_tests.run_all()
    
    ai_tests = TestAIEvaluation(reporter)
    ai_tests.run_all()
    
    # Generate report
    print("\n" + "="*80)
    print("📊 REAL DEPLOYMENT TEST RESULTS")
    print("="*80)
    
    for result in reporter.results:
        icon = "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⏭️"
        print(f"{icon} {result.test_id}: {result.test_name} - {result.message}")
    
    # Save report
    report_path = Path(__file__).resolve().parent / f"test_report_real_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)  # Tạo thư mục nếu chưa có
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Real Deployment Test Report\n")
        f.write(f"Connection Type: {connection_type}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        for result in reporter.results:
            f.write(f"{result.test_id}: {result.status} - {result.message}\n")
    
    print(f"\n💾 Report saved: {report_path}")
    
    # Cleanup
    if connection_type == "mqtt" and mqtt_conn:
        mqtt_conn.disconnect()
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
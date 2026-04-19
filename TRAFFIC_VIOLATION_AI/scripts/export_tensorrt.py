"""
********************************************************************************************************************
Project:      Traffic Violation Detection
File:         scripts/export_tensorrt.py
Description:  Export YOLO models to TensorRT engine for Jetson Nano (FP16/INT8)
Author:       LARRY PHONG TRUC
Updated:      2026-04-19
********************************************************************************************************************
"""

import os
import sys
import torch
from ultralytics import YOLO
import argparse

# ====================== CONFIG ======================
MODELS_CONFIG = {
    "vehicle": {
        "pt_path": "../edge/models/yolo12n.pt",
        "engine_name": "yolo12n.engine",
        "imgsz": 640,
        "half": True,       # FP16
        "int8": False,
        "batch": 1
    },
    "traffic_light": {
        "pt_path": "../edge/models/model_detect_traffic_light.pt",
        "engine_name": "traffic_light.engine",
        "imgsz": 640,
        "half": True,
        "int8": False,
        "batch": 1
    },
    "plate": {
        "pt_path": "../server/models/model_detect_license_plate.pt",
        "engine_name": "plate_detect.engine",
        "imgsz": 640,
        "half": True,
        "int8": False,
        "batch": 1
    }
}

def export_to_tensorrt(model_name: str):
    """Export single model to TensorRT"""
    config = MODELS_CONFIG.get(model_name)
    if not config:
        print(f"Model {model_name} not found!")
        return False

    pt_path = config["pt_path"]
    engine_name = config["engine_name"]
    imgsz = config["imgsz"]
    half = config["half"]
    int8 = config["int8"]

    if not os.path.exists(pt_path):
        print(f"Không tìm thấy file model: {pt_path}")
        return False

    print(f"Đang export {model_name} → TensorRT ({'INT8' if int8 else 'FP16' if half else 'FP32'})...")

    try:
        # Load model
        model = YOLO(pt_path)

        # Export to TensorRT
        success = model.export(
            format="engine",           # TensorRT
            imgsz=imgsz,
            half=half,
            int8=int8,
            batch=config["batch"],
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            workspace=8,               # GB workspace
            simplify=True
        )

        if success:
            # Rename output file
            default_engine = pt_path.replace(".pt", ".engine")
            if os.path.exists(default_engine):
                target_path = f"../edge/models/{engine_name}"
                os.rename(default_engine, target_path)
                print(f"Export thành công: {target_path}")
                print(f"   Size: {os.path.getsize(target_path) / (1024*1024):.2f} MB")
            return True
        else:
            print("Export thất bại!")
            return False

    except Exception as e:
        print(f"Lỗi khi export {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["vehicle", "traffic_light", "plate", "all"],
                        default="all", help="Model to export")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantization (slower export, faster inference)")
    args = parser.parse_args()

    if args.int8:
        for model in MODELS_CONFIG.values():
            model["int8"] = True
            model["half"] = False

    print("========================================")
    print("🚀 TENSORRT EXPORT TOOL FOR JETSON NANO")
    print("========================================")

    if args.model == "all":
        for model_name in ["vehicle", "traffic_light", "plate"]:
            export_to_tensorrt(model_name)
    else:
        export_to_tensorrt(args.model)

    print("\nHoàn tất export TensorRT!")
    print("Khuyến nghị: Sử dụng FP16 cho tốc độ tốt nhất trên Jetson Nano")


if __name__ == "__main__":
    # Kiểm tra GPU
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("Không tìm thấy GPU! TensorRT export sẽ rất chậm.")

    main()
"""
*******************************************************************************************************************
General Information
********************************************************************************************************************
Project:        DETECT LICENSE PLATES AND TRAFFIC PENALTY INTERGRATION
File:          PRETRAIN.py
Description:   This file is used to train model YOLO before intergrating to main code.

Author:        LARRY PHONG TRUC
Email:         vanphongtruc1808@gmail.com
Created:       26/10/2025
Last Update:   26/10
Version:       1.0

Python:        3.10.11
Dependencies:  - Model YOLOv12 (yolo12x.pt) "x" can be l,n,m,...
               - Dataset for training (include images + labels)

Copyright:     (c) 2025 IOE INNOVATION Team
License:       [LICENSE_TYPE]

Notes:         - Use for pretrain, after training, use file best.pt as model YOLO (AI)
               - SHOULD USE COLAB OR KAGGLE NOTES TO TRAIN INSTEAD OF ON TERMINAL
*******************************************************************************************************************
"""

#######################################################################################################################
# Imports
#######################################################################################################################
# Standard library imports
from ultralytics import YOLO
import os, shutil


#######################################################################################################################
# Constants and Configuration
#######################################################################################################################
# Configuration constants
RUNS_DIR = "/content/Project2/main_code/runs/detect"
TRAIN_DIR = os.path.join(RUNS_DIR, "train")
DATASET_YAML = "/content/Project2/big_dataset/my_dataset.yaml"

#######################################################################################################################
# Module
#######################################################################################################################
def delete_old_train():
    # Xoá thư mục train cũ (nếu có)
    if os.path.exists(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
        print(f"Deleted folder: {TRAIN_DIR}")


#######################################################################################################################
# Main Classes
#######################################################################################################################
def main_function():
    # Load model YOLOv12
    model = YOLO("yolo12n.pt")

    # Train
    results = model.train(
        data=DATASET_YAML,  #link to dataset
        epochs=30,
        imgsz=640,      #Giảm về 512 nếu train trực tiếp trên laptop
        batch=4,        # có thể tăng batch size vì GPU Colab mạnh hơn
        workers=2,
        device=0,
        cache=True
    )


#######################################################################################################################
# Main Execution
#######################################################################################################################
if __name__ == "__main__":
    delete_old_train()
    main_function()

# End of File

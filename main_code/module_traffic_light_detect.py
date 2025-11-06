"""
*******************************************************************************************************************
General Information
********************************************************************************************************************
Project:       DETECT LICENSE PLATES AND TRAFFIC PENALTY INTERGRATION
File:          TRAFFIC_LIGHT_DETECT.py
Description:   This file is used to test model YOLO after training to detect traffic light.

Author:        LARRY PHONG TRUC
Email:         vanphongtruc1808@gmail.com
Created:       26/10/2025
Last Update:   26/10
Version:       1.0

Python:        3.10.11
Dependencies:  - Model YOLOv12 trained for detect light traffic
               - Test image/video

Copyright:     (c) 2025 IOE INNOVATION Team
License:       [LICENSE_TYPE]

Notes:         - Just for test, may use for module
*******************************************************************************************************************
"""

#######################################################################################################################
# Imports
#######################################################################################################################
# Standard library imports
import os
from ultralytics import YOLO
from PIL import Image


#######################################################################################################################
# Constants and Configuration
#######################################################################################################################
# Configuration constants
YOLO_PATH = r"./runs/detect/traffic_light/weights/best.pt"
TEST_IMAGE = "D:\VSCode\DCLP/big_dataset/test\light.png"
SAVE_PATH = "D:\VSCode\DCLP\main_code/result/vehicle_detect"


#######################################################################################################################
# Main Classes
#######################################################################################################################
#Load trained model YOLOv12 after epoched - choose the best.pt
def main_function():
    model = YOLO(YOLO_PATH)
    result = model(TEST_IMAGE)

    #Print the result to screen
    for r in result:
        print(r.boxes)
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1]) #RGB PIL image
        im.show()
        #save_path = os.path.join(SAVE_PATH, 'result.jpg')
        #im.save(save_path)


#######################################################################################################################
# Main Execution
#######################################################################################################################
if __name__ == "__main__":
    main_function()

# End of File
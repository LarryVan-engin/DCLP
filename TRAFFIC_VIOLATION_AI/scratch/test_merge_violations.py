"""
Test script to verify merging of multiple violations, directory renaming, and MongoDB path updates.
"""
import asyncio
import os
import sys
import shutil
import base64
import cv2
import numpy as np

# Thêm đường dẫn dự án
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'server')))

from server.api_main import process_violation, violations_col, VIOLATION_DIR

# Create a sample frame for base64
dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(dummy_img, (10, 10), (90, 90), (0, 0, 255), 2)
_, buf = cv2.imencode(".jpg", dummy_img)
dummy_b64 = base64.b64encode(buf).decode()

# Mock packets
packet1 = {
    "camera_id": "TEST_CAM_01",
    "timestamp": "2026-05-22T12:00:00.000Z",
    "violation_type": "VƯỢT ĐÈN ĐỎ",
    "confidence": 0.85,
    "track_id": 999,
    "mode": "video",
    "vehicle_crop_base64": dummy_b64,
    "vehicle_crop_wide_base64": dummy_b64,
    "full_frame_a_base64": dummy_b64,
    "full_frame_b_base64": dummy_b64
}

packet2 = {
    "camera_id": "TEST_CAM_01",
    "timestamp": "2026-05-22T12:00:05.000Z",
    "violation_type": "SAI LÀN",
    "confidence": 0.92,
    "track_id": 999,
    "mode": "video",
    "vehicle_crop_base64": dummy_b64,
    "vehicle_crop_wide_base64": dummy_b64,
    "full_frame_a_base64": dummy_b64,
    "full_frame_b_base64": dummy_b64
}

async def run_test():
    print("[TEST] 1. Cleaning any old test violations from MongoDB...")
    await violations_col.delete_many({"camera_id": "TEST_CAM_01", "track_id": 999})
    
    # Clean up test directories
    for folder in os.listdir(VIOLATION_DIR):
        if folder.startswith("ID999_"):
            shutil.rmtree(os.path.join(VIOLATION_DIR, folder))
            print(f"[TEST] Cleaned up folder: {folder}")
            
    print("[TEST] 2. Processing first violation packet ('VƯỢT ĐÈN ĐỎ')...")
    await process_violation(packet1)
    
    # Verify DB
    doc1 = await violations_col.find_one({"camera_id": "TEST_CAM_01", "track_id": 999})
    assert doc1 is not None, "First violation was not saved to DB"
    assert doc1["violation_type"] == "VƯỢT ĐÈN ĐỎ", f"Expected 'VƯỢT ĐÈN ĐỎ', got {doc1['violation_type']}"
    folder_path1 = doc1["image_folder"]
    print(f"[TEST] First folder path: {folder_path1}")
    assert os.path.exists(folder_path1), "First violation folder does not exist"
    assert os.path.exists(doc1["vehicle_img_path"]), "Vehicle crop does not exist"
    
    print("[TEST] 3. Processing second violation packet ('SAI LÀN')...")
    await process_violation(packet2)
    
    # Verify DB again - should be updated (not duplicated)
    count = await violations_col.count_documents({"camera_id": "TEST_CAM_01", "track_id": 999})
    assert count == 1, f"Expected exactly 1 document in DB, found {count}"
    
    doc2 = await violations_col.find_one({"camera_id": "TEST_CAM_01", "track_id": 999})
    print(f"[TEST] Combined violation type: {doc2['violation_type']}")
    assert doc2["violation_type"] == "VƯỢT ĐÈN ĐỎ+SAI LÀN", f"Expected merged violations, got: {doc2['violation_type']}"
    assert doc2["confidence"] == 0.92, f"Expected updated confidence to be 0.92, got: {doc2['confidence']}"
    
    # Verify directory rename
    folder_path2 = doc2["image_folder"]
    print(f"[TEST] New folder path: {folder_path2}")
    assert "VUOT_DEN_DO+SAI_LAN" in folder_path2 or "VƯỢT_ĐÈN_ĐỎ+SAI_LÀN" in folder_path2, "New folder path doesn't contain both violations"
    assert os.path.exists(folder_path2), "Renamed folder path does not physically exist"
    assert not os.path.exists(folder_path1), "Old folder path should have been renamed/deleted"
    
    # Check that paths in DB match the renamed folder
    for path_key in ["vehicle_img_path", "vehicle_wide_img_path", "full_frame_a_path", "full_frame_b_path"]:
        path_val = doc2[path_key]
        print(f"[TEST] Checking path {path_key}: {path_val}")
        assert path_val.startswith(folder_path2), f"Path {path_key} ({path_val}) does not start with new folder path ({folder_path2})"
        assert os.path.exists(path_val), f"Physical file at {path_val} does not exist!"

    print("[TEST] ✅ ALL MERGING AND RENAMING TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(run_test())

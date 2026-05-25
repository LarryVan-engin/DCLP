import os
import cv2
import json
import easyocr
import argparse
from tqdm import tqdm

def normalize_gt(text):
    if text is None: return ""
    return text.replace(" ", "").replace("-", "").replace(".", "").upper()

def split_gt_string(gt, top_pred, bot_pred):
    """
    Sử dụng kết quả OCR nháp để quyết định cách chia chuỗi GT.
    Mục đích: Nếu GT là 8 ký tự (VD: 51G51936), ta không biết là 3/5 hay 4/4.
    Ta dùng độ dài của top_pred để nội suy.
    """
    gt = normalize_gt(gt)
    L = len(gt)
    
    # Simple cases
    if L <= 6:
        # Lỗi GT? Chia đôi bừa
        return gt[:3], gt[3:]
    if L == 7:
        # Chắc chắn 3/4
        return gt[:3], gt[3:]
    if L == 9:
        # Chắc chắn 4/5
        return gt[:4], gt[4:]
        
    # L == 8 (có thể là 3/5 hoặc 4/4)
    # Loại bỏ các khoảng trắng trong dự đoán
    top_pred = normalize_gt(top_pred)
    
    if len(top_pred) >= 4:
        return gt[:4], gt[4:]
    else:
        return gt[:3], gt[3:]

from ultralytics import YOLO

# Khởi tạo YOLO model giống evaluate_ocr.py
try:
    plate_model = YOLO("runs/detect/train/weights/best.pt")
except Exception as e:
    print(f"Warning: Cannot load YOLO model: {e}")
    plate_model = None

def get_plate_crop(img):
    if plate_model is None:
        return img
    results = plate_model(img, verbose=False)
    best_det = None
    for r in results:
        for box in r.boxes:
            if best_det is None or box.conf[0] > best_det.conf[0]:
                best_det = box
    
    if best_det is not None:
        x1, y1, x2, y2 = map(int, best_det.xyxy[0].cpu().numpy())
        # Thêm padding giống module_utils
        h, w = img.shape[:2]
        pad = int(0.18 * (y2 - y1))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        return img[y1:y2, x1:x2]
    return img

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True, help="Thư mục chứa ảnh gốc")
    parser.add_argument("--gt", required=True, help="File ground_truth.txt")
    parser.add_argument("--out_dir", default="dataset_easyocr", help="Thư mục lưu dataset")
    args = parser.parse_args()

    out_images = os.path.join(args.out_dir, "images")
    os.makedirs(out_images, exist_ok=True)
    
    label_file_path = os.path.join(args.out_dir, "labels.txt")
    
    # Load GT
    gt_data = {}
    with open(args.gt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                gt_data[parts[0]] = parts[1]
                
    reader = easyocr.Reader(['en'])
    
    print(f"Total GT loaded: {len(gt_data)}")
    
    with open(label_file_path, "w", encoding="utf-8") as out_f:
        for filename, gt_text in tqdm(gt_data.items()):
            img_path = os.path.join(args.img_dir, filename)
            if not os.path.exists(img_path):
                continue
                
            full_img = cv2.imread(img_path)
            if full_img is None: continue
            
            # Dùng YOLO cắt biển số ra trước
            img = get_plate_crop(full_img)
            
            # Khử nhiễu nhẹ
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            h, w = img.shape[:2]
            ratio = w / h
            
            # Phân loại biển 1 dòng hay 2 dòng dựa trên tỷ lệ
            if ratio > 2.0:
                # Biển 1 dòng
                out_name = f"1L_{filename}"
                out_path = os.path.join(out_images, out_name)
                cv2.imwrite(out_path, gray)
                
                norm_gt = normalize_gt(gt_text)
                out_f.write(f"{out_name}\t{norm_gt}\n")
            else:
                # Biển 2 dòng
                split = int(h * 0.48)
                top = gray[:split, :]
                bottom = gray[split:, :]
                
                # Chạy OCR nháp để lấy gợi ý chia dòng
                res_top = reader.readtext(top, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                res_bot = reader.readtext(bottom, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                
                top_text = res_top[0] if res_top else ""
                bot_text = res_bot[0] if res_bot else ""
                
                gt_top, gt_bot = split_gt_string(gt_text, top_text, bot_text)
                
                # Save top
                top_name = f"2L_top_{filename}"
                cv2.imwrite(os.path.join(out_images, top_name), top)
                out_f.write(f"{top_name}\t{gt_top}\n")
                
                # Save bottom
                bot_name = f"2L_bot_{filename}"
                cv2.imwrite(os.path.join(out_images, bot_name), bottom)
                out_f.write(f"{bot_name}\t{gt_bot}\n")

    print(f"Dataset generated at: {args.out_dir}")

if __name__ == "__main__":
    run()

import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import os

# Import từ source code của bạn
from model import SCNN
from utils.prob2lines import getLane

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN 
# ==========================================
MODEL_WEIGHTS = r'exp10_best.pth'  # Thay bằng 'exp10/best.pth' hoặc đường dẫn model của bạn
IMAGE_PATH = r'E:\Video\train\Demo_image_redpassing.png'           # Bức ảnh bạn muốn test
INPUT_SIZE = (800, 288)                 # (Width, Height) của CULane theo định nghĩa trong model
# ==========================================

def preprocess_image(img_bgr):
    """Tiền xử lý ảnh giống hệt như lúc train/test CULane"""
    # ImageNet mean, std (lấy từ test_CULane.py)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((INPUT_SIZE[1], INPUT_SIZE[0])), # PyTorch Resize dùng (H, W) -> (288, 800)
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0) # Thêm chiều Batch (1, C, H, W)
    return img_tensor

def draw_lanes(original_img, lane_coords):
    """Vẽ các làn đường lên ảnh gốc"""
    img_draw = original_img.copy()
    # Định nghĩa 4 màu cho 4 làn đường (Xanh lá, Đỏ, Xanh dương, Vàng)
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
    
    for i, lane in enumerate(lane_coords):
        if len(lane) == 0:
            continue
            
        color = colors[i % len(colors)]
        pts = np.array(lane, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Vẽ đường nối các điểm
        cv2.polylines(img_draw, [pts], isClosed=False, color=color, thickness=5)
        
        # (Tùy chọn) Vẽ các điểm chấm tròn
        for pt in lane:
            cv2.circle(img_draw, (int(pt[0]), int(pt[1])), 5, color, -1)
            
    return img_draw

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Lỗi: Không tìm thấy ảnh tại {IMAGE_PATH}")
        return
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Lỗi: Không tìm thấy file weights tại {MODEL_WEIGHTS}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên thiết bị: {device}")

    # 1. Khởi tạo Model
    print("Đang load model...")
    net = SCNN(input_size=INPUT_SIZE, pretrained=False)
    
    # Xử lý trường hợp model được lưu bằng DataParallel (thường có tiền tố 'module.')
    state_dict = torch.load(MODEL_WEIGHTS, map_location=device)
    if 'net' in state_dict: # Giống format của test_CULane.py
        state_dict = state_dict['net']
        
    # Xóa 'module.' nếu có
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    net.load_state_dict(new_state_dict)
    net.to(device)
    net.eval()
    print("Load model thành công!\n")

    # 2. Đọc và Tiền xử lý ảnh
    img = cv2.imread(IMAGE_PATH)
    original_h, original_w = img.shape[:2]
    img_tensor = preprocess_image(img).to(device)

    # 3. Chạy Suy luận (Inference)
    print("Đang phân tích hình ảnh...")
    start_time = time.time()
    with torch.no_grad():
        seg_pred, exist_pred = net(img_tensor)[:2]
        
        # Post-processing giống hệt test_CULane.py
        seg_pred = F.softmax(seg_pred, dim=1)
        seg_pred = seg_pred.detach().cpu().numpy()[0] # Lấy batch đầu tiên
        exist_pred = exist_pred.detach().cpu().numpy()[0]
        
    inference_time = time.time() - start_time

    # 4. Giải mã kết quả (Decoding)
    # Xác định làn nào tồn tại (ngưỡng > 0.5)
    exist = [1 if exist_pred[i] > 0.5 else 0 for i in range(4)]
    
    # Chuyển đổi probability map thành tọa độ x, y trên kích thước ảnh gốc
    # Hàm prob2lines_CULane tự động nội suy tọa độ về resize_shape
    lane_coords = getLane.prob2lines_CULane(
        seg_pred, 
        exist, 
        resize_shape=(original_h, original_w), # Truyền kích thước ảnh gốc của bạn vào đây
        y_px_gap=20, 
        pts=18
    )

    # 5. Trực quan hóa và Lưu kết quả
    result_img = draw_lanes(img, lane_coords)
    
    print(f"Hoàn thành! Thời gian xử lý: {inference_time*1000:.2f} ms")
    print(f"Số làn đường phát hiện được: {sum(exist)}")

    # Hiển thị ảnh
    cv2.imshow('SCNN Lane Detection Demo', result_img)
    
    # Lưu ảnh ra file
    output_filename = 'result_' + os.path.basename(IMAGE_PATH)
    cv2.imwrite(output_filename, result_img)
    print(f"Đã lưu kết quả tại: {output_filename}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
"""
🧩 1. Mục đích của code

Thuật toán SORT không tự phát hiện (detect) đối tượng, mà nhận đầu vào là các bounding box từ mô hình detector như YOLO, Faster R-CNN, CenterNet,…, sau đó:

🔹 Gán ID và theo dõi liên tục các đối tượng giữa các khung hình (frames).

Tức là:

YOLO cung cấp:
→ [x1, y1, x2, y2, confidence] ở từng frame

SORT nhận vào các bbox đó
→ Trả ra: [x1, y1, x2, y2, ID] để biết đối tượng nào là ai theo thời gian.

⚙️ 2. Kiến trúc của SORT

SORT có 3 phần chính:

Kalman Filter
    Dự đoán vị trí tiếp theo của bbox khi không có detection (motion model)	
        filterpy.kalman
IOU Association	
    So khớp detection mới với các tracker đang tồn tại dựa trên IOU (Intersection over Union)	
        iou_batch()
Hungarian Algorithm	
    Tối ưu việc gán detection ↔ tracker	
        scipy.optimize.linear_sum_assignment hoặc lap.lapjv

🔍 3. Giải thích luồng hoạt động của mã chính
3.1. KalmanBoxTracker
Là một đối tượng theo dõi đơn lẻ (1 người, 1 xe, …)

    self.kf = KalmanFilter(dim_x=7, dim_z=4)
→ Mô hình chuyển động 7 trạng thái: [x, y, s, r, vx, vy, vs]
trong đó:

    x, y: tâm bbox

    s: diện tích bbox

    r: tỉ lệ khung (aspect ratio)

    vx, vy, vs: vận tốc ẩn

    Hàm predict() dự đoán vị trí tiếp theo

    Hàm update() cập nhật vị trí thật dựa trên detection mới

3.2. associate_detections_to_trackers

Tính IOU giữa tất cả detections và trackers

Gán chúng lại với nhau dựa trên ngưỡng IOU threshold

Nếu không có detection nào khớp → tracker đó sẽ sống thêm vài frame (max_age) trước khi bị xóa.

3.3. Sort class

Quản lý nhiều KalmanBoxTracker.

Trong mỗi frame:

Gọi predict() cho tất cả tracker để dự đoán vị trí mới.

Dùng associate_detections_to_trackers() để gán detection mới vào các tracker.

Nếu detection mới mà không có tracker khớp → tạo tracker mới.

Nếu tracker không có detection lâu quá (max_age) → xóa đi.

Trả ra danh sách bbox kèm ID.

3.4. Chương trình chính (if __name__ == '__main__':)

Đọc file .txt chứa các detection có sẵn (thường từ YOLO hoặc MOT dataset)

    frame, id, x, y, w, h, score, ...

Mỗi frame gọi mot_tracker.update(dets)

Xuất kết quả output/seq.txt gồm:

    frame,id,x,y,w,h,1,-1,-1,-1

Tùy chọn --display để hiển thị bằng matplotlib.

⚙️ 5. Tham số quan trọng
max_age	:
    Số frame mà tracker "sống sót" khi mất detection	1–5
min_hits :	
    Số lần detection liên tiếp để xác nhận tracker mới	1–3
iou_threshold :	
    Ngưỡng IOU để coi là cùng một đối tượng	            0.3–0.5
"""

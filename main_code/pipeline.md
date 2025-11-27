```mermaid
Camera (Full Frame)
        │
        ▼
+-------------------+
| Nhận diện đèn giao|
| thông (YOLO nhỏ / |
| classifier riêng) |
+-------------------+
        │
        ├── Đèn xanh → Không xử lý
        │
        └── Đèn vàng/đỏ → Kích hoạt xử lý phương tiện
                          │
                          ▼
                +-------------------+
                | Crop frame theo   |
                | polygon (ROI)     |
                +-------------------+
                          │
                          ▼
                +-------------------+
                | YOLOv12 detect    |
                | phương tiện trong |
                | ROI               |
                +-------------------+
                          │
                          ▼
                +-------------------+
                | DeepSORT tracking |
                | ID xe trong ROI   |
                +-------------------+
                          │
                          ▼
                +-------------------+
                | EasyOCR đọc biển  |
                | số nếu xe đi vào  |
                | ROI khi đèn đỏ    |
                +-------------------+
                          │
                          ▼
                +-------------------+
                | Logic kiểm tra    |
                | vi phạm           |
                +-------------------+
                          │
                          ▼
                +-------------------+
                | Overlay kết quả   |
                | lên full frame GUI|
                +-------------------+
```
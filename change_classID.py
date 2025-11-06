import os

folder = "D:/VSCode/DCLP/big_dataset/LP_detection/labels/val"

for filename in os.listdir(folder):
    if not filename.endswith(".txt"):
        continue

    path = os.path.join(folder, filename)
    new_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            # Ví dụ: đổi class 0 → 6
            if cls_id == 0:
                parts[0] = "6"
            new_lines.append(" ".join(parts))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

print("Toàn bộ file txt trong thư mục đã được chỉnh sửa!")

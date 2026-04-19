#!/bin/bash
# ================================================
# BACKUP DATABASE & DATA HỆ THỐNG TRAFFIC AI
# Author: LARRY PHONG TRUC
# Version: 1.2 - 2026

# ================================================
# CHÚ Ý:
# 1. Cần cài đặt MongoDB Database Tools để sử dụng mongodump
# 2. Cách sử dụng:
# cd scripts
# chmod +x backup_db.sh
# Sau đó chạy: ./backup_db.sh
# ================================================

set -e  # Dừng script nếu có lỗi

# ====================== CONFIG ======================
BACKUP_DIR="./backups"
DATE=$(date +"%Y-%m-%d_%H-%M-%S")
BACKUP_NAME="traffic_ai_backup_${DATE}"

# MongoDB Atlas Config (thay bằng thông tin thật của anh)
MONGODB_URI="mongodb+srv://admin:admin123@cluster0.teleibk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME="traffic_violation_db"

# Local paths
LOCAL_DB_DIR="../server/database"
VIOLATIONS_DIR="../server/violations"
UPLOADS_DIR="../server/uploads"

# Tạo thư mục backup
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"
cd "${BACKUP_DIR}/${BACKUP_NAME}" || exit

echo "========================================"
echo "🚀 BẮT ĐẦU BACKUP HỆ THỐNG - ${DATE}"
echo "========================================"

# 1. Backup MongoDB Atlas
echo "📦 Đang backup MongoDB Atlas..."
mongodump --uri="${MONGODB_URI}" --db="${DB_NAME}" --out="./mongodb" --gzip

if [ $? -eq 0 ]; then
    echo "✅ Backup MongoDB thành công"
else
    echo "❌ Lỗi backup MongoDB!"
fi

# 2. Backup Local Database (CSV + owners)
echo "📄 Đang backup Local Database..."
cp -r "${LOCAL_DB_DIR}" ./local_database/

# 3. Backup Violations (ảnh + dữ liệu)
echo "📸 Đang backup Violations..."
cp -r "${VIOLATIONS_DIR}" ./violations_backup/

# 4. Backup Uploads (video gốc)
echo "📤 Đang backup Uploads folder..."
cp -r "${UPLOADS_DIR}" ./uploads_backup/ 2>/dev/null || echo "⚠️ Uploads folder trống hoặc không tồn tại"

# 5. Tạo file metadata
cat > backup_info.txt << EOF
BACKUP DATE: ${DATE}
SYSTEM: Traffic AI Edge-Server
DATABASE: ${DB_NAME}
MONGODB: Yes
LOCAL_CSV: Yes
VIOLATIONS: Yes
CREATED_BY: backup_db.sh
EOF

# 6. Nén toàn bộ backup
echo "🗜️ Đang nén backup..."
cd ..
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"

# 7. Dọn dẹp
rm -rf "${BACKUP_NAME}"

echo "========================================"
echo "🎉 BACKUP HOÀN TẤT!"
echo "📁 File backup: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo "========================================"

# Optional: Xóa backup cũ hơn 30 ngày
find . -name "*.tar.gz" -mtime +30 -exec rm {} \;

exit 0
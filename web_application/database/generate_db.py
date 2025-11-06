import random
import csv
from datetime import datetime, timedelta

# === 63 TỈNH THÀNH VIỆT NAM ===
vietnam_provinces = [
    ("11", "Hà Nội"), ("12", "Hà Giang"), ("14", "Cao Bằng"), ("15", "Bắc Kạn"),
    ("17", "Tuyên Quang"), ("18", "Lào Cai"), ("19", "Điện Biên"), ("20", "Lai Châu"),
    ("21", "Phú Thọ"), ("22", "Yên Bái"), ("23", "Sơn La"), ("24", "Hòa Bình"),
    ("25", "Thái Nguyên"), ("26", "Lạng Sơn"), ("27", "Bắc Giang"), ("28", "Phú Thọ"),
    ("29", "Bắc Ninh"), ("30", "Hải Dương"), ("31", "Hải Phòng"), ("33", "Hưng Yên"),
    ("34", "Thái Bình"), ("35", "Hà Nam"), ("36", "Nam Định"), ("37", "Ninh Bình"),
    ("38", "Thanh Hóa"), ("40", "Nghệ An"), ("41", "Hà Tĩnh"), ("42", "Quảng Bình"),
    ("43", "Quảng Trị"), ("44", "Thừa Thiên Huế"), ("45", "Đà Nẵng"), ("46", "Quảng Nam"),
    ("47", "Quảng Ngãi"), ("48", "Bình Định"), ("49", "Phú Yên"), ("50", "Khánh Hòa"),
    ("51", "Ninh Thuận"), ("52", "Bình Thuận"), ("53", "Kon Tum"), ("54", "Gia Lai"),
    ("55", "Đắk Lắk"), ("56", "Đắk Nông"), ("57", "Lâm Đồng"), ("58", "Bình Phước"),
    ("59", "Tây Ninh"), ("60", "Bình Dương"), ("61", "Đồng Nai"), ("62", "Bà Rịa - Vũng Tàu"),
    ("63", "TP. Hồ Chí Minh"), ("64", "Long An"), ("65", "Tiền Giang"), ("66", "Bến Tre"),
    ("67", "Trà Vinh"), ("68", "Vĩnh Long"), ("69", "Đồng Tháp"), ("70", "An Giang"),
    ("71", "Kiên Giang"), ("72", "Cần Thơ"), ("73", "Hậu Giang"), ("74", "Sóc Trăng"),
    ("75", "Bạc Liêu"), ("76", "Cà Mau")
]

province_codes = [code for code, _ in vietnam_provinces]
province_names = [name for _, name in vietnam_provinces]

# === LOẠI XE ===
vehicle_classes = [
    "Xe máy", "Ô tô con", "Ô tô tải", "Ô tô khách", "Xe bán tải", "Xe chuyên dụng"
]

# === CHỦ XE ===
owners = [
    "Nguyễn Văn A", "Trần Thị B", "Lê Văn C", "Phạm Thị D", "Hoàng Văn E", "Vũ Thị F", "Đặng Văn G", "Bùi Thị H",
    "Ngô Văn I", "Đỗ Thị J", "Lý Văn K", "Mai Thị L", "Đinh Văn M", "Hồ Thị N", "Võ Văn O", "Phan Thị P",
    "Nguyễn Văn Q", "Trần Văn R", "Lê Thị S", "Phạm Văn T", "Hoàng Thị U", "Vũ Văn V", "Đặng Thị W",
    "Bùi Văn X", "Ngô Thị Y", "Đỗ Văn Z", "Lý Thị AA", "Mai Văn BB", "Đinh Thị CC", "Hồ Văn DD", "Võ Thị EE",
    "Phan Văn FF", "Nguyễn Thị GG", "Trần Văn HH", "Lê Thị II", "Phạm Văn JJ", "Hoàng Thị KK", "Vũ Văn LL",
    "Đặng Thị MM", "Bùi Văn NN", "Ngô Thị OO", "Đỗ Văn PP", "Lý Thị QQ", "Mai Văn RR", "Đinh Thị SS", "Hồ Văn TT",
    "Võ Thị UU", "Phan Văn VV", "Nguyễn Thị WW", "Trần Văn XX", "Lê Thị YY", "Phạm Văn ZZ"
]

# === SỐ ĐIỆN THOẠI ===
phone_prefixes = [
    "090", "091", "092", "093", "094", "095", "096", "097", "098", "099",
    "032", "033", "034", "035", "036", "037", "038", "039",
    "070", "076", "077", "078", "079",
    "088", "081", "082", "083", "084", "085", "086", "087"
]
phones = [prefix + ''.join(random.choices('0123456789', k=7)) for prefix in phone_prefixes for _ in range(10)]

# === TẠO CCCD 12 SỐ ===
def generate_id_card():
    province_id = random.choice([
        "001", "002", "004", "006", "008", "010", "011", "012", "014", "015",
        "017", "019", "020", "022", "024", "025", "026", "027", "030", "031",
        "033", "034", "035", "036", "037", "038", "040", "042", "043", "044",
        "045", "046", "047", "048", "049", "051", "052", "054", "056", "058",
        "060", "062", "064", "066", "067", "068", "070", "072", "074", "075", "077"
    ])
    gender_century = str(random.randint(0, 9))
    birth_year = random.randint(0, 99)
    birth_year_str = f"{birth_year:02d}"
    random_part = ''.join(random.choices('0123456789', k=6))
    return province_id + gender_century + birth_year_str + random_part

# === TẠO NGÀY ĐĂNG KÝ ===
def generate_registration_date():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 11, 5)
    random_days = random.randint(0, (end_date - start_date).days)
    return (start_date + timedelta(days=random_days)).strftime("%d/%m/%Y")

# === TẠO BIỂN SỐ – CHUẨN 100% VIỆT NAM ===
def generate_plate(province_code, class_vehicle):
    if class_vehicle == "Xe chuyên dụng":
        # TH-18-17
        prefix = "TH"
        part1 = f"{random.randint(10, 99):02d}"
        part2 = f"{random.randint(10, 99):02d}"
        return f"{prefix}-{part1}-{part2}"

    elif class_vehicle == "Xe máy":
        # 63-T1.4567 hoặc 29-A1.12345
        if random.random() < 0.6:  # 60% là 4 số sau dấu chấm
            letters = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + random.choice('123456789')  # T1, A1,...
            numbers = f"{random.randint(1000, 9999)}"
            return f"{province_code}-{letters}.{numbers}"
        else:  # 40% là 5 số
            letters = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + random.choice('123456789')
            numbers = f"{random.randint(10000, 99999)}"
            return f"{province_code}-{letters}.{numbers}"

    else:  # Ô tô con, tải, khách, bán tải
        # 30E-1234 hoặc 51D-56789
        letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if random.random() < 0.7:  # 70% là 4 số
            numbers = f"{random.randint(1000, 9999)}"
        else:  # 30% là 5 số
            numbers = f"{random.randint(10000, 99999)}"
        return f"{province_code}{letter}-{numbers}"

# === TẠO 1000 BẢN GHI ===
data = []
for i in range(1000):
    province_code = random.choice(province_codes)
    province_name = province_names[province_codes.index(province_code)]
    class_vehicle = random.choice(vehicle_classes)
    plate = generate_plate(province_code, class_vehicle)
    owner = random.choice(owners)
    phone = random.choice(phones)
    id_card = generate_id_card()
    registration_date = generate_registration_date()
    
    data.append([plate, owner, phone, class_vehicle, province_name, registration_date, id_card])

# === GHI FILE CSV ===
with open('owners_sample.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['plate', 'owner', 'phone', 'class_vehicle', 'province', 'registration_date', 'id_card'])
    writer.writerows(data)

print("Đã tạo thành công 'owners_sample.csv' – 1000 bản ghi")
print("Biển số ĐÚNG CHUẨN Việt Nam:")
print("   Xe máy: 63-T1.4567, 29-A1.12345")
print("   Ô tô: 30E-1234, 51D-56789")
print("   Xe chuyên dụng: TH-18-17")
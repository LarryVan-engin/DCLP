/**
 * ********************************************************************************************************************
 * Project:      Traffic Violation Detection (Pro Version - MQTT Hybrid)
 * File:         server/static/app.js
 * Description:  Logic điều khiển Dashboard, kết nối WebSocket và cập nhật UI Realtime.
 * ********************************************************************************************************************
 */

let socket;
let stage, layer;
let forbidenShape;
let violationHistory = {};
const streamImg = document.getElementById('mjpeg-stream');
const streamPlaceholder = document.getElementById('stream-placeholder');
const connectionBadge = document.getElementById('connection-status');
const systemLog = document.getElementById('system-log');

// ==========================================
// 1. KẾT NỐI WEBSOCKET
// ==========================================
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        logSystem("Đã kết nối thành công tới Server Dashboard.");
        connectionBadge.className = "badge bg-success me-3";
        connectionBadge.innerText = "MQTT: Connected";
    };

    socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "realtime_update") {
            handleRealtimeStream(message);
        } else if (message.type === "violation") {
            violationHistory.push(message.data); // Lưu vào bộ nhớ tạm
            handleNewViolation(message.data);
        } else if (message.type === "file_list") {
            updateVideoSelect(message.files);
        }
    };

    socket.onclose = () => {
        logSystem("Mất kết nối tới Server. Đang thử lại sau 3 giây...");
        connectionBadge.className = "badge bg-danger me-3";
        connectionBadge.innerText = "MQTT: Disconnected";
        setTimeout(connectWebSocket, 3000);
    };
}

// ==========================================
// 2. XỬ LÝ LUỒNG STREAM & STATS
// ==========================================
// Cập nhật danh sách video từ Edge
function updateVideoSelect(files) {
    const select = document.getElementById('video-file-select');
    select.innerHTML = ""; // Xóa cũ
    files.forEach(file => {
        let opt = document.createElement('option');
        opt.value = file;
        opt.text = f;
        // opt.innerHTML = file;
        select.appendChild(opt);
    });
    logSystem(`Đã cập nhật danh sách ${files.length} video từ Edge.`);
}

function handleRealtimeStream(data) {
    // Cập nhật Frame ảnh (Base64)
    if (data.stream) {
        streamPlaceholder.style.display = 'none';
        streamImg.style.display = 'block';
        streamImg.src = `data:image/jpeg;base64,${data.stream}`;
    }

    // Cập nhật Thống kê (Lấy từ Heartbeat của Edge đang chọn)
    const activeCamera = document.getElementById('camera-select').value;
    const heartbeat = data.heartbeats[activeCamera];

    if (heartbeat) {
        document.getElementById('stat-car').innerText = heartbeat.stats.car || 0;
        document.getElementById('stat-motorcycle').innerText = heartbeat.stats.motorcycle || 0;
        document.getElementById('fps-counter').innerText = `FPS: ${heartbeat.fps}`;

        // Cập nhật Badge đèn giao thông
        const lightBadge = document.getElementById('stat-light');
        const lightColor = heartbeat.lights.straight.toLowerCase();
        lightBadge.innerText = lightColor.toUpperCase();
        
        if (lightColor === 'red') lightBadge.className = "badge bg-danger";
        else if (lightColor === 'green') lightBadge.className = "badge bg-success";
        else if (lightColor === 'yellow') lightBadge.className = "badge bg-warning text-dark";
    }
}

// ==========================================
// 3. XỬ LÝ VI PHẠM MỚI
// ==========================================
function handleNewViolation(violation) {
    const list = document.getElementById('violation-list');
    
    // Xóa dòng "Đang chờ dữ liệu" nếu có
    if (list.innerHTML.includes("Đang chờ dữ liệu")) list.innerHTML = "";

    const card = document.createElement('div');
    card.className = 'violation-card pointer-cursor';
    card.onclick = () => showViolationDetail(violation.track_id); // Click để xem chi tiết

    card.innerHTML = `
        <div class="d-flex justify-content-between">
            <span class="violation-type">${violation.violation_type}</span>
            <span class="violation-time">${new Date(violation.timestamp).toLocaleTimeString()}</span>
        </div>
        <div class="violation-plate">${violation.plate_read || 'CHƯA ĐỌC ĐƯỢC BIỂN SỐ'}</div>
        <div class="small text-muted mb-1">ID Xe: ${violation.track_id} | Chủ xe: ${violation.owner || 'N/A'}</div>
        <img src="data:image/jpeg;base64,${violation.vehicle_crop_base64}" alt="Violation Crop">
    `;

    // Chèn lên đầu danh sách
    list.prepend(card);
    logSystem(`🚨 Phát hiện vi phạm: ${violation.violation_type} - Biển số: ${violation.plate_read}`);
}

function openViolationModal(trackId) {
    const v = violationHistory[trackId];
    if (!v) return;

    document.getElementById('modal-id').innerText = `#${trackId}`;
    document.getElementById('modal-type').innerText = v.violation_type;
    document.getElementById('modal-plate-text').innerText = v.plate_read || "CHƯA ĐỌC ĐƯỢC";
    document.getElementById('modal-owner').innerText = v.owner || "Không có dữ liệu";
    document.getElementById('modal-time').innerText = v.timestamp;
    document.getElementById('modal-camera').innerText = v.camera_id;
    document.getElementById('modal-img-context').src = `data:image/jpeg;base64,${v.vehicle_crop_base64}`;
    document.getElementById('modal-img-plate').src = `data:image/jpeg;base64,${v.plate_img_base64 || ''}`;

    new bootstrap.Modal(document.getElementById('violationModal')).show();
}

// Khởi tạo Canvas Konva
function initROICanvas() {
    const container = document.getElementById('roi-canvas');
    stage = new Konva.Stage({
        container: 'roi-canvas',
        width: container.offsetWidth,
        height: container.offsetHeight
    });

    layer = new Konva.Layer();
    stage.add(layer);

    // Mặc định tạo 1 ROI hình thang (Lấy từ logic Auto-ROI của Larry Van)
    createDraggablePolygon('roi_lane', [
        { x: 200, y: 150 }, { x: 440, y: 150 }, 
        { x: 550, y: 400 }, { x: 50, y: 400 }
    ], '#00ff00');

    // Tạo 1 vùng ĐƯỜNG CẤM (Màu đỏ)
    createDraggablePolygon('forbidden_zone', [
        { x: 450, y: 100 }, { x: 600, y: 100 }, 
        { x: 600, y: 250 }, { x: 450, y: 250 }
    ], '#ff0000');
}

// Hàm tạo đa giác có thể kéo dãn các góc
function createDraggablePolygon(id, points, color) {
    const poly = new Konva.Line({
        points: points.flatMap(p => [p.x, p.y]),
        fill: color + '33', // Trong suốt 20%
        stroke: color,
        strokeWidth: 2,
        closed: true,
        name: 'poly-' + id,
        id: id
    });

    layer.add(poly);

    // Tạo các điểm neo (Anchors) ở mỗi góc
    points.forEach((p, index) => {
        const anchor = new Konva.Circle({
            x: p.x, y: p.y,
            radius: 6,
            fill: '#ffffff',
            stroke: color,
            strokeWidth: 2,
            draggable: true,
            name: `anchor-${id}-${index}`
        });

        anchor.on('dragmove', function() {
            const newPoints = poly.points().slice();
            newPoints[index * 2] = anchor.x();
            newPoints[index * 2 + 1] = anchor.y();
            poly.points(newPoints);
            layer.draw();
        });

        layer.add(anchor);
    });

    layer.draw();
}

// Ẩn/Hiện lớp vẽ ROI
function toggleROILayer() {
    layer.visible(!layer.visible());
    layer.draw();
}

function toggleForbiddenMode() {
    // Lấy trạng thái của nút Switch
    const chkForbidden = document.getElementById('chk-forbidden');
    const isForbiddenMode = chkForbidden.checked;
    
    // Tìm các đối tượng trên Layer (dựa trên ID đã đặt ở bước trước)
    const roiShape = layer.findOne('#roi_lane');
    const forbiddenShape = layer.findOne('#forbidden_zone');
    
    // Tìm tất cả các điểm neo (anchors) để ẩn/hiện theo vùng
    const roiAnchors = layer.find('.anchor-roi_lane');
    const forbiddenAnchors = layer.find('.anchor-forbidden_zone');

    if (isForbiddenMode) {
        // CHẾ ĐỘ ĐƯỜNG CẤM: Hiện đỏ, ẩn xanh
        if (roiShape) roiShape.visible(false);
        if (forbiddenShape) forbiddenShape.visible(true);
        
        roiAnchors.forEach(a => a.visible(false));
        forbiddenAnchors.forEach(a => a.visible(true));
        
        logSystem("⚠️ Đã chuyển sang chế độ GIÁM SÁT ĐƯỜNG CẤM.");
    } else {
        // CHẾ ĐỘ BÌNH THƯỜNG: Hiện xanh, ẩn đỏ
        if (roiShape) roiShape.visible(true);
        if (forbiddenShape) forbiddenShape.visible(false);
        
        roiAnchors.forEach(a => a.visible(true));
        forbiddenAnchors.forEach(a => a.visible(false));
        
        logSystem("ℹ️ Đã chuyển về chế độ ĐƯỜNG BÌNH THƯỜNG.");
    }
    
    layer.draw();
}

// LƯU VÀ GỬI XUỐNG EDGE
function saveROI() {
    // 1. Lấy tọa độ từ các đa giác đang vẽ trên Konva
    const roiPoly = layer.findOne('#roi_lane');
    const forbiddenPoly = layer.findOne('#forbidden_zone');

    let polygonsToSend = [];

    // Nếu người dùng đang bật/vẽ vùng cấm, chỉ gửi vùng cấm
    if (forbiddenPoly && forbiddenPoly.visible()) {
        polygonsToSend.push({
            label: "forbidden",
            points: formatPoints(forbiddenPoly.points())
        });
    } else if (roiPoly) {
        // Ngược lại gửi vùng ROI bình thường
        polygonsToSend.push({
            label: "roi_lane",
            points: formatPoints(roiPoly.points())
        });
    }

    const config = {
        action: "update_zones",
        mode: currentMode, // 'realtime' hoặc 'video'
        polygons: polygonsToSend,
        lines: getStopLinesFromROI() // Tự động lấy cạnh trên làm vạch dừng
    };

    sendControlToEdge(config);
    logSystem("Cấu hình đã được phê duyệt và đồng bộ xuống Edge.");
}

function formatPoints(rawPoints) {
    let pts = [];
    for (let i = 0; i < rawPoints.length; i += 2) {
        pts.push({ x: Math.round(rawPoints[i]), y: Math.round(rawPoints[i+1]) });
    }
    return pts;
}

// Yêu cầu danh sách file khi đổi Camera
document.getElementById('camera-select').onchange = function() {
    const camId = this.value;
    logSystem(`Đang yêu cầu danh sách video từ ${camId}...`);
    // Gửi lệnh list_files qua Server
    fetch(`/api/refresh_videos/${camId}`);
};

// ==========================================
// 4. ĐIỀU KHIỂN (SEND COMMAND)
// ==========================================
async function sendControl(action, mode = 'realtime') {
    const cameraId = document.getElementById('camera-select').value;
    const videoName = document.getElementById('video-file-select').value;

    const payload = {
        action: action,
        mode: mode,
        video_name: (mode === 'video') ? videoName : null,
        lines: [], // Bạn có thể tích hợp Konva.js để vẽ tọa độ ở đây
        polygons: [],
        light_zones: []
    };

    try {
        const response = await fetch(`/api/control_edge?camera_id=${cameraId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            logSystem(`Lệnh [${action.toUpperCase()}] đã gửi tới ${cameraId} thành công.`);
            if (action === 'stop') {
                streamImg.style.display = 'none';
                streamPlaceholder.style.display = 'block';
            }
        }
    } catch (error) {
        logSystem(`Lỗi gửi lệnh: ${error}`);
    }
}

async function refreshEdgeFiles() {
    const cameraId = document.getElementById('camera-select').value;
    logSystem(`Đang yêu cầu danh sách file từ ${cameraId}...`);
    
    // Server sẽ trung chuyển lệnh này qua MQTT
    const res = await fetch(`/api/refresh_videos/${cameraId}`);
    if (res.ok) logSystem("Lệnh quét file đã được gửi.");
}

// Tiện ích Log
function logSystem(msg) {
    const now = new Date().toLocaleTimeString();
    systemLog.innerHTML += `<div>> [${now}] ${msg}</div>`;
    systemLog.scrollTop = systemLog.scrollHeight;
}

// Khởi tạo khi trang web load xong
document.addEventListener('DOMContentLoaded', connectWebSocket);
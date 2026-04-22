/**
 * ********************************************************************************************************************
 * Project:      Traffic Violation Detection (Pro Version - MQTT Hybrid)
 * File:         server/static/app.js
 * Description:  Logic điều khiển Dashboard, kết nối WebSocket và cập nhật UI Realtime (Tích hợp Konva Editable ROI).
 * ********************************************************************************************************************
 */

let socket;
let stage, layer;
let violationHistory = {}; // Chuyển thành Object để dễ truy xuất theo trackId
const streamImg = document.getElementById('mjpeg-stream');
const streamPlaceholder = document.getElementById('stream-placeholder');
const connectionBadge = document.getElementById('connection-status');
const systemLog = document.getElementById('system-log');

// Khởi tạo ngay khi web load
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    initKonva(); // Khởi tạo Canvas để sẵn sàng hứng dữ liệu vẽ
});

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
            // Lưu đệm thông tin lỗi để mở Modal
            violationHistory[message.data.track_id] = message.data; 
            handleNewViolation(message.data);
        } else if (message.type === "file_list") {
            updateVideoSelect(message.files);
        } else if (message.type === "auto_roi_proposal") {
            // Nhận tọa độ đề xuất từ AI và vẽ lên màn hình để chỉnh sửa
            loadAutoROI(message.points);
            logSystem("🤖 AI đã đề xuất vùng giám sát (Auto-ROI). Hãy kéo thả để tinh chỉnh.");
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
function updateVideoSelect(files) {
    const select = document.getElementById('video-file-select');
    select.innerHTML = "";
    files.forEach(file => {
        let opt = document.createElement('option');
        opt.value = file;
        opt.text = file;
        select.appendChild(opt);
    });
    logSystem(`Đã cập nhật danh sách ${files.length} video từ Edge.`);
}

function handleRealtimeStream(data) {
    if (data.stream) {
        streamPlaceholder.style.display = 'none';
        streamImg.style.display = 'block';
        streamImg.src = `data:image/jpeg;base64,${data.stream}`;
    }

    const activeCamera = document.getElementById('camera-select').value;
    const heartbeat = data.heartbeats && data.heartbeats[activeCamera];

    if (heartbeat) {
        document.getElementById('stat-car').innerText = heartbeat.stats.car || 0;
        document.getElementById('stat-motorcycle').innerText = heartbeat.stats.motorcycle || 0;
        document.getElementById('fps-counter').innerText = `FPS: ${heartbeat.fps}`;

        const lightBadge = document.getElementById('stat-light');
        const lightColor = heartbeat.lights.straight.toLowerCase();
        lightBadge.innerText = lightColor.toUpperCase();
        
        if (lightColor === 'red') lightBadge.className = "badge bg-danger";
        else if (lightColor === 'green') lightBadge.className = "badge bg-success";
        else if (lightColor === 'yellow') lightBadge.className = "badge bg-warning text-dark";
    }
}

// ==========================================
// 3. XỬ LÝ VI PHẠM & MODAL
// ==========================================
function handleNewViolation(violation) {
    const list = document.getElementById('violation-list');
    if (list.innerHTML.includes("Đang chờ dữ liệu")) list.innerHTML = "";

    const card = document.createElement('div');
    card.className = 'violation-card pointer-cursor';
    // Đã gọi đúng ID để mở popup Modal
    card.onclick = () => openViolationModal(violation.track_id); 

    card.innerHTML = `
        <div class="d-flex justify-content-between">
            <span class="violation-type">${violation.violation_type}</span>
            <span class="violation-time">${new Date(violation.timestamp).toLocaleTimeString()}</span>
        </div>
        <div class="violation-plate">${violation.plate_read || 'CHƯA ĐỌC ĐƯỢC BIỂN SỐ'}</div>
        <div class="small text-muted mb-1">ID Xe: ${violation.track_id} | Chủ xe: ${violation.owner || 'N/A'}</div>
        <img src="data:image/jpeg;base64,${violation.vehicle_crop_base64}" alt="Violation Crop">
    `;
    list.prepend(card);
    logSystem(`🚨 Phát hiện vi phạm: ${violation.violation_type} - Biển số: ${violation.plate_read}`);
}

function openViolationModal(trackId) {
    const v = violationHistory[trackId];
    if (!v) return;

    document.getElementById('modal-id').innerText = `#${trackId}`;
    document.getElementById('modal-type').innerText = v.violation_type;
    document.getElementById('modal-plate-text').innerText = v.plate_read || "CHƯA ĐỌC ĐƯỢC";
    document.getElementById('modal-owner').innerText = v.owner;
    document.getElementById('m-phone').innerText = v.phone;
    document.getElementById('m-class').innerText = v.class_vehicle; // Lấy đúng "Xe chuyên dụng"
    document.getElementById('m-province').innerText = v.province;
    document.getElementById('m-date').innerText = v.registration_date;
    document.getElementById('m-id').innerText = v.id_card; // Khớp với cột id_card trong CSV
    document.getElementById('modal-time').innerText = new Date(v.timestamp).toLocaleTimeString();
    document.getElementById('modal-camera').innerText = v.camera_id || "JETSON_01";

    document.getElementById('modal-img-context').src = `data:image/jpeg;base64,${v.vehicle_crop_base64}`;
    if (v.plate_img_base64) {
        document.getElementById('modal-img-plate').src = `data:image/jpeg;base64,${v.plate_img_base64}`;
    }

    const modalElement = document.getElementById('violationModal');
    const bsModal = bootstrap.Modal.getInstance(modalElement) || new bootstrap.Modal(modalElement);
    bsModal.show();
}

function closeViolationModal() {
    document.getElementById('violation-modal').classList.add('hidden');
}

// ==========================================
// 2. KONVA.JS - TINH CHỈNH AUTO-ROI (KHÔNG VẼ TỰ DO)
// ==========================================
function initKonva() {
    const container = document.getElementById('canvas-container');
    stage = new Konva.Stage({
        container: 'canvas-container',
        width: container.offsetWidth || 800,
        height: container.offsetHeight || 450
    });
    layer = new Konva.Layer();
    stage.add(layer);
    
    // Tự động scale Canvas khi resize màn hình
    const resizeObserver = new ResizeObserver(() => {
        if(streamImg.clientWidth > 0) {
            stage.width(streamImg.clientWidth);
            stage.height(streamImg.clientHeight);
        }
    });
    resizeObserver.observe(streamImg);
}

// Xóa tất cả và load ROI do AI đề xuất
function loadAutoROI(normalizedPointsArray) {
    layer.destroyChildren(); // Xóa sạch hình cũ
    
    const w = stage.width();
    const h = stage.height();
    
    // Đổi tọa độ 0.0->1.0 thành Pixel trên trình duyệt
    const absolutePoints = normalizedPointsArray.map(p => ({ x: p.x * w, y: p.y * h }));
    
    // Tạo vùng Đường Bình Thường (Xanh) - Mặc định
    createDraggablePolygon('roi_lane', absolutePoints, '#22c55e');
    
    // Tạo sẵn vùng Đường Cấm (Đỏ) ẩn đi - Dùng chung tọa độ đề xuất ban đầu
    createDraggablePolygon('forbidden_zone', absolutePoints, '#ef4444');
    
    toggleForbiddenMode(); // Ẩn hiện đúng theo nút Switch
}

function createDraggablePolygon(id, points, color) {
    const poly = new Konva.Line({
        points: points.flatMap(p => [p.x, p.y]),
        fill: color + '33', // Trong suốt 20%
        stroke: color,
        strokeWidth: 3,
        closed: true,
        id: id
    });
    layer.add(poly);

    // CHỈ CHO PHÉP KÉO THẢ CÁC GÓC (TINH CHỈNH)
    points.forEach((p, index) => {
        const anchor = new Konva.Circle({
            x: p.x, y: p.y,
            radius: 7,
            fill: '#ffffff',
            stroke: color,
            strokeWidth: 2,
            draggable: true,
            name: `anchor-${id}`
        });

        anchor.on('dragmove', function() {
            const newPoints = poly.points().slice();
            newPoints[index * 2] = anchor.x();
            newPoints[index * 2 + 1] = anchor.y();
            poly.points(newPoints);
            layer.draw();
        });

        anchor.on('mouseover', () => document.body.style.cursor = 'grab');
        anchor.on('mousedown', () => document.body.style.cursor = 'grabbing');
        anchor.on('mouseup mouseout', () => document.body.style.cursor = 'default');

        layer.add(anchor);
    });
    layer.draw();
}

// ==========================================
// 3. ĐÓNG GÓI JSON VÀ GỬI XUỐNG SERVER
// ==========================================
function toggleForbiddenMode() {
    const isForbidden = document.getElementById('chk-forbidden') ? document.getElementById('chk-forbidden').checked : false;
    
    const roiShape = layer.findOne('#roi_lane');
    const forbiddenShape = layer.findOne('#forbidden_zone');
    
    if (roiShape) roiShape.visible(!isForbidden);
    if (forbiddenShape) forbiddenShape.visible(isForbidden);
    
    layer.find('.anchor-roi_lane').forEach(a => a.visible(!isForbidden));
    layer.find('.anchor-forbidden_zone').forEach(a => a.visible(isForbidden));
    
    layer.draw();
}

function toggleROILayer() {
    if(layer) {
        layer.visible(!layer.visible());
        layer.draw();
        logSystem(layer.visible() ? "👁️ Đã hiển thị khung ROI." : "🙈 Đã ẩn khung ROI.");
    }
}

function resetROI() {
    // Gửi lệnh yêu cầu AI ở Edge học lại ROI từ đầu
    const cameraId = document.getElementById('camera-select').value;
    fetch(`/api/control_edge?camera_id=${cameraId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: "reset_roi" })
    }).then(() => logSystem("🔄 Đã yêu cầu AI học lại ROI từ đầu."));
}

function saveROI() {
    const isForbidden = document.getElementById('chk-forbidden') ? document.getElementById('chk-forbidden').checked : false;
    const targetId = isForbidden ? '#forbidden_zone' : '#roi_lane';
    const poly = layer.findOne(targetId);

    if (!poly) return alert("❌ Chưa có vùng ROI nào do AI đề xuất!");

    const rawPoints = poly.points();
    const w = stage.width();
    const h = stage.height();
    
    // Quy đổi lại tỷ lệ 0.0 -> 1.0 kèm định dạng chuẩn {x: val, y: val} cho Python
    let normalizedPoints = [];
    for (let i = 0; i < rawPoints.length; i += 2) {
        normalizedPoints.push({
            x: Number((rawPoints[i] / w).toFixed(4)),
            y: Number((rawPoints[i+1] / h).toFixed(4))
        });
    }

    // Đóng gói JSON chính xác như Python mong muốn
    const payload = {
        action: "update_zones",
        polygons: [{
            label: isForbidden ? "forbidden" : "roi_lane",
            points: normalizedPoints
        }],
        lines: [{
            label: "stop_line",
            points: [normalizedPoints[0], normalizedPoints[1]] // Hai điểm đầu tiên làm vạch dừng
        }]
    };

    fetch(`/api/control_edge?camera_id=JETSON_NANO_01`, { // Thay bằng biến camera thực tế nếu có
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    }).then(() => {
        logSystem(`✅ Đã đồng bộ ROI mới [${isForbidden ? 'ĐƯỜNG CẤM' : 'BÌNH THƯỜNG'}] xuống Edge AI.`);
        alert("Lưu cấu hình thành công!");
    });
}

// ==========================================
// 5. ĐIỀU KHIỂN & TIỆN ÍCH
// ==========================================
async function sendControl(action, mode = 'realtime') {
    const cameraId = document.getElementById('camera-select').value;
    const videoName = document.getElementById('video-file-select').value;

    const payload = {
        action: action,
        mode: mode,
        video_name: (mode === 'video') ? videoName : null
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
    const res = await fetch(`/api/refresh_videos/${cameraId}`);
    if (res.ok) logSystem("Lệnh quét file đã được gửi.");
}

function logSystem(msg) {
    const now = new Date().toLocaleTimeString();
    systemLog.innerHTML += `<div>> [${now}] ${msg}</div>`;
    systemLog.scrollTop = systemLog.scrollHeight;
}
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
let pendingVideoStart = null;
let lastStartPayload = null;
let isStopped = false;
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
        } else if (message.type === "camera_list") {
            updateCameraSelect(message.cameras);
        } else if (message.type === "auto_roi_proposal") {
            // Nhận tọa độ đề xuất từ AI và vẽ lên màn hình để chỉnh sửa
            loadAutoROI(message.points);
            const currentMode = pendingVideoStart ? pendingVideoStart.mode : "video";
            pendingVideoStart = {
                action: "start",
                mode: currentMode,
                video_name: message.video_name || document.getElementById('video-file-select').value
            };
            logSystem("🤖 AI đã đề xuất vùng giám sát (Auto-ROI). Hãy kéo thả để tinh chỉnh.");
        } else if (message.type === "video_ready") {
            logSystem(`✅ Video local đã xử lý xong. Thời gian Inference: ${message.processing_time}s`);
            streamImg.style.display = 'none';
            streamPlaceholder.style.display = 'none';
            
            let videoPlayer = document.getElementById('local-video-player');
            if (!videoPlayer) {
                videoPlayer = document.createElement('video');
                videoPlayer.id = 'local-video-player';
                videoPlayer.controls = true;
                videoPlayer.autoplay = true;
                videoPlayer.style.width = '100%';
                videoPlayer.style.height = '100%';
                videoPlayer.style.objectFit = 'contain';
                document.getElementById('stream-container').appendChild(videoPlayer);
            }
            videoPlayer.style.display = 'block';
            videoPlayer.src = message.video_url;
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
function updateCameraSelect(cameras) {
    const select = document.getElementById('camera-select');
    const currentValue = select.value;
    
    select.innerHTML = "";
    
    if (!cameras || cameras.length === 0) {
        let opt = document.createElement('option');
        opt.value = "";
        opt.disabled = true;
        opt.selected = true;
        opt.text = "⏳ Chờ kết nối từ Edge...";
        select.appendChild(opt);
        logSystem("⚠️ Chưa có Edge device kết nối. Vui lòng bật Jetson hoặc camera AI.");
    } else {
        cameras.forEach((camera, index) => {
            let opt = document.createElement('option');
            opt.value = camera.id;
            opt.text = `${camera.id} (${camera.location || 'Chưa cấu hình vị trí'})`;
            if (index === 0) opt.selected = true;
            select.appendChild(opt);
        });
        logSystem(`✅ Đã phát hiện ${cameras.length} camera Edge. Sẵn sàng để điều khiển.`);
    }
}

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
        const videoPlayer = document.getElementById('local-video-player');
        if (videoPlayer) videoPlayer.style.display = 'none';
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
        const lightColor = ((heartbeat.lights && heartbeat.lights.straight) || "unknown").toLowerCase();
        lightBadge.innerText = lightColor.toUpperCase();
        
        if (lightColor === 'red') lightBadge.className = "badge bg-danger";
        else if (lightColor === 'green') lightBadge.className = "badge bg-success";
        else if (lightColor === 'yellow') lightBadge.className = "badge bg-warning text-dark";
        else lightBadge.className = "badge bg-secondary";
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

    const setText = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.innerText = value || "N/A";
    };

    setText('modal-id', `#${trackId}`);
    setText('modal-type', v.violation_type);
    setText('modal-plate-text', v.plate_read || "CHUA DOC DUOC");
    setText('modal-owner', v.owner);
    setText('modal-phone', v.phone);
    setText('modal-class', v.class_vehicle);
    setText('modal-province', v.province);
    setText('modal-registration-date', v.registration_date);
    setText('modal-id-card', v.id_card);
    setText('modal-time', new Date(v.timestamp).toLocaleTimeString());
    setText('modal-camera', v.camera_id || "JETSON_01");

    document.getElementById('modal-img-context').src = `data:image/jpeg;base64,${v.vehicle_crop_base64}`;
    const plateImg = document.getElementById('modal-img-plate');
    if (v.plate_img_base64) {
        plateImg.style.display = 'inline-block';
        plateImg.src = `data:image/jpeg;base64,${v.plate_img_base64}`;
    } else {
        plateImg.style.display = 'none';
        plateImg.removeAttribute('src');
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
    const container = document.getElementById('roi-canvas');
    stage = new Konva.Stage({
        container: 'roi-canvas',
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

function syncStageToStream() {
    if (!stage) return;
    const canvas = document.getElementById('roi-canvas');
    const streamContainer = document.getElementById('stream-container');
    const streamBox = streamContainer.getBoundingClientRect();
    const imgBox = streamImg.getBoundingClientRect();
    const boxWidth = imgBox.width || streamBox.width || 800;
    const boxHeight = imgBox.height || streamBox.height || 450;
    const naturalWidth = streamImg.naturalWidth || 640;
    const naturalHeight = streamImg.naturalHeight || 360;
    const imgAspect = naturalWidth / Math.max(naturalHeight, 1);
    const boxAspect = boxWidth / Math.max(boxHeight, 1);

    let width = boxWidth;
    let height = boxHeight;
    let left = imgBox.left - streamBox.left;
    let top = imgBox.top - streamBox.top;

    if (boxAspect > imgAspect) {
        width = boxHeight * imgAspect;
        left += (boxWidth - width) / 2;
    } else {
        height = boxWidth / imgAspect;
        top += (boxHeight - height) / 2;
    }

    canvas.style.left = `${left}px`;
    canvas.style.top = `${top}px`;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    stage.width(width);
    stage.height(height);
    layer.draw();
}

streamImg.onload = syncStageToStream;

// Xóa tất cả và load ROI do AI đề xuất
function loadAutoROI(normalizedPointsArray) {
    syncStageToStream();
    const canvas = document.getElementById('roi-canvas');
    if (canvas) canvas.style.display = 'block';
    if (layer) layer.visible(true);
    if (!normalizedPointsArray || normalizedPointsArray.length !== 4) {
        normalizedPointsArray = [
            { x: 0.1, y: 0.3 },
            { x: 0.9, y: 0.3 },
            { x: 1.0, y: 1.0 },
            { x: 0.0, y: 1.0 }
        ];
    }
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
    const canvas = document.getElementById('roi-canvas');
    if (!canvas || !layer) return;

    const nextVisible = canvas.style.display === 'none';
    canvas.style.display = nextVisible ? 'block' : 'none';
    layer.visible(nextVisible);
    layer.draw();
    logSystem(nextVisible ? "Da hien thi khung ROI." : "Da an khung ROI.");
}

function getCurrentROIPayload() {
    const isForbidden = document.getElementById('chk-forbidden') ? document.getElementById('chk-forbidden').checked : false;
    const targetId = isForbidden ? '#forbidden_zone' : '#roi_lane';
    const poly = layer && layer.findOne(targetId);

    if (!poly) return null;

    const rawPoints = poly.points();
    const w = stage.width();
    const h = stage.height();
    const normalizedPoints = [];

    for (let i = 0; i < rawPoints.length; i += 2) {
        normalizedPoints.push({
            x: Number((rawPoints[i] / w).toFixed(4)),
            y: Number((rawPoints[i + 1] / h).toFixed(4))
        });
    }

    return {
        roi: normalizedPoints.map(p => [p.x, p.y]),
        polygons: [{
            label: isForbidden ? "forbidden" : "roi_lane",
            points: normalizedPoints
        }],
        lines: [{
            label: "stop_line",
            points: [normalizedPoints[0], normalizedPoints[1]]
        }]
    };
}

function resetROI() {
    const cameraId = document.getElementById('camera-select').value;
    const selectedVideoName = document.getElementById('video-file-select').value;
    const videoName = (lastStartPayload && lastStartPayload.video_name) ||
        (pendingVideoStart && pendingVideoStart.video_name) ||
        selectedVideoName;
    const resetMode = (pendingVideoStart && (pendingVideoStart.mode === "video" || pendingVideoStart.mode === "video_local")) ||
        (lastStartPayload && (lastStartPayload.mode === "video" || lastStartPayload.mode === "video_local")) ? "video" : "realtime";

    if (layer) {
        layer.destroyChildren();
        layer.draw();
    }

    pendingVideoStart = resetMode === "video"
        ? { action: "start", mode: "video", video_name: videoName }
        : null;
    lastStartPayload = null;
    violationHistory = {};
    setStopButton(false);
    document.getElementById('violation-list').innerHTML =
        '<div class="text-center text-muted mt-5 small">Đang chờ dữ liệu...</div>';
    document.getElementById('stat-car').innerText = "0";
    document.getElementById('stat-motorcycle').innerText = "0";
    document.getElementById('fps-counter').innerText = "FPS: 0";

    fetch(`/api/control_edge?camera_id=${cameraId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: "reset_roi", mode: resetMode, video_name: videoName })
    }).then(() => {
        if (resetMode === "video") {
            logSystem("Da dung video dang xu ly, xoa buffer ROI cu va quay lai buoc preview ROI.");
        } else {
            logSystem("Da xoa ROI cu va reset ROI ve mac dinh.");
        }
    }).catch((error) => logSystem(`Loi reset ROI: ${error}`));
}

function setStopButton(stopped) {
    const btn = document.getElementById('stop-continue-btn');
    if (!btn) return;
    isStopped = stopped;
    btn.innerText = stopped ? "TIẾP TỤC" : "DỪNG TẤT CẢ";
    btn.className = stopped ? "btn btn-success btn-sm w-100 border-2 fw-bold" : "btn btn-outline-dark btn-sm w-100 border-2 fw-bold";
}

function saveROI() {
    const cameraId = document.getElementById('camera-select').value;
    const roiPayload = getCurrentROIPayload();

    if (!roiPayload) return alert("Chua co ROI de luu. Hay chay preview video truoc.");

    const payload = pendingVideoStart
        ? { ...pendingVideoStart, ...roiPayload }
        : { action: "update_zones", mode: "realtime", ...roiPayload };

    fetch(`/api/control_edge?camera_id=${cameraId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    }).then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        lastStartPayload = payload.action === "start" ? payload : lastStartPayload;
        pendingVideoStart = null;
        setStopButton(false);
        logSystem("Da luu ROI va ap dung xuong Edge.");
        alert("Luu cau hinh thanh cong!");
    }).catch((error) => logSystem(`Loi luu ROI: ${error}`));
}

async function toggleStopContinue() {
    if (isStopped) {
        if (lastStartPayload) {
            try {
                await sendPayload(lastStartPayload);
                setStopButton(false);
                logSystem("Da gui lenh tiep tuc xu ly.");
            } catch (error) {
                logSystem(`Loi gui lenh tiep tuc: ${error}`);
            }
        } else {
            await sendControl("start", "realtime");
        }
    } else {
        await sendControl("stop");
    }
}

// ==========================================
// 5. ĐIỀU KHIỂN & TIỆN ÍCH
// ==========================================
async function sendPayload(payload) {
    const cameraId = document.getElementById('camera-select').value;
    const response = await fetch(`/api/control_edge?camera_id=${cameraId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response;
}

async function sendControl(action, mode = 'realtime') {
    const cameraId = document.getElementById('camera-select').value;
    const videoName = document.getElementById('video-file-select').value;

    const payload = {
        action: action,
        mode: mode,
        video_name: (mode === 'video' || mode === 'video_local') ? videoName : null
    };

    if (action === 'start' && (mode === 'video' || mode === 'video_local')) {
        payload.action = 'preview_video';
        pendingVideoStart = { action: 'start', mode: 'video', video_name: videoName };
        lastStartPayload = null;
        setStopButton(false);
        if (layer) {
            layer.destroyChildren();
            layer.draw();
        }
        logSystem("Dang doc 5 frame dau de hieu chinh ROI truoc khi xu ly video...");
    }

    try {
        await sendPayload(payload);
        logSystem(`Lenh [${payload.action.toUpperCase()}] da gui toi ${cameraId} thanh cong.`);

        if (action === 'start' && mode !== 'video') {
            lastStartPayload = payload;
            setStopButton(false);
        }

        if (action === 'stop') {
            setStopButton(true);
            streamImg.style.display = 'none';
            const videoPlayer = document.getElementById('local-video-player');
            if (videoPlayer) {
                videoPlayer.pause();
                videoPlayer.style.display = 'none';
            }
            streamPlaceholder.style.display = 'block';
        }
    } catch (error) {
        logSystem(`Loi gui lenh: ${error}`);
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
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
let localVideoViolations = []; // Mảng chứa toàn bộ vi phạm của video xử lý local
let pendingVideoStart = null;
let lastStartPayload = null;
let isStopped = false;
let roiSaved = false; // Flag: true sau khi user đã Lưu ROI → ngăn auto_roi_proposal hiện lại canvas
const streamImg = document.getElementById('mjpeg-stream');
const streamPlaceholder = document.getElementById('stream-placeholder');
const connectionBadge = document.getElementById('connection-status');
const systemLog = document.getElementById('system-log');

// Khởi tạo ngay khi web load
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    initKonva(); // Khởi tạo Canvas để sẵn sàng hứng dữ liệu vẽ

    // Tự động clear dashboard state khi chuyển tab/mode
    const modeTabs = document.querySelectorAll('#modeTab button');
    modeTabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', (e) => {
            clearDashboardState();
        });
    });
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
            const data = message.data;
            const tid = data.track_id;

            if (data.mode === "video_local") {
                // Cập nhật/gộp lỗi cho các gói tin vi phạm local nhận từ MQTT
                const existingIndex = localVideoViolations.findIndex(v => v.track_id === tid);
                if (existingIndex !== -1) {
                    const existing = localVideoViolations[existingIndex].violation_type || "";
                    const incoming = data.violation_type || "";
                    const mergedSet = new Set(
                        [...existing.split("+"), ...incoming.split("+")]
                            .map(s => s.trim()).filter(Boolean)
                    );
                    data.violation_type = Array.from(mergedSet).join(" + ");
                    
                    // Giữ lại trạng thái shownOnPlayer cũ nếu có
                    data.shownOnPlayer = localVideoViolations[existingIndex].shownOnPlayer || false;
                    localVideoViolations[existingIndex] = data;
                } else {
                    data.shownOnPlayer = false;
                    localVideoViolations.push(data);
                }
            } else {
                if (violationHistory[tid]) {
                    // Cùng ID: merge violation_type (tránh trùng lặp)
                    const existing = violationHistory[tid].violation_type || "";
                    const incoming = data.violation_type || "";
                    const mergedSet = new Set(
                        [...existing.split("+"), ...incoming.split("+")]
                            .map(s => s.trim()).filter(Boolean)
                    );
                    data.violation_type = Array.from(mergedSet).join(" + ");
                }
                violationHistory[tid] = data;
                handleNewViolation(data);
            }
        } else if (message.type === "file_list") {
            updateVideoSelect(message.files);
        } else if (message.type === "camera_list") {
            updateCameraSelect(message.cameras);
        } else if (message.type === "auto_roi_proposal") {
            // Nhận tọa độ đề xuất từ AI và vẽ lên màn hình để chỉnh sửa
            // right_turn_zone_bottom_y: tỉ lệ [0-1] từ server (mặc định 0.7 nếu chưa có)
            const currentMode = pendingVideoStart ? pendingVideoStart.mode : "video";
            pendingVideoStart = {
                action: "start",
                mode: currentMode,
                video_name: message.video_name || document.getElementById('video-file-select').value
            };
            // Chỉ vẽ lên canvas nếu user chưa Lưu ROI (tránh chồng lấn 2 lớp)
            if (!roiSaved) {
                loadAutoROI(message.points, message.right_turn_zone_bottom_y ?? 0.15);
                logSystem("🤖 AI đã đề xuất vùng giám sát (Auto-ROI). Hãy kéo thả để tinh chỉnh.");
            }
        } else if (message.type === "video_ready") {
            logSystem(`✅ Video local đã xử lý xong. Thời gian Inference: ${message.processing_time}s`);
            streamImg.style.display = 'none';
            streamPlaceholder.style.display = 'none';

            // Ẩn ROI canvas (Konva stage position:absolute phủ toàn container)
            // để video kết quả không bị che khuất phía dưới.
            const roiCanvasEl = document.getElementById('roi-canvas');
            if (roiCanvasEl) roiCanvasEl.style.display = 'none';

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
            videoPlayer.muted = true; // MUST be set before src for autoplay to work
            videoPlayer.src = message.video_url;
            videoPlayer.load(); // Force reload so browser sees muted=true before playing

            // Remove any leftover play overlay from previous run
            const existingOverlay = document.getElementById('play-btn-overlay');
            if (existingOverlay) existingOverlay.remove();

            videoPlayer.play().then(() => {
                logSystem("▶️ Bắt đầu tự động phát video kết quả.");
            }).catch(err => {
                logSystem("⚠️ Trình duyệt chặn tự động phát. Nhấn nút ▶ để xem.");
                console.warn("Autoplay blocked:", err);

                const overlay = document.createElement('div');
                overlay.id = 'play-btn-overlay';
                overlay.style.cssText = [
                    'position:absolute', 'inset:0',
                    'display:flex', 'align-items:center', 'justify-content:center',
                    'background:rgba(0,0,0,0.45)', 'cursor:pointer', 'z-index:10'
                ].join(';');
                overlay.innerHTML = `
                    <div style="width:72px;height:72px;border-radius:50%;
                                background:rgba(255,255,255,0.92);
                                display:flex;align-items:center;justify-content:center;
                                box-shadow:0 4px 24px rgba(0,0,0,0.55);
                                transition:transform .12s">
                        <svg width="30" height="30" viewBox="0 0 24 24" fill="#111">
                            <path d="M8 5v14l11-7z"/>
                        </svg>
                    </div>`;
                overlay.onmouseenter = () => overlay.firstElementChild.style.transform = 'scale(1.1)';
                overlay.onmouseleave = () => overlay.firstElementChild.style.transform = '';
                overlay.onclick = () => {
                    videoPlayer.play().then(() => {
                        overlay.remove();
                        logSystem("▶️ Đang phát video kết quả.");
                    });
                };
                document.getElementById('stream-container').appendChild(overlay);
            });

            // Đính kèm sự kiện timeupdate để đồng bộ hiển thị vi phạm theo tiến trình video
            videoPlayer.ontimeupdate = () => {
                const currentTime = videoPlayer.currentTime;

                // 1. Quét xuôi: Hiển thị các vi phạm có video_offset <= currentTime mà chưa hiển thị
                localVideoViolations.forEach(v => {
                    if (v.video_offset !== null && v.video_offset <= currentTime) {
                        if (!v.shownOnPlayer) {
                            v.shownOnPlayer = true;
                            violationHistory[v.track_id] = v;
                            handleNewViolation(v);
                        }
                    }
                });

                // 2. Quét ngược (Tua ngược): Xoá các vi phạm có video_offset > currentTime khỏi DOM & history
                localVideoViolations.forEach(v => {
                    if (v.video_offset !== null && v.video_offset > currentTime) {
                        if (v.shownOnPlayer) {
                            v.shownOnPlayer = false;
                            delete violationHistory[v.track_id];
                            
                            // Xoá card khỏi DOM
                            const card = document.getElementById(`vcard-${v.track_id}`);
                            if (card) {
                                card.remove();
                            }
                        }
                    }
                });

                // Nếu không còn vi phạm nào được hiển thị, hiển thị lại dòng "Đang chờ dữ liệu..."
                const list = document.getElementById('violation-list');
                const activeCards = list.querySelectorAll('.violation-card');
                if (activeCards.length === 0 && !list.innerHTML.includes("Đang chờ dữ liệu")) {
                    list.innerHTML = '<div class="text-center text-muted mt-5 small">Đang chờ dữ liệu...</div>';
                }

                // 3. Tính toán lại thống kê (Ô tô, Xe máy) dựa trên các vi phạm đang hiển thị
                let carCount = 0;
                let motoCount = 0;
                const countedTracks = new Set();

                localVideoViolations.forEach(v => {
                    if (v.shownOnPlayer && !countedTracks.has(v.track_id)) {
                        countedTracks.add(v.track_id);
                        
                        // Xác định class xe để đếm
                        const vehicleClass = (v.class_vehicle || "").toLowerCase();
                        const isMoto = vehicleClass.includes("xe máy") || vehicleClass.includes("xe may") || 
                                       vehicleClass.includes("motorcycle") || vehicleClass.includes("moto");
                        
                        if (isMoto) {
                            motoCount++;
                        } else {
                            carCount++;
                        }
                    }
                });

                document.getElementById('stat-car').innerText = carCount;
                document.getElementById('stat-motorcycle').innerText = motoCount;
            };
        };
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
    const activeCamera = document.getElementById('camera-select').value;
    const heartbeat = data.heartbeats && data.heartbeats[activeCamera];

    if (heartbeat && heartbeat.mode === 'video_local') {
        streamImg.style.display = 'none';
        const videoPlayer = document.getElementById('local-video-player');
        const isShowingResult = videoPlayer && videoPlayer.style.display === 'block';
        // Không override nếu video kết quả đã sẵn sàng phát — heartbeat "đang xử lý"
        // tới mỗi ~1s và sẽ liên tục ẩn video nếu không có guard này.
        if (!isShowingResult) {
            if (videoPlayer) videoPlayer.style.display = 'none';
            streamPlaceholder.style.display = 'flex';
            streamPlaceholder.innerHTML = '<div class="text-center"><h4 class="text-warning fw-bold">🚀 ĐANG XỬ LÝ LOCAL TRÊN KIT</h4><p class="text-muted small">Edge đang phân tích cục bộ toàn bộ file video.<br>Màn hình stream bị tắt để tối đa hóa tài nguyên phần cứng.<br>Vui lòng theo dõi các thông số phân tích bên dưới.</p></div>';
        }
    } else if (data.stream) {
        const videoPlayer = document.getElementById('local-video-player');
        const isShowingResult = videoPlayer && videoPlayer.style.display === 'block';
        if (!isShowingResult) {
            streamPlaceholder.style.display = 'none';
            if (videoPlayer) videoPlayer.style.display = 'none';
            streamImg.style.display = 'block';
            streamImg.src = `data:image/jpeg;base64,${data.stream}`;
        }
    }

    if (heartbeat) {
        // Chỉ cập nhật thống kê (car, motorcycle) từ heartbeat khi local-video-player KHÔNG hiển thị và hoạt động
        const videoPlayer = document.getElementById('local-video-player');
        const isPlayingVideo = videoPlayer && videoPlayer.style.display === 'block';
        if (!isPlayingVideo) {
            document.getElementById('stat-car').innerText = heartbeat.stats.car || 0;
            document.getElementById('stat-motorcycle').innerText = heartbeat.stats.motorcycle || 0;
        }
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

    const cardId = `vcard-${violation.track_id}`;
    let card = document.getElementById(cardId);

    const cardHTML = `
        <div class="d-flex justify-content-between">
            <span class="violation-type">${violation.violation_type}</span>
            <span class="violation-time">${new Date(violation.timestamp).toLocaleTimeString()}</span>
        </div>
        <div class="violation-plate">${violation.plate_read || 'CHƯA ĐỌC ĐƯỢC BIỂN SỐ'}</div>
        <div class="small text-muted mb-1">ID Xe: ${violation.track_id} | Chủ xe: ${violation.owner || 'N/A'}</div>
        <img src="data:image/jpeg;base64,${violation.vehicle_crop_base64}" alt="Violation Crop">
    `;

    if (card) {
        // Cập nhật card hiện có (cùng track_id) thay vì tạo card mới
        card.innerHTML = cardHTML;
    } else {
        card = document.createElement('div');
        card.id = cardId;
        card.className = 'violation-card pointer-cursor';
        card.onclick = () => openViolationModal(violation.track_id);
        card.innerHTML = cardHTML;
        list.prepend(card);
    }
    logSystem(`🚨 Vi phạm ID ${violation.track_id}: ${violation.violation_type} - Biển: ${violation.plate_read || 'N/A'}`);
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
            syncStageToStream();
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
// rightTurnZoneY: tọa độ Y chuẩn hóa [0-1] của đường Right Turn Zone (mặc định 0.7)
function loadAutoROI(normalizedPointsArray, rightTurnZoneY = 0.15) {
    syncStageToStream();
    const canvas = document.getElementById('roi-canvas');
    if (canvas) canvas.style.display = 'block'; // Đảm bảo canvas không bị ẩn từ lần Save trước
    if (layer) layer.visible(true);
    if (!normalizedPointsArray || normalizedPointsArray.length !== 4) {
        normalizedPointsArray = [
            { x: 0.3146, y: 0.6241 }, // 604/1920
            { x: 0.7625, y: 0.6102 }, // 1464/1920
            { x: 0.9865, y: 0.9796 }, // 1894/1920
            { x: 0.1271, y: 0.9917 }  // 244/1920
        ];
    }
    layer.destroyChildren(); // Xóa sạch hình cũ

    const w = stage.width();
    const h = stage.height();

    // Đổi tọa độ 0.0->1.0 thành pixel trên trình duyệt
    const absolutePoints = normalizedPointsArray.map(p => ({ x: p.x * w, y: p.y * h }));

    // Vùng Đường Bình Thường (xanh lá) — mặc định hiển thị
    const isForbidden = document.getElementById('chk-forbidden') ? document.getElementById('chk-forbidden').checked : false;
    createDraggablePolygon('roi_lane', absolutePoints, '#22c55e', !isForbidden);

    // Vùng Đường Cấm (đỏ) — ẩn ngay khi tạo nếu không ở chế độ Đường Cấm
    createDraggablePolygon('forbidden_zone', absolutePoints, '#ef4444', isForbidden);

    // Đường Right Turn Zone (xanh neon) — kéo theo trục Y
    createRightTurnZoneLine(rightTurnZoneY * h);

    // Chỉ cần sync visibility anchors (không cần toggle nữa vì đã set từ đầu)
    layer.find('.anchor-roi_lane').forEach(a => a.visible(!isForbidden));
    layer.find('.anchor-forbidden_zone').forEach(a => a.visible(isForbidden));
    const rtzLine = layer.findOne('#rtz_line');
    const rtzLabel = layer.findOne('#rtz_label');
    const rtzOverlay = layer.findOne('#rtz_overlay');
    if (rtzLine)    rtzLine.visible(!isForbidden);
    if (rtzLabel)   rtzLabel.visible(!isForbidden);
    if (rtzOverlay) rtzOverlay.visible(!isForbidden);
    layer.batchDraw();
}

// Tạo vùng theo dõi rẽ phải: overlay màu + đường kẻ kéo được + label hint
// Tối ưu hiệu năng:
//   - listening:false  → bỏ qua hit-graph cho shape không tương tác (overlay, label)
//   - batchDraw()      → gom nhiều lần redraw vào 1 animation frame, tránh vẽ thừa
//   - Konva.Rect       → shape đơn giản nhất (O(1)), không dùng polygon phức tạp
function createRightTurnZoneLine(rtzY) {
    const w = stage.width();
    const h = stage.height();

    // --- Tính toán ranh giới ngang (X) của vùng rẽ phải ---
    // Lấy từ ROI polygon nếu có; khớp với right_turn_lane_min=0.65 mặc định trên edge
    const roiPoly = layer.findOne('#roi_lane');
    let stopLineY  = h * 0.3;   // fallback
    let zoneXStart = w * 0.65;  // fallback — 65% từ trái = phần làn bên phải
    if (roiPoly) {
        const pts = roiPoly.points(); // flat: [TL.x, TL.y, TR.x, TR.y, BR.x, BR.y, BL.x, BL.y]
        stopLineY  = Math.min(pts[1], pts[3]);                      // Y cạnh trên ROI
        zoneXStart = pts[0] + 0.65 * (pts[2] - pts[0]);            // 65% chiều rộng stop line
    }
    // Đảm bảo rtzY luôn trên stop line
    rtzY = Math.min(rtzY, stopLineY - 20);
    if (rtzY < 0) rtzY = 0;

    // --- Overlay bán trong suốt: vùng sẽ được giám sát rẽ phải ---
    // Sử dụng Polygon (Konva.Line) thay vì Rect để mép trên bám sát đường xéo của Stop Line
    const overlay = new Konva.Line({
        points: [
            zoneXStart, stopLineY, // Sẽ được tính lại chính xác trên đường Stop Line
            w, roiPoly ? roiPoly.points()[3] : stopLineY,
            w, rtzY,
            zoneXStart, rtzY
        ],
        fill: '#00e676',
        opacity: 0.12,
        listening: false,
        closed: true,
        id: 'rtz_overlay'
    });

    // --- Đường kẻ ngang: chỉ nửa phải frame (khớp với edge logic) ---
    const line = new Konva.Line({
        points: [zoneXStart, 0, w, 0],
        x: 0, y: rtzY,
        stroke: '#00e676',
        strokeWidth: 3,
        dash: [14, 6],
        id: 'rtz_line',
        draggable: true,
        dragBoundFunc: function(pos) {
            // Tính lại stopLineY động phòng khi ROI đã bị kéo
            const poly = layer.findOne('#roi_lane');
            let currentStopLineY = stopLineY;
            if (poly) {
                const pts = poly.points();
                currentStopLineY = Math.min(pts[1], pts[3]);
            }
            // Chỉ kéo theo trục Y, không vượt quá stop line hoặc cạnh trên (0)
            return { x: 0, y: Math.max(0, Math.min(currentStopLineY - 20, pos.y)) };
        }
    });

    // --- Label với hint kéo thả ---
    // listening:false → không tốn hit-test
    const label = new Konva.Text({
        x: zoneXStart + 8,
        y: rtzY - 18,
        text: '▶ RIGHT TURN ZONE  (kéo lên/xuống để điều chỉnh)',
        fill: '#00e676',
        fontSize: 11,
        fontStyle: 'bold',
        listening: false,
        id: 'rtz_label'
    });

    // Cursor hint khi hover vào đường kẻ
    line.on('mouseover', () => { document.body.style.cursor = 'ns-resize'; });
    line.on('mouseout',  () => { document.body.style.cursor = 'default'; });

    // Khi kéo: chỉ cập nhật 3 thuộc tính tối thiểu, dùng batchDraw để gom redraws
    line.on('dragmove', function () {
        const ny = line.y();
        const poly = layer.findOne('#roi_lane');
        let currentStopLineY = stopLineY;
        if (poly) {
            const pts = poly.points();
            currentStopLineY = Math.min(pts[1], pts[3]);
            
            // Cập nhật đáy của polygon overlay
            if (overlay.className === 'Line') {
                const overPts = overlay.points();
                overPts[5] = ny; // Y top right
                overPts[7] = ny; // Y top left
                overlay.points(overPts);
            }
        } else {
            overlay.y(ny);
            overlay.height(Math.max(0, currentStopLineY - ny)); // fallback nếu là Rect
        }
        label.y(ny - 18);
        layer.batchDraw(); // deferred — không vẽ ngay mà chờ frame tiếp theo
    });

    layer.add(overlay);
    layer.add(line);
    layer.add(label);
    layer.batchDraw();
}

function createDraggablePolygon(id, points, color, visible = true) {
    const poly = new Konva.Line({
        points: points.flatMap(p => [p.x, p.y]),
        fill: color + '33', // Trong suốt 20%
        stroke: color,
        strokeWidth: 3,
        closed: true,
        id: id,
        visible: visible
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
            name: `anchor-${id}`,
            visible: visible  // Anchor cũng ẩn/hiện cùng polygon
        });

        anchor.on('dragmove', function() {
            const newPoints = poly.points().slice();
            newPoints[index * 2]     = anchor.x();
            newPoints[index * 2 + 1] = anchor.y();
            poly.points(newPoints);
            
            // Cập nhật lại Right Turn Zone nếu đang kéo vùng ROI chính
            if (id === 'roi_lane') {
                const rtzOverlay = layer.findOne('#rtz_overlay');
                const rtzLine = layer.findOne('#rtz_line');
                const rtzLabel = layer.findOne('#rtz_label');
                
                if (rtzOverlay && rtzLine) {
                    const stopLineY = Math.min(newPoints[1], newPoints[3]);
                    const zoneXStart = newPoints[0] + 0.65 * (newPoints[2] - newPoints[0]);
                    const w = stage.width();
                    
                    let currentRtzY = rtzLine.y();
                    if (currentRtzY > stopLineY - 20) {
                        currentRtzY = stopLineY - 20;
                        if (currentRtzY < 0) currentRtzY = 0;
                        rtzLine.y(currentRtzY);
                        if (rtzLabel) rtzLabel.y(currentRtzY - 18);
                    }
                    rtzLine.points([zoneXStart, 0, w, 0]);
                    
                    // Cập nhật hình dáng overlay (Polygon)
                    if (rtzOverlay.className === 'Line') {
                        // Tính tọa độ Y của giao điểm giữa x=zoneXStart và đoạn thẳng Stop Line
                        const x1 = newPoints[0], y1 = newPoints[1];
                        const x2 = newPoints[2], y2 = newPoints[3];
                        let yAtZoneStart = y1;
                        if (x2 !== x1) {
                            yAtZoneStart = y1 + (y2 - y1) * (zoneXStart - x1) / (x2 - x1);
                        }
                        
                        rtzOverlay.points([
                            zoneXStart, yAtZoneStart,
                            w, y2,
                            w, currentRtzY,
                            zoneXStart, currentRtzY
                        ]);
                    } else {
                        // Cập nhật vị trí overlay nếu là Rect (fallback)
                        rtzOverlay.x(zoneXStart);
                        rtzOverlay.y(currentRtzY);
                        rtzOverlay.width(w - zoneXStart);
                        rtzOverlay.height(Math.max(0, stopLineY - currentRtzY));
                    }
                    
                    // Cập nhật lại nhãn
                    if (rtzLabel) rtzLabel.x(zoneXStart + 8);
                }
            }
            
            layer.batchDraw(); // gom redraw — tránh vẽ mỗi pixel kéo
        });

        anchor.on('mouseover', () => document.body.style.cursor = 'grab');
        anchor.on('mousedown', () => document.body.style.cursor = 'grabbing');
        anchor.on('mouseup mouseout', () => document.body.style.cursor = 'default');

        layer.add(anchor);
    });
    layer.batchDraw();
}

// ==========================================
// 3. ĐÓNG GÓI JSON VÀ GỬI XUỐNG SERVER
// ==========================================
function toggleForbiddenMode() {
    const isForbidden = document.getElementById('chk-forbidden') ? document.getElementById('chk-forbidden').checked : false;

    const roiShape       = layer.findOne('#roi_lane');
    const forbiddenShape = layer.findOne('#forbidden_zone');
    const rtzLine        = layer.findOne('#rtz_line');
    const rtzLabel       = layer.findOne('#rtz_label');
    const rtzOverlay     = layer.findOne('#rtz_overlay');  // overlay bán trong suốt

    if (roiShape)      roiShape.visible(!isForbidden);
    if (forbiddenShape) forbiddenShape.visible(isForbidden);
    // RTZ chỉ hiển thị ở chế độ bình thường (không phải đường cấm)
    if (rtzLine)    rtzLine.visible(!isForbidden);
    if (rtzLabel)   rtzLabel.visible(!isForbidden);
    if (rtzOverlay) rtzOverlay.visible(!isForbidden);

    layer.find('.anchor-roi_lane').forEach(a => a.visible(!isForbidden));
    layer.find('.anchor-forbidden_zone').forEach(a => a.visible(isForbidden));

    layer.batchDraw(); // gom redraw vào 1 frame
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

    // Lấy vị trí Y chuẩn hóa của đường Right Turn Zone
    // line.y() là vị trí pixel hiện tại (do dùng position thay vì points[1])
    const rtzLine = layer && layer.findOne('#rtz_line');
    const rightTurnZoneBottomY = rtzLine
        ? Number((rtzLine.y() / h).toFixed(4))
        : 0.15;   // fallback mặc định

    return {
        roi: normalizedPoints.map(p => [p.x, p.y]),
        polygons: [{
            label: isForbidden ? "forbidden" : "roi_lane",
            points: normalizedPoints
        }],
        lines: [{
            label: "stop_line",
            points: [normalizedPoints[0], normalizedPoints[1]]
        }],
        right_turn_zone_bottom_y: rightTurnZoneBottomY   // gửi xuống edge
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

    // Reset flag — sau khi reset, auto_roi_proposal được phép hiện canvas lại
    roiSaved = false;

    const canvas = document.getElementById('roi-canvas');
    if (canvas) canvas.style.display = 'block';

    if (layer) {
        layer.destroyChildren();
        layer.draw();
    }

    pendingVideoStart = resetMode === "video"
        ? { action: "start", mode: "video", video_name: videoName }
        : null;
    lastStartPayload = null;
    violationHistory = {};
    localVideoViolations = [];
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
    }).then(r => r.json()).then(res => {
        logSystem(`Da gui lenh ROI thanh cong: ${JSON.stringify(payload.action)}`);
        
        // Ẩn canvas và đánh dấu đã lưu — ngăn auto_roi_proposal hiện lại canvas
        const canvas = document.getElementById('roi-canvas');
        if (canvas) canvas.style.display = 'none';
        roiSaved = true;

        if (payload.action === "start") {
            lastStartPayload = payload;
            pendingVideoStart = null;
            logSystem(`Da start camera/video voi ROI moi.`);
        }
        setStopButton(false);
        alert("Luu cau hinh thanh cong!");
    }).catch(e => logSystem(`Loi Save ROI: ${e}`));
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
    if (!cameraId) {
        alert("Vui lòng chọn Camera trước khi điều khiển!");
        return;
    }
    const videoName = document.getElementById('video-file-select').value;

    const payload = {
        action: action,
        mode: mode,
        video_name: (mode === 'video' || mode === 'video_local') ? videoName : null
    };

    if (action === 'start' && (mode === 'video' || mode === 'video_local')) {
        payload.action = 'preview_video';
        pendingVideoStart = { action: 'start', mode: mode, video_name: videoName };
        lastStartPayload = null;
        roiSaved = false; // Reset để loadAutoROI được phép vẽ lại canvas
        const roiCanvas = document.getElementById('roi-canvas');
        if (roiCanvas) roiCanvas.style.display = 'block'; // Hiện lại canvas đã bị ẩn sau lần Save trước
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
            // Hiện lại ROI canvas sau khi dừng
            const roiCanvasEl = document.getElementById('roi-canvas');
            if (roiCanvasEl) roiCanvasEl.style.display = 'block';
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

async function testOCR() {
    const fileInput = document.getElementById('ocr-image-upload');
    if (!fileInput.files.length) {
        alert("Vui lòng chọn một ảnh để test OCR!");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    logSystem("Đang gửi ảnh lên Server để phân tích OCR...");
    
    try {
        const response = await fetch('/api/test_ocr', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (response.ok) {
            logSystem(`[OCR SUCCESS] Biển số: ${result.plate_text}`);
            
            // Re-use the violation modal UI for OCR result display
            const v = {
                violation_type: "TEST OCR",
                plate_read: result.plate_text,
                owner: result.owner_info.owner,
                phone: result.owner_info.phone,
                class_vehicle: result.owner_info.class_vehicle,
                province: result.owner_info.province,
                registration_date: result.owner_info.registration_date,
                id_card: result.owner_info.id_card,
                timestamp: new Date().toISOString(),
                camera_id: "LOCAL_TEST",
                vehicle_crop_base64: result.image_base64, // We can return the bounding box drawn image
                plate_img_base64: result.plate_crop_base64
            };
            
            violationHistory["test_ocr"] = v;
            openViolationModal("test_ocr");

        } else {
            logSystem(`[OCR FAILED] ${result.detail || "Không nhận diện được"}`);
            alert(`Lỗi OCR: ${result.detail || "Không nhận diện được"}`);
        }
    } catch (e) {
        logSystem(`[OCR ERROR] Mất kết nối: ${e}`);
    }
}

async function reloadVehicleDB() {
    const statusEl = document.getElementById("reload-db-status");
    statusEl.textContent = "Đang reload...";
    statusEl.className = "small text-muted mt-1";
    try {
        const res = await fetch("/api/reload_db", { method: "POST" });
        const data = await res.json();
        if (res.ok) {
            statusEl.textContent = `✅ ${data.message}`;
            statusEl.className = "small text-success mt-1";
            logSystem(`[DB RELOAD] ${data.message}`);
        } else {
            statusEl.textContent = `❌ Lỗi: ${data.detail || "Không xác định"}`;
            statusEl.className = "small text-danger mt-1";
        }
    } catch (e) {
        statusEl.textContent = `❌ Mất kết nối: ${e}`;
        statusEl.className = "small text-danger mt-1";
    }
}

function showExportModal() {
    const modal = new bootstrap.Modal(document.getElementById('exportModal'));
    // Set default dates (today)
    const now = new Date();
    const todayStr = now.toISOString().slice(0, 16);
    const yesterday = new Date(now.getTime() - 24*60*60*1000);
    const yesterdayStr = yesterday.toISOString().slice(0, 16);
    
    document.getElementById('export-start').value = yesterdayStr;
    document.getElementById('export-end').value = todayStr;
    
    modal.show();
}

async function doExport() {
    const start = document.getElementById('export-start').value;
    const end = document.getElementById('export-end').value;
    const format = document.querySelector('input[name="exportFormat"]:checked').value;
    
    if (!start || !end) {
        alert("Vui lòng chọn khoảng thời gian!");
        return;
    }
    
    // Tạo URL với query params
    const url = `/api/export_violations?start_date=${encodeURIComponent(start)}&end_date=${encodeURIComponent(end)}&format=${format}`;
    
    logSystem(`Đang chuẩn bị xuất file ${format.toUpperCase()}...`);
    
    // Tải file
    window.location.href = url;
    
    // Đóng modal
    bootstrap.Modal.getInstance(document.getElementById('exportModal')).hide();
}

function clearDashboardState() {
    // 1. Reset các biến trạng thái
    violationHistory = {};
    localVideoViolations = [];
    pendingVideoStart = null;
    lastStartPayload = null;
    roiSaved = false;

    // 2. Reset Konva Canvas
    if (layer) {
        layer.destroyChildren();
        layer.draw();
    }

    // 3. Reset các con số thống kê và Đèn giao thông
    document.getElementById('stat-car').innerText = "0";
    document.getElementById('stat-motorcycle').innerText = "0";
    document.getElementById('fps-counter').innerText = "FPS: 0";
    const lightBadge = document.getElementById('stat-light');
    if (lightBadge) {
        lightBadge.innerText = "UNKNOWN";
        lightBadge.className = "badge bg-secondary";
    }

    // 4. Reset danh sách vi phạm
    document.getElementById('violation-list').innerHTML =
        '<div class="text-center text-muted mt-5 small">Đang chờ dữ liệu...</div>';

    // 5. Reset stream views và video player
    streamImg.src = "";
    streamImg.style.display = 'none';
    const videoPlayer = document.getElementById('local-video-player');
    if (videoPlayer) {
        try {
            videoPlayer.pause();
            videoPlayer.src = "";
            videoPlayer.load();
        } catch (e) {}
        videoPlayer.style.display = 'none';
    }

    // Hiện lại ROI canvas (bị ẩn khi phát video kết quả)
    const roiCanvasEl = document.getElementById('roi-canvas');
    if (roiCanvasEl) roiCanvasEl.style.display = 'block';

    streamPlaceholder.style.display = 'flex';
    streamPlaceholder.innerHTML = '<div class="text-center"><h4 class="text-primary fw-bold">HỆ THỐNG GIÁM SÁT VI PHẠM GIAO THÔNG AI</h4><p class="text-muted small">Vui lòng bấm BẮT ĐẦU GIÁM SÁT hoặc CHẠY XỬ LÝ PHẠT để bắt đầu luồng dữ liệu.</p></div>';

    logSystem("🔄 Đã chuyển tab. Đã làm sạch trạng thái và dữ liệu Dashboard.");
}
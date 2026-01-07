// ==============================
// 1) GLOBAL STATE
// ==============================
let stage, layer;
let mode = null; // 'line', 'polygon', 'light_zone'
let isDrawing = false;
let currentShape = null;

// Cấu trúc dữ liệu mới: Lưu cả tọa độ và hướng (label)
// zones = { 
//    lines: [ {points:[], label:'left'}, ... ], 
//    polygons: [], 
//    light_zones: [ {points:[], label:'straight'} ] 
// }
let zones = { lines: [], polygons: [], light_zones: [] };
let vehicles = {};
let violations = [];
let isPaused = false;
let ws = null;

// ==============================
// 2) WEBSOCKET
// ==============================
function connectWS() {
    const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            updateTrackingPanel(data.vehicles);
            updateViolationPanel(data.violations);
            updateStats(data.stats);
            updateLights(data.lights);
            updateFPS(data.fps);

            if (data.is_paused !== undefined) {
                syncPauseButton(data.is_paused);
            }
        } catch(err) {}
    };
    ws.onclose = () => setTimeout(connectWS, 2000);
}

// ==============================
// 3) DRAWING LOGIC (CORE)
// ==============================
function initCanvas() {
    const img = document.getElementById('video-stream');
    stage = new Konva.Stage({ container: 'canvas-container', width: 800, height: 450 });
    layer = new Konva.Layer();
    stage.add(layer);
    
    const resize = () => {
        if(img.clientWidth > 0) {
            stage.width(img.clientWidth);
            stage.height(img.clientHeight);
        }
    };
    new ResizeObserver(resize).observe(img);
    img.onload = resize;
}

function attachDrawingEvents() {
    stage.off('mousedown touchstart mousemove touchmove contextmenu');

    stage.on('mousedown touchstart', (e) => {
        if (e.evt.button === 2 && isDrawing) { finishDrawing(); return; }
        if (!mode) return;

        const pos = stage.getPointerPosition();
        if (!isDrawing) {
            isDrawing = true;
            let color = (mode === 'line') ? 'red' : ((mode === 'light_zone') ? '#00ffea' : 'yellow');
            
            // Vẽ Line hoặc Polygon
            currentShape = new Konva.Line({
                points: [pos.x, pos.y, pos.x, pos.y],
                stroke: color, strokeWidth: (mode === 'line') ? 4 : 2,
                closed: false, 
                fill: (mode !== 'line') ? color.replace(')', ',0.2)').replace('rgb', 'rgba') : null
            });
            if(mode !== 'line') currentShape.fill(`${color}33`); // Hack màu fill
            layer.add(currentShape);
        } else {
            const pts = currentShape.points();
            if (mode === 'line') {
                // Line: Điểm đầu -> Điểm cuối (XONG LUÔN)
                const label = document.getElementById('direction-select').value;
                const scale = getScaleFactor(); // Lấy tỷ lệ scale hiện tại
                
                // Quy đổi toạ độ màn hình -> toạ độ video thực
                const start = [pts[0] * scale.x, pts[1] * scale.y];
                const end = [pos.x * scale.x, pos.y * scale.y];

                zones.lines.push({ points: [start, end], label: label });
                
                // Gửi dữ liệu đi
                sendZones();
                
                // QUAN TRỌNG: Hủy hình vẽ trên client để tránh bị 2 hình đè lên nhau
                currentShape.destroy(); 
                isDrawing = false;
                currentShape = null;
            } else {
                currentShape.points([...pts, pos.x, pos.y]);
            }
        }
        layer.draw();
    });

    stage.on('mousemove touchmove', () => {
        if (!isDrawing || !currentShape) return;
        const pos = stage.getPointerPosition();
        const pts = currentShape.points();
        pts[pts.length-2] = pos.x;
        pts[pts.length-1] = pos.y;
        currentShape.points(pts);
        layer.batchDraw();
    });

    stage.on('contextmenu', (e) => e.evt.preventDefault());
}

function finishDrawing() {
    if (!currentShape || !isDrawing) return;
    const pts = currentShape.points();
    const cleanPts = pts.slice(0, -2); // Bỏ điểm thừa theo chuột

    if (cleanPts.length >= 6) { // Ít nhất 3 điểm
        const scale = getScaleFactor();
        const label = document.getElementById('direction-select').value;
        
        // Quy đổi toàn bộ điểm sang toạ độ thực
        const realPoly = [];
        for(let i=0; i<cleanPts.length; i+=2) {
            realPoly.push([ cleanPts[i] * scale.x, cleanPts[i+1] * scale.y ]);
        }

        if (mode === 'light_zone') zones.light_zones.push({ points: realPoly, label: label });
        else if (mode === 'polygon') zones.polygons.push(realPoly);
        
        sendZones();
    }
    
    // Xóa hình vẽ tạm trên client -> Video stream sẽ hiển thị hình đã vẽ từ server
    currentShape.destroy(); 
    isDrawing = false;
    currentShape = null;
    layer.draw();
}

function addLabelText(x, y, text) {
    const txt = new Konva.Text({
        x: x, y: y - 15,
        text: text.toUpperCase(),
        fontSize: 12, fill: 'white',
        shadowColor: 'black', shadowBlur: 2
    });
    layer.add(txt);
}

window.toggleTrafficMode = function() {
    const isChecked = document.getElementById('traffic-switch').checked;
    fetch('/api/set_option', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({use_traffic_light: isChecked})
    });
};

function syncPauseButton(serverPausedState) {
    // Chỉ cập nhật nếu trạng thái local khác trạng thái server
    if (isPaused !== serverPausedState) {
        isPaused = serverPausedState;
        const btn = document.getElementById('pause-btn');
        if (isPaused) {
            btn.innerHTML = '<i class="fas fa-play"></i> Tiếp tục';
            btn.style.backgroundColor = '#f59e0b';
            btn.style.color = '#000';
        } else {
            btn.innerHTML = "<i class='fas fa-pause'></i> Tạm dừng";
            btn.style.backgroundColor = '';
            btn.style.color = '';
        }
    }
}

// ==============================
// 4) HELPER FUNCTIONS
// ==============================
function setMode(m) {
    mode = m;
    isDrawing = false;
    currentShape = null;
    attachDrawingEvents();
    document.getElementById('canvas-container').style.cursor = 'crosshair';
    console.log("Mode:", m);
}

function clearDraw() {
    if(layer) layer.destroyChildren(); 
    layer.draw();
    zones = { lines: [], polygons: [], light_zones: [] };
    sendZones();
}

function sendZones() {
    fetch('/api/zones', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(zones)
    });
}

// ==============================
// 5) UI HELPERS
// ==============================
window.togglePause = function() {
    isPaused = !isPaused;

    fetch('/api/pause', {
        method: 'POST', 
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pause: isPaused})
    }).then(r=>r.json()).then(data=>console.log("Pause status:", data)).catch(console.error);
};

window.exportViolations = function() {
    if(!violations.length) return alert("Không có dữ liệu!");
    let csv = "ID,BienSo,LoaiViPham,ThoiGian,ChuXe\n";
    violations.forEach(v => { csv += `${v.id},${v.plate},${v.type},${v.time},${v.owner||''}\n`; });
    const link = document.createElement("a");
    link.href = 'data:text/csv;charset=utf-8,' + encodeURI(csv);
    link.download = `ViPham_${new Date().toISOString().slice(0,10)}.csv`;
    link.click();
};
function openViolationModal(v) {
    document.getElementById('violation-modal').classList.remove('hidden');

    document.getElementById('m-plate').innerText = v.plate || '';
    document.getElementById('m-owner').innerText = v.owner || 'Không xác định';
    document.getElementById('m-phone').innerText = v.phone || '';
    document.getElementById('m-class').innerText = v.class_vehicle || '';
    document.getElementById('m-province').innerText = v.province || '';
    document.getElementById('m-date').innerText = v.registration_date || '';
    document.getElementById('m-id').innerText = v.id_card || '';
    document.getElementById('m-time').innerText = v.time || '';
    document.getElementById('m-type').innerText = v.type || '';

    document.getElementById('m-vehicle-img').src = v.img || '';
    document.getElementById('m-plate-img').src = v.plate_img || '';
}

function closeViolationModal() {
    document.getElementById('violation-modal').classList.add('hidden');
}

function updateTrackingPanel(vs) {
    const list = document.getElementById('vehicle-list');
    if(!vs) return;
    const items = Object.entries(vs).map(([id, v]) => {
        const plateHtml = v.plate === 'Reading...' ? `<span class="reading-text">Reading...</span>` : `<span class="plate-box">${v.plate}</span>`;
        return `<div class="vehicle-item"><img src="${v.img}" class="vehicle-img"><div class="vehicle-info"><b>ID: ${id}</b> <small>${v.time}</small><br>${v.type} | ${plateHtml}</div></div>`;
    }).join('');
    list.innerHTML = items || '<p style="text-align:center;color:#666">Trống</p>';
}

function updateViolationPanel(viols) {
    violations = viols || [];
    const list = document.getElementById('violation-list');
    const items = violations.map(v => `
        <div class="violation-item" onclick='openViolationModal(${JSON.stringify(v)})'>
            <img src="${v.img}" class="violation-img" alt="Ảnh vi phạm xe ${v.plate}" title="Vi phạm ${v.type}">
            <div class="violation-info"><b style="color:var(--danger)">${v.type}</b><br>ID: ${v.id} | <b>${v.plate}</b><br><small>${v.time}</small>${v.plate_img?`<br><img src="${v.plate_img}" class="plate-mini">`:''}</div>
        </div>`).join('');
    list.innerHTML = items || '<p style="text-align:center;color:#666">Chưa có vi phạm</p>';
}

function updateStats(s) { 
  if(s) { 
    document.getElementById('count-car').innerText=s.car; 
    document.getElementById('count-motorcycle').innerText=s.motorcycle; 
    document.getElementById('count-truck').innerText=s.truck; 
  } 
}

function updateLights(l) {
    if(!l) return;
    const setL = (id, c) => { document.getElementById(id).className = `light ${c} active`; };
    setL('light-left', l.left);
    setL('light-straight', l.straight);
}

function updateFPS(fps) {
    const el = document.getElementById('fps-value');
    if (!el) return;
    el.innerText = fps ? fps.toFixed(1) : '0.0';
}


window.onload = function() {
    initCanvas();
    connectWS();
    window.addEventListener('keydown', (e) => { if(e.key === 'Enter') finishDrawing(); });
    document.getElementById('upload-form').onsubmit = (e) => {
        e.preventDefault();
        const f = document.getElementById('video-file').files[0];
        const fd = new FormData(); fd.append("file", f);
        fetch('/upload_video', {method:'POST', body:fd}).then(r=>r.json()).then(d=>alert(d.message));
    };
    
    const tools = document.querySelector('.tools');
    const header = document.querySelector('.tool-header');

    let isDragging = false;
    let startX, startY, initialLeft, initialTop;

    header.onmousedown = (e) => {
        e.preventDefault();
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;

        const rect = tools.getBoundingClientRect();
        initialLeft = rect.left;
        initialTop = rect.top;

        tools.style.cursor = 'grabbing';
    };

    document.onmouseup = () => {
        isDragging = false;
        tools.style.cursor = 'default';
    };

    document.onmousemove = (e) => {
        if (!isDragging) return;

        const dx = e.clientX - startX;
        const dy = e.clientY - startY;

        tools.style.left = `${initialLeft + dx}px`;
        tools.style.top = `${initialTop + dy}px`;

        tools.style.bottom = 'auto';
        tools.style.right = 'auto';
    };
}

function getScaleFactor() {
    const img = document.getElementById('video-stream');
    if (img && img.naturalWidth && img.clientWidth) {
        return {
            x: img.naturalWidth / img.clientWidth,
            y: img.naturalHeight / img.clientHeight
        };
    }
    return { x: 1, y: 1 };
}
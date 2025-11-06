// ==============================
// 1. KHAI BÁO BIẾN TOÀN CỤC (TRÊN CÙNG)
// ==============================
let stage, layer, mode = null;
const zones = { lines: [], polygons: [] };
let vehicles = {};
let violations = [];
let isPaused = false;  // ← Đặt ở đây

// ==============================
// 2. WEBSOCKET
// ==============================
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  // console.log("WebSocket update:", data); // thêm dòng này để theo dõi
  updateTrackingPanel(data.vehicles);
  updateViolationPanel(data.violations);
  updateStats(data.stats);
  updateTrafficLights(data.lights);
};

// ==============================
// 3. CANVAS + DRAG TOOLS
// ==============================
function initCanvas() {
  const img = document.getElementById('video-stream');
  
  stage = new Konva.Stage({
    container: 'canvas-container',
    width: img.clientWidth || 800,
    height: img.clientHeight || 450
  });
  layer = new Konva.Layer();
  stage.add(layer);

  const updateSize = () => {
    const width = img.clientWidth;
    const height = img.clientHeight;
    if (width > 0 && height > 0) {
      stage.width(width);
      stage.height(height);
      stage.scale({ x: 1, y: 1 });
      layer.draw();
    }
  };

  img.onload = updateSize;
  new ResizeObserver(updateSize).observe(img);
  updateSize();
}

// Gắn sự kiện vẽ sau khi stage sẵn sàng
function attachDrawingEvents() {
  if (!stage || !layer) return;

  let currentShape;
  stage.on('mousedown touchstart', (e) => {
    if (!mode) return;
    const pos = stage.getPointerPosition();
    const points = [pos.x, pos.y];
    currentShape = mode === 'line'
      ? new Konva.Line({ points, stroke: 'red', strokeWidth: 4 })
      : new Konva.Line({ points, stroke: 'yellow', strokeWidth: 3, closed: false });
    layer.add(currentShape);
  });

  stage.on('mousemove touchmove', () => {
    if (!currentShape) return;
    const pos = stage.getPointerPosition();
    const pts = currentShape.points();
    currentShape.points(pts.concat([pos.x, pos.y]));
  });

  stage.on('mouseup touchend', () => {
    if (!currentShape) return;
    if (mode === 'polygon') {
      currentShape.closed(true);
      currentShape.fill('rgba(255,255,0,0.3)');
    }
    const pts = currentShape.points();
    if (mode === 'line') zones.lines.push([[pts[0],pts[1]], [pts[2],pts[3]]]);
    else zones.polygons.push(pts.reduce((a,b,i)=> i%2===0 ? [...a, [b, pts[i+1]]] : a, []));
    sendZones();
    currentShape = null;
  });
}

// ==============================
// 4. CÁC HÀM HỖ TRỢ
// ==============================
function setMode(m) { mode = m; }
function clearDraw() {
  if (layer) {
    layer.destroyChildren();
    zones.lines = [];
    zones.polygons = [];
    sendZones();
  }
}
function sendZones() {
  fetch('/api/zones', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(zones)
  });
}

// Pause/Play
function togglePause() {
  isPaused = !isPaused;
  const btn = document.getElementById('pause-btn');
  const icon = btn.querySelector('i');
  const text = btn.querySelector('.btn-text');

  btn.classList.toggle('paused', isPaused);
  if (isPaused) {
    icon.className = 'fas fa-pause';
    text.textContent = 'Tiếp tục';
  } else {
    icon.className = 'fas fa-play';
    text.textContent = 'Tạm dừng';
  }

  fetch('/api/pause', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({pause: isPaused})
  });
}

// Export CSV
function exportViolations() {
  const csv = [
    "ID,Xe,Biển số,Vi phạm,Thời gian,Chủ xe",
    ...violations.map(v => `${v.id},${v.type},${v.plate},${v.type},${v.time},${v.owner || ''}`)
  ].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'violations.csv'; a.click();
}

// ======================
// CẬP NHẬT GIAO DIỆN UI
// ======================

// === THEO DÕI PHƯƠNG TIỆN ===
function updateTrackingPanel(vehicles) {
  const list = document.getElementById('vehicle-list');
  if (!list) return;

  if (!vehicles || Object.keys(vehicles).length === 0) {
    list.innerHTML = '<p style="color:#aaa;text-align:center;">Không có phương tiện nào đang được theo dõi</p>';
    return;
  }

  list.innerHTML = Object.entries(vehicles).map(([id, v]) => `
    <div class="vehicle-item">
      <img class="vehicle-img" src="${v.img}" alt="vehicle"/>
      <div class="vehicle-info">
        <b>ID: ${id}</b><br>
        ${v.type} | ${v.plate}<br>
        <small>${v.time}</small>
      </div>
    </div>
  `).join('');
}


// === DANH SÁCH VI PHẠM ===
function updateViolationPanel(viols) {
  violations = viols || [];
  const list = document.getElementById('violation-list');
  if (!list) return;

  if (violations.length === 0) {
    list.innerHTML = '<p style="color:#aaa;text-align:center;">Chưa có vi phạm nào được ghi nhận</p>';
    return;
  }

  list.innerHTML = violations.map(v => `
    <div class="violation-item">
      <img class="violation-img" src="${v.img}" alt="violation"/>
      <div class="violation-info">
        <b>Xe ID: ${v.id}</b> | ${v.plate}<br>
        <b>Vi phạm:</b> ${v.type}<br>
        <small>Thời gian: ${v.time}</small><br>
        <small style="color:#aaa">Chủ xe: ${v.owner || 'Chưa tra cứu'}</small>
      </div>
    </div>
  `).join('');
}


// === THỐNG KÊ SỐ LƯỢNG ===
function updateStats(stats) {
  if (!stats) return;
  document.getElementById('count-car').textContent = stats.car || 0;
  document.getElementById('count-motorcycle').textContent = stats.motorcycle || 0;
  document.getElementById('count-bus').textContent = stats.bus || 0;
  document.getElementById('count-truck').textContent = stats.truck || 0;
}


// === ĐÈN GIAO THÔNG (REAL-TIME) ===
function updateTrafficLights(lights) {
  if (!lights) return;
  const left = document.getElementById('light-left');
  const straight = document.getElementById('light-straight');
  if (!left || !straight) return;

  // Reset class
  left.className = 'light';
  straight.className = 'light';

  // Gán trạng thái mới
  if (lights.left) left.classList.add(lights.left);
  if (lights.straight) straight.classList.add(lights.straight);
}


// ==============================
// 5. DRAGGABLE TOOLS
// ==============================
function makeDraggable(element) {
  let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
  let isDragging = false;

  const dragMouseDown = (e) => {
    e.preventDefault();
    e.stopPropagation();
    pos3 = e.clientX || e.touches[0].clientX;
    pos4 = e.clientY || e.touches[0].clientY;
    document.onmouseup = closeDrag;
    document.onmousemove = elementDrag;
    isDragging = true;
  };

  const elementDrag = (e) => {
    if (!isDragging) return;
    e.preventDefault();
    const clientX = e.clientX || e.touches[0].clientX;
    const clientY = e.clientY || e.touches[0].clientY;
    pos1 = pos3 - clientX;
    pos2 = pos4 - clientY;
    pos3 = clientX;
    pos4 = clientY;
    element.style.top = (element.offsetTop - pos2) + "px";
    element.style.left = (element.offsetLeft - pos1) + "px";
  };

  const closeDrag = () => {
    document.onmouseup = null;
    document.onmousemove = null;
    isDragging = false;
  };

  element.onmousedown = dragMouseDown;
  element.addEventListener('touchstart', dragMouseDown, { passive: false });
}

function setOption(useLight) {
  fetch('/api/set_option', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ use_traffic_light: useLight })
  }).then(res => res.json())
    .then(data => console.log('Option set:', data.use_traffic_light))
    .catch(err => console.error('Error setting option:', err));
}


// ==============================
// 6. WINDOW.ONLOAD – GỌI TẤT CẢ
// ==============================
window.onload = function() {
  initCanvas();
  attachDrawingEvents();  // ← Gắn sau khi stage sẵn sàng

  // Tools draggable
  const tools = document.querySelector('.tools');
  if (tools) makeDraggable(tools);

  // Switch
  document.getElementById('traffic-switch')?.addEventListener('change', function() {
    setOption(this.checked);
  });

  // Upload
  document.getElementById('upload-form')?.addEventListener('submit', function(e) {
    e.preventDefault();
    const file = document.getElementById('video-file').files[0];
    if (!file) return alert('Chọn video!');
    const form = new FormData();
    form.append('file', file);
    fetch('/upload_video', { method: 'POST', body: form })
      .then(r => r.json())
      .then(d => alert(d.status === 'ok' ? d.message : 'Lỗi: ' + d.message));
  });
};
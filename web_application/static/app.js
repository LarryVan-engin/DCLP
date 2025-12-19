// ==============================
// 1) STATE
// ==============================
let stage, layer, mode = null;
const zones = { lines: [], polygons: [] };

let vehicles = {};
let violations = [];
let isPaused = false;

// ==============================
// 2) WEBSOCKET (auto reconnect + ws/wss)
// ==============================
let ws = null;
let wsRetry = 0;
let wsRetryTimer = null;

function setWsStatus(text) {
  const el = document.getElementById('ws-status');
  if (el) el.textContent = text;
}

function wsUrl() {
  const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
  return `${proto}://${location.host}/ws`;
}

function connectWS() {
  if (ws) {
    try { ws.close(); } catch {}
    ws = null;
  }
  if (wsRetryTimer) {
    clearTimeout(wsRetryTimer);
    wsRetryTimer = null;
  }

  setWsStatus('connecting');
  ws = new WebSocket(wsUrl());

  ws.onopen = () => {
    wsRetry = 0;
    setWsStatus('connected');
  };

  ws.onclose = () => {
    setWsStatus('disconnected');
    const backoff = Math.min(3000, 300 * (2 ** wsRetry));
    wsRetry = Math.min(wsRetry + 1, 5);
    wsRetryTimer = setTimeout(connectWS, backoff);
  };

  ws.onerror = () => {
    setWsStatus('error');
  };

  ws.onmessage = (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }

    // Nếu đã nhận data thì chắc chắn là connected
    setWsStatus('connected');

    updateTrackingPanel(data.vehicles);
    updateViolationPanel(data.violations);
    updateStats(data.stats);
    updateTrafficLights(data.lights);
    updateFPS(data.fps);
  };

}

// ==============================
// 3) CANVAS INIT
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

// ==============================
// 4) DRAWING (clean, no double-submit)
// - line: click 2 điểm để chốt
// - polygon: click để thêm điểm, double click để chốt
// ==============================
function attachDrawingEvents() {
  if (!stage || !layer) return;

  let currentShape = null;
  let isDrawing = false;
  let currentMode = null;

  const resetCurrent = () => {
    currentShape = null;
    isDrawing = false;
    currentMode = null;
  };

  stage.off('mousedown touchstart');
  stage.off('mousemove touchmove');
  stage.off('doubleclick doubletap');

  stage.on('mousedown touchstart', () => {
    if (!mode) return;
    const pos = stage.getPointerPosition();
    if (!pos) return;

    // start
    if (!isDrawing) {
      isDrawing = true;
      currentMode = mode;

      if (currentMode === 'line') {
        currentShape = new Konva.Line({
          points: [pos.x, pos.y, pos.x, pos.y],
          stroke: 'red',
          strokeWidth: 4
        });
      } else if (currentMode === 'polygon') {
        currentShape = new Konva.Line({
          points: [pos.x, pos.y, pos.x, pos.y],
          stroke: 'yellow',
          strokeWidth: 3,
          closed: false
        });
      }

      layer.add(currentShape);
      layer.draw();
      return;
    }

    // continue
    if (!currentShape) return;

    if (currentMode === 'line') {
      // click thứ 2 -> chốt line
      const pts = currentShape.points();
      pts[2] = pos.x;
      pts[3] = pos.y;
      currentShape.points(pts);
      layer.draw();

      zones.lines.push([[pts[0], pts[1]], [pts[2], pts[3]]]);
      sendZones();
      resetCurrent();
      return;
    }

    if (currentMode === 'polygon') {
      // click -> thêm điểm, giữ 1 điểm preview cuối
      const pts = currentShape.points();
      pts[pts.length - 2] = pos.x;
      pts[pts.length - 1] = pos.y;
      currentShape.points(pts.concat([pos.x, pos.y]));
      layer.draw();
    }
  });

  stage.on('mousemove touchmove', () => {
    if (!isDrawing || !currentShape) return;
    const pos = stage.getPointerPosition();
    if (!pos) return;

    const pts = currentShape.points();
    pts[pts.length - 2] = pos.x;
    pts[pts.length - 1] = pos.y;
    currentShape.points(pts);
    layer.batchDraw();
  });

  const finishPolygon = () => {
    if (!isDrawing || currentMode !== 'polygon' || !currentShape) return;

    const pts = currentShape.points();
    if (pts.length < 8) {
      // < 3 điểm thật -> hủy
      currentShape.destroy();
      layer.draw();
      resetCurrent();
      return;
    }

    const cleanPts = pts.slice(0, -2); // bỏ preview
    currentShape.points(cleanPts);
    currentShape.closed(true);
    currentShape.fill('rgba(255,255,0,0.3)');
    layer.draw();

    const poly = [];
    for (let i = 0; i < cleanPts.length; i += 2) {
      poly.push([cleanPts[i], cleanPts[i + 1]]);
    }

    zones.polygons.push(poly);
    sendZones();
    resetCurrent();
  };

  stage.on('doubleclick doubletap', finishPolygon);
}

// ==============================
// 5) HELPERS
// ==============================
function setMode(m) { mode = m; }

function clearDraw() {
  if (!layer) return;
  layer.destroyChildren();
  zones.lines = [];
  zones.polygons = [];
  layer.draw();
  sendZones();
}

function sendZones() {
  fetch('/api/zones', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(zones)
  }).catch(() => {});
}

function togglePause() {
  isPaused = !isPaused;

  const btn = document.getElementById('pause-btn');
  const icon = btn?.querySelector('i');
  const text = btn?.querySelector('.btn-text');

  btn?.classList.toggle('paused', isPaused);
  if (icon) icon.className = isPaused ? 'fas fa-pause' : 'fas fa-play';
  if (text) text.textContent = isPaused ? 'Tiếp tục' : 'Tạm dừng';

  fetch('/api/pause', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pause: isPaused })
  }).catch(() => {});
}

function exportViolations() {
  const header = "id,plate,type,time,owner,phone,class_vehicle,province,registration_date,id_card,match_type";
  const rows = (violations || []).map(v => ([
    v.id ?? '',
    csvSafe(v.plate ?? ''),
    csvSafe(v.type ?? ''),
    csvSafe(v.time ?? ''),
    csvSafe(v.owner ?? ''),
    csvSafe(v.phone ?? ''),
    csvSafe(v.class_vehicle ?? ''),
    csvSafe(v.province ?? ''),
    csvSafe(v.registration_date ?? ''),
    csvSafe(v.id_card ?? ''),
    v.match_type ? 'true' : 'false'
  ].join(',')));

  const csv = [header, ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'violations.csv';
  a.click();
  URL.revokeObjectURL(url);
}

function csvSafe(s) {
  const str = String(s).replace(/"/g, '""');
  return `"${str}"`;
}

function setOption(useLight) {
  fetch('/api/set_option', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ use_traffic_light: useLight })
  }).catch(() => {});
}

// ======================
// 6) UI RENDER
// ======================
function updateTrackingPanel(vs) {
  vehicles = vs || {};
  const list = document.getElementById('vehicle-list');
  if (!list) return;

  const keys = Object.keys(vehicles);
  if (keys.length === 0) {
    list.innerHTML = '<p style="color:#aaa;text-align:center;">Không có phương tiện nào đang được theo dõi</p>';
    return;
  }

  list.innerHTML = Object.entries(vehicles).map(([id, v]) => {
    const img = v?.img || '';
    const type = v?.type || 'unknown';
    const plate = v?.plate || '';
    const time = v?.time || '';
    const owner = v?.owner || ''; // nếu backend bổ sung sau thì UI tự hiện

    return `
      <div class="vehicle-item">
        <img class="vehicle-img" src="${img}" alt="vehicle" onerror="this.style.display='none'"/>
        <div class="vehicle-info">
          <b>ID: ${id}</b><br>
          ${type} | ${plate}<br>
          ${owner ? `<small style="color:#aaa">Chủ xe: ${owner}</small><br>` : ``}
          <small>${time}</small>
        </div>
      </div>
    `;
  }).join('');
}

function updateViolationPanel(viols) {
  violations = viols || [];
  const list = document.getElementById('violation-list');
  if (!list) return;

  if (violations.length === 0) {
    list.innerHTML = '<p style="color:#aaa;text-align:center;">Chưa có vi phạm nào được ghi nhận</p>';
    return;
  }

  list.innerHTML = violations.map(v => {
    const badge = v.match_type ? `<span class="match-badge ok">MATCH TYPE</span>` : `<span class="match-badge warn">TYPE?</span>`;
    const plateImg = v.plate_img ? `<img class="plate-img" src="${v.plate_img}" alt="plate" onerror="this.style.display='none'"/>` : '';

    return `
      <div class="violation-item">
        <img class="violation-img" src="${v.img || ''}" alt="violation" onerror="this.style.display='none'"/>
        <div class="violation-info">
          <div class="violation-row">
            <b>Xe ID: ${v.id}</b> | ${v.plate || ''}
            ${badge}
          </div>
          <div><b>Vi phạm:</b> ${v.type || ''}</div>
          <div class="violation-sub">
            <small>Thời gian: ${v.time || ''}</small>
            ${plateImg}
          </div>
          <div class="violation-sub">
            <small style="color:#aaa">Chủ xe: ${v.owner || 'Chưa tra cứu'}</small>
            ${v.class_vehicle ? `<small style="color:#aaa"> | Loại: ${v.class_vehicle}</small>` : ``}
          </div>
          ${v.phone ? `<div class="violation-sub"><small style="color:#aaa">SĐT: ${v.phone}</small></div>` : ``}
        </div>
      </div>
    `;
  }).join('');
}

function updateStats(stats) {
  if (!stats) return;
  document.getElementById('count-car').textContent = stats.car || 0;
  document.getElementById('count-motorcycle').textContent = stats.motorcycle || 0;
  document.getElementById('count-bus').textContent = stats.bus || 0;
  document.getElementById('count-truck').textContent = stats.truck || 0;
}

function updateTrafficLights(lights) {
  if (!lights) return;
  const left = document.getElementById('light-left');
  const straight = document.getElementById('light-straight');
  if (!left || !straight) return;

  left.className = 'light';
  straight.className = 'light';

  if (lights.left) left.classList.add(lights.left, 'active');
  if (lights.straight) straight.classList.add(lights.straight, 'active');
}

function updateFPS(fps) {
  const el = document.getElementById('fps-value');
  if (!el) return;

  const val = Number.isFinite(fps) ? fps : parseFloat(fps);
  el.textContent = (Number.isFinite(val) ? val : 0).toFixed(1);
}


// ==============================
// 7) ONLOAD
// ==============================
window.onload = function() {
  initCanvas();
  attachDrawingEvents();
  connectWS();

  const tools = document.querySelector('.tools');
  if (tools) makeDraggable(tools);

  document.getElementById('traffic-switch')?.addEventListener('change', function() {
    setOption(this.checked);
  });

  document.getElementById('upload-form')?.addEventListener('submit', function(e) {
    e.preventDefault();
    const file = document.getElementById('video-file').files[0];
    if (!file) return alert('Chọn video!');
    const form = new FormData();
    form.append('file', file);

    fetch('/upload_video', { method: 'POST', body: form })
      .then(r => r.json())
      .then(d => alert(d.status === 'ok' ? d.message : 'Lỗi: ' + d.message))
      .catch(err => alert('Lỗi upload: ' + err));
  });
};

// ==============================
// 8) DRAGGABLE TOOLS
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
// End of app.js
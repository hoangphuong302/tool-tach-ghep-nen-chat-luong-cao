(() => {
  const API_BASE = window.location.pathname.startsWith('/matting') ? '/matting/api' : '/api';
  const CHUNK_SIZE = 4 * 1024 * 1024;
  const IS_EMBED = new URLSearchParams(window.location.search).get('embed') === '1';

  const state = {
    sid: null,
    srcMeta: null,
    bgMeta: null,
    currentStep: 'cut',
    cutView: 'overlay',
    cutFrame: 0,
    cutFrameReqId: 0,
    cutZoom: 1,
    cutPanX: 0,
    cutPanY: 0,
    brushMode: 'fg',
    brushSize: 22,
    isDrawing: false,
    isPanningCut: false,
    isApplyingStrokes: false,
    cursorCanvasPt: null,
    boxDraftImg: null,
    cutPanStart: { x: 0, y: 0, px: 0, py: 0 },
    hasPendingStrokes: false,
    cutImg: null,

    compFrame: 0,
    diagMode: 'split',
    compZoom: 1,
    compPanX: 0,
    compPanY: 0,
    wipeRatio: 0.5,
    isDraggingWipe: false,
    isPanningComp: false,
    compPanStart: { x: 0, y: 0, px: 0, py: 0 },
    compLeftImg: null,
    compRightImg: null,
    previewAbort: null,
    previewTimer: null,
    lastElapsedMs: 0,
    lastAutoTracked: false,
    spaceDown: false,
    renderTimer: null,
  };

  // Offscreen stroke layer (at full source resolution once loaded)
  const strokeFgCanvas = document.createElement('canvas');
  const strokeBgCanvas = document.createElement('canvas');

  const $ = (id) => document.getElementById(id);

  function showToast(msg, kind = 'info', durationMs = 3400) {
    const el = $('toast');
    if (!el) return;
    el.textContent = msg;
    el.className = 'toast' + (kind === 'ok' ? ' ok' : kind === 'err' ? ' err' : '');
    el.hidden = false;
    clearTimeout(el._t);
    el._t = setTimeout(() => { el.hidden = true; }, durationMs);
  }

  function applyTheme(theme) {
    const isDark = theme === 'dark';
    document.body.classList.toggle('dark', isDark);
    const btn = $('themeBtn');
    if (btn) btn.textContent = isDark ? '☀️ Sáng' : '🌙 Tối';
  }

  function initTheme() {
    const qTheme = new URLSearchParams(window.location.search).get('theme');
    const saved = qTheme || localStorage.getItem('ui_theme') || 'light';
    applyTheme(saved);
    if (IS_EMBED && $('themeBtn')) $('themeBtn').style.display = 'none';
    window.addEventListener('message', (ev) => {
      if (ev.origin !== window.location.origin) return;
      if (ev.data && ev.data.type === 'nhasam-theme') {
        applyTheme(ev.data.theme);
      }
    });
  }

  async function apiJson(path, opts = {}) {
    const res = await fetch(`${API_BASE}${path}`, opts);
    const text = await res.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch (_) {}
    if (!res.ok) {
      const err = new Error(data.detail || data.error || `HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return data;
  }

  async function ensureSession() {
    if (state.sid) return state.sid;
    const data = await apiJson('/session', { method: 'POST' });
    state.sid = data.session_id;
    return state.sid;
  }

  function updateDiskBadge(mb) {
    const el = $('diskBadge');
    if (el) el.textContent = `Phiên: ${Number(mb || 0).toFixed(1)} MB`;
  }

  function switchStep(step) {
    state.currentStep = step;
    document.querySelectorAll('.step-tab').forEach((btn) => {
      const on = btn.dataset.step === step;
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    const map = { cut: 'viewCut', prompt: 'viewPrompt', composite: 'viewComposite' };
    Object.entries(map).forEach(([k, id]) => {
      const panel = $(id);
      if (panel) {
        panel.classList.toggle('active', k === step);
        panel.hidden = k !== step;
      }
    });
    if (step === 'cut') {
      requestAnimationFrame(drawCutCanvas);
    } else if (step === 'composite') {
      requestAnimationFrame(() => {
        drawCompCanvas();
        if (state.sid && state.srcMeta) schedulePreview(true);
      });
    }
  }

  // ------------------------------------------------------------------
  // Chunked Remote Upload
  // ------------------------------------------------------------------
  async function uploadFileChunked(role, file) {
    if (!file) return;
    if (file.size <= 0) {
      showToast('File được chọn bị rỗng (0 byte). Hãy chọn file khác.', 'err');
      return;
    }
    let sid = await ensureSession();
    const wrap = $(role === 'source' ? 'srcProgressWrap' : 'bgProgressWrap');
    const fill = $(role === 'source' ? 'srcProgressFill' : 'bgProgressFill');
    const txt = $(role === 'source' ? 'srcProgressText' : 'bgProgressText');
    const tag = $(role === 'source' ? 'srcMetaTag' : 'bgMetaTag');
    wrap.hidden = false;
    fill.style.width = '2%';
    txt.textContent = 'Đang khởi tạo...';

    try {
      let initRes;
      try {
        initRes = await apiJson(`/session/${sid}/upload/init`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ role, filename: file.name, total_size: file.size }),
        });
      } catch (initErr) {
        if (initErr.status === 404) {
          state.sid = null;
          sid = await ensureSession();
          initRes = await apiJson(`/session/${sid}/upload/init`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ role, filename: file.name, total_size: file.size }),
          });
        } else {
          throw initErr;
        }
      }
      const uid = initRes.upload_id;
      const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
      for (let i = 0; i < totalChunks; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(file.size, start + CHUNK_SIZE);
        const blob = file.slice(start, end);
        let res = null;
        let lastErr = null;
        for (let attempt = 0; attempt < 2; attempt++) {
          try {
            res = await fetch(`${API_BASE}/session/${sid}/upload/${uid}/chunk?index=${i}`, {
              method: 'PUT',
              headers: { 'Content-Type': 'application/octet-stream' },
              body: blob,
            });
            if (res.ok || res.status < 500) break;
          } catch (netErr) {
            lastErr = netErr;
          }
        }
        if (!res) throw lastErr || new Error(`Mất kết nối ở mảnh ${i}`);
        if (!res.ok) {
          const errText = await res.text().catch(() => '');
          let errJson = {};
          try { errJson = errText ? JSON.parse(errText) : {}; } catch (_) {}
          throw new Error(errJson.detail || errJson.error || `Chunk ${i} thất bại (${res.status})`);
        }
        const pct = Math.round(((i + 1) / totalChunks) * 88);
        fill.style.width = `${pct}%`;
        txt.textContent = `Tải lên ${pct}% (${(end / 1048576).toFixed(1)}/${(file.size / 1048576).toFixed(1)} MB)`;
      }
      txt.textContent = 'Đang tách khung hình trên máy chủ...';
      fill.style.width = '94%';
      const comp = await apiJson(`/session/${sid}/upload/${uid}/complete`, { method: 'POST' });
      fill.style.width = '100%';
      txt.textContent = 'Hoàn tất!';
      updateDiskBadge(comp.disk_mb);

      if (role === 'source') {
        state.srcMeta = comp;
        state.cutFrame = 0;
        state.compFrame = 0;
        const maxF = Math.max(0, comp.frame_count - 1);
        $('cutFrameSlider').max = maxF;
        $('cutFrameSlider').value = 0;
        $('compFrameSlider').max = maxF;
        $('compFrameSlider').value = 0;
        $('cutFrameText').textContent = `0 / ${maxF}`;
        $('compFrameText').textContent = `0 / ${maxF}`;
        tag.textContent = `${comp.width}×${comp.height} · ${comp.frame_count}f`;
        tag.classList.add('ok');
        $('cutEmptyOverlay').hidden = true;
        strokeFgCanvas.width = comp.width;
        strokeFgCanvas.height = comp.height;
        strokeBgCanvas.width = comp.width;
        strokeBgCanvas.height = comp.height;
        clearStrokeLayers();
        renderConfirmedPills([]);
        state.compLeftImg = null;
        state.compRightImg = null;
        $('compEmptyOverlay').hidden = false;
        $('downloadBox').hidden = true;
        $('renderJobBox').hidden = true;
        $('sampledThumbs').innerHTML = '';
        $('telemetryGrid').innerHTML = '';
        drawCompCanvas();
        await loadCutFrame(0);
        showToast(`Đã nạp ${comp.filename} (${comp.width}×${comp.height}, ${comp.frame_count} frames)`, 'ok');
      } else {
        state.bgMeta = comp;
        tag.textContent = `${comp.width}×${comp.height} · ${comp.frame_count}f`;
        tag.classList.add('ok');
        showToast(`Đã nạp nền mới ${comp.filename}`, 'ok');
        if (state.currentStep === 'composite') schedulePreview(true);
      }
    } catch (err) {
      showToast(`Lỗi tải file: ${err.message}`, 'err', 5000);
    } finally {
      setTimeout(() => { wrap.hidden = true; }, 1400);
    }
  }

  function clearStrokeLayers() {
    [strokeFgCanvas, strokeBgCanvas].forEach((c) => {
      if (c.width > 0 && c.height > 0) c.getContext('2d').clearRect(0, 0, c.width, c.height);
    });
    state.hasPendingStrokes = false;
  }

  function renderConfirmedPills(frames) {
    const box = $('confirmedPills');
    if (!box) return;
    box.innerHTML = '';
    (frames || []).forEach((f) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'kf-pill';
      b.textContent = `KF ${f}`;
      b.onclick = () => {
        state.cutFrame = f;
        $('cutFrameSlider').value = f;
        loadCutFrame(f);
      };
      box.appendChild(b);
    });
  }

  // ------------------------------------------------------------------
  // Step 1 Canvas: View, Zoom/Pan & Roto-Brush Drawing
  // ------------------------------------------------------------------
  async function loadCutFrame(idx) {
    if (!state.sid || !state.srcMeta) return;
    state.cutFrame = Number(idx);
    const reqId = ++state.cutFrameReqId;
    const maxF = Math.max(0, state.srcMeta.frame_count - 1);
    $('cutFrameText').textContent = `${state.cutFrame} / ${maxF}`;
    const url = `${API_BASE}/session/${state.sid}/frame/${state.cutFrame}?view=${state.cutView}&max_dim=1440&_t=${Date.now()}`;
    const img = new Image();
    img.onload = () => {
      if (reqId !== state.cutFrameReqId) return;
      state.cutImg = img;
      drawCutCanvas();
    };
    img.onerror = () => {
      if (reqId !== state.cutFrameReqId) return;
      showToast('Không tải được khung hình xem trước.', 'err');
    };
    img.src = url;
  }

  function getFitTransform(canvas, imgW, imgH, zoom, panX, panY) {
    const cw = canvas.width;
    const ch = canvas.height;
    const baseScale = Math.min((cw - 16) / Math.max(1, imgW), (ch - 16) / Math.max(1, imgH));
    const scale = baseScale * zoom;
    const drawW = imgW * scale;
    const drawH = imgH * scale;
    const ox = (cw - drawW) / 2 + panX;
    const oy = (ch - drawH) / 2 + panY;
    return { scale, ox, oy, drawW, drawH };
  }

  function syncCanvasSize(canvas) {
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = Math.max(200, Math.round(rect.width));
    const h = Math.max(200, Math.round(rect.height));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function drawCutCanvas() {
    const canvas = $('cutCanvas');
    if (!canvas) return;
    syncCanvasSize(canvas);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!state.cutImg) return;

    const iw = state.srcMeta ? state.srcMeta.width : state.cutImg.width;
    const ih = state.srcMeta ? state.srcMeta.height : state.cutImg.height;
    const tr = getFitTransform(canvas, iw, ih, state.cutZoom, state.cutPanX, state.cutPanY);

    ctx.drawImage(state.cutImg, tr.ox, tr.oy, tr.drawW, tr.drawH);
    if (strokeFgCanvas.width > 0) {
      ctx.save();
      ctx.globalAlpha = 0.72;
      ctx.drawImage(strokeFgCanvas, tr.ox, tr.oy, tr.drawW, tr.drawH);
      ctx.drawImage(strokeBgCanvas, tr.ox, tr.oy, tr.drawW, tr.drawH);
      ctx.restore();
    }
    if (state.boxDraftImg) {
      const x0 = Math.min(state.boxDraftImg.x0, state.boxDraftImg.x1);
      const y0 = Math.min(state.boxDraftImg.y0, state.boxDraftImg.y1);
      const bw = Math.abs(state.boxDraftImg.x1 - state.boxDraftImg.x0);
      const bh = Math.abs(state.boxDraftImg.y1 - state.boxDraftImg.y0);
      const rx = tr.ox + (x0 / Math.max(1, iw)) * tr.drawW;
      const ry = tr.oy + (y0 / Math.max(1, ih)) * tr.drawH;
      const rw = (bw / Math.max(1, iw)) * tr.drawW;
      const rh = (bh / Math.max(1, ih)) * tr.drawH;
      ctx.save();
      ctx.fillStyle = 'rgba(20, 184, 166, 0.14)';
      ctx.fillRect(rx, ry, rw, rh);
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = '#14b8a6';
      ctx.lineWidth = 2;
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.restore();
    } else if (state.cursorCanvasPt && !state.isPanningCut && !state.spaceDown && state.brushMode !== 'box') {
      const r = Math.max(2, (state.brushSize * tr.scale) / 2);
      ctx.save();
      ctx.beginPath();
      ctx.arc(state.cursorCanvasPt.x, state.cursorCanvasPt.y, r, 0, Math.PI * 2);
      ctx.strokeStyle = state.brushMode === 'fg' ? 'rgba(34, 197, 94, 0.92)' : 'rgba(239, 68, 68, 0.92)';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();
    }
    const badge = $('cutZoomBadge');
    if (badge) badge.textContent = `${state.cutZoom.toFixed(2)}× · Lăn chuột Zoom · Space+Kéo Di chuyển · [ ] Cỡ cọ`;
  }

  function clientToImageXY(canvas, clientX, clientY, imgW, imgH, zoom, panX, panY) {
    const rect = canvas.getBoundingClientRect();
    const cx = clientX - rect.left;
    const cy = clientY - rect.top;
    const tr = getFitTransform(canvas, imgW, imgH, zoom, panX, panY);
    return {
      x: ((cx - tr.ox) / tr.drawW) * imgW,
      y: ((cy - tr.oy) / tr.drawH) * imgH,
      tr,
    };
  }

  function paintStrokeSegment(x0, y0, x1, y1) {
    const target = state.brushMode === 'fg' ? strokeFgCanvas : strokeBgCanvas;
    const color = state.brushMode === 'fg' ? '#22c55e' : '#ef4444';
    const ctx = target.getContext('2d');
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = state.brushSize;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    state.hasPendingStrokes = true;
  }

  function setupCutCanvasEvents() {
    const canvas = $('cutCanvas');
    let lastPt = null;

    canvas.addEventListener('wheel', (e) => {
      if (!state.cutImg) return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.18 : 1 / 1.18;
      const next = Math.min(8, Math.max(1, state.cutZoom * factor));
      if (next <= 1.02) {
        state.cutZoom = 1;
        state.cutPanX = 0;
        state.cutPanY = 0;
      } else {
        state.cutZoom = next;
      }
      drawCutCanvas();
    }, { passive: false });

    canvas.addEventListener('pointerdown', (e) => {
      if (!state.cutImg || !state.srcMeta) return;
      if (e.button === 1 || state.spaceDown || e.shiftKey) {
        state.isPanningCut = true;
        state.cutPanStart = { x: e.clientX, y: e.clientY, px: state.cutPanX, py: state.cutPanY };
        canvas.setPointerCapture(e.pointerId);
        return;
      }
      if (e.button !== 0) return;
      const pt = clientToImageXY(canvas, e.clientX, e.clientY, state.srcMeta.width, state.srcMeta.height, state.cutZoom, state.cutPanX, state.cutPanY);
      state.isDrawing = true;
      lastPt = pt;
      if (state.brushMode === 'box') {
        state.boxDraftImg = { x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y };
      } else {
        paintStrokeSegment(pt.x, pt.y, pt.x + 0.1, pt.y + 0.1);
      }
      drawCutCanvas();
      canvas.setPointerCapture(e.pointerId);
    });

    canvas.addEventListener('pointermove', (e) => {
      const rect = canvas.getBoundingClientRect();
      state.cursorCanvasPt = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      if (state.isPanningCut) {
        state.cutPanX = state.cutPanStart.px + (e.clientX - state.cutPanStart.x);
        state.cutPanY = state.cutPanStart.py + (e.clientY - state.cutPanStart.y);
        drawCutCanvas();
        return;
      }
      if (!state.isDrawing || !state.srcMeta) {
        drawCutCanvas();
        return;
      }
      const pt = clientToImageXY(canvas, e.clientX, e.clientY, state.srcMeta.width, state.srcMeta.height, state.cutZoom, state.cutPanX, state.cutPanY);
      if (state.brushMode === 'box' && state.boxDraftImg) {
        state.boxDraftImg.x1 = pt.x;
        state.boxDraftImg.y1 = pt.y;
      } else {
        paintStrokeSegment(lastPt.x, lastPt.y, pt.x, pt.y);
      }
      lastPt = pt;
      drawCutCanvas();
    });

    canvas.addEventListener('pointerleave', () => {
      state.cursorCanvasPt = null;
      drawCutCanvas();
    });

    const endCut = () => {
      const wasDrawing = state.isDrawing;
      const draftBox = state.boxDraftImg;
      state.isDrawing = false;
      state.isPanningCut = false;
      state.boxDraftImg = null;
      lastPt = null;
      drawCutCanvas();
      if (wasDrawing && state.brushMode === 'box' && draftBox && !state.isApplyingStrokes) {
        const bx0 = Math.round(Math.min(draftBox.x0, draftBox.x1));
        const by0 = Math.round(Math.min(draftBox.y0, draftBox.y1));
        const bx1 = Math.round(Math.max(draftBox.x0, draftBox.x1));
        const by1 = Math.round(Math.max(draftBox.y0, draftBox.y1));
        if (bx1 - bx0 >= 8 && by1 - by0 >= 8) {
          submitBoxSelect([bx0, by0, bx1, by1]);
        } else {
          showToast('Khung khoanh vùng quá nhỏ, hãy kéo rộng quanh chủ thể.', 'info');
        }
        return;
      }
      if (wasDrawing && state.hasPendingStrokes && $('chkAutoApplyBrush')?.checked && !state.isApplyingStrokes) {
        submitStrokes();
      }
    };
    canvas.addEventListener('pointerup', endCut);
    canvas.addEventListener('pointercancel', endCut);
    canvas.addEventListener('dblclick', () => {
      state.cutZoom = 1;
      state.cutPanX = 0;
      state.cutPanY = 0;
      drawCutCanvas();
    });
  }

  async function runAutoCut(mode) {
    if (!state.sid || !state.srcMeta) return showToast('Hãy tải Video hoặc Ảnh gốc lên trước.', 'err');
    const btnMap = { birefnet: 'btnAutoBirefnet', hands_product: 'btnAutoHands', chroma: 'btnAutoChroma' };
    const btn = $(btnMap[mode]);
    if (btn) btn.disabled = true;
    showToast('Đang chạy AI tách nền chi tiết...', 'info');
    try {
      const res = await apiJson(`/session/${state.sid}/auto-cut`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frame_idx: state.cutFrame,
          mode,
          engine: $('engineSelect').value,
          use_pymatting_fg: $('chkPyMatting').checked,
        }),
      });
      clearStrokeLayers();
      renderConfirmedPills(res.confirmed_frames);
      updateDiskBadge(res.disk_mb);
      await loadCutFrame(state.cutFrame);
      showToast(`Tách nền xong khung ${res.frame_idx} (${res.fg_pct}% vùng chọn)!`, 'ok');
    } catch (err) {
      showToast(`Lỗi tách nền: ${err.message}`, 'err', 5000);
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function submitBoxSelect(boxXyxy) {
    if (!state.sid || !state.srcMeta || state.isApplyingStrokes) return;
    state.isApplyingStrokes = true;
    $('btnApplyStrokes').disabled = true;
    showToast('AI đang tách chủ thể trong vùng bạn khoanh...', 'info');
    try {
      const res = await apiJson(`/session/${state.sid}/strokes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frame_idx: state.cutFrame,
          box_xyxy: boxXyxy,
          smart_red_subtract: $('chkSmartRed').checked,
          engine: $('engineSelect').value,
          use_pymatting_fg: $('chkPyMatting').checked,
        }),
      });
      clearStrokeLayers();
      renderConfirmedPills(res.confirmed_frames);
      updateDiskBadge(res.disk_mb);
      await loadCutFrame(state.cutFrame);
      showToast(`Đã tách chủ thể vùng khoanh khung ${res.frame_idx} (${res.fg_pct}%)`, 'ok');
    } catch (err) {
      showToast(`Lỗi tách vùng khoanh: ${err.message}`, 'err', 5000);
    } finally {
      state.isApplyingStrokes = false;
      $('btnApplyStrokes').disabled = false;
    }
  }

  async function submitStrokes() {
    if (!state.sid || !state.srcMeta) return showToast('Hãy tải Video hoặc Ảnh gốc lên trước.', 'err');
    if (!state.hasPendingStrokes || state.isApplyingStrokes) return showToast('Hãy tô nét Xanh hoặc Đỏ trên ảnh trước.', 'err');
    state.isApplyingStrokes = true;
    $('btnApplyStrokes').disabled = true;
    try {
      const res = await apiJson(`/session/${state.sid}/strokes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frame_idx: state.cutFrame,
          fg_mask_b64: strokeFgCanvas.toDataURL('image/png'),
          bg_mask_b64: strokeBgCanvas.toDataURL('image/png'),
          smart_red_subtract: $('chkSmartRed').checked,
          engine: $('engineSelect').value,
          use_pymatting_fg: $('chkPyMatting').checked,
        }),
      });
      clearStrokeLayers();
      renderConfirmedPills(res.confirmed_frames);
      updateDiskBadge(res.disk_mb);
      await loadCutFrame(state.cutFrame);
      showToast(`Đã bắt chi tiết & cập nhật khung ${res.frame_idx} (${res.fg_pct}%)`, 'ok');
    } catch (err) {
      showToast(`Lỗi áp dụng nét: ${err.message}`, 'err', 5000);
    } finally {
      state.isApplyingStrokes = false;
      $('btnApplyStrokes').disabled = false;
    }
  }

  // ------------------------------------------------------------------
  // Step 2: Optical Multi-Frame Prompt Analyzer
  // ------------------------------------------------------------------
  async function analyzeAndGeneratePrompts() {
    if (!state.sid || !state.srcMeta) return showToast('Hãy tải Video hoặc Ảnh gốc ở Bước 1 trước.', 'err');
    const btn = $('btnAnalyzePrompts');
    btn.disabled = true;
    btn.textContent = '⏳ Đang phân tích quang học...';
    try {
      const res = await apiJson(`/session/${state.sid}/analyze-prompts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_background_vi: $('targetBgText').value,
          subject_description_vi: $('subjectDescText').value,
          use_local_llm: $('chkLocalLlm').checked,
          use_vision_ai: $('chkVisionAi').checked,
        }),
      });

      const strip = $('sampledThumbs');
      strip.innerHTML = '';
      (res.sampled_indices || []).forEach((idx) => {
        const div = document.createElement('div');
        div.className = 'sampled-thumb';
        const im = document.createElement('img');
        im.src = `${API_BASE}/session/${state.sid}/frame/${idx}?view=raw&max_dim=400`;
        im.alt = `Frame ${idx}`;
        const sp = document.createElement('span');
        sp.textContent = `Frame ${idx}`;
        div.append(im, sp);
        strip.appendChild(div);
      });

      const grid = $('telemetryGrid');
      grid.innerHTML = '';
      (res.telemetry_cards || []).forEach((c) => {
        const card = document.createElement('div');
        card.className = 'tele-card';
        card.innerHTML = `
          <div class="tele-top"><span>${c.icon || '📊'}</span><span></span></div>
          <div class="tele-val"></div>
          <div class="tele-det"></div>
        `;
        card.querySelector('.tele-top span:last-child').textContent = c.title || '';
        card.querySelector('.tele-val').textContent = c.value || '';
        card.querySelector('.tele-det').textContent = c.detail || '';
        grid.appendChild(card);
      });

      $('outCleanPrompt').value = res.clean_plate_prompt || '';
      $('outV2vPrompt').value = res.v2v_prompt || '';
      $('outStillPrompt').value = res.still_bg_prompt || '';
      showToast('Đã phân tích quang học và tạo 3 Prompt chuẩn phối cảnh!', 'ok');
    } catch (err) {
      showToast(`Lỗi phân tích Prompt: ${err.message}`, 'err', 5000);
    } finally {
      btn.disabled = false;
      btn.textContent = '🔍 Phân tích Frame & Tạo Prompt';
    }
  }

  // ------------------------------------------------------------------
  // Step 3: Live Split-Wipe Preview & Zoom/Pan Stage
  // ------------------------------------------------------------------
  function buildPreviewPayload(extra = {}) {
    return {
      frame_idx: Number($('compFrameSlider').value || 0),
      diag_mode: state.diagMode,
      max_preview_dim: 1080,
      matting_engine: $('engineSelect').value,
      use_pymatting_fg: $('chkPyMatting').checked,
      trimap_radius: Number($('slTrimap').value),
      guided_radius: Number($('slGuided').value),
      decont_strength: Number($('slDecont').value),
      despill_green: $('chkDespill').checked,
      erode_dilate: Number($('slErode').value),
      edge_preserve: Number($('slEdgePreserve').value),
      edge_blur_match: Number($('slEdgeBlur').value),
      ambient_color_strength: Number($('slAmbient').value),
      directional_light_strength: Number($('slDirLight').value),
      black_point_match: Number($('slBlackPoint').value),
      light_wrap_radius: Number($('slLwRad').value),
      light_wrap_strength: Number($('slLwStr').value),
      protect_screen: $('chkProtectScreen').checked,
      screen_glare: Number($('slGlare').value),
      grain_match: Number($('slGrain').value),
      bg_defocus: Number($('slBgDefocus').value),
      clean_ai_ghost: $('chkCleanGhost').checked,
      linear_compositing: $('chkLinearComp').checked,
      pass2_legacy: $('chkPass2Legacy').checked,
      ...extra,
    };
  }

  function schedulePreview(immediate = false, delayMs = 60) {
    if (!state.sid || !state.srcMeta) return;
    if (!immediate && !$('chkLivePreview').checked) return;
    clearTimeout(state.previewTimer);
    state.previewTimer = setTimeout(fetchLivePreview, immediate ? 0 : delayMs);
  }

  async function fetchLivePreview() {
    if (!state.sid || !state.srcMeta) return;
    if (state.previewAbort) state.previewAbort.abort();
    const ctrl = new AbortController();
    state.previewAbort = ctrl;
    const badge = $('compZoomBadge');
    if (badge && state.compLeftImg) badge.textContent = '⏳ Đang cập nhật preview...';
    try {
      const res = await fetch(`${API_BASE}/session/${state.sid}/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPreviewPayload()),
        signal: ctrl.signal,
      });
      const text = await res.text();
      let data = {};
      try { data = text ? JSON.parse(text) : {}; } catch (_) {}
      if (!res.ok) {
        drawCompCanvas();
        if (res.status === 400) showToast(data.detail || 'Chưa có mask', 'info');
        else showToast(data.detail || data.error || `Preview lỗi (HTTP ${res.status})`, 'err');
        return;
      }
      state.lastElapsedMs = data.elapsed_ms || 0;
      state.lastAutoTracked = Boolean(data.auto_tracked);
      $('compEmptyOverlay').hidden = true;

      const lImg = new Image();
      const rImg = new Image();
      let loaded = 0;
      const onBoth = () => {
        loaded++;
        if (loaded === 2) {
          state.compLeftImg = lImg;
          state.compRightImg = rImg;
          drawCompCanvas();
        }
      };
      lImg.onload = onBoth;
      rImg.onload = onBoth;
      lImg.src = data.left_b64;
      rImg.src = data.right_b64;
    } catch (err) {
      if (err.name !== 'AbortError') {
        drawCompCanvas();
        showToast(`Preview: ${err.message}`, 'err');
      }
    }
  }

  function drawCompCanvas() {
    const canvas = $('compCanvas');
    if (!canvas) return;
    syncCanvasSize(canvas);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!state.compLeftImg || !state.compRightImg) return;

    const iw = state.compLeftImg.width;
    const ih = state.compLeftImg.height;
    const tr = getFitTransform(canvas, iw, ih, state.compZoom, state.compPanX, state.compPanY);
    const splitX = Math.max(0, Math.min(canvas.width, Math.round(canvas.width * state.wipeRatio)));

    // Left panel (clipped to [0, splitX])
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, splitX, canvas.height);
    ctx.clip();
    ctx.drawImage(state.compLeftImg, tr.ox, tr.oy, tr.drawW, tr.drawH);
    ctx.restore();

    // Right panel (clipped to [splitX, canvas.width])
    ctx.save();
    ctx.beginPath();
    ctx.rect(splitX, 0, canvas.width - splitX, canvas.height);
    ctx.clip();
    ctx.drawImage(state.compRightImg, tr.ox, tr.oy, tr.drawW, tr.drawH);
    ctx.restore();

    // Crisp Wipe Divider & Handle
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(splitX, 0);
    ctx.lineTo(splitX, canvas.height);
    ctx.stroke();

    const hy = Math.round(canvas.height / 2);
    ctx.fillStyle = '#0f766e';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(splitX, hy, 14, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('⇆', splitX, hy);
    ctx.restore();

    const badge = $('compZoomBadge');
    if (badge) {
      const autoPrefix = state.lastAutoTracked ? `⚡ Tự bắt khung ${state.compFrame} · ` : '';
      badge.textContent = `${autoPrefix}${state.compZoom.toFixed(2)}× · ${state.lastElapsedMs}ms · Kéo vạch chia | Lăn chuột Zoom`;
    }
  }

  function applyCompZoomPreset(zoom, px, py) {
    const canvas = $('compCanvas');
    if (!canvas || !state.compLeftImg) {
      state.compZoom = zoom;
      state.compPanX = 0;
      state.compPanY = 0;
      drawCompCanvas();
      return;
    }
    syncCanvasSize(canvas);
    const iw = state.compLeftImg.width;
    const ih = state.compLeftImg.height;
    const baseScale = Math.min((canvas.width - 16) / iw, (canvas.height - 16) / ih);
    const drawW = iw * baseScale * zoom;
    const drawH = ih * baseScale * zoom;
    state.compZoom = zoom;
    state.compPanX = (0.5 - px) * Math.max(0, drawW - canvas.width * 0.6);
    state.compPanY = (0.5 - py) * Math.max(0, drawH - canvas.height * 0.6);
    drawCompCanvas();
  }

  function setupCompCanvasEvents() {
    const canvas = $('compCanvas');
    canvas.addEventListener('wheel', (e) => {
      if (!state.compLeftImg) return;
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const ox = e.clientX - (rect.left + rect.width / 2);
      const oy = e.clientY - (rect.top + rect.height / 2);
      const factor = e.deltaY < 0 ? 1.18 : 1 / 1.18;
      const next = Math.min(8, Math.max(1, state.compZoom * factor));
      if (next <= 1.02) {
        state.compZoom = 1;
        state.compPanX = 0;
        state.compPanY = 0;
      } else {
        const ratio = next / state.compZoom;
        state.compPanX = ox - (ox - state.compPanX) * ratio;
        state.compPanY = oy - (oy - state.compPanY) * ratio;
        state.compZoom = next;
      }
      drawCompCanvas();
    }, { passive: false });

    canvas.addEventListener('pointerdown', (e) => {
      if (!state.compLeftImg) return;
      if (e.button === 1 || state.spaceDown || e.shiftKey) {
        state.isPanningComp = true;
        state.compPanStart = { x: e.clientX, y: e.clientY, px: state.compPanX, py: state.compPanY };
      } else {
        state.isDraggingWipe = true;
        const rect = canvas.getBoundingClientRect();
        state.wipeRatio = Math.max(0.02, Math.min(0.98, (e.clientX - rect.left) / Math.max(1, rect.width)));
        drawCompCanvas();
      }
      canvas.setPointerCapture(e.pointerId);
    });

    canvas.addEventListener('pointermove', (e) => {
      if (state.isPanningComp) {
        state.compPanX = state.compPanStart.px + (e.clientX - state.compPanStart.x);
        state.compPanY = state.compPanStart.py + (e.clientY - state.compPanStart.y);
        drawCompCanvas();
      } else if (state.isDraggingWipe) {
        const rect = canvas.getBoundingClientRect();
        state.wipeRatio = Math.max(0.02, Math.min(0.98, (e.clientX - rect.left) / Math.max(1, rect.width)));
        drawCompCanvas();
      }
    });

    const stopComp = () => { state.isDraggingWipe = false; state.isPanningComp = false; };
    canvas.addEventListener('pointerup', stopComp);
    canvas.addEventListener('pointercancel', stopComp);
    canvas.addEventListener('dblclick', () => {
      state.compZoom = 1;
      state.compPanX = 0;
      state.compPanY = 0;
      state.wipeRatio = 0.5;
      drawCompCanvas();
    });
  }

  // ------------------------------------------------------------------
  // Final Render & Auto-Cleanup Download
  // ------------------------------------------------------------------
  async function startFinalRender() {
    if (!state.sid || !state.srcMeta) return showToast('Hãy tải Video/Ảnh gốc ở Bước 1 trước.', 'err');
    const btn = $('btnRenderFinal');
    const jobBox = $('renderJobBox');
    const dlBox = $('downloadBox');
    const qaBox = $('qaReportBox');
    btn.disabled = true;
    jobBox.hidden = false;
    dlBox.hidden = true;
    if (qaBox) qaBox.hidden = true;
    $('renderProgressFill').style.width = '4%';
    $('renderStatusMsg').textContent = 'Đang gửi tác vụ render...';

    try {
      const payload = buildPreviewPayload({
        tracking_mode: $('selTracking').value,
        audio_choice: $('selAudio').value,
        matanyone_max_size: Number($('selMatRes').value),
        auto_qa_heal: Boolean($('chkAutoQaHeal')?.checked ?? true),
      });
      const res = await apiJson(`/session/${state.sid}/render`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      pollRenderJob(res.job_id);
    } catch (err) {
      btn.disabled = false;
      showToast(`Lỗi bắt đầu render: ${err.message}`, 'err', 5000);
    }
  }

  async function pollRenderJob(jobId) {
    clearTimeout(state.renderTimer);
    try {
      const st = await apiJson(`/session/${state.sid}/render/${jobId}`);
      $('renderProgressFill').style.width = `${Math.round((st.progress || 0) * 100)}%`;
      $('renderStatusMsg').textContent = st.message || '';
      if (st.disk_mb !== undefined) updateDiskBadge(st.disk_mb);

      if (st.qa_report) renderQaReport(st.qa_report);

      if (st.status === 'done') {
        $('btnRenderFinal').disabled = false;
        $('downloadBox').hidden = false;
        $('downloadVideoLink').href = `${API_BASE}/session/${state.sid}/download?kind=video&job_id=${jobId}&cleanup=1`;
        showToast('Xuất video hoàn tất! Bấm tải về để lưu máy và xóa sạch dữ liệu tạm.', 'ok', 6000);
        return;
      }
      if (st.status === 'error') {
        $('btnRenderFinal').disabled = false;
        showToast(st.message || 'Render lỗi', 'err', 6000);
        return;
      }
      state.renderTimer = setTimeout(() => pollRenderJob(jobId), 1200);
    } catch (err) {
      $('btnRenderFinal').disabled = false;
      showToast(`Lỗi kiểm tra tiến trình: ${err.message}`, 'err');
    }
  }

  function renderQaReport(qa) {
    const box = $('qaReportBox');
    const badge = $('qaScoreBadge');
    const summary = $('qaSummaryMsg');
    const grid = $('qaThumbGrid');
    if (!box || !qa || !qa.audited_count) return;
    box.hidden = false;
    badge.textContent = `${qa.integrity_score ?? 100}%`;
    badge.className = 'badge ' + ((qa.flagged_frames || 0) > 0 ? 'warn' : 'ok');
    summary.textContent = qa.summary || '';
    grid.innerHTML = '';
    (qa.samples || []).forEach((s) => {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'qa-sample-card' + (s.status === 'healed' ? ' healed' : '');
      card.title = 'Bấm để nhảy tới khung hình này ở Bước 1';
      if (s.diff_b64) {
        const im = document.createElement('img');
        im.src = s.diff_b64;
        im.alt = `Frame ${s.frame_idx}`;
        card.appendChild(im);
      }
      const cap = document.createElement('span');
      cap.textContent = s.status === 'healed' ? `F${s.frame_idx} · Đã vá` : `F${s.frame_idx} · OK`;
      card.appendChild(cap);
      card.addEventListener('click', () => {
        switchStep('cut');
        $('cutFrameSlider').value = s.frame_idx;
        loadCutFrame(s.frame_idx);
      });
      grid.appendChild(card);
    });
  }

  async function manualCleanup() {
    if (!state.sid) return;
    try {
      const res = await apiJson(`/session/${state.sid}/cleanup`, { method: 'POST' });
      state.sid = null;
      state.srcMeta = null;
      state.bgMeta = null;
      state.cutImg = null;
      state.compLeftImg = null;
      state.compRightImg = null;
      clearStrokeLayers();
      renderConfirmedPills([]);
      updateDiskBadge(0);
      $('srcMetaTag').textContent = 'Chưa có file';
      $('srcMetaTag').classList.remove('ok');
      $('bgMetaTag').textContent = 'Mặc định';
      $('bgMetaTag').classList.remove('ok');
      $('cutEmptyOverlay').hidden = false;
      $('compEmptyOverlay').hidden = false;
      $('downloadBox').hidden = true;
      $('renderJobBox').hidden = true;
      drawCutCanvas();
      drawCompCanvas();
      showToast(`Đã xóa sạch ${res.freed_mb} MB dữ liệu tạm trên máy chủ.`, 'ok');
    } catch (err) {
      showToast(`Lỗi xóa tạm: ${err.message}`, 'err');
    }
  }

  function wireDropZone(zoneId, inputId, role) {
    const zone = $(zoneId);
    const input = $(inputId);
    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) uploadFileChunked(role, e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => {
      if (input.files && input.files[0]) uploadFileChunked(role, input.files[0]);
      input.value = '';
    });
  }

  // ------------------------------------------------------------------
  // Bind Events on Boot
  // ------------------------------------------------------------------
  function bindAll() {
    initTheme();
    wireDropZone('srcDropZone', 'srcFileInput', 'source');
    wireDropZone('bgDropZone', 'bgFileInput', 'background');
    setupCutCanvasEvents();
    setupCompCanvasEvents();

    $('btnPickSourceHero').addEventListener('click', () => $('srcFileInput').click());
    $('themeBtn').addEventListener('click', () => {
      const next = document.body.classList.contains('dark') ? 'light' : 'dark';
      localStorage.setItem('ui_theme', next);
      applyTheme(next);
    });
    $('cleanupBtn').addEventListener('click', manualCleanup);

    document.querySelectorAll('.step-tab').forEach((btn) => {
      btn.addEventListener('click', () => switchStep(btn.dataset.step));
    });

    $('btnAutoBirefnet').addEventListener('click', () => runAutoCut('birefnet'));
    $('btnAutoHands').addEventListener('click', () => runAutoCut('hands_product'));
    $('btnAutoChroma').addEventListener('click', () => runAutoCut('chroma'));
    $('btnApplyStrokes').addEventListener('click', submitStrokes);

    document.querySelectorAll('[data-brush]').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.brushMode = btn.dataset.brush;
        document.querySelectorAll('[data-brush]').forEach((b) => b.classList.toggle('active', b === btn));
      });
    });

    $('brushSizeInput').addEventListener('input', (e) => {
      state.brushSize = Number(e.target.value);
      $('brushSizeVal').textContent = `${state.brushSize}px`;
    });

    $('btnUndoMask').addEventListener('click', async () => {
      if (!state.sid) return;
      try {
        const res = await apiJson(`/session/${state.sid}/undo`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frame_idx: state.cutFrame }),
        });
        renderConfirmedPills(res.confirmed_frames);
        if (res.disk_mb !== undefined) updateDiskBadge(res.disk_mb);
        await loadCutFrame(state.cutFrame);
        showToast(res.ok ? 'Đã hoàn tác nét trước.' : (res.message || 'Không còn bước hoàn tác.'), res.ok ? 'ok' : 'info');
      } catch (err) {
        showToast(`Lỗi hoàn tác: ${err.message}`, 'err');
      }
    });

    $('btnResetMask').addEventListener('click', async () => {
      clearStrokeLayers();
      drawCutCanvas();
      if (!state.sid) return;
      try {
        const res = await apiJson(`/session/${state.sid}/reset-mask`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frame_idx: state.cutFrame }),
        });
        renderConfirmedPills(res.confirmed_frames);
        if (res.disk_mb !== undefined) updateDiskBadge(res.disk_mb);
        await loadCutFrame(state.cutFrame);
        showToast('Đã xóa vùng chọn khung hiện tại.', 'ok');
      } catch (err) {
        showToast(`Lỗi xóa vùng chọn: ${err.message}`, 'err');
      }
    });

    document.querySelectorAll('[data-cutview]').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.cutView = btn.dataset.cutview;
        document.querySelectorAll('[data-cutview]').forEach((b) => b.classList.toggle('active', b === btn));
        loadCutFrame(state.cutFrame);
      });
    });

    $('cutZoomFitBtn').addEventListener('click', () => { state.cutZoom = 1; state.cutPanX = 0; state.cutPanY = 0; drawCutCanvas(); });
    $('cutZoom2xBtn').addEventListener('click', () => { state.cutZoom = 2; drawCutCanvas(); });

    $('cutFrameSlider').addEventListener('input', (e) => {
      const f = Number(e.target.value);
      $('compFrameSlider').value = f;
      $('compFrameText').textContent = `${f} / ${$('compFrameSlider').max}`;
      loadCutFrame(f);
    });

    $('btnDownloadCutout').addEventListener('click', async () => {
      if (!state.sid || !state.srcMeta) return showToast('Hãy tải ảnh/video lên trước.', 'err');
      try {
        const eng = encodeURIComponent($('engineSelect').value);
        const pymat = $('chkPyMatting').checked ? 1 : 0;
        const url = `${API_BASE}/session/${state.sid}/download?kind=cutout&frame_idx=${state.cutFrame}&cleanup=0&engine=${eng}&use_pymatting_fg=${pymat}`;
        const res = await fetch(url);
        if (!res.ok) {
          const text = await res.text().catch(() => '');
          let data = {};
          try { data = text ? JSON.parse(text) : {}; } catch (_) {}
          throw new Error(data.detail || `HTTP ${res.status}`);
        }
        const blob = await res.blob();
        const objUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = objUrl;
        a.download = `tach_nen_f${String(state.cutFrame).padStart(5, '0')}.png`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(objUrl), 2000);
      } catch (err) {
        showToast(`Lỗi tải PNG tách nền: ${err.message}`, 'err', 4500);
      }
    });

    $('btnGoPrompt').addEventListener('click', () => switchStep('prompt'));
    $('btnGoComposite').addEventListener('click', () => switchStep('composite'));

    // Step 2
    $('btnAnalyzePrompts').addEventListener('click', analyzeAndGeneratePrompts);
    document.querySelectorAll('#bgPresets .chip').forEach((c) => {
      c.addEventListener('click', () => { $('targetBgText').value = c.dataset.bg; });
    });
    document.querySelectorAll('.copy-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const target = $(btn.dataset.copy);
        if (!target || !target.value) return showToast('Chưa có Prompt để sao chép.', 'err');
        try {
          await navigator.clipboard.writeText(target.value);
          showToast('Đã copy Prompt vào Clipboard!', 'ok');
        } catch (_) {
          target.select();
          document.execCommand('copy');
          showToast('Đã copy Prompt!', 'ok');
        }
      });
    });

    // Step 3
    document.querySelectorAll('.itab').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.itab').forEach((b) => b.classList.toggle('active', b === btn));
        document.querySelectorAll('.itab-pane').forEach((p) => p.classList.toggle('active', p.dataset.pane === btn.dataset.itab));
      });
    });

    document.querySelectorAll('#diagModeGroup .seg-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.diagMode = btn.dataset.diag;
        document.querySelectorAll('#diagModeGroup .seg-btn').forEach((b) => b.classList.toggle('active', b === btn));
        schedulePreview(true);
      });
    });

    document.querySelectorAll('.comp-zoom-preset').forEach((btn) => {
      btn.addEventListener('click', () => applyCompZoomPreset(Number(btn.dataset.zoom), Number(btn.dataset.px), Number(btn.dataset.py)));
    });

    $('compFrameSlider').addEventListener('input', (e) => {
      const f = Number(e.target.value);
      $('compFrameText').textContent = `${f} / ${$('compFrameSlider').max}`;
      state.compFrame = f;
      schedulePreview(false, 180);
    });
    $('compFrameSlider').addEventListener('change', (e) => {
      const f = Number(e.target.value);
      $('compFrameText').textContent = `${f} / ${$('compFrameSlider').max}`;
      state.compFrame = f;
      schedulePreview(true);
    });

    $('btnRefreshPreview').addEventListener('click', () => schedulePreview(true));

    document.querySelectorAll('.live-param').forEach((el) => {
      const valSpan = document.querySelector(`[data-val-for="${el.id}"]`);
      const syncVal = () => {
        if (valSpan) valSpan.textContent = el.value;
      };
      syncVal();
      el.addEventListener('input', () => {
        syncVal();
        schedulePreview(false);
      });
      if (el.type === 'checkbox') {
        el.addEventListener('change', () => schedulePreview(true));
      }
    });

    $('btnRenderFinal').addEventListener('click', startFinalRender);
    $('downloadVideoLink').addEventListener('click', () => {
      showToast('Đang tải video về máy — dữ liệu trên máy chủ sẽ tự động xóa sau khi tải xong.', 'ok', 6500);
      setTimeout(async () => {
        try {
          const h = await apiJson('/health');
          updateDiskBadge(0);
          if (h.active_sessions === 0) $('diskBadge').textContent = '✅ Đã dọn máy chủ';
        } catch (_) {}
      }, 4500);
    });

    window.addEventListener('keydown', (e) => {
      const tag = document.activeElement?.tagName;
      const inInput = tag === 'TEXTAREA' || tag === 'INPUT';
      if (e.code === 'Space' && !inInput) {
        state.spaceDown = true;
      } else if (!inInput && state.currentStep === 'cut') {
        if (e.key === '[') {
          state.brushSize = Math.max(4, state.brushSize - 4);
          $('brushSizeInput').value = state.brushSize;
          $('brushSizeVal').textContent = `${state.brushSize}px`;
          drawCutCanvas();
        } else if (e.key === ']') {
          state.brushSize = Math.min(72, state.brushSize + 4);
          $('brushSizeInput').value = state.brushSize;
          $('brushSizeVal').textContent = `${state.brushSize}px`;
          drawCutCanvas();
        } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
          e.preventDefault();
          $('btnUndoMask').click();
        }
      }
    });
    window.addEventListener('keyup', (e) => {
      if (e.code === 'Space') state.spaceDown = false;
    });
    window.addEventListener('resize', () => {
      drawCutCanvas();
      drawCompCanvas();
    });

    // Clean up orphaned temp session when user closes the tab
    window.addEventListener('pagehide', () => {
      if (state.sid && navigator.sendBeacon) {
        navigator.sendBeacon(`${API_BASE}/session/${state.sid}/cleanup`, new Blob());
      }
    });

    apiJson('/health').then((h) => {
      const gpuEl = $('gpuBadge');
      if (gpuEl) gpuEl.textContent = h.cuda_available ? 'RTX CUDA Ready' : 'CPU Mode';
    }).catch(() => {});
  }

  bindAll();
})();

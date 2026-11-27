from __future__ import annotations

import argparse
import shutil
import os
import subprocess
import tempfile
import json
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import UnidentifiedImageError

from src.alpha_matting import (
    MattingConfig,
    auto_chroma_key_mask,
    auto_seed_roto_strokes,
    detect_active_frame_box,
    expand_mask_with_color_evidence,
    _propagate_local_colors,
    decontaminate_foreground_edges,
    extract_pixel_accurate_matte,
    extract_strokes_from_editor,
    sample_roto_anchor_points,
    stabilize_alpha_temporal,
)
from src.flow_prompt_engine import analyze_scene_telemetry, generate_flow_prompts
from src.neural_matting import (
    estimate_foreground_pymatting,
    run_birefnet_matting,
    run_matanyone2_video,
)
from src.relight_harmonizer import (
    RelightConfig,
    composite_with_relighting,
    detect_emissive_screen_mask,
)
from src.segmentation import (
    clear_vram,
    preview_single_frame_mask,
    refine_mask_with_sam2_negative,
    temporal_smooth_mask,
    track_roto_video_masks,
)
from src.spatial_align import (
    estimate_pair_similarity,
    estimate_source_camera_trajectory,
    smooth_transforms,
    warp_ai_frame,
    warp_bg_with_camera_motion,
)
from src.temporal_sync import compute_temporal_mapping
from src.video_io import encode_video, extract_frames, get_video_metadata, load_mask, save_mask

MATTING_ENGINE_MATANYONE2 = "MatAnyone 2 + ViTMatte-Base (Tốt nhất Video & Ảnh)"
MATTING_ENGINE_BIREFNET = "BiRefNet HR-Matting (MIT Thương mại — 1-Click siêu sạch)"
MATTING_ENGINE_VITMATTE = "ViTMatte-Base + SAM 2.1 (MIT/Apache Thương mại)"
MATTING_ENGINE_CLASSIC = "Classic CPU Guided Filter (Chế độ cũ)"
MATTING_ENGINE_CHOICES = [
    MATTING_ENGINE_MATANYONE2,
    MATTING_ENGINE_BIREFNET,
    MATTING_ENGINE_VITMATTE,
    MATTING_ENGINE_CLASSIC,
]


def _is_ai_engine(engine_label: str) -> bool:
    return engine_label != MATTING_ENGINE_CLASSIC


def _resolve_matting_cfg(
    engine_label: str,
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    use_pymatting_fg: bool,
    two_pass_verify: bool,
) -> MattingConfig:
    if engine_label == MATTING_ENGINE_BIREFNET:
        engine_key = "birefnet"
    elif engine_label in (MATTING_ENGINE_MATANYONE2, MATTING_ENGINE_VITMATTE):
        engine_key = "vitmatte"
    else:
        engine_key = "classic"
    return MattingConfig(
        trimap_radius=int(trimap_radius),
        guided_radius=int(guided_radius),
        decontaminate_strength=float(decont_strength),
        despill_green=bool(despill_green),
        erode_dilate=int(erode_dilate),
        two_pass_verify=bool(two_pass_verify),
        use_vitmatte_gpu=(engine_key == "vitmatte"),
        engine=engine_key,
        use_pymatting_fg=bool(use_pymatting_fg),
    )


def _alpha_preview_from_cutout_path(png_path) -> object:
    if not isinstance(png_path, (str, Path)):
        return gr.update()
    p = Path(png_path)
    if not p.is_file():
        return gr.update()
    rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        return gr.update()
    alpha_u8 = rgba[..., 3]
    return np.stack([alpha_u8, alpha_u8, alpha_u8], axis=-1)

APP_WORK_DIR = Path(__file__).resolve().parent / "workspace" / "roto_studio_run"
_PREVIEW_CACHE = OrderedDict()
_PREVIEW_CACHE_LOCK = threading.Lock()
_BG_FRAME_CACHE = OrderedDict()
_BG_FRAME_CACHE_LOCK = threading.Lock()
_COMPOSITE_RESULT_CACHE = OrderedDict()
_COMPOSITE_RESULT_CACHE_LOCK = threading.Lock()

PREVIEW_DIAG_SPLIT = "Gốc ⟷ Ghép nền"
PREVIEW_DIAG_BLACK = "Check Halo — Nền Đen"
PREVIEW_DIAG_WHITE = "Check Halo — Nền Trắng"
PREVIEW_DIAG_SCREEN = "Soi Vùng Màn Hình (Check lấn tay)"
PREVIEW_DIAG_CHOICES = [
    PREVIEW_DIAG_SPLIT,
    PREVIEW_DIAG_BLACK,
    PREVIEW_DIAG_WHITE,
    PREVIEW_DIAG_SCREEN,
]


class PreviewMaskUnavailable(Exception):
    """Raised when a non-anchor preview has no current-frame tracked mask."""


CUSTOM_CSS = """
.gradio-container { max-width: 1620px !important; margin: 0 auto !important; padding: 12px 18px !important; font-family: 'Inter', 'Segoe UI', sans-serif !important; }
.hero-header { background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(16,185,129,0.10)); border: 1px solid rgba(59,130,246,0.28); border-radius: 14px; padding: 14px 18px; margin-bottom: 10px; }
.step-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(148,163,184,0.24); border-radius: 14px; padding: 14px !important; margin-bottom: 12px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.step-title { font-size: 1.05rem !important; font-weight: 700 !important; margin-bottom: 6px !important; }
.compact-btn { min-height: 38px !important; font-weight: 600 !important; border-radius: 10px !important; }
.preset-row button { border-radius: 999px !important; font-size: 12px !important; padding: 4px 10px !important; }
.status-bar textarea { font-weight: 600 !important; font-size: 13.5px !important; border-radius: 10px !important; }
.zoom-inspect-box { overflow: hidden !important; position: relative !important; border-radius: 12px !important; border: 1px solid rgba(59,130,246,0.32) !important; background: #0f172a !important; }
.zoom-inspect-box img { transform-origin: center center !important; will-change: transform !important; }
.zoom-hint-badge { position: absolute; right: 10px; bottom: 10px; z-index: 25; pointer-events: none; background: rgba(15,23,42,0.78); color: #e2e8f0; border: 1px solid rgba(148,163,184,0.35); border-radius: 999px; padding: 3px 10px; font-size: 11px; font-weight: 600; backdrop-filter: blur(4px); }
"""

ZOOM_JS = """
() => {
  const bindZoomBox = (box) => {
    if (!box || box.dataset.zoomBound === '1') return;
    box.dataset.zoomBound = '1';
    let scale = 1.0, tx = 0.0, ty = 0.0;
    let dragging = false, startX = 0, startY = 0;

    let badge = box.querySelector('.zoom-hint-badge');
    if (!badge) {
      badge = document.createElement('div');
      badge.className = 'zoom-hint-badge';
      box.appendChild(badge);
    }

    const apply = () => {
      const imgs = box.querySelectorAll('img');
      imgs.forEach((img) => {
        img.style.transformOrigin = 'center center';
        img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
      });
      badge.textContent = scale <= 1.01
        ? '1.0x | Lăn chuột: Zoom | Click đúp: Reset'
        : `${scale.toFixed(2)}x | Shift+Kéo: Di chuyển | Click đúp: Reset`;
    };
    apply();

    box.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.18 : 1 / 1.18;
      const next = Math.min(6.0, Math.max(1.0, scale * factor));
      if (next <= 1.02) {
        scale = 1.0; tx = 0.0; ty = 0.0;
      } else {
        const rect = box.getBoundingClientRect();
        const ox = e.clientX - (rect.left + rect.width / 2);
        const oy = e.clientY - (rect.top + rect.height / 2);
        const ratio = next / scale;
        tx = ox - (ox - tx) * ratio;
        ty = oy - (oy - ty) * ratio;
        scale = next;
      }
      apply();
    }, { passive: false });

    box.addEventListener('pointerdown', (e) => {
      if ((e.shiftKey || e.button === 1) && scale > 1.01) {
        dragging = true;
        startX = e.clientX - tx;
        startY = e.clientY - ty;
        e.preventDefault();
      }
    });

    window.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      tx = e.clientX - startX;
      ty = e.clientY - startY;
      apply();
    });

    window.addEventListener('pointerup', () => { dragging = false; });

    box.addEventListener('dblclick', () => {
      scale = 1.0; tx = 0.0; ty = 0.0;
      apply();
    });
  };

  const tryBindAll = () => {
    const box = document.getElementById('comp-zoom-box');
    if (box) bindZoomBox(box);
  };
  tryBindAll();
  const obs = new MutationObserver(tryBindAll);
  obs.observe(document.body, { childList: true, subtree: true });
}
"""


class StrokeImageEditor(gr.ImageEditor):
    """Decode only brush layers; the source frame is loaded from session state."""

    def get_block_name(self):
        return "imageeditor"

    def preprocess(self, payload):
        if payload is None:
            return None
        cached = self.blob_storage.get(payload.id) if payload.id is not None else payload
        try:
            layers = [self.convert_and_format_image(layer) for layer in cached.layers] if cached and cached.layers else []
        except (OSError, UnidentifiedImageError) as exc:
            raise gr.Error("Không thể đọc nét cọ. Hãy thử vẽ lại nét vừa rồi.") from exc
        finally:
            if payload.id is not None:
                self.blob_storage.pop(payload.id, None)
        return {"background": None, "layers": [layer for layer in layers if layer is not None], "composite": None}


def _load_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {path}")
    return img


def _load_rgb(path: str | Path) -> np.ndarray:
    return cv2.cvtColor(_load_bgr(path), cv2.COLOR_BGR2RGB)


def _confirmed_reference_frame(confirmed_masks, frame_idx: int):
    return min(
        confirmed_masks,
        key=lambda k: (int(k) > frame_idx, abs(int(k) - frame_idx)),
        default=None,
    )


def _make_checkerboard_cutout(fg_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Render sub-pixel alpha cutout over a professional dark diagonal/grid checkerboard (returns RGB)."""
    h, w = fg_bgr.shape[:2]
    checker = (np.indices((h, w)).sum(axis=0) // 16) % 2
    bg = np.where(checker[..., None] == 0, 42, 74).astype(np.float32)
    a3 = np.clip(alpha, 0.0, 1.0)[..., None]
    comp_bgr = np.clip(bg * (1.0 - a3) + fg_bgr.astype(np.float32) * a3, 0, 255).astype(np.uint8)
    return cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)


def _render_photoshop_green_selection(rgb: np.ndarray, mask: Optional[np.ndarray], alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """Photoshop-style cumulative emerald-green overlay + crisp outline directly on the editor canvas."""
    if mask is None or not np.any(mask > 0):
        return rgb
    out = rgb.copy()
    coverage = np.clip(alpha, 0.0, 1.0) if alpha is not None else (mask > 0).astype(np.float32)
    green_tint = np.array([0, 235, 90], dtype=np.float32)
    weight = coverage[..., None] * 0.35
    out = np.clip(out.astype(np.float32) * (1.0 - weight) + green_tint * weight, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours((coverage >= 0.5).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 110), 2, lineType=cv2.LINE_AA)
    return out


def _saved_selection_alpha(state: dict, frame_idx: int, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    confirmed = state.get("confirmed_masks", {}).get(int(frame_idx))
    if not confirmed:
        return None
    png = Path(state.get("work_dir", APP_WORK_DIR)) / f"subpixel_cutout_f{int(frame_idx):05d}.png"
    mask_path = Path(confirmed)
    if not png.exists() or not mask_path.exists() or png.stat().st_mtime_ns < mask_path.stat().st_mtime_ns:
        return None
    saved_mask = load_mask(mask_path)
    if saved_mask.shape != mask.shape or not np.array_equal(saved_mask > 0, mask > 0):
        return None
    rgba = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.shape[:2] != mask.shape:
        return None
    return rgba[..., 3].astype(np.float32) / 255.0


def _draw_roto_confirm_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    fg_stroke: Optional[np.ndarray] = None,
    bg_stroke: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw After Effects Roto-Brush style confirmation boundary (Magenta contour + dimmed exterior, returns RGB)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    m = (mask > 0).astype(np.uint8)
    if np.any(m > 0):
        # Dim background slightly so the confirmed Roto foreground pops clearly
        bg_zone = m == 0
        rgb[bg_zone] = (rgb[bg_zone].astype(np.float32) * 0.55).astype(np.uint8)
        contours, _ = cv2.findContours(m * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (255, 40, 220), 2)

    if fg_stroke is not None and np.any(fg_stroke > 0):
        overlay = rgb.copy()
        overlay[fg_stroke > 0] = (40, 245, 95)
        rgb = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)
    if bg_stroke is not None and np.any(bg_stroke > 0):
        overlay = rgb.copy()
        overlay[bg_stroke > 0] = (245, 55, 55)
        rgb = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)
    return rgb


def _extract_media_frames(
    media_path: str,
    out_dir: Path,
    auto_crop_bars: bool = True,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Path], float, int, int, bool]:
    """Support both Video (.mp4/.mov/.avi) and single Image (.jpg/.png/.webp) inputs seamlessly."""
    p = Path(media_path)
    ext = p.suffix.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        bgr = _load_bgr(p)
        was_cropped = False
        if auto_crop_bars:
            x0, y0, x1, y1 = detect_active_frame_box(bgr)
            if (x0, y0, x1, y1) != (0, 0, bgr.shape[1], bgr.shape[0]):
                pad_x = 2 if (x1 - x0) > 40 else 0
                bgr = bgr[y0:y1, x0 + pad_x: x1 - pad_x].copy()
                was_cropped = True
        if target_size is not None and (bgr.shape[1], bgr.shape[0]) != target_size:
            bgr = cv2.resize(bgr, target_size, interpolation=cv2.INTER_CUBIC)
        fp = out_dir / "00000.jpg"
        cv2.imwrite(str(fp), bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
        return [fp], 30.0, bgr.shape[1], bgr.shape[0], was_cropped

    meta = get_video_metadata(p)
    raw_frames = extract_frames(p, out_dir, target_size=target_size)
    was_cropped = False
    if auto_crop_bars and target_size is None and raw_frames:
        sample_bgr = _load_bgr(raw_frames[0])
        x0, y0, x1, y1 = detect_active_frame_box(sample_bgr)
        if (x0, y0, x1, y1) != (0, 0, sample_bgr.shape[1], sample_bgr.shape[0]):
            pad_x = 2 if (x1 - x0) > 40 else 0
            cx0, cy0, cx1, cy1 = x0 + pad_x, y0, x1 - pad_x, y1
            for fp in raw_frames:
                f_bgr = _load_bgr(fp)[cy0:cy1, cx0:cx1]
                cv2.imwrite(str(fp), f_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
            was_cropped = True
            sample_after = _load_bgr(raw_frames[0])
            return raw_frames, meta.fps, sample_after.shape[1], sample_after.shape[0], True

    sample_final = _load_bgr(raw_frames[0])
    return raw_frames, meta.fps, sample_final.shape[1], sample_final.shape[0], was_cropped


# =====================================================================
# BƯỚC 1: NẠP VIDEO GỐC, CẮT ẢNH THAM CHIẾU & PHÂN TÍCH QUANG HỌC
# =====================================================================
def on_step1_load_source(src_file: Optional[str], auto_crop_bars: bool, state: dict):
    if not src_file:
        raise gr.Error("Vui lòng chọn Video gốc (hoặc Ảnh gốc) cần tách chủ thể.")

    work = APP_WORK_DIR / uuid.uuid4().hex
    work.mkdir(parents=True, exist_ok=False)

    try:
        src_frames, fps, w, h, was_cropped = _extract_media_frames(
            src_file, work / "src_frames", auto_crop_bars=auto_crop_bars
        )
    except (OSError, RuntimeError, ValueError, IndexError, cv2.error) as exc:
        raise gr.Error("Không thể đọc video/ảnh. Hãy kiểm tra tệp và thử nạp lại.") from exc
    n_frames = len(src_frames)

    # Pick up to 6 key reference frames evenly spaced across the video
    ref_count = min(6, n_frames)
    ref_indices = sorted({int(round(i)) for i in np.linspace(0, n_frames - 1, ref_count)})
    ref_dir = work / "reference_frames"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_gallery_items = []
    for idx in ref_indices:
        ref_path = ref_dir / f"ref_frame_{idx:05d}.jpg"
        shutil.copy2(src_frames[idx], ref_path)
        ref_gallery_items.append((_load_rgb(ref_path), f"Frame #{idx}"))

    # Optical telemetry analysis on first/middle reference frame
    mid_idx = ref_indices[len(ref_indices) // 2]
    telemetry = analyze_scene_telemetry(_load_bgr(src_frames[mid_idx]))

    state = {
        "work_dir": str(work),
        "src_media": src_file,
        "src_frames": [str(p) for p in src_frames],
        "ref_indices": ref_indices,
        "fps": fps,
        "width": w,
        "height": h,
        "telemetry": telemetry,
        "confirmed_masks": {},      # frame_idx -> mask path
        "active_mask_by_frame": {}, # frame_idx -> cumulative binary np.ndarray mask (Photoshop 1-stroke)
        "roto_strokes": {},         # frame_idx -> (fg_stroke, bg_stroke)
        "tracked_masks": None,      # list of all per-frame mask paths
        "alpha_mattes_dir": None,
        "bg_media": None,
        "bg_frames": None,
    }

    crop_note = " (Đã tự động cắt bỏ viền đen 2 bên)" if was_cropped else ""
    telemetry_md = (
        f"**Thông số Video/Ảnh gốc đã phân tích:**\n"
        f"- **Số khung hình:** `{n_frames} frames` @ `{fps:.1f} FPS` | **Kích thước khung thật:** `{w}x{h}` (`{telemetry['active_aspect_ratio']}`){crop_note}\n"
        f"- **Tiêu cự & Tỷ lệ phối cảnh:** `{telemetry['focal_length_mm']}mm` (~`{telemetry['subject_distance_cm']}cm`) — {telemetry['scale_calibration_vi']}\n"
        f"- **Góc máy & Bố cục:** {telemetry['camera_angle_vi']}\n"
        f"- **Quang học Điện thoại:** {telemetry['bokeh_blur_level']}\n"
        f"- **Ánh sáng & Nhiệt độ màu gốc:** {telemetry['lighting_direction']} | {telemetry['color_temperature']}"
    )

    f0_rgb = _load_rgb(src_frames[0])
    return (
        state,
        ref_gallery_items,
        telemetry_md,
        gr.Slider(minimum=0, maximum=max(1, n_frames - 1), step=1, value=0),
        f0_rgb,
        f"Bước 1 hoàn tất: Đã trích xuất {n_frames} frames và cắt {len(ref_indices)} ảnh tham chiếu. Bôi 1 nét xanh lên máy/tay để tự động chọn toàn bộ vật thể!",
    )


def on_select_ref_gallery(state: dict, evt: gr.SelectData):
    if not state.get("src_frames") or not state.get("ref_indices"):
        return 0, None
    ref_indices = state["ref_indices"]
    sel = int(evt.index) if evt.index is not None else 0
    f_idx = ref_indices[min(sel, len(ref_indices) - 1)]
    rgb = _load_rgb(state["src_frames"][f_idx])
    active_m = state.get("active_mask_by_frame", {}).get(int(f_idx))
    return f_idx, _render_photoshop_green_selection(rgb, active_m, _saved_selection_alpha(state, f_idx, active_m))


def on_frame_slider_change(frame_idx: int, state: dict):
    frames = state.get("src_frames")
    if not frames:
        return None
    f_idx = int(np.clip(frame_idx, 0, len(frames) - 1))
    rgb = _load_rgb(frames[f_idx])
    active_m = state.get("active_mask_by_frame", {}).get(int(f_idx))
    return _render_photoshop_green_selection(rgb, active_m, _saved_selection_alpha(state, f_idx, active_m))


# =====================================================================
# BƯỚC 2: CÔNG CỤ BÔI CHỌN ROTO-BRUSH & CONFIRM VÙNG CHỌN THÔNG MINH
# =====================================================================
AUTO_ROTO_GRID_STEP = 32


def _protect_manual_background(alpha, state, frame_idx):
    bg = state.get("manual_strokes", {}).get(int(frame_idx), {}).get("bg")
    if bg is None or not np.any(bg):
        return alpha
    result = alpha.copy()
    result[bg > 0] = 0
    return result


def _run_sam2matting_tiny(src_frames, tracked_masks, confirmed_masks, output_dir: Path, frame_shape, progress):
    """Run the installed Tiny helper in bounded chunks with one predictor load."""
    repo_root = Path(__file__).resolve().parent
    config_path = repo_root / "workspace" / "sam2matting_local.json"
    home = os.environ.get("SAM2MATTING_HOME")
    if not home and config_path.is_file():
        home = json.loads(config_path.read_text(encoding="utf-8")).get("home")
    if not home:
        raise FileNotFoundError(f"Chưa cấu hình SAM2Matting. Tạo {config_path} với trường home hoặc đặt SAM2MATTING_HOME.")
    home = Path(home)
    runner = Path(os.environ.get("SAM2MATTING_RUNNER", home / "sam2_eval" / "run_video_matting.py"))
    python = Path(os.environ.get("SAM2MATTING_PYTHON", home / "sam2matting_env" / "Scripts" / "python.exe"))
    for path in (runner, python):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp SAM2Matting: {path}")

    anchors = {int(k): Path(v) for k, v in confirmed_masks.items()}
    if not anchors:
        raise ValueError("SAM2Matting Tiny cần ít nhất một vùng chọn đã xác nhận.")
    n_frames = len(src_frames)
    bounds = sorted(anchors)
    intervals = []
    if bounds[0] > 0:
        intervals.append((0, bounds[0], None))
    for i, start in enumerate(bounds):
        intervals.append((start, bounds[i + 1] if i + 1 < len(bounds) else n_frames, anchors[start]))

    stage = output_dir / "inputs"
    stage.mkdir(parents=True, exist_ok=True)
    manifest = []
    for start, stop, anchor_mask in intervals:
        for chunk_start in range(start, stop, 28):
            chunk_stop = min(stop, chunk_start + 32)
            input_dir = stage / f"segment_{chunk_start:05d}"
            input_dir.mkdir(parents=True, exist_ok=True)
            for local_idx, frame_idx in enumerate(range(chunk_start, chunk_stop)):
                shutil.copy2(src_frames[frame_idx], input_dir / f"{local_idx:05d}.jpg")
            seed_path = stage / f"seed_{chunk_start:05d}.png"
            seed = load_mask(anchor_mask) if chunk_start == start and anchor_mask is not None else load_mask(tracked_masks[chunk_start])
            seed = (seed > 0).astype(np.uint8) * 255
            if not np.any(seed):
                raise ValueError(f"Seed mask SAM2Matting rỗng tại frame {chunk_start}.")
            if not cv2.imwrite(str(seed_path), seed):
                raise OSError(f"Không ghi được seed mask frame {chunk_start}.")
            manifest.append({"frames": str(input_dir), "mask": str(seed_path), "output": f"segment_{chunk_start:05d}", "start": chunk_start})
    manifest_path = stage / "segments.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    progress(0.40, desc=f"SAM2Matting Tiny GPU: {len(manifest)} đoạn tối đa 32 frame...")
    sub_kw = {}
    if os.name == "nt":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        sub_kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000), "startupinfo": si}
    result = subprocess.run(
        [str(python), str(runner), "--segments-json", str(manifest_path), "--output-dir", str(output_dir / "alpha")],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, **sub_kw,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "SAM2Matting Tiny thất bại.").strip()[-1200:])
    alpha_paths = [output_dir / "alpha" / f"{idx:05d}.png" for idx in range(n_frames)]
    covered_frames = set()
    for segment in manifest:
        segment_frames = sorted(Path(segment["frames"]).glob("*.jpg"))
        seg_start = int(segment["start"])
        seg_out = output_dir / "alpha" / segment["output"]
        for local_idx, _ in enumerate(segment_frames):
            source_alpha = cv2.imread(str(seg_out / f"{local_idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
            frame_idx = seg_start + local_idx
            if source_alpha is None or source_alpha.shape != frame_shape:
                raise ValueError(f"Alpha SAM2Matting thiếu hoặc sai kích thước tại frame {frame_idx}.")
            target = alpha_paths[frame_idx]
            covered_frames.add(frame_idx)
            if target.exists() and local_idx < 4:
                previous = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                new_weight = (local_idx + 1) / 5.0
                merged = np.round(previous * (1.0 - new_weight) + source_alpha * new_weight).astype(np.uint8)
                if not cv2.imwrite(str(target), merged):
                    raise OSError(f"Không ghi được alpha ghép tại frame {frame_idx}.")
            else:
                shutil.copy2(seg_out / f"{local_idx:05d}.png", target)
    if len(covered_frames) != n_frames:
        raise ValueError(f"SAM2Matting mới tạo alpha cho {len(covered_frames)}/{n_frames} frame.")
    return alpha_paths


def apply_photoshop_stroke_to_mask(
    rgb: np.ndarray,
    prev_mask: Optional[np.ndarray],
    fg_stroke: np.ndarray,
    bg_stroke: np.ndarray,
    manual_prompts: bool = False,
    grid_step: int = 28,
    *,
    snap_boundary: bool = True,
    prefer_best_checkpoint: bool = False,
) -> np.ndarray:
    """
    Photoshop Object Selection 1-Stroke Cumulative Logic:
      - Green stroke (`fg_stroke`): auto-detect the entire clicked/stroked object and UNION into `prev_mask`.
      - Red stroke (`bg_stroke`): auto-detect the unwanted region and SUBTRACT from `prev_mask`.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    has_fg = bool(np.any(fg_stroke > 0))
    has_bg = bool(np.any(bg_stroke > 0))

    if manual_prompts:
        if not has_fg:
            raise gr.Error("Hãy vẽ nét xanh lên chủ thể trước khi dùng nét đỏ.")
        pts = sample_roto_anchor_points(fg_stroke, bg_stroke, bgr, grid_step=grid_step, preserve_stroke_components=True)
        if not has_bg:
            pts = [point for point in pts if point[2] == 1]
        kw = {"manual_prompts": True}
        if not snap_boundary:
            kw["snap_boundary"] = False
        if prefer_best_checkpoint:
            kw["prefer_best_checkpoint"] = True
        return (preview_single_frame_mask(rgb, pts, **kw) > 0).astype(np.uint8)

    if has_fg:
        pts = sample_roto_anchor_points(fg_stroke, bg_stroke, bgr, grid_step=grid_step)
        seg_m = preview_single_frame_mask(rgb, pts) if any(p[2] == 1 for p in pts) else np.zeros(rgb.shape[:2], dtype=np.uint8)
        if prev_mask is not None and np.any(prev_mask > 0):
            mask = ((prev_mask > 0) | (seg_m > 0)).astype(np.uint8)
        else:
            mask = (seg_m > 0).astype(np.uint8)
        if has_bg:
            rm_dil = cv2.dilate((bg_stroke > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            mask[rm_dil > 0] = 0
        return mask

    if has_bg and prev_mask is not None and np.any(prev_mask > 0):
        # Segment the clicked unwanted part (passing deep core anchors as negatives if distant from stroke)
        empty = np.zeros_like(fg_stroke)
        rm_pts = sample_roto_anchor_points(bg_stroke, empty, bgr, grid_step=24)
        rm_mask = preview_single_frame_mask(rgb, rm_pts) if rm_pts else np.zeros_like(fg_stroke)
        rm_dil = cv2.dilate((bg_stroke > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        mask = ((prev_mask > 0) & (rm_mask == 0) & (rm_dil == 0)).astype(np.uint8)
        return mask

    return prev_mask.astype(np.uint8) if prev_mask is not None else np.zeros(rgb.shape[:2], dtype=np.uint8)


def _run_confirm_and_preview(
    f_idx: int,
    bgr: np.ndarray,
    fg_stroke: np.ndarray,
    bg_stroke: np.ndarray,
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    state: dict,
    direct_mask: Optional[np.ndarray] = None,
    manual_prompts: bool = False,
    grid_step: int = 28,
    explicit_bg: Optional[np.ndarray] = None,
    *,
    matting_engine: str = MATTING_ENGINE_CLASSIC,
    use_pymatting_fg: bool = False,
    two_pass_verify: bool = True,
    precomputed_alpha: Optional[np.ndarray] = None,
):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prev_mask = state.get("active_mask_by_frame", {}).get(int(f_idx))
    if direct_mask is not None:
        mask = (direct_mask > 0).astype(np.uint8)
    else:
        if not np.any(fg_stroke > 0) and not np.any(bg_stroke > 0):
            if prev_mask is not None and np.any(prev_mask > 0):
                mask = prev_mask
            else:
                raise gr.Error("Chưa có nét cọ Xanh Lá (+ Giữ lại). Hãy bôi 1 nét lên thiết bị/bàn tay hoặc bấm '✨ Tự Động Tách Nền AI 1-Click'!")
        else:
            mask = apply_photoshop_stroke_to_mask(
                rgb,
                prev_mask,
                fg_stroke,
                bg_stroke,
                manual_prompts=manual_prompts,
                grid_step=grid_step,
                snap_boundary=not _is_ai_engine(matting_engine),
                prefer_best_checkpoint=_is_ai_engine(matting_engine),
            )

    if explicit_bg is not None:
        mask = mask.copy()
        mask[explicit_bg > 0] = 0

    # Save confirmed binary mask & cumulative active mask for this keyframe
    state.setdefault("active_mask_by_frame", {})[int(f_idx)] = (mask > 0).astype(np.uint8)
    work = Path(state.get("work_dir", APP_WORK_DIR))
    conf_dir = work / "confirmed_masks"
    conf_dir.mkdir(parents=True, exist_ok=True)
    mask_path = conf_dir / f"conf_{f_idx:05d}.png"
    save_mask(mask, mask_path)
    if int(f_idx) not in state["confirmed_masks"] or prev_mask is None or not np.array_equal(mask > 0, prev_mask > 0):
        state["tracked_masks"] = None
    state["confirmed_masks"][int(f_idx)] = str(mask_path)
    state["roto_strokes"][int(f_idx)] = (fg_stroke, bg_stroke)

    if precomputed_alpha is not None:
        alpha = np.clip(precomputed_alpha.astype(np.float32), 0.0, 1.0)
        if explicit_bg is not None:
            alpha[explicit_bg > 0] = 0.0
        if use_pymatting_fg:
            fg_clean_bgr = estimate_foreground_pymatting(
                bgr, alpha, strength=float(decont_strength), despill_green=bool(despill_green)
            )
        else:
            img_f = bgr.astype(np.float32) / 255.0
            fg_core = alpha >= 0.95
            if np.count_nonzero(fg_core) < 64:
                fg_core = mask > 0
            fg_est = _propagate_local_colors(img_f, fg_core, sigma=5.0)
            fg_clean_bgr = np.clip(
                decontaminate_foreground_edges(
                    img_f,
                    alpha,
                    fg_est=fg_est,
                    strength=float(np.clip(decont_strength, 0.0, 1.0)),
                    despill_green=bool(despill_green),
                )
                * 255.0,
                0,
                255,
            ).astype(np.uint8)
        fg_clean_rgb = cv2.cvtColor(fg_clean_bgr, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([fg_clean_rgb, np.clip(alpha * 255.0, 0, 255).astype(np.uint8)])
        miss_px, exc_px = 0, 0
        state.setdefault("birefnet_alpha_by_frame", {})[int(f_idx)] = alpha
    else:
        state.get("birefnet_alpha_by_frame", {}).pop(int(f_idx), None)
        # Run Sub-Pixel Alpha Matting & Edge Color Decontamination on this frame
        mcfg = _resolve_matting_cfg(
            matting_engine,
            trimap_radius,
            guided_radius,
            decont_strength,
            despill_green,
            erode_dilate,
            use_pymatting_fg,
            two_pass_verify,
        )
        mres = extract_pixel_accurate_matte(bgr, mask, mcfg)
        if explicit_bg is not None:
            mres.alpha[explicit_bg > 0] = 0
            mres.rgba[explicit_bg > 0, 3] = 0
        alpha = mres.alpha
        fg_clean_bgr = mres.fg_clean_bgr
        rgba = mres.rgba
        miss_px = mres.missing_restored_px
        exc_px = mres.excess_removed_px

    # Save transparent RGBA PNG for instant download/inspection
    cutout_png_path = work / f"subpixel_cutout_f{f_idx:05d}.png"
    cv2.imwrite(
        str(cutout_png_path),
        cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
    )

    editor_green_rgb = _render_photoshop_green_selection(rgb, mask, alpha)
    roto_vis = _draw_roto_confirm_overlay(bgr, mask, fg_stroke, bg_stroke)
    cutout_vis = _make_checkerboard_cutout(fg_clean_bgr, alpha)

    # Refresh telemetry using actual confirmed foreground mask
    state["telemetry"] = analyze_scene_telemetry(bgr, mask)

    n_conf = len(state["confirmed_masks"])
    if _is_ai_engine(matting_engine):
        msg = (
            f"Đã tách AI ({matting_engine.split(' (')[0]}) tại Frame #{f_idx} (Tổng {n_conf} keyframe)"
            + (" + PyMatting bóc màu viền!" if use_pymatting_fg else "!")
        )
    else:
        msg = (
            f"Đã tự động phủ xanh toàn bộ vật thể tại Frame #{f_idx} (Tổng {n_conf} keyframe) | "
            f"[Vòng 2 Kiểm Định Pixel]: Bù đắp +{miss_px} px & cắt bỏ -{exc_px} px thừa!"
        )
    return editor_green_rgb, roto_vis, cutout_vis, str(cutout_png_path), state, msg


def _apply_manual_editor_strokes(
    f_idx,
    bgr,
    editor_val,
    trimap_radius,
    guided_radius,
    decont_strength,
    despill_green,
    erode_dilate,
    state,
    *,
    smart_red_subtract: bool = False,
    matting_engine: str = MATTING_ENGINE_CLASSIC,
    use_pymatting_fg: bool = False,
    two_pass_verify: bool = True,
):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fg_now, bg_now = extract_strokes_from_editor(editor_val, rgb)
    history = state.get("manual_strokes", {}).get(f_idx, {})
    empty = np.zeros(bgr.shape[:2], dtype=np.uint8)
    fg_old = history.get("fg", empty)
    bg_old = history.get("bg", empty)
    old_fg_snapshot = history.get("snapshot_fg", empty) > 0
    old_bg_snapshot = history.get("snapshot_bg", empty) > 0
    current_fg, current_bg = fg_now > 0, bg_now > 0
    retained = np.all(~old_fg_snapshot | current_fg | current_bg) and np.all(~old_bg_snapshot | current_bg | current_fg)
    fg_delta = current_fg & ~old_fg_snapshot if retained else current_fg
    bg_delta = current_bg & ~old_bg_snapshot if retained else current_bg
    if not np.any(fg_delta) and not np.any(bg_delta):
        return None
    fg = ((fg_old > 0) | fg_delta) & ~bg_delta
    bg = ((bg_old > 0) | bg_delta) & ~fg_delta
    prev = state.get("active_mask_by_frame", {}).get(int(f_idx))
    if prev is None and not np.any(fg_delta):
        raise gr.Error("Hãy vẽ nét xanh lên chủ thể trước khi dùng nét đỏ.")
    mask = (prev > 0).astype(np.uint8) if prev is not None else empty.copy()
    use_ai = _is_ai_engine(matting_engine)
    if np.any(fg_delta):
        stroke_kw = {"manual_prompts": True}
        if smart_red_subtract and use_ai:
            stroke_kw["snap_boundary"] = False
            stroke_kw["prefer_best_checkpoint"] = True
        candidate = apply_photoshop_stroke_to_mask(
            rgb, None, fg_delta.astype(np.uint8), bg.astype(np.uint8), **stroke_kw
        )
        mask |= candidate
    if smart_red_subtract and np.any(bg_delta) and np.any(mask > 0):
        mask = refine_mask_with_sam2_negative(
            rgb,
            mask,
            fg.astype(np.uint8),
            bg_delta.astype(np.uint8),
            prefer_best_checkpoint=use_ai,
        )
    mask[bg] = 0
    confirm_kw = {"direct_mask": mask, "explicit_bg": bg.astype(np.uint8)}
    if smart_red_subtract:
        confirm_kw["matting_engine"] = matting_engine
        confirm_kw["use_pymatting_fg"] = use_pymatting_fg
        confirm_kw["two_pass_verify"] = two_pass_verify
    result = _run_confirm_and_preview(
        f_idx, bgr, fg.astype(np.uint8), bg.astype(np.uint8),
        trimap_radius, guided_radius, decont_strength, despill_green, erode_dilate,
        state, **confirm_kw
    )
    state.setdefault("manual_strokes", {})[f_idx] = {
        "fg": fg.astype(np.uint8), "bg": bg.astype(np.uint8),
        "snapshot_fg": fg_now, "snapshot_bg": bg_now,
    }
    return result


def on_step2_quick_stroke(
    editor_val: Optional[dict],
    frame_idx: int,
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    state: dict,
    matting_engine: str = MATTING_ENGINE_CLASSIC,
    use_pymatting_fg: bool = False,
    two_pass_verify: bool = True,
):
    """Photoshop 1-Stroke Auto-Fill trigger on `roto_editor.input` / `apply`: fills object green immediately."""
    if not state.get("src_frames") or not editor_val:
        return gr.update(), gr.update(), gr.update(), gr.update(), state, gr.update()
    f_idx = int(np.clip(frame_idx, 0, len(state["src_frames"]) - 1))
    bgr = _load_bgr(state["src_frames"][f_idx])
    result = _apply_manual_editor_strokes(
        f_idx,
        bgr,
        editor_val,
        trimap_radius,
        guided_radius,
        decont_strength,
        despill_green,
        erode_dilate,
        state,
        smart_red_subtract=True,
        matting_engine=matting_engine,
        use_pymatting_fg=use_pymatting_fg,
        two_pass_verify=two_pass_verify,
    )
    return result if result is not None else (gr.update(), gr.update(), gr.update(), gr.update(), state, gr.update())


def on_step2_reset_selection(frame_idx: int, state: dict):
    """Clear Photoshop emerald-green cumulative selection on current frame."""
    if not state.get("src_frames"):
        return None, None, None, None, state, "Chưa có video/ảnh gốc."
    f_idx = int(np.clip(frame_idx, 0, len(state["src_frames"]) - 1))
    state.get("active_mask_by_frame", {}).pop(f_idx, None)
    state.get("birefnet_alpha_by_frame", {}).pop(f_idx, None)
    if state.get("confirmed_masks", {}).pop(f_idx, None) is not None:
        state["tracked_masks"] = None
    state.get("manual_strokes", {}).pop(f_idx, None)
    rgb = _load_rgb(state["src_frames"][f_idx])
    return rgb, rgb, None, None, state, f"Đã xóa vùng chọn tại Frame #{f_idx}. Hãy bôi 1 nét cọ xanh mới!"


def on_step2_confirm_roto_brush(
    editor_val: Optional[dict],
    frame_idx: int,
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    state: dict,
    matting_engine: str = MATTING_ENGINE_CLASSIC,
    use_pymatting_fg: bool = False,
    two_pass_verify: bool = True,
):
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")
    f_idx = int(np.clip(frame_idx, 0, len(state["src_frames"]) - 1))
    bgr = _load_bgr(state["src_frames"][f_idx])
    prev_mask = state.get("active_mask_by_frame", {}).get(f_idx)
    result = _apply_manual_editor_strokes(
        f_idx,
        bgr,
        editor_val,
        trimap_radius,
        guided_radius,
        decont_strength,
        despill_green,
        erode_dilate,
        state,
        smart_red_subtract=True,
        matting_engine=matting_engine,
        use_pymatting_fg=use_pymatting_fg,
        two_pass_verify=two_pass_verify,
    )
    if result is not None:
        return result
    if prev_mask is not None:
        return gr.update(), gr.update(), gr.update(), gr.update(), state, "Đã xác nhận vùng chọn hiện tại."
    if prev_mask is None:
        # If user didn't paint a stroke yet, fall back to smart auto-roto strokes so button works immediately
        fg_stroke, bg_stroke = auto_seed_roto_strokes(bgr)

    return _run_confirm_and_preview(
        f_idx, bgr, fg_stroke, bg_stroke,
        trimap_radius, guided_radius, decont_strength, despill_green, erode_dilate, state,
        grid_step=AUTO_ROTO_GRID_STEP,
        matting_engine=matting_engine,
        use_pymatting_fg=use_pymatting_fg,
        two_pass_verify=two_pass_verify,
    )


def on_step2_auto_birefnet(
    frame_idx: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    use_pymatting_fg: bool,
    state: dict,
):
    """Universal 1-Click AI Matting powered by BiRefNet_HR-matting + PyMatting."""
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")
    f_idx = int(np.clip(frame_idx, 0, len(state["src_frames"]) - 1))
    bgr = _load_bgr(state["src_frames"][f_idx])
    alpha = run_birefnet_matting(bgr, max_size=1024)
    mask = (alpha >= 0.5).astype(np.uint8)
    ed = int(erode_dilate)
    if ed != 0:
        k = abs(ed) * 2 + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, ker) if ed > 0 else cv2.erode(mask, ker)

    manual_bg = state.get("manual_strokes", {}).get(f_idx, {}).get("bg")
    if manual_bg is not None and np.any(manual_bg > 0):
        alpha = alpha.copy()
        alpha[manual_bg > 0] = 0.0
        mask[manual_bg > 0] = 0

    fg_stroke = np.zeros(bgr.shape[:2], dtype=np.uint8)
    bg_stroke = np.zeros(bgr.shape[:2], dtype=np.uint8)
    return _run_confirm_and_preview(
        f_idx,
        bgr,
        fg_stroke,
        bg_stroke,
        8,
        2,
        decont_strength,
        despill_green,
        erode_dilate,
        state,
        direct_mask=mask,
        explicit_bg=manual_bg,
        matting_engine=MATTING_ENGINE_BIREFNET,
        use_pymatting_fg=use_pymatting_fg,
        two_pass_verify=False,
        precomputed_alpha=alpha,
    )


def on_step2_auto_roto(
    frame_idx: int,
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    state: dict,
    matting_engine: str = MATTING_ENGINE_CLASSIC,
    use_pymatting_fg: bool = False,
    two_pass_verify: bool = True,
):
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")
    f_idx = int(np.clip(frame_idx, 0, len(state["src_frames"]) - 1))
    bgr = _load_bgr(state["src_frames"][f_idx])
    fg_stroke, bg_stroke = auto_seed_roto_strokes(bgr)
    manual_bg = state.get("manual_strokes", {}).get(f_idx, {}).get("bg")
    auto_state = dict(state)
    auto_masks = dict(state.get("active_mask_by_frame", {}))
    auto_masks.pop(f_idx, None)
    auto_state["active_mask_by_frame"] = auto_masks
    kw = {}
    if _is_ai_engine(matting_engine) or use_pymatting_fg or not two_pass_verify:
        kw = {
            "matting_engine": matting_engine,
            "use_pymatting_fg": use_pymatting_fg,
            "two_pass_verify": two_pass_verify,
        }
    return _run_confirm_and_preview(
        f_idx, bgr, fg_stroke, bg_stroke,
        trimap_radius, guided_radius, decont_strength, despill_green, erode_dilate, auto_state,
        grid_step=AUTO_ROTO_GRID_STEP,
        explicit_bg=manual_bg,
        **kw,
    )


def on_step2_auto_chroma_key(
    frame_idx: int,
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    erode_dilate: int,
    state: dict,
    use_pymatting_fg: bool = False,
):
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")
    f_idx = int(np.clip(frame_idx, 0, len(state["src_frames"]) - 1))
    bgr = _load_bgr(state["src_frames"][f_idx])
    mask = auto_chroma_key_mask(bgr, key_color="auto")
    fg_stroke = np.zeros(bgr.shape[:2], dtype=np.uint8)
    bg_stroke = np.zeros(bgr.shape[:2], dtype=np.uint8)
    return _run_confirm_and_preview(
        f_idx, bgr, fg_stroke, bg_stroke,
        trimap_radius, guided_radius, decont_strength, True, erode_dilate, state,
        direct_mask=mask,
        use_pymatting_fg=use_pymatting_fg,
    )


# =====================================================================
# BƯỚC 3: PHÂN TÍCH & TÁCH TỪNG KHUNG HÌNH Ở CẤP ĐỘ SUB-PIXEL
# =====================================================================
def on_step3_track_and_matte_all_frames(
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    state: dict,
    progress=gr.Progress(),
    matting_engine: str = MATTING_ENGINE_CLASSIC,
    use_pymatting_fg: bool = False,
    pass2_legacy: bool = True,
    matanyone_max_size: int = 720,
):
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")
    if not state.get("confirmed_masks"):
        raise gr.Error("Vui lòng Confirm vùng chọn ở Bước 2 trước khi tách toàn bộ khung hình.")

    work = Path(state.get("work_dir", APP_WORK_DIR))
    src_frames = [Path(p) for p in state["src_frames"]]
    n_frames = len(src_frames)

    confirmed_np: Dict[int, np.ndarray] = {
        int(k): load_mask(v) for k, v in state["confirmed_masks"].items()
    }

    cutout_frames_dir = work / "cutout_preview_frames"
    cutout_frames_dir.mkdir(parents=True, exist_ok=True)
    for old in cutout_frames_dir.glob("*.png"):
        old.unlink()

    matanyone_alpha_paths = None
    if n_frames > 1 and matting_engine == MATTING_ENGINE_MATANYONE2:
        try:
            progress(0.10, desc="Đang chạy MatAnyone 2 GPU tách alpha toàn video...")
            matanyone_alpha_paths = run_matanyone2_video(
                src_frames,
                confirmed_np,
                work / "matanyone2_preview_alpha",
                max_internal_size=int(matanyone_max_size),
                progress_cb=lambda done, tot: progress(
                    0.10 + 0.45 * (done / max(tot, 1)),
                    desc=f"MatAnyone 2 GPU: {done}/{tot} frames...",
                ),
            )
        except Exception:
            matanyone_alpha_paths = None

    if matanyone_alpha_paths is not None:
        prev_alpha_st = None
        prev_bgr_st = None
        for t in range(n_frames):
            progress(0.55 + 0.35 * (t / max(n_frames, 1)), desc=f"PyMatting bóc màu viền frame {t + 1}/{n_frames}...")
            bgr = _load_bgr(src_frames[t])
            alpha_u8 = cv2.imread(str(matanyone_alpha_paths[t]), cv2.IMREAD_GRAYSCALE)
            alpha = alpha_u8.astype(np.float32) / 255.0
            if t in confirmed_np:
                prev_alpha_st = prev_bgr_st = None
            alpha_st = stabilize_alpha_temporal(alpha, prev_alpha_st, bgr, prev_bgr_st, temporal_weight=0.25)
            if t in confirmed_np:
                alpha_st = _protect_manual_background(alpha_st, state, t)
            prev_alpha_st = alpha_st
            prev_bgr_st = bgr
            fg_clean = (
                estimate_foreground_pymatting(bgr, alpha_st, strength=float(decont_strength), despill_green=bool(despill_green))
                if use_pymatting_fg
                else bgr
            )
            checker_rgb = _make_checkerboard_cutout(fg_clean, alpha_st)
            cv2.imwrite(str(cutout_frames_dir / f"{t:05d}.png"), cv2.cvtColor(checker_rgb, cv2.COLOR_RGB2BGR))
    else:
        if n_frames == 1:
            only_idx = next(iter(confirmed_np.keys()))
            state["tracked_masks"] = [state["confirmed_masks"][only_idx]]
        else:
            progress(0.1, desc="Đang chạy SAM 2.1 Roto-Brush Video Tracking qua từng khung hình...")
            tracked_paths = track_roto_video_masks(
                work / "src_frames",
                work / "tracked_masks",
                confirmed_masks=confirmed_np,
                progress_cb=lambda f, tot: progress(
                    0.1 + 0.25 * ((f - min(confirmed_np) + 1) / max(tot - min(confirmed_np), 1)),
                    desc=f"SAM lượt xuôi: khung {f + 1}/{tot}; có thể còn lượt ngược...",
                ),
                prefer_best_checkpoint=_is_ai_engine(matting_engine),
            )
            state["tracked_masks"] = [str(p) for p in tracked_paths]
            progress(0.55, desc="Đã hoàn tất theo dõi SAM; chuẩn bị tách nền...")

        mcfg = _resolve_matting_cfg(
            matting_engine,
            trimap_radius,
            guided_radius,
            decont_strength,
            despill_green,
            erode_dilate,
            use_pymatting_fg,
            pass2_legacy,
        )

        mask_paths = [Path(p) for p in state["tracked_masks"]]
        prev_alpha_st = None
        prev_bgr_st = None
        for t in range(n_frames):
            progress(0.55 + 0.35 * (t / max(n_frames, 1)), desc=f"Tách Sub-Pixel Alpha Matting & Khử viền frame {t + 1}/{n_frames}...")
            bgr = _load_bgr(src_frames[t])
            if t in confirmed_np:
                m_smooth = confirmed_np[t]
                prev_alpha_st = prev_bgr_st = None
            else:
                m_smooth = temporal_smooth_mask(mask_paths, t, frame_paths=src_frames, current_bgr=bgr) if n_frames > 3 else load_mask(mask_paths[t])
            mres = extract_pixel_accurate_matte(bgr, m_smooth, mcfg)
            alpha_st = stabilize_alpha_temporal(mres.alpha, prev_alpha_st, bgr, prev_bgr_st)
            if t in confirmed_np:
                alpha_st = _protect_manual_background(alpha_st, state, t)
            prev_alpha_st = alpha_st
            prev_bgr_st = bgr
            checker_rgb = _make_checkerboard_cutout(mres.fg_clean_bgr, alpha_st)
            cv2.imwrite(str(cutout_frames_dir / f"{t:05d}.png"), cv2.cvtColor(checker_rgb, cv2.COLOR_RGB2BGR))

    if n_frames > 1:
        progress(0.92, desc="Đang đóng gói Video Preview Tách Nền Sub-Pixel...")
        cutout_mp4 = encode_video(cutout_frames_dir, work / "roto_cutout_preview.mp4", fps=state["fps"])
        return state, str(cutout_mp4), f"Bước 3 hoàn tất: Đã tách chuẩn Sub-Pixel toàn bộ {n_frames} khung hình!"
    else:
        single_mp4 = encode_video(cutout_frames_dir, work / "roto_cutout_preview.mp4", fps=1.0)
        return state, str(single_mp4), "Bước 3 hoàn tất: Đã tách chuẩn Sub-Pixel khung hình!"


# =====================================================================
# BƯỚC 4: AGY SUY LUẬN & VIẾT PROMPT TẠO VIDEO PHÔNG NỀN MỚI (FLOW)
# =====================================================================
def on_step4_generate_prompt(
    target_bg_vi: str,
    subject_vi: str,
    use_local_agy_llm: bool,
    state: dict,
):
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng nạp Video/Ảnh gốc ở Bước 1 trước khi tạo Prompt.")

    telemetry = state.get("telemetry")
    if not telemetry:
        bgr = _load_bgr(state["src_frames"][0])
        telemetry = analyze_scene_telemetry(bgr)
        state["telemetry"] = telemetry

    res = generate_flow_prompts(
        telemetry=telemetry,
        target_background_vi=target_bg_vi,
        subject_description_vi=subject_vi,
        use_local_llm=bool(use_local_agy_llm),
    )
    return (
        res["clean_plate_prompt"],
        res["v2v_prompt"],
        res["full_markdown"],
        "Bước 4 hoàn tất: Đã tạo xong Prompt chuẩn Google Flow khớp 100% góc máy, tiêu cự Bokeh & ánh sáng gốc!",
    )


# =====================================================================
# BƯỚC 5: GHÉP NỀN MỚI — TỰ KHỚP ÁNH SÁNG, CHUYỂN ĐỘNG & TRACKING
# =====================================================================
def _prepare_single_preview_composite(
    frame_idx: int,
    bg_file: Optional[str],
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    ambient_color_strength: float,
    directional_light_strength: float,
    light_wrap_radius: int,
    light_wrap_strength: float,
    protect_screen: bool,
    screen_glare: float,
    grain_match: float,
    bg_defocus: float,
    clean_ai_ghost: bool,
    state: dict,
    matting_engine: str = MATTING_ENGINE_CLASSIC,
    use_pymatting_fg: bool = False,
    linear_compositing: bool = False,
    pass2_legacy: bool = True,
    edge_blur_match: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")

    src_frames = state["src_frames"]
    f_idx = int(np.clip(frame_idx, 0, len(src_frames) - 1))
    src_bgr = _load_bgr(src_frames[f_idx])
    h, w = src_bgr.shape[:2]

    confirmed = state.get("confirmed_masks", {})
    ref_k = _confirmed_reference_frame(confirmed, f_idx)
    is_confirmed = ref_k is not None and int(ref_k) == f_idx
    tracked = state.get("tracked_masks")
    use_tracking = tracked and f_idx < len(tracked) and not is_confirmed

    if use_tracking:
        mask_paths = [Path(p) for p in tracked]
        mask = temporal_smooth_mask(mask_paths, f_idx, frame_paths=src_frames, current_bgr=src_bgr) if len(mask_paths) > 3 else load_mask(mask_paths[f_idx])
    elif ref_k is not None:
        if not is_confirmed:
            raise PreviewMaskUnavailable(
                f"Khung {f_idx} chưa có mask. Bấm ‘✨ Tự Động Tách Tay + Sản Phẩm’ ở khung này "
                "hoặc chạy ‘Chạy Tách Sub-Pixel Toàn Bộ Video (SAM 2.1)’ trước khi xem trước."
            )
        mask = load_mask(confirmed[ref_k])
    else:
        fg_st, bg_st = auto_seed_roto_strokes(src_bgr)
        mask = apply_photoshop_stroke_to_mask(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB), None, fg_st, bg_st, grid_step=AUTO_ROTO_GRID_STEP)

def _load_preview_bg_frame(
    bg_file: Optional[str],
    f_idx: int,
    src_fps: float,
    src_bgr: np.ndarray,
) -> Tuple[np.ndarray, Tuple[object, ...]]:
    """Load/resize preview background frame with an LRU cache to avoid per-drag VideoCapture seeks."""
    h, w = src_bgr.shape[:2]
    if not bg_file:
        sample_bg = Path("workspace/taxi_test/ai_f0.jpg")
        if sample_bg.exists():
            bg_bgr = cv2.GaussianBlur(_load_bgr(sample_bg), (0, 0), 10.0)
        else:
            bg_bgr = cv2.GaussianBlur(src_bgr, (0, 0), 25.0)
        return cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_CUBIC), ("synth", w, h)

    p_bg = Path(bg_file)
    stat = p_bg.stat()
    bg_key = (str(p_bg.resolve()), stat.st_size, stat.st_mtime_ns, int(f_idx), round(float(src_fps or 30.0), 3), h, w)
    with _BG_FRAME_CACHE_LOCK:
        cached_bg = _BG_FRAME_CACHE.get(bg_key)
        if cached_bg is not None:
            _BG_FRAME_CACHE.move_to_end(bg_key)
            return cached_bg, bg_key

    if p_bg.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        bg_bgr = _load_bgr(p_bg)
    else:
        cap = cv2.VideoCapture(str(p_bg))
        total_bg = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        bg_fps = float(cap.get(cv2.CAP_PROP_FPS) or src_fps or 30.0)
        mapped_bg = int(round(f_idx * (bg_fps / max(float(src_fps or 30.0), 1.0))))
        t_bg = mapped_bg % max(1, total_bg)
        cap.set(cv2.CAP_PROP_POS_FRAMES, t_bg)
        ret, frame_bg = cap.read()
        cap.release()
        if not ret or frame_bg is None:
            raise gr.Error("Không thể đọc khung hình từ Video nền mới.")
        bg_bgr = frame_bg

    bg_resized = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    with _BG_FRAME_CACHE_LOCK:
        _BG_FRAME_CACHE[bg_key] = bg_resized
        _BG_FRAME_CACHE.move_to_end(bg_key)
        while len(_BG_FRAME_CACHE) > 8:
            _BG_FRAME_CACHE.popitem(last=False)
    return bg_resized, bg_key


def _render_preview_view(
    src_rgb: np.ndarray,
    comp_rgb: np.ndarray,
    fg_clean_rgb: np.ndarray,
    alpha: np.ndarray,
    scr_mask: Optional[np.ndarray],
    diag_mode: str = PREVIEW_DIAG_SPLIT,
    zoom: float = 1.0,
    pan_x: float = 0.5,
    pan_y: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply lossless server-side zoom/pan crop and diagnostic inspection views."""
    h, w = src_rgb.shape[:2]
    z = float(np.clip(zoom, 1.0, 4.0))
    if z > 1.01:
        cw = max(64, int(round(w / z)))
        ch = max(64, int(round(h / z)))
        cx = int(round(np.clip(pan_x, 0.0, 1.0) * max(0, w - cw)))
        cy = int(round(np.clip(pan_y, 0.0, 1.0) * max(0, h - ch)))
        src_v = src_rgb[cy:cy + ch, cx:cx + cw]
        comp_v = comp_rgb[cy:cy + ch, cx:cx + cw]
        fg_v = fg_clean_rgb[cy:cy + ch, cx:cx + cw]
        alpha_v = alpha[cy:cy + ch, cx:cx + cw]
        scr_v = scr_mask[cy:cy + ch, cx:cx + cw] if scr_mask is not None else None
    else:
        src_v = src_rgb
        comp_v = comp_rgb
        fg_v = fg_clean_rgb
        alpha_v = alpha
        scr_v = scr_mask

    if diag_mode == PREVIEW_DIAG_BLACK:
        a3 = np.clip(alpha_v, 0.0, 1.0)[..., None]
        black_v = np.clip(fg_v.astype(np.float32) * a3, 0, 255).astype(np.uint8)
        return src_v, black_v
    if diag_mode == PREVIEW_DIAG_WHITE:
        a3 = np.clip(alpha_v, 0.0, 1.0)[..., None]
        white_v = np.clip(fg_v.astype(np.float32) * a3 + 255.0 * (1.0 - a3), 0, 255).astype(np.uint8)
        return src_v, white_v
    if diag_mode == PREVIEW_DIAG_SCREEN:
        overlay_v = src_v.copy()
        if scr_v is not None and np.any(scr_v > 0):
            s_bool = scr_v > 0
            overlay_v[s_bool] = np.clip(
                overlay_v[s_bool].astype(np.float32) * 0.45 + np.array([35, 130, 255], dtype=np.float32) * 0.55,
                0,
                255,
            ).astype(np.uint8)
            contours, _ = cv2.findContours(s_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_v, contours, -1, (0, 255, 180), 2)
        return overlay_v, comp_v
    return src_v, comp_v


def _compute_preview_bundle(
    frame_idx: int,
    bg_file: Optional[str],
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    ambient_color_strength: float,
    directional_light_strength: float,
    light_wrap_radius: int,
    light_wrap_strength: float,
    protect_screen: bool,
    screen_glare: float,
    grain_match: float,
    bg_defocus: float,
    clean_ai_ghost: bool,
    state: dict,
    matting_engine: str = MATTING_ENGINE_CLASSIC,
    use_pymatting_fg: bool = False,
    linear_compositing: bool = False,
    pass2_legacy: bool = True,
    edge_blur_match: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")

    src_frames = state["src_frames"]
    f_idx = int(np.clip(frame_idx, 0, len(src_frames) - 1))
    src_bgr = _load_bgr(src_frames[f_idx])
    h, w = src_bgr.shape[:2]

    confirmed = state.get("confirmed_masks", {})
    ref_k = _confirmed_reference_frame(confirmed, f_idx)
    is_confirmed = ref_k is not None and int(ref_k) == f_idx
    tracked = state.get("tracked_masks")
    use_tracking = tracked and f_idx < len(tracked) and not is_confirmed

    if use_tracking:
        mask_paths = [Path(p) for p in tracked]
        mask = temporal_smooth_mask(mask_paths, f_idx, frame_paths=src_frames, current_bgr=src_bgr) if len(mask_paths) > 3 else load_mask(mask_paths[f_idx])
    elif ref_k is not None:
        if not is_confirmed:
            raise PreviewMaskUnavailable(
                f"Khung {f_idx} chưa có mask. Bấm ‘✨ Tự Động Tách Tay + Sản Phẩm’ ở khung này "
                "hoặc chạy ‘Chạy Tách Sub-Pixel Toàn Bộ Video (SAM 2.1)’ trước khi xem trước."
            )
        mask = load_mask(confirmed[ref_k])
    else:
        fg_st, bg_st = auto_seed_roto_strokes(src_bgr)
        mask = apply_photoshop_stroke_to_mask(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB), None, fg_st, bg_st, grid_step=AUTO_ROTO_GRID_STEP)

    bg_bgr, bg_sig = _load_preview_bg_frame(bg_file, f_idx, float(state.get("fps") or 30.0), src_bgr)

    mcfg = _resolve_matting_cfg(
        matting_engine,
        trimap_radius,
        guided_radius,
        decont_strength,
        despill_green,
        erode_dilate,
        use_pymatting_fg,
        pass2_legacy,
    )
    def signature(path):
        p = Path(path)
        stat = p.stat()
        return str(p.resolve()), stat.st_size, stat.st_mtime_ns

    if use_tracking:
        relevant = tracked[max(0, f_idx - 2):min(len(tracked), f_idx + 3)]
        relevant_src = src_frames[max(0, f_idx - 2):min(len(src_frames), f_idx + 3)]
    elif ref_k is not None:
        relevant = [confirmed[ref_k]]
        relevant_src = []
    else:
        relevant = []
        relevant_src = []
    cache_key = (state.get("work_dir"), f_idx, signature(src_frames[f_idx]),
                 tuple(signature(p) for p in relevant), tuple(signature(p) for p in relevant_src),
                 signature(src_frames[int(ref_k)]) if ref_k is not None else None,
                 signature(confirmed[ref_k]) if ref_k is not None else None,
                 matting_engine, bool(use_pymatting_fg), bool(pass2_legacy),
                 int(trimap_radius), int(guided_radius), float(decont_strength), bool(despill_green), int(erode_dilate))

    comp_key = (
        cache_key,
        bg_sig,
        round(float(ambient_color_strength), 4),
        round(float(directional_light_strength), 4),
        int(light_wrap_radius),
        round(float(light_wrap_strength), 4),
        bool(protect_screen),
        round(float(screen_glare), 4),
        round(float(grain_match), 4),
        round(float(bg_defocus), 3),
        bool(clean_ai_ghost),
        bool(linear_compositing),
        round(float(edge_blur_match), 3),
        bool(is_confirmed),
    )
    if relevant:
        with _COMPOSITE_RESULT_CACHE_LOCK:
            bundle = _COMPOSITE_RESULT_CACHE.get(comp_key)
            if bundle is not None:
                _COMPOSITE_RESULT_CACHE.move_to_end(comp_key)
                return bundle

    with _PREVIEW_CACHE_LOCK:
        cached = _PREVIEW_CACHE.get(cache_key) if relevant else None
        if cached is not None:
            _PREVIEW_CACHE.move_to_end(cache_key)
    if cached is None:
        cached_biref_alpha = (
            state.get("birefnet_alpha_by_frame", {}).get(f_idx)
            if is_confirmed and matting_engine == MATTING_ENGINE_BIREFNET
            else None
        )
        if cached_biref_alpha is not None and cached_biref_alpha.shape == (h, w):
            fg_clean = (
                estimate_foreground_pymatting(
                    src_bgr, cached_biref_alpha, strength=float(decont_strength), despill_green=bool(despill_green)
                )
                if use_pymatting_fg
                else src_bgr
            )
            cached = (fg_clean, cached_biref_alpha)
        else:
            if ref_k is not None:
                mres = extract_pixel_accurate_matte(src_bgr, mask, mcfg, ref_bgr=_load_bgr(src_frames[int(ref_k)]), ref_mask=load_mask(confirmed[ref_k]))
            else:
                mres = extract_pixel_accurate_matte(src_bgr, mask, mcfg)
            cached = (mres.fg_clean_bgr, mres.alpha)
        if relevant:
            with _PREVIEW_CACHE_LOCK:
                _PREVIEW_CACHE[cache_key] = cached
                _PREVIEW_CACHE.move_to_end(cache_key)
                while len(_PREVIEW_CACHE) > 2:
                    _PREVIEW_CACHE.popitem(last=False)

    relight_kw = {}
    if _is_ai_engine(matting_engine) or linear_compositing or float(edge_blur_match) > 0.0:
        relight_kw = {
            "pass3_aa_and_restore": bool(pass2_legacy) if _is_ai_engine(matting_engine) else True,
            "linear_compositing": bool(linear_compositing),
            "edge_blur_match": float(edge_blur_match),
        }
    rcfg = RelightConfig(
        ambient_color_strength=float(ambient_color_strength),
        directional_light_strength=float(directional_light_strength),
        light_wrap_Radius=int(light_wrap_radius),
        light_wrap_strength=float(light_wrap_strength),
        protect_emissive_screen=bool(protect_screen),
        screen_glare_opacity=float(screen_glare),
        grain_match=float(grain_match),
        bg_defocus_sigma=float(bg_defocus),
        clean_ai_ghost=bool(clean_ai_ghost),
        **relight_kw,
    )
    preview_alpha = _protect_manual_background(cached[1], state, f_idx) if is_confirmed else cached[1]
    scr_mask = (
        detect_emissive_screen_mask(cached[0], (preview_alpha > 0.15).astype(np.uint8))
        if protect_screen
        else np.zeros((h, w), dtype=np.uint8)
    )
    comp_kw = {"frame_idx": f_idx} if _is_ai_engine(matting_engine) or linear_compositing else {}
    comp_bgr = composite_with_relighting(
        fg_clean_bgr=cached[0],
        alpha=preview_alpha,
        new_bg_bgr=bg_bgr,
        src_full_bgr=src_bgr,
        screen_mask=scr_mask,
        cfg=rcfg,
        **comp_kw,
    )
    bundle = (
        cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cached[0], cv2.COLOR_BGR2RGB),
        preview_alpha,
        scr_mask,
    )
    if relevant:
        with _COMPOSITE_RESULT_CACHE_LOCK:
            _COMPOSITE_RESULT_CACHE[comp_key] = bundle
            _COMPOSITE_RESULT_CACHE.move_to_end(comp_key)
            while len(_COMPOSITE_RESULT_CACHE) > 4:
                _COMPOSITE_RESULT_CACHE.popitem(last=False)
    return bundle


def _prepare_single_preview_composite(
    frame_idx: int,
    bg_file: Optional[str],
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    ambient_color_strength: float,
    directional_light_strength: float,
    light_wrap_radius: int,
    light_wrap_strength: float,
    protect_screen: bool,
    screen_glare: float,
    grain_match: float,
    bg_defocus: float,
    clean_ai_ghost: bool,
    state: dict,
    matting_engine: str = MATTING_ENGINE_CLASSIC,
    use_pymatting_fg: bool = False,
    linear_compositing: bool = False,
    pass2_legacy: bool = True,
    edge_blur_match: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    src_rgb, comp_rgb, _, _, _ = _compute_preview_bundle(
        frame_idx, bg_file, trimap_radius, guided_radius, decont_strength, despill_green, erode_dilate,
        ambient_color_strength, directional_light_strength, light_wrap_radius, light_wrap_strength,
        protect_screen, screen_glare, grain_match, bg_defocus, clean_ai_ghost, state,
        matting_engine=matting_engine,
        use_pymatting_fg=use_pymatting_fg,
        linear_compositing=linear_compositing,
        pass2_legacy=pass2_legacy,
        edge_blur_match=edge_blur_match,
    )
    return src_rgb, comp_rgb


def on_step5_preview_composite(
    frame_idx: int,
    bg_file: Optional[str],
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    ambient_color_strength: float,
    directional_light_strength: float,
    light_wrap_radius: int,
    light_wrap_strength: float,
    protect_screen: bool,
    screen_glare: float,
    grain_match: float,
    bg_defocus: float,
    clean_ai_ghost: bool,
    state: dict,
    matting_engine: str = MATTING_ENGINE_CLASSIC,
    use_pymatting_fg: bool = False,
    linear_compositing: bool = False,
    pass2_legacy: bool = True,
    edge_blur_match: float = 0.0,
    diag_mode: str = PREVIEW_DIAG_SPLIT,
    zoom: float = 1.0,
    pan_x: float = 0.5,
    pan_y: float = 0.5,
):
    try:
        src_rgb, comp_rgb, fg_clean_rgb, alpha, scr_mask = _compute_preview_bundle(
            frame_idx, bg_file, trimap_radius, guided_radius, decont_strength, despill_green, erode_dilate,
            ambient_color_strength, directional_light_strength, light_wrap_radius, light_wrap_strength,
            protect_screen, screen_glare, grain_match, bg_defocus, clean_ai_ghost, state,
            matting_engine=matting_engine,
            use_pymatting_fg=use_pymatting_fg,
            linear_compositing=linear_compositing,
            pass2_legacy=pass2_legacy,
            edge_blur_match=edge_blur_match,
        )
    except PreviewMaskUnavailable as exc:
        return None, str(exc)
    left_rgb, right_rgb = _render_preview_view(
        src_rgb,
        comp_rgb,
        fg_clean_rgb,
        alpha,
        scr_mask,
        diag_mode=diag_mode,
        zoom=float(zoom),
        pan_x=float(pan_x),
        pan_y=float(pan_y),
    )
    if float(zoom) > 1.01 or diag_mode != PREVIEW_DIAG_SPLIT:
        return (left_rgb, right_rgb), (
            f"⚡ Live Preview | Khung {int(frame_idx)} | {diag_mode} | Zoom {float(zoom):.2g}×"
        )
    return (left_rgb, right_rgb), "Đã tạo bản xem trước Ghép Nền & Tự Khớp Ánh Sáng (Kéo thanh trượt để so sánh Gốc vs Đã Ghép)!"


def on_step5_render_final_video(
    bg_file: Optional[str],
    tracking_mode: str,
    audio_choice: str,
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    ambient_color_strength: float,
    directional_light_strength: float,
    light_wrap_radius: int,
    light_wrap_strength: float,
    protect_screen: bool,
    screen_glare: float,
    grain_match: float,
    bg_defocus: float,
    clean_ai_ghost: bool,
    state: dict,
    progress=gr.Progress(),
    use_sam2matting_tiny: bool = False,
    fast_tiny: bool = False,
    matting_engine: str = MATTING_ENGINE_CLASSIC,
    use_pymatting_fg: bool = False,
    linear_compositing: bool = False,
    pass2_legacy: bool = True,
    matanyone_max_size: int = 720,
    edge_blur_match: float = 0.0,
):
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")
    if not bg_file:
        raise gr.Error("Vui lòng nạp Video (hoặc Ảnh) Bối Cảnh Mới tạo từ Google Flow ở Bước 5.")

    work = Path(state.get("work_dir", APP_WORK_DIR))
    src_frames = [Path(p) for p in state["src_frames"]]
    n_src = len(src_frames)
    w, h = state["width"], state["height"]

    # Ensure at least one confirmed keyframe exists (auto-seed with BiRefNet for AI engines or legacy auto-roto)
    if not state.get("confirmed_masks"):
        if _is_ai_engine(matting_engine):
            alpha0 = run_birefnet_matting(_load_bgr(src_frames[0]), max_size=1024)
            m0 = (alpha0 >= 0.5).astype(np.uint8)
        else:
            fg_st, bg_st = auto_seed_roto_strokes(_load_bgr(src_frames[0]))
            m0 = apply_photoshop_stroke_to_mask(_load_rgb(src_frames[0]), None, fg_st, bg_st, grid_step=AUTO_ROTO_GRID_STEP)
        m0_path = work / "confirmed_masks" / "conf_00000.png"
        save_mask(m0, m0_path)
        state.setdefault("confirmed_masks", {})[0] = str(m0_path)

    confirmed_np = {int(k): load_mask(v) for k, v in state["confirmed_masks"].items()}
    matanyone2_alpha_paths = None
    matanyone2_note = ""

    if n_src > 1 and not use_sam2matting_tiny and matting_engine == MATTING_ENGINE_MATANYONE2:
        try:
            progress(0.08, desc="Đang chạy MatAnyone 2 GPU Video Matting...")
            matanyone2_alpha_paths = run_matanyone2_video(
                src_frames,
                confirmed_np,
                work / "matanyone2_alpha",
                max_internal_size=int(matanyone_max_size),
                progress_cb=lambda done, tot: progress(
                    0.08 + 0.16 * (done / max(tot, 1)),
                    desc=f"MatAnyone 2 GPU: {done}/{tot} frames...",
                ),
            )
            matanyone2_note = f" | Engine: MatAnyone 2 GPU (max_internal_size={int(matanyone_max_size)})" + (" + PyMatting" if use_pymatting_fg else "")
        except Exception as exc:
            matanyone2_note = f" | MatAnyone 2 fallback ({str(exc)[:120]}) -> SAM 2.1 + ViTMatte-Base"
            matanyone2_alpha_paths = None

    # Ensure we have per-frame masks across the entire video (needed for SAM2/Classic/Camera Motion Lock)
    if matanyone2_alpha_paths is not None:
        mask_paths = [Path(p) for p in matanyone2_alpha_paths]
    else:
        if not state.get("tracked_masks") or len(state["tracked_masks"]) != n_src:
            if n_src == 1:
                state["tracked_masks"] = [next(iter(state["confirmed_masks"].values()))]
            else:
                progress(0.08, desc="Đang chạy SAM 2.1 Roto-Brush Tracking cho toàn bộ video...")
                tracked_paths = track_roto_video_masks(
                    work / "src_frames",
                    work / "tracked_masks",
                    confirmed_masks=confirmed_np,
                    progress_cb=lambda f, tot: progress(
                        0.08 + 0.08 * ((f - min(confirmed_np) + 1) / max(tot - min(confirmed_np), 1)),
                        desc=f"SAM lượt xuôi: khung {f + 1}/{tot}; có thể còn lượt ngược...",
                    ),
                    prefer_best_checkpoint=_is_ai_engine(matting_engine),
                )
                state["tracked_masks"] = [str(p) for p in tracked_paths]
                progress(0.24, desc="Đã hoàn tất theo dõi SAM; chuẩn bị nền mới...")
        mask_paths = [Path(p) for p in state["tracked_masks"]]

    sam2_alpha_paths = None
    sam2_fallback_note = ""
    sam2_run_dir = None

    # Extract background video/image frames resized to (w, h)
    progress(0.25, desc="Đang trích xuất & chuẩn hóa khung hình Bối Cảnh Mới...")
    bg_frames, bg_fps, _, _, _ = _extract_media_frames(
        bg_file, work / "bg_frames", auto_crop_bars=True, target_size=(w, h)
    )
    n_bg = len(bg_frames)

    # Compute Camera Motion Trajectory if Camera Motion Lock is selected
    cam_transforms = None
    v2v_transforms = None
    if "Camera Motion Lock" in tracking_mode and n_src > 1:
        progress(0.35, desc="Đang phân tích chuyển động Camera gốc (Background Optical Flow RANSAC)...")
        cam_transforms = estimate_source_camera_trajectory(src_frames, mask_paths)
    elif "V2V Spatial" in tracking_mode and n_src > 1:
        progress(0.35, desc="Đang căn khớp không gian V2V (SIFT + Sub-Pixel ECC)...")
        raw_tf = []
        for t in range(n_src):
            t_bg = min(t, n_bg - 1)
            s_bgr = _load_bgr(src_frames[t])
            b_bgr = _load_bgr(bg_frames[t_bg])
            m_fg = load_mask(mask_paths[t])
            raw_tf.append(estimate_pair_similarity(s_bgr, b_bgr, m_fg, m_fg))
        v2v_transforms = smooth_transforms(raw_tf)

    mcfg = _resolve_matting_cfg(
        matting_engine,
        trimap_radius,
        guided_radius,
        decont_strength,
        despill_green,
        erode_dilate,
        use_pymatting_fg,
        pass2_legacy,
    )
    relight_kw = {}
    if _is_ai_engine(matting_engine) or linear_compositing or float(edge_blur_match) > 0.0:
        relight_kw = {
            "pass3_aa_and_restore": bool(pass2_legacy) if _is_ai_engine(matting_engine) else True,
            "linear_compositing": bool(linear_compositing),
            "edge_blur_match": float(edge_blur_match),
        }
    rcfg = RelightConfig(
        ambient_color_strength=float(ambient_color_strength),
        directional_light_strength=float(directional_light_strength),
        light_wrap_Radius=int(light_wrap_radius),
        light_wrap_strength=float(light_wrap_strength),
        protect_emissive_screen=bool(protect_screen),
        screen_glare_opacity=float(screen_glare),
        grain_match=float(grain_match),
        bg_defocus_sigma=float(bg_defocus),
        clean_ai_ghost=bool(clean_ai_ghost or ("V2V" in tracking_mode)),
        **relight_kw,
    )

    if use_sam2matting_tiny:
        try:
            sam2_run_dir = work / "sam2matting_tiny" / uuid.uuid4().hex
            sam2_alpha_paths = _run_sam2matting_tiny(
                src_frames, mask_paths, state["confirmed_masks"], sam2_run_dir, (h, w), progress
            )
            sam2_fallback_note = " | Alpha: SAM2Matting Tiny GPU (chunk 32 frame, overlap blend 4 frame)."
            if fast_tiny:
                sam2_fallback_note += " Màu tiền cảnh: Fast từ alpha Tiny (chất lượng viền có thể khác matte CPU)."
        except Exception as exc:
            sam2_fallback_note = f" | SAM2Matting Tiny không khả dụng ({str(exc)[:240]}); đã dùng matte CPU mặc định cho toàn bộ video."
            if sam2_run_dir is not None:
                shutil.rmtree(sam2_run_dir, ignore_errors=True)
                sam2_run_dir = None

    out_dir = work / "final_composite_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    confirmed = state["confirmed_masks"]
    total_restored = 0
    total_removed = 0
    fast_tiny_active = bool(sam2_alpha_paths is not None and fast_tiny)
    prev_alpha_stable = None
    prev_src_bgr = None
    confirmed_paths = {int(key): value for key, value in confirmed.items()}
    reference_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def prepare_frame(frame_idx: int):
        src_bgr = _load_bgr(src_frames[frame_idx])
        if matanyone2_alpha_paths is not None:
            alpha_u8 = cv2.imread(str(matanyone2_alpha_paths[frame_idx]), cv2.IMREAD_GRAYSCALE)
            if alpha_u8 is None or alpha_u8.shape != (h, w):
                raise RuntimeError(f"Thiếu alpha MatAnyone 2 tại frame {frame_idx}.")
            alpha = alpha_u8.astype(np.float32) / 255.0
            fg_clean_bgr = (
                estimate_foreground_pymatting(
                    src_bgr, alpha, strength=float(decont_strength), despill_green=bool(despill_green)
                )
                if use_pymatting_fg
                else src_bgr
            )
            ref_idx = int(frame_idx) if frame_idx in confirmed_paths else None
            return src_bgr, (alpha, fg_clean_bgr), ref_idx, (alpha >= 0.5).astype(np.uint8), None, None

        reference_idx = _confirmed_reference_frame(confirmed, frame_idx)
        ref_bgr = ref_mask = None

        if reference_idx is not None:
            reference_idx = int(reference_idx)
            if reference_idx in reference_cache:
                ref_bgr, ref_mask = reference_cache.pop(reference_idx)
                reference_cache[reference_idx] = (ref_bgr, ref_mask)
            else:
                ref_bgr = src_bgr if reference_idx == frame_idx else _load_bgr(src_frames[reference_idx])
                ref_mask = load_mask(confirmed_paths[reference_idx])
                reference_cache[reference_idx] = (ref_bgr, ref_mask)
                if len(reference_cache) > 2:
                    reference_cache.popitem(last=False)

        if reference_idx is not None and reference_idx == frame_idx:
            m_fg = ref_mask
        else:
            m_fg = temporal_smooth_mask(mask_paths, frame_idx, frame_paths=src_frames, current_bgr=src_bgr) if n_src > 3 else load_mask(mask_paths[frame_idx])

        tiny_margin_mask = m_fg
        tiny_recovered_silver = np.zeros(m_fg.shape, dtype=bool)
        if (fast_tiny_active or sam2_alpha_paths is not None) and int(erode_dilate) > 0:
            tiny_margin_mask, tiny_recovered_silver = expand_mask_with_color_evidence(
                src_bgr, m_fg, pixels=int(erode_dilate)
            )

        if fast_tiny_active:
            predicted = cv2.imread(str(sam2_alpha_paths[frame_idx]), cv2.IMREAD_GRAYSCALE)
            if predicted is None or predicted.shape != (h, w):
                raise RuntimeError(f"SAM2Matting alpha đã xác thực bị thiếu tại frame {frame_idx}.")
            alpha = predicted.astype(np.float32) / 255.0
            support = cv2.dilate(
                (m_fg > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            ) > 0
            alpha[~support] = 0.0
            tiny_added = (tiny_margin_mask > 0) & (m_fg == 0)
            alpha[tiny_added] = np.maximum(alpha[tiny_added], 0.35)
            img_f = src_bgr.astype(np.float32) / 255.0
            fg_core = (m_fg > 0) & (alpha >= 0.98)
            if np.count_nonzero(fg_core) < 64:
                fg_core = m_fg > 0
            fg_est = _propagate_local_colors(img_f, fg_core, sigma=7.0)
            fg_clean_bgr = np.clip(
                decontaminate_foreground_edges(
                    img_f,
                    alpha,
                    fg_est=fg_est,
                    strength=float(np.clip(mcfg.decontaminate_strength, 0.0, 1.0)),
                    despill_green=bool(mcfg.despill_green),
                ) * 255.0,
                0,
                255,
            ).astype(np.uint8)
            return src_bgr, (alpha, fg_clean_bgr), reference_idx, m_fg, tiny_margin_mask, tiny_recovered_silver
        if ref_bgr is None:
            mres = extract_pixel_accurate_matte(src_bgr, m_fg, mcfg)
        else:
            mres = extract_pixel_accurate_matte(src_bgr, m_fg, mcfg, ref_bgr=ref_bgr, ref_mask=ref_mask)
        return src_bgr, mres, reference_idx, m_fg, tiny_margin_mask, tiny_recovered_silver

    use_matanyone2 = matanyone2_alpha_paths is not None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="matte-prefetch") as pool:
        pending = pool.submit(prepare_frame, 0)
        for t in range(n_src):
            progress(
                0.42 + 0.48 * (t / max(n_src, 1)),
                desc=(
                    f"MatAnyone 2 + PyMatting & Ghép Linear-RGB frame {t + 1}/{n_src}..." if use_matanyone2
                    else (f"Fast Tiny + ghép sáng frame {t + 1}/{n_src}..." if fast_tiny_active
                          else f"Tách Alpha & Ghép Khớp Sáng frame {t + 1}/{n_src}...")
                ),
            )
            src_bgr, mres, prepared_ref, m_fg, tiny_margin_mask, tiny_recovered_silver = pending.result()
            if t + 1 < n_src:
                pending = pool.submit(prepare_frame, t + 1)

            if prepared_ref is not None and prepared_ref == t:
                prev_alpha_stable = prev_src_bgr = None
            if use_matanyone2 or fast_tiny_active:
                alpha, fg_clean_bgr = mres
            else:
                alpha = mres.alpha
            if sam2_alpha_paths is not None and not fast_tiny_active and not use_matanyone2:
                predicted = cv2.imread(str(sam2_alpha_paths[t]), cv2.IMREAD_GRAYSCALE)
                if predicted is None or predicted.shape != (h, w):
                    raise RuntimeError(f"SAM2Matting alpha đã xác thực bị thiếu tại frame {t}.")
                alpha = predicted.astype(np.float32) / 255.0
                support = cv2.dilate(
                    (m_fg > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                ) > 0
                alpha[~support] = 0.0
                tiny_added = (tiny_margin_mask > 0) & (m_fg == 0)
                alpha[tiny_added] = np.maximum(alpha[tiny_added], 0.35)
            if not fast_tiny_active and not use_matanyone2:
                fg_clean_bgr = mres.fg_clean_bgr
            if tiny_recovered_silver is not None and np.any(tiny_recovered_silver):
                fg_clean_bgr[tiny_recovered_silver] = src_bgr[tiny_recovered_silver]
            if use_matanyone2:
                alpha_stable = stabilize_alpha_temporal(
                    alpha, prev_alpha_stable, src_bgr, prev_src_bgr, temporal_weight=0.25
                )
            else:
                alpha_stable = stabilize_alpha_temporal(alpha, prev_alpha_stable, src_bgr, prev_src_bgr)
            if prepared_ref is not None and prepared_ref == t:
                alpha_stable = _protect_manual_background(alpha_stable, state, t)
            prev_alpha_stable = alpha_stable
            prev_src_bgr = src_bgr
            if not fast_tiny_active and not use_matanyone2:
                total_restored += mres.missing_restored_px
                total_removed += mres.excess_removed_px

            # Time-map background frame by real FPS ratio to prevent speed mismatch or loop jumps
            if n_bg > 1:
                src_fps = float(state.get("fps") or 30.0)
                mapped_bg = int(round(t * (float(bg_fps or src_fps) / max(src_fps, 1.0))))
                t_bg = mapped_bg % n_bg if mapped_bg >= n_bg else mapped_bg
            else:
                t_bg = 0
            bg_bgr = _load_bgr(bg_frames[t_bg])

            if cam_transforms is not None:
                bg_bgr = warp_bg_with_camera_motion(bg_bgr, cam_transforms[t], (w, h))
            elif v2v_transforms is not None:
                bg_bgr, _ = warp_ai_frame(bg_bgr, v2v_transforms[t].to_matrix(), (w, h))

            comp_kw = {"frame_idx": t} if _is_ai_engine(matting_engine) or linear_compositing else {}
            comp_bgr = composite_with_relighting(
                fg_clean_bgr=fg_clean_bgr,
                alpha=alpha_stable,
                new_bg_bgr=bg_bgr,
                src_full_bgr=src_bgr,
                cfg=rcfg,
                **comp_kw,
            )
            cv2.imwrite(str(out_dir / f"{t:05d}.png"), comp_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    progress(0.93, desc="Đang xuất Video MP4 chất lượng cao (FFmpeg H.264 CRF 16)...")
    audio_src = (
        state["src_media"] if audio_choice == "Giữ âm thanh Video Gốc"
        else (bg_file if audio_choice == "Lấy âm thanh Video Nền Mới" else None)
    )
    out_mp4 = encode_video(out_dir, work / "AGY_Final_Relit_Composite.mp4", fps=state["fps"], audio_source_path=audio_src)
    if sam2_run_dir is not None:
        shutil.rmtree(sam2_run_dir, ignore_errors=True)
    progress(1.0, desc="Hoàn tất!")
    if fast_tiny_active:
        status_msg = f"Render hoàn tất ({n_src} frames) -> {out_mp4} | Fast Tiny: alpha GPU + màu tiền cảnh rút gọn; không chạy kiểm định Vòng 2.{sam2_fallback_note}"
    elif use_matanyone2 or _is_ai_engine(matting_engine):
        status_msg = f"Render hoàn tất ({n_src} frames) -> {out_mp4}{matanyone2_note or (' | Engine: ' + matting_engine)}{sam2_fallback_note}"
    else:
        status_msg = f"Render hoàn tất ({n_src} frames) -> {out_mp4} | [Báo cáo Kiểm Định Vòng 2]: Đã tự động bù đắp +{total_restored:,} px bị thiếu & cắt bỏ -{total_removed:,} px bị thừa!{sam2_fallback_note}"
    return str(out_mp4), status_msg


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Tool Tách Ghép Nền Chất Lượng Cao — AGY Roto-Brush 3D Relighting Studio") as demo:
        state = gr.State({})

        with gr.Group(elem_classes="hero-header"):
            gr.Markdown(
                "## 🎬 Tool Tách Ghép Nền Chất Lượng Cao — AGY 3D Roto-Brush & Google Flow Studio (100% Offline)\n"
                "**3 Bước Đơn Giản:** **① Nạp Video & Chạm Tách Chủ Thể** ➔ **② Tạo Prompt Đổi Bối Cảnh (Google Flow)** ➔ **③ Ghép Nền Mới Tự Khớp Ánh Sáng & Đổ Bóng 3D**"
            )
            status_banner = gr.Textbox(
                label="Trạng thái xử lý",
                value="Sẵn sàng! Hãy nạp Video gốc (hoặc Ảnh gốc) ở Bước ① bên dưới để bắt đầu.",
                interactive=False,
                lines=1,
                elem_classes="status-bar",
            )

        # =========================================================
        # BƯỚC 1: TÁCH CHỦ THỂ THÔNG MINH (SOTA AI MATTING + ROTO-BRUSH)
        # =========================================================
        with gr.Group(elem_classes="step-card"):
            gr.Markdown(
                "### ① Bước 1: Tách Chủ Thể AI Siêu Sạch (BiRefNet HR / MatAnyone 2 / ViTMatte-Base + SAM 2.1 Roto-Brush)",
                elem_classes="step-title",
            )
            matting_engine = gr.Dropdown(
                choices=MATTING_ENGINE_CHOICES,
                value=MATTING_ENGINE_MATANYONE2,
                label="AI Matting Engine (Khuyên dùng: MatAnyone 2 + ViTMatte-Base hoặc BiRefNet HR 1-Click)",
            )
            with gr.Row():
                # Cột Trái: Nạp Video + Gallery Ảnh Tham Chiếu + Thông số
                with gr.Column(scale=3):
                    src_input = gr.File(
                        label="Nạp Video Gốc (.mp4, .mov) hoặc Ảnh (.jpg, .png)",
                        file_types=["video", "image"],
                    )
                    auto_crop_chk = gr.Checkbox(
                        value=True,
                        label="Tự động cắt bỏ viền đen 2 bên (9:16 trong 16:9)",
                    )
                    step1_btn = gr.Button(
                        "1. Nạp Video & Phân Tích",
                        variant="primary",
                        elem_classes="compact-btn",
                    )
                    ref_gallery = gr.Gallery(
                        label="Ảnh Tham Chiếu Tự Cắt (Click để nhảy tới frame)",
                        columns=3,
                        height=180,
                        allow_preview=False,
                    )
                    with gr.Accordion("Thông số góc máy & ánh sáng tự đo", open=False):
                        telemetry_box = gr.Markdown("*Chưa có dữ liệu. Hãy bấm '1. Nạp Video & Phân Tích'.*")

                # Cột Giữa: Bàn Vẽ Roto-Brush
                with gr.Column(scale=5):
                    frame_slider = gr.Slider(0, 1, step=1, value=0, label="Chọn Khung Hình (Frame Scrubber)")
                    roto_editor = StrokeImageEditor(
                        label="Bôi 1 Nét Cọ Xanh (Chọn thông minh kiểu Photoshop) hoặc Cọ Đỏ (Trừ thông minh vùng thừa)",
                        type="numpy",
                        image_mode="RGBA",
                        height=440,
                        brush=gr.Brush(colors=["#00ff66", "#ff3333"], default_size=18, color_mode="fixed"),
                        eraser=gr.Eraser(default_size=18),
                        transforms=(),
                        layers=False,
                    )
                    with gr.Row():
                        auto_birefnet_btn = gr.Button(
                            "✨ Tự Động Tách Nền AI 1-Click (BiRefNet HR)",
                            variant="primary",
                            elem_classes="compact-btn",
                        )
                        auto_roto_btn = gr.Button(
                            "🖐️ Tự Động Tách Tay + Sản Phẩm",
                            variant="secondary",
                            elem_classes="compact-btn",
                        )
                    with gr.Row():
                        confirm_roto_btn = gr.Button(
                            "🖌️ Xác Nhận Nét Cọ",
                            variant="secondary",
                            elem_classes="compact-btn",
                        )
                        auto_chroma_btn = gr.Button(
                            "🟩 Tách Phông Xanh",
                            variant="secondary",
                            elem_classes="compact-btn",
                        )
                        reset_roto_btn = gr.Button(
                            "🔄 Xóa Vùng Chọn",
                            variant="secondary",
                            elem_classes="compact-btn",
                        )

                # Cột Phải: Tabs Kết Quả Tách Sub-Pixel + Alpha Matte + Viền Confirm + Tách Toàn Bộ Video
                with gr.Column(scale=4):
                    with gr.Tabs():
                        with gr.Tab("Bản Tách Sub-Pixel (Nền Caro)"):
                            subpixel_cutout_img = gr.Image(
                                label="Kết Quả Tách Sub-Pixel & Khử Viền Màu",
                                type="numpy",
                                interactive=False,
                                height=400,
                            )
                            cutout_download = gr.File(label="Tải ảnh PNG Tách Nền Trong Suốt (RGBA)", height=68)

                        with gr.Tab("Alpha Matte (Đen/Trắng)"):
                            alpha_matte_img = gr.Image(
                                label="Bản Đồ Alpha Matte (Trắng = Giữ lại, Đen = Xóa nền)",
                                type="numpy",
                                interactive=False,
                                height=450,
                            )

                        with gr.Tab("Viền Roto-Brush Confirm"):
                            roto_confirm_img = gr.Image(
                                label="Đường Viền Xác Nhận (Magenta)",
                                type="numpy",
                                interactive=False,
                                height=450,
                            )

                        with gr.Tab("Tách Toàn Bộ Video"):
                            gr.Markdown("*Tùy chọn: Xem trước video đã tách sạch toàn bộ các khung hình trước khi ghép nền.*")
                            step3_track_btn = gr.Button(
                                "Chạy Tách Sub-Pixel Toàn Bộ Video (AI Matting)",
                                variant="primary",
                                elem_classes="compact-btn",
                            )
                            cutout_preview_vid = gr.Video(
                                label="Video Đã Tách Nền Trên Nền Caro",
                                height=360,
                            )

        # =========================================================
        # BƯỚC 2: VIẾT PROMPT TẠO BỐI CẢNH MỚI (GOOGLE FLOW)
        # =========================================================
        with gr.Group(elem_classes="step-card"):
            with gr.Accordion(
                "② Bước 2: Viết Prompt Tạo Video Bối Cảnh Mới (Google Flow — Bấm để mở rộng)",
                open=False,
            ):
                with gr.Row(elem_classes="preset-row"):
                    preset_airport = gr.Button("✈️ Sảnh Sân Bay", size="sm")
                    preset_cafe = gr.Button("☕ Quán Cafe Tokyo", size="sm")
                    preset_hotel = gr.Button("🏨 Khách Sạn 5 Sao", size="sm")
                    preset_street = gr.Button("🚶 Phố Đi Bộ", size="sm")
                    preset_taxi = gr.Button("🚕 Ghế Sau Taxi", size="sm")

                with gr.Row():
                    with gr.Column(scale=8):
                        target_bg_input = gr.Textbox(
                            label="Nhập mô tả bối cảnh mới muốn đổi sang (Tiếng Việt tự nhiên)",
                            value="Sảnh sân bay quốc tế hiện đại, không gian rộng thoáng, ánh sáng ban ngày tự nhiên xuyên qua vách kính, có hành khách kéo vali mờ nhẹ ở xa",
                            lines=2,
                        )
                    with gr.Column(scale=4):
                        step4_prompt_btn = gr.Button(
                            "2. Viết Prompt Tạo Video Nền Mới (Google Flow)",
                            variant="primary",
                            elem_classes="compact-btn",
                        )

                with gr.Row():
                    clean_plate_prompt_box = gr.Textbox(
                        label="PROMPT 1 (Khuyên dùng): Tạo Video Phông Nền Sạch (Không dính tay/máy ảo, khớp góc máy & Bokeh)",
                        lines=5,
                        buttons=["copy"],
                    )
                    v2v_prompt_box = gr.Textbox(
                        label="PROMPT 2: Google Flow Omni 1.2 V2V (Đổi nền trực tiếp từ video gốc trên Flow)",
                        lines=5,
                        buttons=["copy"],
                    )

                with gr.Accordion("Xem báo cáo chiến lược quang học chi tiết", open=False):
                    prompt_strategy_md = gr.Markdown("*Bấm '2. Viết Prompt Tạo Video Nền Mới' để xem báo cáo chiến lược.*")

        # =========================================================
        # BƯỚC 3: GHÉP VÀ XEM TRƯỚC LIVE
        # =========================================================
        with gr.Group(elem_classes="step-card"):
            gr.Markdown("### ③ Ghép Nền & Soi Lỗi Cận Pixel (Live Preview)", elem_classes="step-title")
            with gr.Row():
                # Cột Trái: Viewer Lớn + Zoom/Pan + Chế Độ Soi Lỗi
                with gr.Column(scale=7, min_width=460):
                    with gr.Row():
                        bg_input = gr.File(
                            label="Video hoặc ảnh nền mới",
                            file_types=["video", "image"],
                            scale=5,
                            height=76,
                        )
                        preview_frame = gr.Slider(
                            0, 1, step=1, value=0,
                            label="Khung hình xem trước (đồng bộ Bước 1)",
                            scale=5,
                        )
                        live_preview_chk = gr.Checkbox(
                            value=True,
                            label="⚡ Live Preview",
                            scale=2,
                        )

                    preview_diag_mode = gr.Radio(
                        choices=PREVIEW_DIAG_CHOICES,
                        value=PREVIEW_DIAG_SPLIT,
                        label="Chế độ soi kiểm tra chi tiết",
                    )

                    with gr.Row(elem_classes="preset-row"):
                        zoom_1x_btn = gr.Button("1× Toàn khung", size="sm")
                        zoom_2x_btn = gr.Button("2× Soi viền", size="sm")
                        zoom_3x_btn = gr.Button("3× Cận pixel", size="sm")
                        zoom_thumb_btn = gr.Button("🔍 Soi Ngón Cái (Đáy)", size="sm")
                        zoom_top_btn = gr.Button("🔍 Soi Viền Trên", size="sm")

                    with gr.Row():
                        preview_zoom = gr.Slider(1.0, 4.0, value=1.0, step=0.25, label="Zoom cận pixel (1×–4×)", scale=4)
                        preview_pan_x = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Pan X", scale=3)
                        preview_pan_y = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Pan Y", scale=3)

                    comp_compare_slider = gr.ImageSlider(
                        label="Gốc / Soi Lỗi ⟷ Ghép Nền",
                        type="numpy",
                        height=580,
                        max_height=680,
                        elem_id="comp-zoom-box",
                        elem_classes=["zoom-inspect-box"],
                        buttons=["fullscreen"],
                    )
                    with gr.Row():
                        preview_status = gr.Markdown(
                            "Chọn nền và kéo bất kỳ thanh chỉnh nào để xem trước trực tiếp (Lăn chuột lên ảnh để zoom 1×–6×).",
                        )
                        preview_comp_btn = gr.Button("👁️ Làm mới Preview", variant="secondary", scale=1)

                # Cột Phải: 3 Tab Gọn Sạch
                with gr.Column(scale=5, min_width=340):
                    with gr.Tabs():
                        with gr.Tab("🎨 Ánh sáng & Ghép (Live)"):
                            ambient_slider = gr.Slider(0.0, 1.0, value=0.68, step=0.04, label="Khớp Nhiệt Độ Màu & Phơi Sáng Nền (Lab)")
                            dir_light_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.04, label="Khớp Hướng Nguồn Sáng 2D")
                            with gr.Row():
                                lw_rad_slider = gr.Slider(0, 15, value=4, step=1, label="Bán kính Đổ Bóng Viền (px)")
                                lw_str_slider = gr.Slider(0.0, 1.0, value=0.65, step=0.04, label="Cường độ Đổ Bóng Mép Viền")
                            with gr.Row():
                                protect_screen_chk = gr.Checkbox(value=True, label="Bảo vệ màn hình phát sáng (Chữ UI nét & tránh lấn tay)")
                                screen_glare_slider = gr.Slider(0.0, 0.40, value=0.06, step=0.02, label="Độ bóng kính màn hình")
                            with gr.Row():
                                grain_slider = gr.Slider(0.0, 0.50, value=0.12, step=0.03, label="Khớp hạt nhiễu (Grain)")
                                bg_defocus_slider = gr.Slider(0.0, 15.0, value=0.6, step=0.2, label="Mờ nhẹ hậu cảnh (px)")
                            with gr.Row():
                                linear_comp_chk = gr.Checkbox(value=True, label="Ghép Linear-RGB chuẩn điện ảnh (Chống viền tối)")
                                clean_ghost_chk = gr.Checkbox(value=False, label="Xóa bóng chủ thể AI ở nền (Ghost Inpaint)")

                        with gr.Tab("✂️ Viền & Bóc màu"):
                            use_pymatting_fg_chk = gr.Checkbox(value=True, label="Bóc màu viền vật lý PyMatting (Hết sạch viền/halo)")
                            decont_slider = gr.Slider(0.0, 1.0, value=0.95, step=0.04, label="Khử màu viền nền cũ (Decontaminate)")
                            with gr.Row():
                                trimap_rad = gr.Slider(1, 24, value=8, step=1, label="Bán kính Sub-Pixel Trimap (px)")
                                guided_rad = gr.Slider(1, 10, value=2, step=1, label="Guided Filter Edge Smooth (px)")
                            with gr.Row():
                                erode_dilate = gr.Slider(-6, 6, value=0, step=1, label="Co/Giãn viền theo màu nguồn (px)")
                                edge_blur_slider = gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="Khớp độ mềm mép viền theo nền (px)")
                            despill_chk = gr.Checkbox(value=False, label="Ép bật Khử ám xanh (Despill)")
                            matanyone_res_dd = gr.Dropdown(
                                choices=[480, 720, 1080, -1],
                                value=720,
                                label="Độ phân giải nội bộ MatAnyone 2 (720 tối ưu 8GB VRAM)",
                            )

                        with gr.Tab("🎬 Xuất Video & Nâng cao"):
                            tracking_mode = gr.Radio(
                                choices=[
                                    "Camera Motion Lock (Khóa rung/chuyển động Camera gốc vào Nền mới)",
                                    "Flow V2V Spatial Sync + Ghost Clean (Đồng bộ V2V & Xóa bóng thiết bị AI)",
                                    "Lock 1:1 (Giữ nguyên chuyển động của Video nền mới)",
                                ],
                                value="Camera Motion Lock (Khóa rung/chuyển động Camera gốc vào Nền mới)",
                                label="Chế độ Tracking & Đồng bộ Chuyển động",
                            )
                            audio_choice = gr.Radio(
                                choices=["Giữ âm thanh Video Gốc", "Lấy âm thanh Video Nền Mới", "Không âm thanh"],
                                value="Giữ âm thanh Video Gốc",
                                label="Nguồn âm thanh xuất ra",
                            )
                            render_final_btn = gr.Button("🎬 3. Xuất video ghép hoàn chỉnh (MP4)", variant="primary", elem_classes="compact-btn")
                            final_video_out = gr.Video(label="Video thành phẩm", height=320)

                            with gr.Accordion("Tùy chọn Nâng cao / Legacy CPU & Mô tả Prompt", open=False):
                                pass2_legacy_chk = gr.Checkbox(value=False, label="Bật Kiểm định Pass-2/Pass-3 cũ (Chỉ cho Classic CPU)")
                                sam2matting_tiny_chk = gr.Checkbox(
                                    value=False,
                                    label="SAM2Matting Tiny GPU (phi thương mại, chỉ thay alpha)",
                                )
                                fast_tiny_chk = gr.Checkbox(
                                    value=False,
                                    label="Fast (cần bật Tiny GPU): màu tiền cảnh từ alpha Tiny (viền có thể khác matte CPU)",
                                )
                                subject_desc_input = gr.Textbox(
                                    label="Mô tả chủ thể giữ nguyên (KEEP) cho Prompt",
                                    value="Hai bàn tay đang cầm máy phiên dịch cầm tay màu đen có màn hình hiển thị giao diện Tiếng Việt - Tiếng Trung sắc nét",
                                    lines=2,
                                )
                                use_agy_llm_chk = gr.Checkbox(
                                    value=True,
                                    label="Sử dụng AGY Local AI Router để làm giàu ngôn ngữ điện ảnh cho Prompt",
                                )


        # =========================================================
        # KẾT NỐI SỰ KIỆN (EVENT WIRING)
        # =========================================================
        loaded_source = step1_btn.click(
            on_step1_load_source,
            inputs=[src_input, auto_crop_chk, state],
            outputs=[state, ref_gallery, telemetry_box, frame_slider, roto_editor, status_banner],
        )

        ref_gallery.select(
            on_select_ref_gallery,
            inputs=[state],
            outputs=[frame_slider, roto_editor],
        )

        frame_slider.release(
            on_frame_slider_change,
            inputs=[frame_slider, state],
            outputs=[roto_editor],
        )

        def _ui_step2_quick_stroke(*args):
            res = on_step2_quick_stroke(*args)
            return (*res, _alpha_preview_from_cutout_path(res[3]))

        def _ui_step2_confirm_roto(*args):
            res = on_step2_confirm_roto_brush(*args)
            return (*res, _alpha_preview_from_cutout_path(res[3]))

        def _ui_step2_auto_birefnet(*args):
            res = on_step2_auto_birefnet(*args)
            return (*res, _alpha_preview_from_cutout_path(res[3]))

        def _ui_step2_auto_roto(*args):
            res = on_step2_auto_roto(*args)
            return (*res, _alpha_preview_from_cutout_path(res[3]))

        def _ui_step2_auto_chroma(*args):
            res = on_step2_auto_chroma_key(*args)
            return (*res, _alpha_preview_from_cutout_path(res[3]))

        def _ui_step2_reset(*args):
            res = on_step2_reset_selection(*args)
            return (*res, None)

        step2_outputs = [
            roto_editor, roto_confirm_img, subpixel_cutout_img, cutout_download, state, status_banner, alpha_matte_img
        ]

        roto_editor.input(
            _ui_step2_quick_stroke,
            inputs=[
                roto_editor, frame_slider, trimap_rad, guided_rad, decont_slider, despill_chk, erode_dilate, state,
                matting_engine, use_pymatting_fg_chk, pass2_legacy_chk,
            ],
            outputs=step2_outputs,
        )

        confirm_roto_btn.click(
            _ui_step2_confirm_roto,
            inputs=[
                roto_editor, frame_slider, trimap_rad, guided_rad, decont_slider, despill_chk, erode_dilate, state,
                matting_engine, use_pymatting_fg_chk, pass2_legacy_chk,
            ],
            outputs=step2_outputs,
        )

        auto_birefnet_btn.click(
            _ui_step2_auto_birefnet,
            inputs=[frame_slider, decont_slider, despill_chk, erode_dilate, use_pymatting_fg_chk, state],
            outputs=step2_outputs,
        )

        auto_roto_btn.click(
            _ui_step2_auto_roto,
            inputs=[
                frame_slider, trimap_rad, guided_rad, decont_slider, despill_chk, erode_dilate, state,
                matting_engine, use_pymatting_fg_chk, pass2_legacy_chk,
            ],
            outputs=step2_outputs,
        )

        auto_chroma_btn.click(
            _ui_step2_auto_chroma,
            inputs=[frame_slider, trimap_rad, guided_rad, decont_slider, erode_dilate, state, use_pymatting_fg_chk],
            outputs=step2_outputs,
        )

        reset_roto_btn.click(
            _ui_step2_reset,
            inputs=[frame_slider, state],
            outputs=step2_outputs,
        )

        step3_track_btn.click(
            on_step3_track_and_matte_all_frames,
            inputs=[
                trimap_rad, guided_rad, decont_slider, despill_chk, erode_dilate, state,
                matting_engine, use_pymatting_fg_chk, pass2_legacy_chk, matanyone_res_dd,
            ],
            outputs=[state, cutout_preview_vid, status_banner],
        )

        # Quick background presets
        preset_airport.click(
            lambda: "Sảnh sân bay quốc tế hiện đại, không gian rộng thoáng, ánh sáng ban ngày tự nhiên xuyên qua vách kính, có bảng điện tử chuyến bay và hành khách kéo vali mờ nhẹ ở xa",
            outputs=target_bg_input,
        )
        preset_cafe.click(
            lambda: "Quán cà phê kính sang trọng tại Tokyo buổi sáng, nội thất gỗ ấm áp, ánh nắng nhẹ chiếu xiên qua khung cửa sổ kính lớn, hậu cảnh xóa phông mịn",
            outputs=target_bg_input,
        )
        preset_hotel.click(
            lambda: "Sảnh khách sạn 5 sao cao cấp với sàn đá cẩm thạch bóng, đèn hắt kiến trúc ấm áp sang trọng, cây xanh trang trí tinh tế ở hậu cảnh mờ Bokeh",
            outputs=target_bg_input,
        )
        preset_street.click(
            lambda: "Phố đi bộ thương mại hiện đại phong cách châu Âu / Tokyo dưới ánh sáng ban ngày tự nhiên, hàng cây và cửa hiệu sang trọng mờ Bokeh ở phía xa",
            outputs=target_bg_input,
        )
        preset_taxi.click(
            lambda: "Khoang hành khách ghế sau của xe Taxi / Limousine sang trọng đang di chuyển êm ái trên đại lộ thành phố, ánh sáng tự nhiên chiếu qua cửa sổ xe",
            outputs=target_bg_input,
        )

        step4_prompt_btn.click(
            on_step4_generate_prompt,
            inputs=[target_bg_input, subject_desc_input, use_agy_llm_chk, state],
            outputs=[clean_plate_prompt_box, v2v_prompt_box, prompt_strategy_md, status_banner],
        )

        def sync_preview_frame(value, current_state):
            count = len(current_state.get("src_frames", []))
            return gr.update(maximum=max(1, count - 1), value=value)

        loaded_source.success(sync_preview_frame, inputs=[frame_slider, state], outputs=preview_frame)
        frame_slider.change(sync_preview_frame, inputs=[frame_slider, state], outputs=preview_frame)
        preview_frame.release(on_frame_slider_change, inputs=[preview_frame, state], outputs=roto_editor)
        preview_frame.release(lambda value: value, inputs=preview_frame, outputs=frame_slider)

        preview_inputs = [
            preview_frame, bg_input, trimap_rad, guided_rad, decont_slider, despill_chk, erode_dilate,
            ambient_slider, dir_light_slider, lw_rad_slider, lw_str_slider, protect_screen_chk,
            screen_glare_slider, grain_slider, bg_defocus_slider, clean_ghost_chk, state,
            matting_engine, use_pymatting_fg_chk, linear_comp_chk, pass2_legacy_chk, edge_blur_slider,
            preview_diag_mode, preview_zoom, preview_pan_x, preview_pan_y, live_preview_chk,
        ]

        def manual_preview(*args):
            return on_step5_preview_composite(*args[:26])

        def refresh_preview(*args):
            current_state = args[16]
            live_enabled = bool(args[26])
            if not live_enabled:
                return gr.update(), "Live Preview đang tắt — bấm '👁️ Làm mới Preview' để cập nhật."
            if not args[1]:
                return gr.update(), "Hãy nạp ảnh hoặc video nền mới để xem trước."
            if not current_state.get("src_frames") or not (current_state.get("confirmed_masks") or current_state.get("tracked_masks")):
                return gr.update(), "Hãy nạp video gốc và xác nhận vùng chọn trước khi xem trước."
            return on_step5_preview_composite(*args[:26])

        preview_comp_btn.click(
            manual_preview, inputs=preview_inputs,
            outputs=[comp_compare_slider, preview_status], concurrency_id="preview", concurrency_limit=1,
        )

        # Quick zoom/focus buttons immediately update sliders and refresh preview
        for btn, preset_vals in (
            (zoom_1x_btn, (1.0, 0.5, 0.5)),
            (zoom_2x_btn, (2.0, 0.5, 0.5)),
            (zoom_3x_btn, (3.0, 0.5, 0.5)),
            (zoom_thumb_btn, (2.5, 0.5, 0.85)),
            (zoom_top_btn, (2.5, 0.5, 0.15)),
        ):
            z_val, px_val, py_val = preset_vals
            btn.click(
                lambda zv=z_val, px=px_val, py=py_val: (zv, px, py),
                inputs=None,
                outputs=[preview_zoom, preview_pan_x, preview_pan_y],
                queue=False,
            ).then(
                refresh_preview,
                inputs=preview_inputs,
                outputs=[comp_compare_slider, preview_status],
                concurrency_id="preview",
                concurrency_limit=1,
                show_progress="hidden",
            )

        # Instant live preview triggers (.input) for relighting, zoom/pan, and toggle controls
        live_triggers = [
            ambient_slider.input, dir_light_slider.input, lw_rad_slider.input, lw_str_slider.input,
            screen_glare_slider.input, grain_slider.input, bg_defocus_slider.input, edge_blur_slider.input,
            preview_zoom.input, preview_pan_x.input, preview_pan_y.input,
            preview_diag_mode.input,
            despill_chk.input, protect_screen_chk.input, clean_ghost_chk.input,
            use_pymatting_fg_chk.input, linear_comp_chk.input, pass2_legacy_chk.input,
        ]
        gr.on(
            live_triggers,
            refresh_preview,
            inputs=preview_inputs,
            outputs=[comp_compare_slider, preview_status],
            trigger_mode="always_last",
            concurrency_id="preview",
            concurrency_limit=1,
            show_progress="hidden",
        )

        # Heavy matting / source / engine triggers on release or change
        heavy_triggers = [
            preview_frame.release, bg_input.change, matting_engine.change,
            trimap_rad.release, guided_rad.release, erode_dilate.release, decont_slider.release,
        ]
        gr.on(
            heavy_triggers,
            refresh_preview,
            inputs=preview_inputs,
            outputs=[comp_compare_slider, preview_status],
            trigger_mode="always_last",
            concurrency_id="preview",
            concurrency_limit=1,
            show_progress="minimal",
        )

        render_final_btn.click(
            on_step5_render_final_video,
            inputs=[
                bg_input, tracking_mode, audio_choice, trimap_rad, guided_rad, decont_slider, despill_chk, erode_dilate,
                ambient_slider, dir_light_slider, lw_rad_slider, lw_str_slider, protect_screen_chk,
                screen_glare_slider, grain_slider, bg_defocus_slider, clean_ghost_chk, state, sam2matting_tiny_chk, fast_tiny_chk,
                matting_engine, use_pymatting_fg_chk, linear_comp_chk, pass2_legacy_chk, matanyone_res_dd, edge_blur_slider,
            ],
            outputs=[final_video_out, status_banner],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGY Roto-Brush & SOTA Neural Matting Studio")
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--no-browser", action="store_true")
    args, _unknown = parser.parse_known_args()

    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=not args.no_browser,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        js=ZOOM_JS,
    )

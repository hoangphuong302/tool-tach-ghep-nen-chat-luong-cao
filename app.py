from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from src.alpha_matting import (
    MattingConfig,
    auto_chroma_key_mask,
    auto_seed_roto_strokes,
    detect_active_frame_box,
    extract_pixel_accurate_matte,
    extract_strokes_from_editor,
    sample_roto_anchor_points,
    stabilize_alpha_temporal,
)
from src.flow_prompt_engine import analyze_scene_telemetry, generate_flow_prompts
from src.relight_harmonizer import (
    RelightConfig,
    composite_with_relighting,
    detect_emissive_screen_mask,
)
from src.segmentation import (
    clear_vram,
    preview_single_frame_mask,
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

APP_WORK_DIR = Path(__file__).resolve().parent / "workspace" / "roto_studio_run"

CUSTOM_CSS = """
.gradio-container { max-width: 1480px !important; margin: 0 auto !important; padding: 14px 20px !important; font-family: 'Inter', 'Segoe UI', sans-serif !important; }
.hero-header { background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(16,185,129,0.10)); border: 1px solid rgba(59,130,246,0.28); border-radius: 14px; padding: 16px 20px; margin-bottom: 12px; }
.step-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(148,163,184,0.24); border-radius: 14px; padding: 16px !important; margin-bottom: 14px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.step-title { font-size: 1.08rem !important; font-weight: 700 !important; margin-bottom: 6px !important; }
.compact-btn { min-height: 40px !important; font-weight: 600 !important; border-radius: 10px !important; }
.preset-row button { border-radius: 999px !important; font-size: 12px !important; padding: 4px 12px !important; }
.status-bar textarea { font-weight: 600 !important; font-size: 13.5px !important; border-radius: 10px !important; }
"""


def _load_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {path}")
    return img


def _load_rgb(path: str | Path) -> np.ndarray:
    return cv2.cvtColor(_load_bgr(path), cv2.COLOR_BGR2RGB)


def _make_checkerboard_cutout(fg_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Render sub-pixel alpha cutout over a professional dark diagonal/grid checkerboard (returns RGB)."""
    h, w = fg_bgr.shape[:2]
    checker = (np.indices((h, w)).sum(axis=0) // 16) % 2
    bg = np.where(checker[..., None] == 0, 42, 74).astype(np.float32)
    a3 = np.clip(alpha, 0.0, 1.0)[..., None]
    comp_bgr = np.clip(bg * (1.0 - a3) + fg_bgr.astype(np.float32) * a3, 0, 255).astype(np.uint8)
    return cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)


def _render_photoshop_green_selection(rgb: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Photoshop-style cumulative emerald-green overlay + crisp outline directly on the editor canvas."""
    if mask is None or not np.any(mask > 0):
        return rgb
    out = rgb.copy()
    m = mask > 0
    green_tint = np.array([0, 235, 90], dtype=np.float32)
    out[m] = np.clip(out[m].astype(np.float32) * 0.65 + green_tint * 0.35, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 110), 2)
    return out


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

    work = APP_WORK_DIR
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    src_frames, fps, w, h, was_cropped = _extract_media_frames(
        src_file, work / "src_frames", auto_crop_bars=auto_crop_bars
    )
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
    return f_idx, _render_photoshop_green_selection(rgb, active_m)


def on_frame_slider_change(frame_idx: int, state: dict):
    frames = state.get("src_frames")
    if not frames:
        return None
    f_idx = int(np.clip(frame_idx, 0, len(frames) - 1))
    rgb = _load_rgb(frames[f_idx])
    active_m = state.get("active_mask_by_frame", {}).get(int(f_idx))
    return _render_photoshop_green_selection(rgb, active_m)


# =====================================================================
# BƯỚC 2: CÔNG CỤ BÔI CHỌN ROTO-BRUSH & CONFIRM VÙNG CHỌN THÔNG MINH
# =====================================================================
def apply_photoshop_stroke_to_mask(
    rgb: np.ndarray,
    prev_mask: Optional[np.ndarray],
    fg_stroke: np.ndarray,
    bg_stroke: np.ndarray,
) -> np.ndarray:
    """
    Photoshop Object Selection 1-Stroke Cumulative Logic:
      - Green stroke (`fg_stroke`): auto-detect the entire clicked/stroked object and UNION into `prev_mask`.
      - Red stroke (`bg_stroke`): auto-detect the unwanted region and SUBTRACT from `prev_mask`.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    has_fg = bool(np.any(fg_stroke > 0))
    has_bg = bool(np.any(bg_stroke > 0))

    if has_fg:
        pts = sample_roto_anchor_points(fg_stroke, bg_stroke, bgr, grid_step=28)
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
                raise gr.Error("Chưa có nét cọ Xanh Lá (+ Giữ lại). Hãy bôi 1 nét lên thiết bị/bàn tay hoặc bấm '✨ Tự Động Tách Tay + Sản Phẩm'!")
        else:
            mask = apply_photoshop_stroke_to_mask(rgb, prev_mask, fg_stroke, bg_stroke)

    # Save confirmed binary mask & cumulative active mask for this keyframe
    state.setdefault("active_mask_by_frame", {})[int(f_idx)] = (mask > 0).astype(np.uint8)
    work = APP_WORK_DIR
    conf_dir = work / "confirmed_masks"
    conf_dir.mkdir(parents=True, exist_ok=True)
    mask_path = conf_dir / f"conf_{f_idx:05d}.png"
    save_mask(mask, mask_path)
    state["confirmed_masks"][int(f_idx)] = str(mask_path)
    state["roto_strokes"][int(f_idx)] = (fg_stroke, bg_stroke)

    # Run Sub-Pixel Alpha Matting & Edge Color Decontamination on this frame
    mcfg = MattingConfig(
        trimap_radius=int(trimap_radius),
        guided_radius=int(guided_radius),
        decontaminate_strength=float(decont_strength),
        despill_green=bool(despill_green),
        erode_dilate=int(erode_dilate),
    )
    mres = extract_pixel_accurate_matte(bgr, mask, mcfg)

    # Save transparent RGBA PNG for instant download/inspection
    cutout_png_path = work / f"subpixel_cutout_f{f_idx:05d}.png"
    cv2.imwrite(
        str(cutout_png_path),
        cv2.cvtColor(mres.rgba, cv2.COLOR_RGBA2BGRA),
    )

    editor_green_rgb = _render_photoshop_green_selection(rgb, mask)
    roto_vis = _draw_roto_confirm_overlay(bgr, mask, fg_stroke, bg_stroke)
    cutout_vis = _make_checkerboard_cutout(mres.fg_clean_bgr, mres.alpha)

    # Refresh telemetry using actual confirmed foreground mask
    state["telemetry"] = analyze_scene_telemetry(bgr, mask)

    n_conf = len(state["confirmed_masks"])
    msg = (
        f"Đã tự động phủ xanh toàn bộ vật thể tại Frame #{f_idx} (Tổng {n_conf} keyframe) | "
        f"[Vòng 2 Kiểm Định Pixel]: Bù đắp +{mres.missing_restored_px} px & cắt bỏ -{mres.excess_removed_px} px thừa!"
    )
    return editor_green_rgb, roto_vis, cutout_vis, str(cutout_png_path), state, msg


def on_step2_quick_stroke(
    editor_val: Optional[dict],
    frame_idx: int,
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    state: dict,
):
    """Photoshop 1-Stroke Auto-Fill trigger on `roto_editor.input` / `apply`: fills object green immediately."""
    if not state.get("src_frames") or not editor_val:
        return gr.update(), gr.update(), gr.update(), gr.update(), state, gr.update()
    f_idx = int(np.clip(frame_idx, 0, len(state["src_frames"]) - 1))
    bgr = _load_bgr(state["src_frames"][f_idx])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prev_mask = state.get("active_mask_by_frame", {}).get(f_idx)
    base_rgb = _render_photoshop_green_selection(rgb, prev_mask)

    fg_stroke, bg_stroke = extract_strokes_from_editor(editor_val, base_rgb)
    if not np.any(fg_stroke > 0) and not np.any(bg_stroke > 0):
        return gr.update(), gr.update(), gr.update(), gr.update(), state, gr.update()

    return _run_confirm_and_preview(
        f_idx, bgr, fg_stroke, bg_stroke,
        trimap_radius, guided_radius, decont_strength, despill_green, erode_dilate, state,
    )


def on_step2_reset_selection(frame_idx: int, state: dict):
    """Clear Photoshop emerald-green cumulative selection on current frame."""
    if not state.get("src_frames"):
        return None, None, None, None, state, "Chưa có video/ảnh gốc."
    f_idx = int(np.clip(frame_idx, 0, len(state["src_frames"]) - 1))
    state.get("active_mask_by_frame", {}).pop(f_idx, None)
    state.get("confirmed_masks", {}).pop(f_idx, None)
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
):
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")
    f_idx = int(np.clip(frame_idx, 0, len(state["src_frames"]) - 1))
    bgr = _load_bgr(state["src_frames"][f_idx])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prev_mask = state.get("active_mask_by_frame", {}).get(f_idx)
    base_rgb = _render_photoshop_green_selection(rgb, prev_mask)

    fg_stroke, bg_stroke = extract_strokes_from_editor(editor_val, base_rgb)
    if not np.any(fg_stroke > 0) and not np.any(bg_stroke > 0) and prev_mask is None:
        # If user didn't paint a stroke yet, fall back to smart auto-roto strokes so button works immediately
        fg_stroke, bg_stroke = auto_seed_roto_strokes(bgr)

    return _run_confirm_and_preview(
        f_idx, bgr, fg_stroke, bg_stroke,
        trimap_radius, guided_radius, decont_strength, despill_green, erode_dilate, state,
    )


def on_step2_auto_roto(
    frame_idx: int,
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    despill_green: bool,
    erode_dilate: int,
    state: dict,
):
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")
    f_idx = int(np.clip(frame_idx, 0, len(state["src_frames"]) - 1))
    bgr = _load_bgr(state["src_frames"][f_idx])
    fg_stroke, bg_stroke = auto_seed_roto_strokes(bgr)
    return _run_confirm_and_preview(
        f_idx, bgr, fg_stroke, bg_stroke,
        trimap_radius, guided_radius, decont_strength, despill_green, erode_dilate, state,
    )


def on_step2_auto_chroma_key(
    frame_idx: int,
    trimap_radius: int,
    guided_radius: int,
    decont_strength: float,
    erode_dilate: int,
    state: dict,
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
):
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")
    if not state.get("confirmed_masks"):
        raise gr.Error("Vui lòng Confirm vùng chọn ở Bước 2 trước khi tách toàn bộ khung hình.")

    work = APP_WORK_DIR
    src_frames = [Path(p) for p in state["src_frames"]]
    n_frames = len(src_frames)

    confirmed_np: Dict[int, np.ndarray] = {
        int(k): load_mask(v) for k, v in state["confirmed_masks"].items()
    }

    if n_frames == 1:
        only_idx = next(iter(confirmed_np.keys()))
        state["tracked_masks"] = [state["confirmed_masks"][only_idx]]
    else:
        progress(0.1, desc="Đang chạy SAM 2.1 Roto-Brush Video Tracking qua từng khung hình...")
        tracked_paths = track_roto_video_masks(
            work / "src_frames",
            work / "tracked_masks",
            confirmed_masks=confirmed_np,
            progress_cb=lambda f, tot: progress(0.1 + 0.45 * (f / max(tot, 1)), desc=f"SAM 2.1 Tracking frame {f + 1}/{tot}..."),
        )
        state["tracked_masks"] = [str(p) for p in tracked_paths]

    # Run per-frame Sub-Pixel Alpha Matting & render transparent cutout preview video
    cutout_frames_dir = work / "cutout_preview_frames"
    cutout_frames_dir.mkdir(parents=True, exist_ok=True)
    for old in cutout_frames_dir.glob("*.png"):
        old.unlink()

    mcfg = MattingConfig(
        trimap_radius=int(trimap_radius),
        guided_radius=int(guided_radius),
        decontaminate_strength=float(decont_strength),
        despill_green=bool(despill_green),
        erode_dilate=int(erode_dilate),
    )

    mask_paths = [Path(p) for p in state["tracked_masks"]]
    prev_alpha_st = None
    prev_bgr_st = None
    for t in range(n_frames):
        progress(0.55 + 0.35 * (t / max(n_frames, 1)), desc=f"Tách Sub-Pixel Alpha Matting & Khử viền frame {t + 1}/{n_frames}...")
        bgr = _load_bgr(src_frames[t])
        m_smooth = temporal_smooth_mask(mask_paths, t) if n_frames > 3 else load_mask(mask_paths[t])
        mres = extract_pixel_accurate_matte(bgr, m_smooth, mcfg)
        alpha_st = stabilize_alpha_temporal(mres.alpha, prev_alpha_st, bgr, prev_bgr_st)
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
) -> Tuple[np.ndarray, np.ndarray]:
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")

    src_frames = state["src_frames"]
    f_idx = int(np.clip(frame_idx, 0, len(src_frames) - 1))
    src_bgr = _load_bgr(src_frames[f_idx])
    h, w = src_bgr.shape[:2]

    # Get mask for frame f_idx
    if state.get("tracked_masks") and f_idx < len(state["tracked_masks"]):
        mask = load_mask(state["tracked_masks"][f_idx])
    elif int(f_idx) in state.get("confirmed_masks", {}):
        mask = load_mask(state["confirmed_masks"][int(f_idx)])
    elif state.get("confirmed_masks"):
        nearest_k = min(state["confirmed_masks"].keys(), key=lambda k: abs(int(k) - f_idx))
        mask = load_mask(state["confirmed_masks"][nearest_k])
    else:
        fg_st, bg_st = auto_seed_roto_strokes(src_bgr)
        pts = sample_roto_anchor_points(fg_st, bg_st, src_bgr, grid_step=32)
        mask = preview_single_frame_mask(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB), pts)

    # Load or synthesize new background frame
    if bg_file:
        p_bg = Path(bg_file)
        if p_bg.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            bg_bgr = _load_bgr(p_bg)
        else:
            cap = cv2.VideoCapture(str(p_bg))
            total_bg = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(f_idx, max(0, total_bg - 1)))
            ret, frame_bg = cap.read()
            cap.release()
            if not ret or frame_bg is None:
                raise gr.Error("Không thể đọc khung hình từ Video nền mới.")
            bg_bgr = frame_bg
    else:
        # Built-in fallback demo background if user hasn't uploaded Flow video yet
        sample_bg = Path("workspace/taxi_test/ai_f0.jpg")
        if sample_bg.exists():
            bg_bgr = cv2.GaussianBlur(_load_bgr(sample_bg), (0, 0), 10.0)
        else:
            bg_bgr = cv2.GaussianBlur(src_bgr, (0, 0), 25.0)

    bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    mcfg = MattingConfig(
        trimap_radius=int(trimap_radius),
        guided_radius=int(guided_radius),
        decontaminate_strength=float(decont_strength),
        despill_green=bool(despill_green),
        erode_dilate=int(erode_dilate),
    )
    mres = extract_pixel_accurate_matte(src_bgr, mask, mcfg)

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
    )
    comp_bgr = composite_with_relighting(
        fg_clean_bgr=mres.fg_clean_bgr,
        alpha=mres.alpha,
        new_bg_bgr=bg_bgr,
        src_full_bgr=src_bgr,
        cfg=rcfg,
    )
    return cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)


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
):
    before_rgb, after_rgb = _prepare_single_preview_composite(
        frame_idx, bg_file, trimap_radius, guided_radius, decont_strength, despill_green, erode_dilate,
        ambient_color_strength, directional_light_strength, light_wrap_radius, light_wrap_strength,
        protect_screen, screen_glare, grain_match, bg_defocus, clean_ai_ghost, state,
    )
    return (before_rgb, after_rgb), "Đã tạo bản xem trước Ghép Nền & Tự Khớp Ánh Sáng (Kéo thanh trượt để so sánh Gốc vs Đã Ghép)!"


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
):
    if not state.get("src_frames"):
        raise gr.Error("Vui lòng thực hiện Bước 1 (Nạp Video/Ảnh gốc) trước.")
    if not bg_file:
        raise gr.Error("Vui lòng nạp Video (hoặc Ảnh) Bối Cảnh Mới tạo từ Google Flow ở Bước 5.")

    work = APP_WORK_DIR
    src_frames = [Path(p) for p in state["src_frames"]]
    n_src = len(src_frames)
    w, h = state["width"], state["height"]

    # Ensure we have per-frame masks across the entire video
    if not state.get("tracked_masks") or len(state["tracked_masks"]) != n_src:
        if not state.get("confirmed_masks"):
            fg_st, bg_st = auto_seed_roto_strokes(_load_bgr(src_frames[0]))
            pts = sample_roto_anchor_points(fg_st, bg_st, _load_bgr(src_frames[0]), grid_step=32)
            m0 = preview_single_frame_mask(_load_rgb(src_frames[0]), pts)
            m0_path = work / "confirmed_masks" / "conf_00000.png"
            save_mask(m0, m0_path)
            state["confirmed_masks"][0] = str(m0_path)

        confirmed_np = {int(k): load_mask(v) for k, v in state["confirmed_masks"].items()}
        if n_src == 1:
            state["tracked_masks"] = [next(iter(state["confirmed_masks"].values()))]
        else:
            progress(0.08, desc="Đang chạy SAM 2.1 Roto-Brush Tracking cho toàn bộ video...")
            tracked_paths = track_roto_video_masks(
                work / "src_frames",
                work / "tracked_masks",
                confirmed_masks=confirmed_np,
            )
            state["tracked_masks"] = [str(p) for p in tracked_paths]

    mask_paths = [Path(p) for p in state["tracked_masks"]]

    # Extract background video/image frames resized to (w, h)
    progress(0.25, desc="Đang trích xuất & chuẩn hóa khung hình Bối Cảnh Mới...")
    bg_frames, _, _, _, _ = _extract_media_frames(
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

    mcfg = MattingConfig(
        trimap_radius=int(trimap_radius),
        guided_radius=int(guided_radius),
        decontaminate_strength=float(decont_strength),
        despill_green=bool(despill_green),
        erode_dilate=int(erode_dilate),
    )
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
    )

    out_dir = work / "final_composite_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    # Load reference keyframe (first confirmed keyframe) for Universal Pass-2 Reference-Guided Verification
    ref_k = min(state["confirmed_masks"].keys(), key=lambda k: int(k))
    ref_bgr_kf = _load_bgr(src_frames[int(ref_k)])
    ref_mask_kf = load_mask(state["confirmed_masks"][ref_k])

    total_restored = 0
    total_removed = 0
    prev_alpha_stable = None
    prev_src_bgr = None
    for t in range(n_src):
        progress(0.42 + 0.48 * (t / max(n_src, 1)), desc=f"Vòng 1 + Vòng 2 Kiểm Định Pixel & Ghép Khớp Sáng frame {t + 1}/{n_src}...")
        src_bgr = _load_bgr(src_frames[t])
        m_fg = temporal_smooth_mask(mask_paths, t) if n_src > 3 else load_mask(mask_paths[t])
        mres = extract_pixel_accurate_matte(src_bgr, m_fg, mcfg, ref_bgr=ref_bgr_kf, ref_mask=ref_mask_kf)
        alpha_stable = stabilize_alpha_temporal(mres.alpha, prev_alpha_stable, src_bgr, prev_src_bgr)
        prev_alpha_stable = alpha_stable
        prev_src_bgr = src_bgr
        total_restored += mres.missing_restored_px
        total_removed += mres.excess_removed_px

        # Loop or clamp background frame index smoothly
        t_bg = t % n_bg if n_bg > 1 else 0
        bg_bgr = _load_bgr(bg_frames[t_bg])

        if cam_transforms is not None:
            bg_bgr = warp_bg_with_camera_motion(bg_bgr, cam_transforms[t], (w, h))
        elif v2v_transforms is not None:
            bg_bgr, _ = warp_ai_frame(bg_bgr, v2v_transforms[t].to_matrix(), (w, h))

        comp_bgr = composite_with_relighting(
            fg_clean_bgr=mres.fg_clean_bgr,
            alpha=alpha_stable,
            new_bg_bgr=bg_bgr,
            src_full_bgr=src_bgr,
            cfg=rcfg,
        )
        cv2.imwrite(str(out_dir / f"{t:05d}.png"), comp_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    progress(0.93, desc="Đang xuất Video MP4 chất lượng cao (FFmpeg H.264 CRF 16)...")
    audio_src = (
        state["src_media"] if audio_choice == "Giữ âm thanh Video Gốc"
        else (bg_file if audio_choice == "Lấy âm thanh Video Nền Mới" else None)
    )
    out_mp4 = encode_video(out_dir, work / "AGY_Final_Relit_Composite.mp4", fps=state["fps"], audio_source_path=audio_src)
    progress(1.0, desc="Hoàn tất!")
    return (
        str(out_mp4),
        f"Render hoàn tất ({n_src} frames) -> {out_mp4} | [Báo cáo Kiểm Định Vòng 2]: Đã tự động bù đắp +{total_restored:,} px bị thiếu & cắt bỏ -{total_removed:,} px bị thừa!",
    )


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
        # BƯỚC 1: TÁCH CHỦ THỂ THÔNG MINH (ROTO-BRUSH & KIỂM ĐỊNH 2 VÒNG)
        # =========================================================
        with gr.Group(elem_classes="step-card"):
            gr.Markdown(
                "### ① Bước 1: Tách Chủ Thể Thông Minh (Roto-Brush + Kiểm Định Bù Đắp Pixel 2 Vòng)",
                elem_classes="step-title",
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
                    roto_editor = gr.ImageEditor(
                        label="Bôi 1 Nét Cọ Xanh (Kiểu Photoshop — Tự nhận & phủ xanh toàn vật thể) hoặc Cọ Đỏ (Xóa vùng thừa)",
                        type="numpy",
                        image_mode="RGB",
                        height=440,
                        brush=gr.Brush(colors=["#00ff66", "#ff3333"], default_size=18, color_mode="fixed"),
                        eraser=gr.Eraser(default_size=18),
                        transforms=(),
                        layers=False,
                    )
                    with gr.Row():
                        auto_roto_btn = gr.Button(
                            "✨ Tự Động Tách Tay + Sản Phẩm",
                            variant="primary",
                            elem_classes="compact-btn",
                        )
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

                # Cột Phải: Tabs Kết Quả Tách Sub-Pixel + Viền Confirm + Tách Toàn Bộ Video
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
                                "Chạy Tách Sub-Pixel Toàn Bộ Video (SAM 2.1)",
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
            gr.Markdown(
                "### ② Bước 2: Viết Prompt Tạo Video Bối Cảnh Mới Chuẩn Góc Máy & Ánh Sáng (Google Flow)",
                elem_classes="step-title",
            )
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
        # BƯỚC 3: GHÉP VÀO VIDEO NỀN MỚI & TỰ KHỚP ÁNH SÁNG ĐỔ BÓNG 3D
        # =========================================================
        with gr.Group(elem_classes="step-card"):
            gr.Markdown(
                "### ③ Bước 3: Ghép Nền Mới & Tự Khớp Ánh Sáng, Đổ Bóng Mép Viền 3D + Khóa Chuyển Động Camera",
                elem_classes="step-title",
            )
            with gr.Row():
                with gr.Column(scale=4):
                    bg_input = gr.File(
                        label="Nạp Video Bối Cảnh Mới (.mp4) hoặc Ảnh Nền Mới (.jpg/.png)",
                        file_types=["video", "image"],
                    )
                    preview_comp_btn = gr.Button(
                        "👁️ Xem Trước 1 Khung Hình (Before / After)",
                        variant="secondary",
                        elem_classes="compact-btn",
                    )
                    render_final_btn = gr.Button(
                        "🎬 3. Xuất Video Ghép Hoàn Chỉnh (MP4 Full HD)",
                        variant="primary",
                        elem_classes="compact-btn",
                    )

                with gr.Column(scale=4):
                    comp_compare_slider = gr.ImageSlider(
                        label="So Sánh Trực Tiếp: Trái (Gốc) ⟷ Phải (Đã Tách & Khớp Sáng 3D)",
                        type="numpy",
                        height=420,
                    )

                with gr.Column(scale=4):
                    final_video_out = gr.Video(
                        label="Video Thành Phẩm Cuối Cùng",
                        height=420,
                    )

        # =========================================================
        # NGĂN TÙY CHỈNH KỸ THUẬT NÂNG CAO (THU GỌN MẶC ĐỊNH)
        # =========================================================
        with gr.Accordion(
            "⚙️ Tùy chỉnh kỹ thuật nâng cao (Viền Sub-Pixel, Ánh sáng 3D, Tracking & Âm thanh — Đã tự động tối ưu sẵn)",
            open=False,
        ):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**1. Tinh chỉnh Tách Biên Sub-Pixel & Kiểm Định 2 Vòng**")
                    trimap_rad = gr.Slider(1, 10, value=2, step=1, label="Bán kính Sub-Pixel Trimap (px)")
                    guided_rad = gr.Slider(1, 10, value=2, step=1, label="Guided Filter Edge Smooth (px)")
                    erode_dilate = gr.Slider(-6, 6, value=0, step=1, label="Co/Giãn Viền Mask (px)")
                    decont_slider = gr.Slider(0.0, 1.0, value=0.95, step=0.04, label="Khử màu viền nền cũ (Decontaminate)")
                    despill_chk = gr.Checkbox(value=False, label="Ép bật Khử ám xanh (Despill)")

                with gr.Column():
                    gr.Markdown("**2. Ánh Sáng 3D Mép Viền & Bảo Vệ Màn Hình**")
                    ambient_slider = gr.Slider(0.0, 1.0, value=0.68, step=0.04, label="Khớp Nhiệt Độ Màu & Phơi Sáng Nền (Lab)")
                    dir_light_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.04, label="Khớp Hướng Nguồn Sáng 2D")
                    lw_rad_slider = gr.Slider(0, 15, value=4, step=1, label="Bán kính Đổ Bóng & Bức Xạ Viền 3D (px)")
                    lw_str_slider = gr.Slider(0.0, 1.0, value=0.65, step=0.04, label="Cường độ Đổ Bóng Mép Viền 3D")
                    protect_screen_chk = gr.Checkbox(value=True, label="Bảo vệ màn hình phát sáng (Giữ chữ UI nét 100%)")
                    screen_glare_slider = gr.Slider(0.0, 0.40, value=0.06, step=0.02, label="Độ bóng kính màn hình")

                with gr.Column():
                    gr.Markdown("**3. Camera Tracking, Prompt KEEP & Âm Thanh**")
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
                    grain_slider = gr.Slider(0.0, 0.50, value=0.12, step=0.03, label="Khớp hạt nhiễu cảm biến (Grain)")
                    bg_defocus_slider = gr.Slider(0.0, 15.0, value=0.6, step=0.2, label="Mờ nhẹ hậu cảnh kiểu điện thoại bình dân (px)")
                    clean_ghost_chk = gr.Checkbox(value=False, label="Xóa bóng chủ thể AI ở nền (V2V Ghost Inpaint)")
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
        step1_btn.click(
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

        roto_editor.input(
            on_step2_quick_stroke,
            inputs=[roto_editor, frame_slider, trimap_rad, guided_rad, decont_slider, despill_chk, erode_dilate, state],
            outputs=[roto_editor, roto_confirm_img, subpixel_cutout_img, cutout_download, state, status_banner],
        )

        confirm_roto_btn.click(
            on_step2_confirm_roto_brush,
            inputs=[roto_editor, frame_slider, trimap_rad, guided_rad, decont_slider, despill_chk, erode_dilate, state],
            outputs=[roto_editor, roto_confirm_img, subpixel_cutout_img, cutout_download, state, status_banner],
        )

        auto_roto_btn.click(
            on_step2_auto_roto,
            inputs=[frame_slider, trimap_rad, guided_rad, decont_slider, despill_chk, erode_dilate, state],
            outputs=[roto_editor, roto_confirm_img, subpixel_cutout_img, cutout_download, state, status_banner],
        )

        auto_chroma_btn.click(
            on_step2_auto_chroma_key,
            inputs=[frame_slider, trimap_rad, guided_rad, decont_slider, erode_dilate, state],
            outputs=[roto_editor, roto_confirm_img, subpixel_cutout_img, cutout_download, state, status_banner],
        )

        reset_roto_btn.click(
            on_step2_reset_selection,
            inputs=[frame_slider, state],
            outputs=[roto_editor, roto_confirm_img, subpixel_cutout_img, cutout_download, state, status_banner],
        )

        step3_track_btn.click(
            on_step3_track_and_matte_all_frames,
            inputs=[trimap_rad, guided_rad, decont_slider, despill_chk, erode_dilate, state],
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

        preview_comp_btn.click(
            on_step5_preview_composite,
            inputs=[
                frame_slider, bg_input, trimap_rad, guided_rad, decont_slider, despill_chk, erode_dilate,
                ambient_slider, dir_light_slider, lw_rad_slider, lw_str_slider, protect_screen_chk,
                screen_glare_slider, grain_slider, bg_defocus_slider, clean_ghost_chk, state,
            ],
            outputs=[comp_compare_slider, status_banner],
        )

        render_final_btn.click(
            on_step5_render_final_video,
            inputs=[
                bg_input, tracking_mode, audio_choice, trimap_rad, guided_rad, decont_slider, despill_chk, erode_dilate,
                ambient_slider, dir_light_slider, lw_rad_slider, lw_str_slider, protect_screen_chk,
                screen_glare_slider, grain_slider, bg_defocus_slider, clean_ghost_chk, state,
            ],
            outputs=[final_video_out, status_banner],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from download_checkpoints import ensure_checkpoint
from src.video_io import save_mask

# (video_type, obj_id, frame_idx) -> list of (x, y, label) where label=1 (pos), 0 (neg)
PointList = List[Tuple[int, int, int]]
PointsDict = Dict[Tuple[str, int, int], PointList]

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def _fill_mask_holes(mask: np.ndarray, max_hole_ratio: float = 0.0012) -> np.ndarray:
    """
    Fill only tiny enclosed interior pinholes (e.g. text/icons inside screen) WITHOUT
    filling real finger-to-device gaps or expanding concave outer contours.
    """
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape[:2]
    max_hole_area = int(h * w * max_hole_ratio)

    inv = (1 - m).astype(np.uint8)
    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for cid in range(1, num_cc):
        x0 = stats[cid, cv2.CC_STAT_LEFT]
        y0 = stats[cid, cv2.CC_STAT_TOP]
        cw = stats[cid, cv2.CC_STAT_WIDTH]
        ch = stats[cid, cv2.CC_STAT_HEIGHT]
        area = stats[cid, cv2.CC_STAT_AREA]
        touches_border = (x0 <= 0) or (y0 <= 0) or (x0 + cw >= w) or (y0 + ch >= h)
        if not touches_border and area <= max_hole_area:
            m[labels == cid] = 1
    return m


def temporal_smooth_mask(mask_paths: List[Path], t: int, radius: int = 2, sigma: float = 1.2) -> np.ndarray:
    """Remove per-frame SAM boundary flicker at FULL native resolution:
    Gaussian-weighted mask average over frames t-radius..t+radius followed by sub-pixel anti-alias blur."""
    from src.video_io import load_mask

    lo, hi = max(0, t - radius), min(len(mask_paths) - 1, t + radius)
    acc, wsum = None, 0.0
    for i in range(lo, hi + 1):
        m = load_mask(mask_paths[i]).astype(np.float32)
        wgt = float(np.exp(-0.5 * ((i - t) / sigma) ** 2))
        acc = m * wgt if acc is None else acc + m * wgt
        wsum += wgt
    avg = acc / max(wsum, 1e-6)
    avg_aa = cv2.GaussianBlur(avg, (0, 0), sigmaX=0.75)
    return (avg_aa > 0.5).astype(np.uint8)


def clear_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def roto_snap_color_boundary(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    pos_pts: List[Tuple[int, int]],
    neg_pts: List[Tuple[int, int]],
    band_px: int = 9,
) -> np.ndarray:
    """
    After Effects Roto-Brush style Spatially-Local Color Edge Snap:
    Compares each pixel in the outer boundary band of `mask` against its IMMEDIATELY
    adjacent local interior foreground Lab color (`fg_est`) vs. its local exterior
    background Lab color (`bg_est`). Trims any background rim/pocket (like gray asphalt
    beside a warm skin hand) without cross-talking between distant objects.
    """
    m = (mask > 0).astype(np.uint8)
    if not np.any(m > 0) or not pos_pts:
        return m

    h, w = m.shape[:2]
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    k_band = band_px * 2 + 1
    ker_band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_band, k_band))
    inner_core = cv2.erode(m, ker_band, iterations=1) > 0
    bg_ext = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1) == 0

    if not np.any(inner_core) or not np.any(bg_ext):
        return m

    # Propagate local interior foreground Lab and local exterior background Lab into the boundary band
    sigma = float(max(6.0, band_px * 1.35))
    fg_num = cv2.GaussianBlur(lab * inner_core[..., None].astype(np.float32), (0, 0), sigmaX=sigma)
    fg_den = cv2.GaussianBlur(inner_core.astype(np.float32), (0, 0), sigmaX=sigma)[..., None]
    fg_est = fg_num / np.maximum(fg_den, 1e-4)

    bg_num = cv2.GaussianBlur(lab * bg_ext[..., None].astype(np.float32), (0, 0), sigmaX=sigma)
    bg_den = cv2.GaussianBlur(bg_ext.astype(np.float32), (0, 0), sigmaX=sigma)[..., None]
    bg_est = bg_num / np.maximum(bg_den, 1e-4)

    outer_band = (m > 0) & (~inner_core)
    # Protect direct positive anchor points (radius 6px)
    prot = np.zeros((h, w), dtype=np.uint8)
    for px, py in pos_pts:
        cv2.circle(prot, (px, py), 6, 1, -1)
    outer_band &= (prot == 0)

    # Protect dark/blue electronic device interior from color-snap erosion so device bezels stay razor-straight
    r_ch = frame_rgb[..., 0].astype(np.float32)
    g_ch = frame_rgb[..., 1].astype(np.float32)
    b_ch = frame_rgb[..., 2].astype(np.float32)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    is_device_pixel = (hsv[..., 2] < 70) | ((b_ch > r_ch + 18) & (hsv[..., 1] > 45))
    outer_band &= (~is_device_pixel)

    # Weight chromaticity (a, b) 2.2x higher than luminance (L) so 3D shading on rounded hands/fingers
    # is never confused with gray background
    w_lab = np.array([0.4, 1.2, 1.2], dtype=np.float32)
    d_bg = np.linalg.norm((lab - bg_est) * w_lab, axis=-1)
    d_fg = np.linalg.norm((lab - fg_est) * w_lab, axis=-1)

    # Trim outer band pixels that are much closer in chromatic Lab color to local exterior background than to local interior foreground
    is_bg_leak = outer_band & (d_bg < 11.0) & (d_fg > d_bg * 1.30 + 3.5)
    m_clean = m.copy()
    m_clean[is_bg_leak] = 0

    # Keep only foreground components that still contain at least one positive point
    n_cc, l_cc, _, _ = cv2.connectedComponentsWithStats(m_clean, connectivity=8)
    valid_mask = np.zeros_like(m_clean)
    for px, py in pos_pts:
        if 0 <= py < h and 0 <= px < w and l_cc[py, px] > 0:
            valid_mask[l_cc == l_cc[py, px]] = 1
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    valid_mask = _fill_mask_holes(valid_mask)

    # Punch out any enclosed green-screen finger gaps AFTER pinhole filling so finger gaps stay transparent
    green_excess = g_ch - np.maximum(r_ch, b_ch)
    is_green_screen_gap = (green_excess > 16.0) & (hsv[..., 0] >= 32) & (hsv[..., 0] <= 94) & (hsv[..., 1] >= 38)
    if np.sum(is_green_screen_gap) > int(h * w * 0.04):
        valid_mask[is_green_screen_gap] = 0
    return valid_mask


def preview_single_frame_mask(
    frame_rgb: np.ndarray,
    points: PointList,
    checkpoints_dir: str | Path = DEFAULT_CKPT_DIR,
) -> np.ndarray:
    """
    Run SAM 2.1 image predictor on a single RGB frame with Roto-Brush multi-part
    cluster decomposition and Color Edge Snapping.
    """
    if not points:
        return np.zeros(frame_rgb.shape[:2], dtype=np.uint8)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt_path = ensure_checkpoint(checkpoints_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(SAM2_CONFIG, str(ckpt_path), device=device)
    predictor = SAM2ImagePredictor(model)

    pos_pts = [(int(p[0]), int(p[1])) for p in points if p[2] == 1]
    neg_pts = [(int(p[0]), int(p[1])) for p in points if p[2] == 0]
    h, w = frame_rgb.shape[:2]

    if not pos_pts:
        return np.zeros((h, w), dtype=np.uint8)

    # Group positive points into spatial-color clusters so separate parts (Left Hand, Device, Right Hand)
    # are each queried cleanly without SAM bridging background gaps between them.
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    clusters: List[List[Tuple[int, int]]] = []
    visited = [False] * len(pos_pts)
    for i in range(len(pos_pts)):
        if visited[i]:
            continue
        cluster = [pos_pts[i]]
        visited[i] = True
        queue = [i]
        while queue:
            curr = queue.pop(0)
            cx, cy = pos_pts[curr]
            clab = lab[min(h - 1, max(0, cy)), min(w - 1, max(0, cx))]
            for j in range(len(pos_pts)):
                if not visited[j]:
                    jx, jy = pos_pts[j]
                    jlab = lab[min(h - 1, max(0, jy)), min(w - 1, max(0, jx))]
                    dist_xy = np.hypot(cx - jx, cy - jy)
                    dist_col = float(np.linalg.norm(clab - jlab))
                    if dist_xy <= 58.0 and dist_col <= 42.0:
                        visited[j] = True
                        cluster.append(pos_pts[j])
                        queue.append(j)
        clusters.append(cluster)

    px_all = [p[0] for p in pos_pts]
    py_all = [p[1] for p in pos_pts]
    margin_x = max(55, int((max(px_all) - min(px_all) + 1) * 0.35))
    margin_y = max(55, int((max(py_all) - min(py_all) + 1) * 0.28))
    env_x0 = max(0, min(px_all) - margin_x)
    env_y0 = max(0, min(py_all) - margin_y)
    env_x1 = min(w, max(px_all) + margin_x)
    env_y1 = min(h, max(py_all) + margin_y)

    try:
        with torch.inference_mode():
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else _null_ctx()
            with ctx:
                predictor.set_image(frame_rgb)
                combined = np.zeros((h, w), dtype=np.uint8)

                for cluster in clusters:
                    sub_coords = [[cx, cy] for (cx, cy) in cluster] + [[nx, ny] for (nx, ny) in neg_pts]
                    sub_labels = [1] * len(cluster) + [0] * len(neg_pts)
                    sub_m, sub_s, _ = predictor.predict(
                        point_coords=np.array(sub_coords, dtype=np.float32),
                        point_labels=np.array(sub_labels, dtype=np.int32),
                        multimask_output=True,
                    )

                    best_sub_idx = 0
                    best_sub_val = -1e9
                    for s_idx in range(len(sub_m)):
                        mc = sub_m[s_idx] > 0.0
                        pos_cov = sum(1 for (cx, cy) in cluster if 0 <= cy < h and 0 <= cx < w and mc[cy, cx]) / max(len(cluster), 1)
                        nhit = sum(1 for (nx, ny) in neg_pts if 0 <= ny < h and 0 <= nx < w and mc[ny, nx]) / max(len(neg_pts), 1)
                        outside_pixels = float(np.sum(mc) - np.sum(mc[env_y0:env_y1, env_x0:env_x1]))
                        spill_pen = (outside_pixels / max(float(np.sum(mc)), 1.0)) * 3.5
                        area_ratio = float(np.mean(mc))
                        area_pen = 2.5 if area_ratio > 0.42 else (0.12 * s_idx)
                        sval = float(sub_s[s_idx]) + 1.4 * pos_cov - 2.8 * nhit - spill_pen - area_pen
                        if sval > best_sub_val:
                            best_sub_val = sval
                            best_sub_idx = s_idx

                    part_mask = (sub_m[best_sub_idx] > 0.0).astype(np.uint8)
                    n_cc, l_cc = cv2.connectedComponents(part_mask, connectivity=8)
                    keep_part = np.zeros_like(part_mask)
                    for cx, cy in cluster:
                        if 0 <= cy < h and 0 <= cx < w and l_cc[cy, cx] > 0:
                            keep_part[l_cc == l_cc[cy, cx]] = 1

                    # If this cluster is the rigid electronic device (dark bezel or blue screen),
                    # solidify concave bezel notches via convex hull so all 4 device edges stay straight
                    dev_hits = 0
                    for cx, cy in cluster:
                        r_p, g_p, b_p = int(frame_rgb[cy, cx, 0]), int(frame_rgb[cy, cx, 1]), int(frame_rgb[cy, cx, 2])
                        if (max(r_p, g_p, b_p) < 80) or (b_p > r_p + 20):
                            dev_hits += 1
                    if dev_hits > len(cluster) * 0.5 and np.any(keep_part > 0):
                        cnts, _ = cv2.findContours(keep_part * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            largest_c = max(cnts, key=cv2.contourArea)
                            hull = cv2.convexHull(largest_c)
                            hull_mask = np.zeros_like(keep_part)
                            cv2.drawContours(hull_mask, [hull], -1, 1, thickness=cv2.FILLED)
                            # Only accept hull expansion if it doesn't swallow negative points
                            neg_swallowed = sum(1 for (nx, ny) in neg_pts if 0 <= ny < h and 0 <= nx < w and hull_mask[ny, nx] > 0)
                            if neg_swallowed == 0:
                                keep_part = hull_mask

                    combined |= keep_part

                env_mask = np.zeros_like(combined)
                env_mask[env_y0:env_y1, env_x0:env_x1] = 1
                combined &= env_mask

                for nx, ny in neg_pts:
                    if 0 <= ny < h and 0 <= nx < w:
                        cv2.circle(combined, (nx, ny), 6, 0, -1)

        snapped = roto_snap_color_boundary(frame_rgb, combined, pos_pts, neg_pts, band_px=10)
        return snapped
    finally:
        del predictor, model
        clear_vram()


def track_video_masks(
    frames_dir: str | Path,
    out_masks_dir: str | Path,
    video_type: str,
    points_dict: PointsDict,
    obj_ids: Optional[List[int]] = None,
    checkpoints_dir: str | Path = "checkpoints",
    progress_cb=None,
) -> Dict[int, List[Path]]:
    """
    Track multi-frame masks using SAM2VideoPredictor with CPU offloading
    and bidirectional propagation for multi-frame correction.
    Returns {obj_id: [mask_path_0, mask_path_1, ...]}.
    """
    from sam2.build_sam import build_sam2_video_predictor

    if obj_ids is None:
        obj_ids = sorted({oid for (vt, oid, _) in points_dict.keys() if vt == video_type and points_dict[(vt, oid, _)]})

    out_root = Path(out_masks_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    num_frames = len(frame_files)
    if num_frames == 0 or not obj_ids:
        return {}

    ckpt_path = ensure_checkpoint(checkpoints_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(ckpt_path), device=device)

    # Prepare output directories
    for oid in obj_ids:
        (out_root / f"obj_{oid}").mkdir(parents=True, exist_ok=True)

    # Read frame size for empty fallback
    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]

    video_segments: Dict[int, Dict[int, np.ndarray]] = {f_idx: {} for f_idx in range(num_frames)}

    try:
        with torch.inference_mode():
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else _null_ctx()
            with ctx:
                state = predictor.init_state(
                    video_path=str(frames_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                )

                annotated_frames = []
                for (vt, oid, f_idx), pts in points_dict.items():
                    if vt != video_type or oid not in obj_ids or not pts:
                        continue
                    coords = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
                    labels = np.array([p[2] for p in pts], dtype=np.int32)
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=int(f_idx),
                        obj_id=int(oid),
                        points=coords,
                        labels=labels,
                    )
                    annotated_frames.append(int(f_idx))

                if not annotated_frames:
                    return {}

                min_f = min(annotated_frames)
                max_f = max(annotated_frames)

                # Forward propagation from earliest annotated frame
                for out_f, out_oids, out_logits in predictor.propagate_in_video(
                    state, start_frame_idx=min_f, reverse=False
                ):
                    for i, oid in enumerate(out_oids):
                        video_segments[out_f][int(oid)] = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                    if progress_cb:
                        progress_cb(f"[{video_type}] Forward tracking frame {out_f + 1}/{num_frames}")

                # Backward propagation from latest annotated frame if needed
                if max_f > 0:
                    for out_f, out_oids, out_logits in predictor.propagate_in_video(
                        state, start_frame_idx=max_f, reverse=True
                    ):
                        for i, oid in enumerate(out_oids):
                            # Fill missing frames or refine frames before min_f
                            if int(oid) not in video_segments[out_f] or out_f < min_f:
                                video_segments[out_f][int(oid)] = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                        if progress_cb:
                            progress_cb(f"[{video_type}] Backward tracking frame {out_f + 1}/{num_frames}")

                del state
    finally:
        del predictor
        clear_vram()

    result: Dict[int, List[Path]] = {oid: [] for oid in obj_ids}
    for f_idx in range(num_frames):
        for oid in obj_ids:
            mask = video_segments[f_idx].get(oid)
            if mask is None or mask.shape != (h, w):
                mask = np.zeros((h, w), dtype=np.uint8)
            elif oid == 1:
                mask = _fill_mask_holes(mask)
            mp = out_root / f"obj_{oid}" / f"{f_idx:05d}.png"
            save_mask(mask, mp)
            result[oid].append(mp)

    return result


def track_roto_video_masks(
    frames_dir: str | Path,
    out_masks_dir: str | Path,
    confirmed_masks: Dict[int, np.ndarray],
    points_dict: Optional[PointsDict] = None,
    checkpoints_dir: str | Path = DEFAULT_CKPT_DIR,
    progress_cb=None,
) -> List[Path]:
    """
    Track confirmed Roto-Brush masks across all video frames using SAM2VideoPredictor.add_new_mask.
    Splits disconnected foreground components into sub-object IDs so multi-part subjects
    (e.g. Left Hand + Right Hand + Product Device) are tracked simultaneously without dropout,
    then unions and fills holes per frame.
    """
    from sam2.build_sam import build_sam2_video_predictor

    out_root = Path(out_masks_dir) / "roto_fg"
    out_root.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    num_frames = len(frame_files)
    if num_frames == 0 or not confirmed_masks:
        return []

    ckpt_path = ensure_checkpoint(checkpoints_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(ckpt_path), device=device)

    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]
    video_segments: Dict[int, np.ndarray] = {
        f_idx: np.zeros((h, w), dtype=np.uint8) for f_idx in range(num_frames)
    }

    try:
        with torch.inference_mode():
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else _null_ctx()
            with ctx:
                state = predictor.init_state(
                    video_path=str(frames_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                )

                annotated_frames = sorted(confirmed_masks.keys())
                for f_idx in annotated_frames:
                    m = (confirmed_masks[f_idx] > 0).astype(np.uint8)
                    if m.shape[:2] != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    # Add full unified mask as obj_id=1
                    predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=int(f_idx),
                        obj_id=1,
                        mask=m > 0,
                    )
                    # Also register large connected components (up to 3) so separate hands/device never drop
                    num_cc, labels_cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                    comp_ids = [
                        cid for cid in range(1, num_cc)
                        if stats[cid, cv2.CC_STAT_AREA] > (h * w * 0.008)
                    ]
                    comp_ids = sorted(comp_ids, key=lambda c: stats[c, cv2.CC_STAT_AREA], reverse=True)[:3]
                    if len(comp_ids) > 1:
                        for idx_c, cid in enumerate(comp_ids):
                            predictor.add_new_mask(
                                inference_state=state,
                                frame_idx=int(f_idx),
                                obj_id=10 + idx_c,
                                mask=(labels_cc == cid),
                            )

                min_f = min(annotated_frames)
                max_f = max(annotated_frames)

                for out_f, out_oids, out_logits in predictor.propagate_in_video(
                    state, start_frame_idx=min_f, reverse=False
                ):
                    union_m = np.zeros((h, w), dtype=np.uint8)
                    for i in range(len(out_oids)):
                        union_m |= (out_logits[i] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
                    video_segments[out_f] = _fill_mask_holes(union_m)
                    if progress_cb:
                        progress_cb(out_f, num_frames)

                if max_f > 0 and min_f > 0:
                    for out_f, out_oids, out_logits in predictor.propagate_in_video(
                        state, start_frame_idx=max_f, reverse=True
                    ):
                        if out_f < min_f:
                            union_m = np.zeros((h, w), dtype=np.uint8)
                            for i in range(len(out_oids)):
                                union_m |= (out_logits[i] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
                            video_segments[out_f] = _fill_mask_holes(union_m)

                del state
    finally:
        del predictor
        clear_vram()

    out_paths: List[Path] = []
    for f_idx in range(num_frames):
        mp = out_root / f"{f_idx:05d}.png"
        save_mask(video_segments[f_idx], mp)
        out_paths.append(mp)
    return out_paths


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False

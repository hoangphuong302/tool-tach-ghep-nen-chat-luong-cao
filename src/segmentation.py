from __future__ import annotations

import gc
import hashlib
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from download_checkpoints import ensure_best_sam2_checkpoint, ensure_checkpoint
from src.video_io import save_mask

# (video_type, obj_id, frame_idx) -> list of (x, y, label) where label=1 (pos), 0 (neg)
PointList = List[Tuple[int, int, int]]
PointsDict = Dict[Tuple[str, int, int], PointList]

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

_SAM2_IMG_CACHE: Dict[str, object] = {}


def release_sam2_cache() -> None:
    """Drop cached SAM 2.1 image predictor and free GPU VRAM."""
    if _SAM2_IMG_CACHE:
        _SAM2_IMG_CACHE.clear()
        clear_vram()


def _frame_fingerprint(frame_rgb: np.ndarray) -> tuple:
    sample = np.ascontiguousarray(frame_rgb[::8, ::8])
    digest = hashlib.blake2b(sample.tobytes(), digest_size=16).hexdigest()
    return (frame_rgb.shape, str(frame_rgb.dtype), digest)


class _LazySamFrameSequence:
    """Read normalized SAM frames on demand and retain only a small CPU cache."""

    def __init__(self, frame_files: List[Path], image_size: int, cache_size: int = 2):
        from sam2.utils.misc import _load_img_as_tensor

        self.frame_files = frame_files
        self.image_size = int(image_size)
        self.cache_size = max(1, int(cache_size))
        self._load_img_as_tensor = _load_img_as_tensor
        self._img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        self._img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.frame_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self.frame_files):
            raise IndexError(index)
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        raw, _, _ = self._load_img_as_tensor(str(self.frame_files[index]), self.image_size)
        image = torch.empty((3, self.image_size, self.image_size), dtype=torch.float32)
        image.copy_(raw)
        image.sub_(self._img_mean).div_(self._img_std)
        self._cache[index] = image
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return image


def _init_sam2_video_state_bounded(predictor, frame_files: List[Path], scratch_dir: Path):
    """Use official state initialization, then replace eager frames with a bounded sequence."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sam2_init_", dir=str(scratch_dir)) as temp_dir:
        first_frame = Path(temp_dir) / "00000.jpg"
        shutil.copy2(frame_files[0], first_frame)
        state = predictor.init_state(
            video_path=temp_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

    if not {"images", "num_frames", "video_height", "video_width", "cached_features"}.issubset(state):
        raise RuntimeError("Unsupported SAM2 inference-state layout for bounded frame loading")

    state["images"] = _LazySamFrameSequence(frame_files, predictor.image_size)
    state["num_frames"] = len(frame_files)
    return state


def _fill_mask_holes(
    mask: np.ndarray,
    max_hole_ratio: float = 0.0012,
    bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fill only tiny enclosed interior pinholes (e.g. text/icons inside screen) WITHOUT
    filling real exterior background gaps (e.g. finger-to-device or arm-to-book slits).
    """
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape[:2]
    max_hole_area = int(h * w * max_hole_ratio)

    inv = (1 - m).astype(np.uint8)
    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    use_color_guard = bgr is not None and bgr.shape[:2] == (h, w) and num_cc > 1
    if use_color_guard:
        assert bgr is not None
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        b_ch, g_ch, r_ch = (bgr[..., 0].astype(np.float32), bgr[..., 1].astype(np.float32), bgr[..., 2].astype(np.float32))
        is_green_soft = ((g_ch - np.maximum(r_ch, b_ch)) > 8.0) & (hsv[..., 0] >= 28) & (hsv[..., 0] <= 104) & (hsv[..., 1] >= 18)
        bg_ring = (cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0) & (m == 0)
        bg_zone = bg_ring if int(np.sum(bg_ring)) >= 40 else (m == 0)
        if np.any(bg_zone):
            bg_vals = lab[bg_zone]
            bg_q = np.percentile(bg_vals, [20, 50, 80], axis=0) if len(bg_vals) >= 6 else np.median(bg_vals, axis=0)[None, :]
            ext_is_green = float(np.mean(is_green_soft[bg_zone])) >= 0.15
        else:
            bg_q = None
            ext_is_green = False
        w_lab = np.array([0.65, 1.15, 1.15], dtype=np.float32)
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    else:
        lab = None
        is_green_soft = None
        bg_q = None
        ext_is_green = False
        w_lab = None
        k5 = None

    for cid in range(1, num_cc):
        x0 = stats[cid, cv2.CC_STAT_LEFT]
        y0 = stats[cid, cv2.CC_STAT_TOP]
        cw = stats[cid, cv2.CC_STAT_WIDTH]
        ch = stats[cid, cv2.CC_STAT_HEIGHT]
        area = stats[cid, cv2.CC_STAT_AREA]
        touches_border = (x0 <= 0) or (y0 <= 0) or (x0 + cw >= w) or (y0 + ch >= h)
        if touches_border or area > max_hole_area:
            continue
        if use_color_guard and bg_q is not None:
            comp = labels == cid
            if ext_is_green and float(np.mean(is_green_soft[comp])) > 0.30:
                continue
            hole_med = np.median(lab[comp], axis=0)
            d_bg = float(np.min(np.linalg.norm((bg_q - hole_med[None, :]) * w_lab[None, :], axis=1)))
            fg_ring = (cv2.dilate(comp.astype(np.uint8), k5) > 0) & (m > 0)
            if np.any(fg_ring):
                fg_med = np.median(lab[fg_ring], axis=0)
                d_fg = float(np.linalg.norm((fg_med - hole_med) * w_lab))
            else:
                d_fg = 1e9
            if d_bg < d_fg * 0.95 or d_bg < 22.0:
                continue
            m[comp] = 1
        else:
            m[labels == cid] = 1
    return m


def temporal_smooth_mask(
    mask_paths: List[Path],
    t: int,
    radius: int = 2,
    sigma: float = 1.2,
    *,
    frame_paths: Optional[List[Path]] = None,
    current_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vote on one-pixel mask disagreements along stationary source edges."""
    from src.video_io import load_mask

    current = load_mask(mask_paths[t])
    if frame_paths is not None and current_bgr is None:
        current_bgr = cv2.imread(str(frame_paths[t]))

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current_edge = cv2.dilate(current, kernel) != cv2.erode(current, kernel)
    total_weight = 1.0
    contrary_weight = np.zeros(current.shape, dtype=np.float32)
    lo, hi = max(0, t - radius), min(len(mask_paths) - 1, t + radius)
    for i in range(lo, hi + 1):
        if i == t:
            continue
        neighbor = load_mask(mask_paths[i])
        nearby_edge = cv2.dilate(neighbor, kernel) != cv2.erode(neighbor, kernel)
        disagreement = current_edge & nearby_edge & (neighbor != current)
        ys, xs = np.nonzero(disagreement)
        weight = float(np.exp(-0.5 * ((i - t) / max(sigma, 1e-6)) ** 2))
        total_weight += weight
        if frame_paths is not None:
            neighbor_bgr = cv2.imread(str(frame_paths[i]))
            if neighbor_bgr is None or current_bgr is None or neighbor_bgr.shape != current_bgr.shape:
                continue
            if not ys.size:
                continue
            change = np.max(cv2.absdiff(current_bgr[ys, xs], neighbor_bgr[ys, xs]), axis=1).ravel()
            weight = weight * (1.0 - np.clip((change.astype(np.float32) - 3.0) / 9.0, 0.0, 1.0))

        if ys.size:
            contrary_weight[ys, xs] += weight
    return np.where(contrary_weight <= total_weight * 0.5, current, 1 - current).astype(np.uint8)


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
    valid_mask = _fill_mask_holes(valid_mask, bgr=cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    # Punch out any enclosed green-screen finger gaps AFTER pinhole filling so finger gaps stay transparent
    green_excess = g_ch - np.maximum(r_ch, b_ch)
    is_green_screen_gap = (green_excess > 16.0) & (hsv[..., 0] >= 32) & (hsv[..., 0] <= 94) & (hsv[..., 1] >= 38)
    if np.sum(is_green_screen_gap) > int(h * w * 0.04):
        valid_mask[is_green_screen_gap] = 0
    return valid_mask


def _get_cached_sam2_predictor(
    frame_rgb: np.ndarray,
    checkpoints_dir: str | Path = DEFAULT_CKPT_DIR,
    *,
    use_cache: bool = True,
    prefer_best_checkpoint: bool = False,
):
    """Load or reuse cached SAM 2.1 predictor and embedding state for `frame_rgb`."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if prefer_best_checkpoint:
        ckpt_path, config_name = ensure_best_sam2_checkpoint(checkpoints_dir)
    else:
        ckpt_path, config_name = ensure_checkpoint(checkpoints_dir), SAM2_CONFIG

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = (id(build_sam2), id(SAM2ImagePredictor), str(ckpt_path), config_name, device)
    frame_sig = _frame_fingerprint(frame_rgb)

    predictor = None
    model = None
    same_frame = False
    if use_cache and _SAM2_IMG_CACHE.get("key") == cache_key:
        model = _SAM2_IMG_CACHE.get("model")
        predictor = _SAM2_IMG_CACHE.get("predictor")
        same_frame = _SAM2_IMG_CACHE.get("frame_sig") == frame_sig

    if predictor is None:
        if use_cache:
            try:
                from src.neural_matting import release_neural_models

                release_neural_models()
            except Exception:
                pass
        model = build_sam2(config_name, str(ckpt_path), device=device)
        predictor = SAM2ImagePredictor(model)
        if use_cache:
            _SAM2_IMG_CACHE["key"] = cache_key
            _SAM2_IMG_CACHE["model"] = model
            _SAM2_IMG_CACHE["predictor"] = predictor
            _SAM2_IMG_CACHE["frame_sig"] = None

    return model, predictor, device, same_frame, frame_sig


def _cluster_positive_points(
    pos_pts: List[Tuple[int, int]],
    lab: np.ndarray,
    dist_thresh: float = 58.0,
    col_thresh: float = 42.0,
) -> List[List[Tuple[int, int]]]:
    h, w = lab.shape[:2]
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
                    if np.hypot(cx - jx, cy - jy) <= dist_thresh and float(np.linalg.norm(clab - jlab)) <= col_thresh:
                        visited[j] = True
                        cluster.append(pos_pts[j])
                        queue.append(j)
        clusters.append(cluster)
    return clusters


def preview_single_frame_mask(
    frame_rgb: np.ndarray,
    points: PointList,
    checkpoints_dir: str | Path = DEFAULT_CKPT_DIR,
    *,
    manual_prompts: bool = False,
    snap_boundary: bool = True,
    use_cache: bool = True,
    prefer_best_checkpoint: bool = False,
) -> np.ndarray:
    """
    Run SAM 2.1 image predictor on a single RGB frame with Roto-Brush multi-part
    cluster decomposition and optional Color Edge Snapping.
    """
    if not points:
        return np.zeros(frame_rgb.shape[:2], dtype=np.uint8)

    model, predictor, device, same_frame, frame_sig = _get_cached_sam2_predictor(
        frame_rgb,
        checkpoints_dir,
        use_cache=use_cache,
        prefer_best_checkpoint=prefer_best_checkpoint,
    )

    pos_pts = [(int(p[0]), int(p[1])) for p in points if p[2] == 1]
    neg_pts = [(int(p[0]), int(p[1])) for p in points if p[2] == 0]
    h, w = frame_rgb.shape[:2]

    if not pos_pts:
        return np.zeros((h, w), dtype=np.uint8)

    # Group positive points into spatial-color clusters so separate parts (Left Hand, Device, Right Hand)
    # are each queried cleanly without SAM bridging background gaps between them.
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    clusters = _cluster_positive_points(pos_pts, lab)

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
                if not same_frame:
                    predictor.set_image(frame_rgb)
                    if use_cache:
                        _SAM2_IMG_CACHE["frame_sig"] = frame_sig
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
                        spill_pen = 0.0 if manual_prompts else (outside_pixels / max(float(np.sum(mc)), 1.0)) * 3.5
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

                    combined |= keep_part

                env_mask = np.zeros_like(combined)
                env_mask[env_y0:env_y1, env_x0:env_x1] = 1
                if not manual_prompts:
                    combined &= env_mask

                for nx, ny in neg_pts:
                    if 0 <= ny < h and 0 <= nx < w:
                        cv2.circle(combined, (nx, ny), 6, 0, -1)

        if snap_boundary:
            result = roto_snap_color_boundary(frame_rgb, combined, pos_pts, neg_pts, band_px=10)
        else:
            result = combined
        if manual_prompts:
            # Hole filling must not undo an explicit background brush point.
            for nx, ny in neg_pts:
                if 0 <= ny < h and 0 <= nx < w:
                    cv2.circle(result, (nx, ny), 6, 0, -1)
        return result
    finally:
        if not use_cache:
            del predictor, model
            clear_vram()


def refine_mask_with_sam2_negative(
    frame_rgb: np.ndarray,
    prev_mask: np.ndarray,
    fg_stroke: Optional[np.ndarray],
    bg_delta: np.ndarray,
    checkpoints_dir: str | Path = DEFAULT_CKPT_DIR,
    *,
    prefer_best_checkpoint: bool = True,
) -> np.ndarray:
    """
    Smart Red Brush subtraction (Photoshop '-' / AE Alt-Roto-Brush):
    Segments the unwanted region clicked by `bg_delta` via SAM 2.1 (using distant
    safe foreground points from `prev_mask` / `fg_stroke` as negative anchors) and
    subtracts only the targeted unwanted region from `prev_mask`.
    """
    m_prev = (prev_mask > 0).astype(np.uint8)
    if not np.any(m_prev) or not np.any(bg_delta > 0):
        return m_prev

    from src.alpha_matting import sample_roto_anchor_points

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    empty = np.zeros_like(m_prev)
    bg_pts_raw = sample_roto_anchor_points(
        bg_delta.astype(np.uint8), empty, bgr, grid_step=24, preserve_stroke_components=True
    )
    pos_rm = [(x, y, 1) for (x, y, lbl) in bg_pts_raw if lbl == 1]
    if not pos_rm:
        return m_prev

    # Build negative anchors (pixels of prev_mask that must stay kept) away from bg_delta
    rm_guard = cv2.dilate(
        (bg_delta > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33)),
    ) > 0
    keep_zone = (m_prev > 0) & (~rm_guard)
    neg_keep: PointList = []
    if fg_stroke is not None and np.any((fg_stroke > 0) & (~rm_guard)):
        fg_pts_raw = sample_roto_anchor_points(
            ((fg_stroke > 0) & (~rm_guard)).astype(np.uint8),
            empty,
            bgr,
            grid_step=40,
            preserve_stroke_components=True,
        )
        neg_keep.extend([(x, y, 0) for (x, y, lbl) in fg_pts_raw if lbl == 1][:8])
    if len(neg_keep) < 4 and np.any(keep_zone):
        ys, xs = np.where(keep_zone)
        step = max(1, len(xs) // 6)
        for idx in range(0, len(xs), step):
            neg_keep.append((int(xs[idx]), int(ys[idx]), 0))
            if len(neg_keep) >= 8:
                break

    rm_mask = preview_single_frame_mask(
        frame_rgb,
        pos_rm + neg_keep,
        checkpoints_dir=checkpoints_dir,
        manual_prompts=True,
        snap_boundary=False,
        use_cache=True,
        prefer_best_checkpoint=prefer_best_checkpoint,
    )
    # Keep only connected components of rm_mask that actually intersect bg_delta
    if np.any(rm_mask > 0):
        n_cc, l_cc = cv2.connectedComponents((rm_mask > 0).astype(np.uint8), connectivity=8)
        hit_ids = set(np.unique(l_cc[bg_delta > 0]).tolist()) - {0}
        targeted_rm = np.isin(l_cc, list(hit_ids)).astype(np.uint8) if hit_ids else np.zeros_like(m_prev)
    else:
        targeted_rm = np.zeros_like(m_prev)

    # Guard against over-aggressive subtraction wiping >65% of prev_mask in one tiny click
    if np.sum(targeted_rm & m_prev) > 0.65 * max(1, np.sum(m_prev)):
        targeted_rm = np.zeros_like(m_prev)

    out = m_prev.copy()
    out[targeted_rm > 0] = 0
    out[bg_delta > 0] = 0
    return out


def _detect_hands_device_priors(frame_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (lab_f32, edge_mag_01, skin_bool, device_bool) for material-aware Roto-Brush scoring."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    b_ch = bgr[..., 0].astype(np.float32)
    g_ch = bgr[..., 1].astype(np.float32)
    r_ch = bgr[..., 2].astype(np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    cr, cb = ycc[..., 1], ycc[..., 2]

    skin = (cr >= 133) & (cr <= 178) & (cb >= 77) & (cb <= 129) & ((hue <= 18) | (hue >= 162)) & ((r_ch - g_ch) >= 10.0)
    is_green = (g_ch > np.maximum(r_ch, b_ch) + 12.0) & (hue >= 28) & (hue <= 98) & (sat >= 28)
    device = (~is_green) & (
        (val < 70)
        | ((b_ch > r_ch + 18.0) & (sat > 45))
        | ((sat <= 55) & (val >= 135) & (np.abs(r_ch - b_ch) <= 28.0))
    )

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    edge_mag = np.clip(mag / max(float(np.percentile(mag, 98)), 0.08), 0.0, 1.0)
    return lab, edge_mag, skin, device


def _sample_ring_negatives(
    stroke_mask: np.ndarray,
    lab: np.ndarray,
    exclude_zone: Optional[np.ndarray],
    same_material: Optional[np.ndarray],
    *,
    inner_r: int = 22,
    outer_r: int = 88,
    min_delta_lab: float = 20.0,
    max_pts: int = 8,
) -> List[Tuple[int, int]]:
    """Sample high-contrast negative anchors in a ring around strokes outside protected subject regions."""
    s_u8 = (stroke_mask > 0).astype(np.uint8)
    if not np.any(s_u8):
        return []
    inner_d = cv2.dilate(s_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_r * 2 + 1, inner_r * 2 + 1))) > 0
    outer_d = cv2.dilate(s_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_r * 2 + 1, outer_r * 2 + 1))) > 0
    ring = outer_d & (~inner_d)
    if exclude_zone is not None:
        ring &= ~(exclude_zone > 0)
    if same_material is not None:
        ring &= ~same_material
    if not np.any(ring):
        return []

    ref_lab = np.median(lab[s_u8 > 0], axis=0)
    ry, rx = np.where(ring)
    dists = np.linalg.norm(lab[ry, rx] - ref_lab, axis=1)
    strong = dists >= min_delta_lab
    if np.any(strong):
        ry, rx = ry[strong], rx[strong]
    if len(rx) == 0:
        return []
    # Pick spatially well-separated ring points around stroke center
    cy, cx = float(np.mean(np.where(s_u8 > 0)[0])), float(np.mean(np.where(s_u8 > 0)[1]))
    angles = np.arctan2(ry - cy, rx - cx)
    order = np.argsort(angles)
    step = max(1, len(order) // max_pts)
    return [(int(rx[order[k]]), int(ry[order[k]])) for k in range(0, len(order), step)][:max_pts]


def smart_roto_brush_refine(
    frame_rgb: np.ndarray,
    prev_mask: Optional[np.ndarray],
    prev_alpha: Optional[np.ndarray],
    fg_strokes: np.ndarray,
    bg_strokes: np.ndarray,
    checkpoints_dir: str | Path = DEFAULT_CKPT_DIR,
    *,
    box_xyxy: Optional[Tuple[int, int, int, int]] = None,
    smart_mode: bool = True,
    prefer_best_checkpoint: bool = True,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Universal Multi-Subject Roto-Brush (Works on ANY subject: person, book, product, pet, gadget):
      - Supports freehand Green (+keep) / Red (-subtract) strokes AND drag Box-Select (`box_xyxy`).
      - Scores SAM 2.1 multimask hypotheses by prompt/box fit, edge gradient alignment,
        multi-modal Lab appearance cohesion, and surgical size prior.
      - Red brush segments unwanted background using high-contrast keep anchors so nearby limbs/edges stay safe.
      - Splices soft alpha ONLY inside the localized edit zone so untouched BiRefNet soft edges survive 100%.
    """
    from src.alpha_matting import (
        expand_mask_with_color_evidence,
        refine_alpha_edge_controls,
        sample_roto_anchor_points,
    )

    h, w = frame_rgb.shape[:2]
    fg_u8 = (fg_strokes > 0).astype(np.uint8)
    bg_u8 = (bg_strokes > 0).astype(np.uint8)
    has_fg = bool(np.any(fg_u8 > 0))
    has_bg = bool(np.any(bg_u8 > 0))
    valid_box: Optional[Tuple[int, int, int, int]] = None
    if box_xyxy is not None and len(box_xyxy) == 4:
        bx0 = max(0, min(w - 1, int(box_xyxy[0])))
        by0 = max(0, min(h - 1, int(box_xyxy[1])))
        bx1 = max(0, min(w, int(box_xyxy[2])))
        by1 = max(0, min(h, int(box_xyxy[3])))
        if bx1 - bx0 >= 8 and by1 - by0 >= 8:
            valid_box = (bx0, by0, bx1, by1)

    old_bin = (prev_mask > 0).astype(np.uint8) if (prev_mask is not None and prev_mask.shape[:2] == (h, w)) else np.zeros((h, w), dtype=np.uint8)
    new_mask = old_bin.copy()

    if not has_fg and not has_bg and valid_box is None:
        alpha_out = (
            np.clip(prev_alpha.astype(np.float32), 0.0, 1.0)
            if (prev_alpha is not None and prev_alpha.shape[:2] == (h, w))
            else new_mask.astype(np.float32)
        )
        return new_mask, alpha_out

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    use_sam2 = bool(prefer_best_checkpoint or smart_mode or valid_box is not None)
    if not use_sam2:
        if has_fg:
            fg_exp, _ = expand_mask_with_color_evidence(bgr, fg_u8, pixels=2)
            new_mask = ((new_mask > 0) | (fg_exp > 0)).astype(np.uint8)
        if has_bg:
            new_mask[bg_u8 > 0] = 0
    else:
        lab, edge_mag, skin_map, device_map = _detect_hands_device_priors(frame_rgb)
        empty = np.zeros((h, w), dtype=np.uint8)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        _, predictor, device, same_frame, frame_sig = _get_cached_sam2_predictor(
            frame_rgb,
            checkpoints_dir,
            use_cache=use_cache,
            prefer_best_checkpoint=prefer_best_checkpoint,
        )

        try:
            with torch.inference_mode():
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else _null_ctx()
                with ctx:
                    if not same_frame:
                        predictor.set_image(frame_rgb)
                        if use_cache:
                            _SAM2_IMG_CACHE["frame_sig"] = frame_sig

                    # 0. Universal Box-Select (Khoanh vùng hình chữ nhật cho bất kỳ chủ thể nào)
                    if valid_box is not None:
                        bx0, by0, bx1, by1 = valid_box
                        box_arr = np.array([bx0, by0, bx1, by1], dtype=np.float32)
                        box_area = float(max(1, (bx1 - bx0) * (by1 - by0)))
                        pts_coords = None
                        pts_labels = None
                        if has_fg or has_bg:
                            mix_pts = sample_roto_anchor_points(fg_u8, bg_u8, bgr, grid_step=24, preserve_stroke_components=True)
                            if mix_pts:
                                pts_coords = np.array([[x, y] for (x, y, _) in mix_pts], dtype=np.float32)
                                pts_labels = np.array([lbl for (_, _, lbl) in mix_pts], dtype=np.int32)
                        sub_m, sub_s, _ = predictor.predict(
                            point_coords=pts_coords,
                            point_labels=pts_labels,
                            box=box_arr,
                            multimask_output=True,
                        )
                        best_idx = 0
                        best_score = -1e9
                        for s_idx in range(len(sub_m)):
                            mc = sub_m[s_idx] > 0.0
                            in_box = float(np.sum(mc[by0:by1, bx0:bx1]))
                            out_box = float(np.sum(mc)) - in_box
                            fill_ratio = in_box / box_area
                            spill_ratio = out_box / max(1.0, in_box)
                            mc_u8 = mc.astype(np.uint8)
                            perim = (cv2.dilate(mc_u8, k3) ^ cv2.erode(mc_u8, k3)) > 0
                            edge_align = float(np.mean(edge_mag[perim])) if np.any(perim) else 0.0
                            # Prefer cohesive object filling a healthy portion of user box without huge outside leak
                            size_fit = -1.2 if fill_ratio < 0.12 else (0.5 * min(fill_ratio, 0.85))
                            score = float(sub_s[s_idx]) + size_fit + 0.55 * edge_align - 1.6 * spill_ratio
                            if score > best_score:
                                best_score = score
                                best_idx = s_idx

                        box_mask = (sub_m[best_idx] > 0.0).astype(np.uint8)
                        # Keep components intersecting user box (soft margin 8px)
                        n_cc, l_cc = cv2.connectedComponents(box_mask, connectivity=8)
                        if n_cc > 2:
                            hit_ids = set(np.unique(l_cc[max(0, by0 - 4):min(h, by1 + 4), max(0, bx0 - 4):min(w, bx1 + 4)]).tolist()) - {0}
                            if hit_ids:
                                box_mask = np.isin(l_cc, list(hit_ids)).astype(np.uint8)
                        box_mask = _fill_mask_holes(box_mask, bgr=bgr)
                        if has_bg:
                            box_mask[bg_u8 > 0] = 0
                        box_mask, _ = expand_mask_with_color_evidence(bgr, box_mask, pixels=1)
                        new_mask = ((new_mask > 0) | (box_mask > 0)).astype(np.uint8) if np.any(old_bin) else box_mask

                    # 1. Green Brush (Universal Multi-Modal Appearance Detail Addition)
                    if has_fg:
                        raw_fg_pts = sample_roto_anchor_points(fg_u8, bg_u8, bgr, grid_step=18, preserve_stroke_components=True)
                        pos_pts = [(x, y) for (x, y, lbl) in raw_fg_pts if lbl == 1]
                        explicit_neg = [(x, y) for (x, y, lbl) in raw_fg_pts if lbl == 0]
                        is_skin_stroke = float(np.mean(skin_map[fg_u8 > 0])) > 0.45
                        is_dev_stroke = float(np.mean(device_map[fg_u8 > 0])) > 0.55
                        same_mat = skin_map if is_skin_stroke else (device_map if is_dev_stroke else None)

                        auto_neg: List[Tuple[int, int]] = []
                        if smart_mode:
                            auto_neg = _sample_ring_negatives(
                                fg_u8,
                                lab,
                                exclude_zone=(old_bin | fg_u8) if np.any(old_bin) else fg_u8,
                                same_material=same_mat,
                                inner_r=20,
                                outer_r=84,
                                min_delta_lab=20.0,
                                max_pts=8,
                            )
                        neg_pts = explicit_neg + auto_neg
                        clusters = _cluster_positive_points(pos_pts, lab, dist_thresh=54.0, col_thresh=40.0)
                        stroke_labs = lab[fg_u8 > 0]
                        # Multi-modal color representatives handle patterned/two-tone objects (e.g. open book, striped shirt)
                        q_labs = np.percentile(stroke_labs, [20, 50, 80], axis=0) if len(stroke_labs) >= 6 else np.median(stroke_labs, axis=0)[None, :]
                        small_stroke_with_prior = np.any(old_bin) and (float(np.mean(fg_u8 > 0)) < 0.015)

                        added_total = np.zeros((h, w), dtype=np.uint8)
                        for cluster in clusters:
                            sub_coords = [[cx, cy] for (cx, cy) in cluster] + [[nx, ny] for (nx, ny) in neg_pts]
                            sub_labels = [1] * len(cluster) + [0] * len(neg_pts)
                            sub_m, sub_s, _ = predictor.predict(
                                point_coords=np.array(sub_coords, dtype=np.float32),
                                point_labels=np.array(sub_labels, dtype=np.int32),
                                multimask_output=True,
                            )

                            best_idx = 0
                            best_score = -1e9
                            for s_idx in range(len(sub_m)):
                                mc = sub_m[s_idx] > 0.0
                                pos_cov = sum(1 for (cx, cy) in cluster if 0 <= cy < h and 0 <= cx < w and mc[cy, cx]) / max(len(cluster), 1)
                                neg_hit = sum(1 for (nx, ny) in neg_pts if 0 <= ny < h and 0 <= nx < w and mc[ny, nx]) / max(len(neg_pts), 1)

                                mc_u8 = mc.astype(np.uint8)
                                perim = (cv2.dilate(mc_u8, k3) ^ cv2.erode(mc_u8, k3)) > 0
                                edge_align = float(np.mean(edge_mag[perim])) if np.any(perim) else 0.0

                                newly_added = mc & (old_bin == 0)
                                if np.any(newly_added):
                                    ny, nx = np.where(newly_added)
                                    step = max(1, len(nx) // 800)
                                    sample_px = lab[ny[::step], nx[::step]]
                                    d_modes = np.min(
                                        np.linalg.norm(sample_px[:, None, :] - q_labs[None, :, :], axis=2),
                                        axis=1,
                                    )
                                    col_cohesion = float(np.mean(d_modes < 36.0))
                                    if same_mat is not None:
                                        mat_match = float(np.mean(same_mat[ny[::step], nx[::step]]))
                                        col_cohesion = 0.80 * col_cohesion + 0.20 * mat_match
                                else:
                                    col_cohesion = 0.6

                                added_ratio = float(np.mean(newly_added))
                                size_pen = 1.8 if (small_stroke_with_prior and added_ratio > 0.22) else (0.06 * s_idx)

                                score = (
                                    float(sub_s[s_idx])
                                    + 1.80 * pos_cov
                                    - 3.40 * neg_hit
                                    + 0.55 * edge_align
                                    + 0.65 * col_cohesion
                                    - size_pen
                                )
                                if score > best_score:
                                    best_score = score
                                    best_idx = s_idx

                            part_mask = (sub_m[best_idx] > 0.0).astype(np.uint8)
                            _, l_cc = cv2.connectedComponents(part_mask, connectivity=8)
                            for cx, cy in cluster:
                                if 0 <= cy < h and 0 <= cx < w and l_cc[cy, cx] > 0:
                                    added_total[l_cc == l_cc[cy, cx]] = 1

                        added_total[fg_u8 > 0] = 1
                        added_total = _fill_mask_holes(added_total, bgr=bgr)
                        if has_bg:
                            added_total[bg_u8 > 0] = 0
                        added_total, _ = expand_mask_with_color_evidence(bgr, added_total, pixels=1)
                        new_mask = ((new_mask > 0) | (added_total > 0)).astype(np.uint8)

                    # 2. Red Brush (Smart High-Contrast Protected Subtraction)
                    if has_bg and np.any(new_mask > 0):
                        if smart_mode:
                            bg_pts_raw = sample_roto_anchor_points(bg_u8, empty, bgr, grid_step=18, preserve_stroke_components=True)
                            pos_rm = [(x, y) for (x, y, lbl) in bg_pts_raw if lbl == 1]
                            if pos_rm:
                                rm_guard = cv2.dilate(bg_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))) > 0
                                near_ring = (cv2.dilate(bg_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (141, 141))) > 0) & (~rm_guard)
                                keep_cand = (new_mask > 0) & (~rm_guard)
                                bg_lab_med = np.median(lab[bg_u8 > 0], axis=0)

                                neg_keep: List[Tuple[int, int]] = []
                                for zone in (keep_cand & near_ring, keep_cand):
                                    if np.any(zone):
                                        ky, kx = np.where(zone)
                                        kd = np.linalg.norm(lab[ky, kx] - bg_lab_med, axis=1)
                                        high_diff = kd >= 16.0
                                        if np.any(high_diff):
                                            ky, kx = ky[high_diff], kx[high_diff]
                                        step = max(1, len(kx) // 8)
                                        for idx_k in range(0, len(kx), step):
                                            neg_keep.append((int(kx[idx_k]), int(ky[idx_k])))
                                            if len(neg_keep) >= 8:
                                                break
                                    if len(neg_keep) >= 6:
                                        break

                                sub_coords = [[rx, ry] for (rx, ry) in pos_rm] + [[kx, ky] for (kx, ky) in neg_keep]
                                sub_labels = [1] * len(pos_rm) + [0] * len(neg_keep)
                                rm_m, rm_s, _ = predictor.predict(
                                    point_coords=np.array(sub_coords, dtype=np.float32),
                                    point_labels=np.array(sub_labels, dtype=np.int32),
                                    multimask_output=True,
                                )
                                best_rm_idx = 0
                                best_rm_score = -1e9
                                for s_idx in range(len(rm_m)):
                                    mc = rm_m[s_idx] > 0.0
                                    r_cov = sum(1 for (rx, ry) in pos_rm if 0 <= ry < h and 0 <= rx < w and mc[ry, rx]) / max(len(pos_rm), 1)
                                    k_hit = sum(1 for (kx, ky) in neg_keep if 0 <= ky < h and 0 <= kx < w and mc[ky, kx]) / max(len(neg_keep), 1)
                                    mc_u8 = mc.astype(np.uint8)
                                    perim = (cv2.dilate(mc_u8, k3) ^ cv2.erode(mc_u8, k3)) > 0
                                    edge_align = float(np.mean(edge_mag[perim])) if np.any(perim) else 0.0
                                    score = float(rm_s[s_idx]) + 1.8 * r_cov - 3.2 * k_hit + 0.5 * edge_align - 0.1 * s_idx
                                    if score > best_rm_score:
                                        best_rm_score = score
                                        best_rm_idx = s_idx

                                cand_rm = (rm_m[best_rm_idx] > 0.0).astype(np.uint8)
                                _, l_rm = cv2.connectedComponents(cand_rm, connectivity=8)
                                hit_ids = set(np.unique(l_rm[bg_u8 > 0]).tolist()) - {0}
                                targeted_rm = np.isin(l_rm, list(hit_ids)).astype(np.uint8) if hit_ids else np.zeros_like(old_bin)
                                stroke_frac = float(np.sum((bg_u8 > 0) & (new_mask > 0))) / max(1.0, float(np.sum(new_mask > 0)))
                                if float(np.sum(targeted_rm & new_mask)) > 0.65 * max(1, int(np.sum(new_mask))) and stroke_frac < 0.10:
                                    targeted_rm = np.zeros_like(old_bin)
                                new_mask[targeted_rm > 0] = 0

                        new_mask[bg_u8 > 0] = 0
        finally:
            if not use_cache:
                del predictor
                clear_vram()

    # Localized soft-alpha recomputation & seamless splice (preserves untouched BiRefNet soft edges 100%)
    diff_bin = (new_mask != old_bin) | (fg_u8 > 0) | (bg_u8 > 0)
    local_alpha = refine_alpha_edge_controls(
        bgr,
        new_mask.astype(np.float32),
        erode_dilate=0.0,
        guided_radius=3,
        trimap_radius=6,
        edge_preserve=0.45,
        edge_feather=0.0,
    )
    if prev_alpha is not None and prev_alpha.shape[:2] == (h, w) and valid_box is None:
        edit_zone = cv2.dilate(diff_bin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        edit_w = cv2.GaussianBlur(edit_zone.astype(np.float32), (9, 9), 0)
        soft_alpha = np.clip(prev_alpha.astype(np.float32) * (1.0 - edit_w) + local_alpha * edit_w, 0.0, 1.0)
        soft_alpha[new_mask == 0] = np.minimum(soft_alpha[new_mask == 0], local_alpha[new_mask == 0])
    elif prev_alpha is not None and prev_alpha.shape[:2] == (h, w) and np.any(old_bin):
        # Box-select unioned with existing mask: splice only around the newly added box region
        edit_zone = cv2.dilate(diff_bin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        edit_w = cv2.GaussianBlur(edit_zone.astype(np.float32), (9, 9), 0)
        soft_alpha = np.clip(prev_alpha.astype(np.float32) * (1.0 - edit_w) + local_alpha * edit_w, 0.0, 1.0)
    else:
        soft_alpha = local_alpha

    return (new_mask > 0).astype(np.uint8), soft_alpha.astype(np.float32)


def auto_cut_universal_hybrid(
    src_bgr: np.ndarray,
    checkpoints_dir: str | Path = DEFAULT_CKPT_DIR,
    *,
    max_size: int = 1024,
    prefer_best_checkpoint: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Universal AI 1-Click Hybrid Cutout (works cleanly for ANY subject: person reading a book,
    hands + product/book/gadget, standalone products, pets, portraits):
      1. BiRefNet_HR-matting + native edge preservation + universal structural & pinhole heal.
      2. Free BiRefNet RAM/VRAM, then query SAM 2.1 with the subject bounding box + interior/exterior
         anchors to verify semantic completeness and recover any adjacent dropped object parts
         (e.g., white book pages against light walls or straight metallic edges).
    """
    from src.alpha_matting import refine_alpha_edge_controls
    from src.matte_qa import heal_universal_structural_defects, is_green_screen_scene
    from src.neural_matting import release_neural_models, run_birefnet_matting

    h, w = src_bgr.shape[:2]
    alpha_bi = run_birefnet_matting(src_bgr, max_size=int(max_size))
    release_neural_models()

    alpha_bi = refine_alpha_edge_controls(
        src_bgr,
        alpha_bi,
        erode_dilate=0.0,
        guided_radius=2,
        trimap_radius=6,
        edge_preserve=0.45,
        edge_feather=0.0,
    )
    alpha_bi, _, _ = heal_universal_structural_defects(src_bgr, alpha_bi)
    mask_bi = (alpha_bi >= 0.42).astype(np.uint8)
    fg_frac = float(np.mean(mask_bi > 0))
    if fg_frac < 0.008 or fg_frac > 0.94:
        return mask_bi, alpha_bi

    # Universal BiRefNet + SAM 2.1 completeness fusion (box + core/near-ring background anchors)
    ys, xs = np.where(mask_bi > 0)
    bx0, by0 = max(0, int(xs.min()) - 8), max(0, int(ys.min()) - 8)
    bx1, by1 = min(w, int(xs.max()) + 9), min(h, int(ys.max()) + 9)
    rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    py, px = np.where(alpha_bi >= 0.80)
    pos_pts: List[List[float]] = []
    if len(px) > 0:
        step = max(1, len(px) // 6)
        for i in range(0, len(px), step):
            pos_pts.append([float(px[i]), float(py[i])])
            if len(pos_pts) >= 6:
                break

    ext_zone = cv2.dilate((alpha_bi >= 0.20).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))) == 0
    near_bg_ring = (
        (cv2.dilate(mask_bi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))) > 0)
        & (cv2.dilate(mask_bi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) == 0)
        & (alpha_bi <= 0.08)
    )
    bg_anchor_zone = near_bg_ring if int(np.sum(near_bg_ring)) >= 40 else ext_zone
    ny, nx = np.where(bg_anchor_zone)
    if len(nx) == 0:
        ny, nx = np.where(ext_zone)
    neg_pts: List[List[float]] = []
    if len(nx) > 0:
        step = max(1, len(nx) // 6)
        for i in range(0, len(nx), step):
            neg_pts.append([float(nx[i]), float(ny[i])])
            if len(neg_pts) >= 6:
                break

    mask_out = mask_bi
    alpha_out = alpha_bi
    try:
        _, predictor, device, same_frame, frame_sig = _get_cached_sam2_predictor(
            rgb,
            checkpoints_dir,
            use_cache=True,
            prefer_best_checkpoint=prefer_best_checkpoint,
        )
        with torch.inference_mode():
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else _null_ctx()
            with ctx:
                if not same_frame:
                    predictor.set_image(rgb)
                    _SAM2_IMG_CACHE["frame_sig"] = frame_sig
                coords = np.array(pos_pts + neg_pts, dtype=np.float32) if (pos_pts or neg_pts) else None
                labels = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int32) if (pos_pts or neg_pts) else None
                sam_m, sam_s, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    box=np.array([bx0, by0, bx1, by1], dtype=np.float32),
                    multimask_output=True,
                )
        best_i = 0
        best_iou = -1.0
        bi_b = mask_bi > 0
        for s_idx in range(len(sam_m)):
            mc = sam_m[s_idx] > 0.0
            iou = float(np.sum(mc & bi_b)) / max(1.0, float(np.sum(mc | bi_b)))
            score = iou + 0.15 * float(sam_s[s_idx])
            if score > best_iou:
                best_iou = score
                best_i = s_idx
        sam_best = (sam_m[best_i] > 0.0).astype(np.uint8)

        # Recover adjacent dropped object parts inside subject bbox (e.g. white pages/sleeves/bezels)
        box_mask = np.zeros((h, w), dtype=np.uint8)
        box_mask[by0:by1, bx0:bx1] = 1
        touch_bi = cv2.dilate(mask_bi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
        cand_rec = ((sam_best > 0) & (alpha_bi < 0.42) & (box_mask > 0)).astype(np.uint8)
        n_rc, l_rc, st_rc, _ = cv2.connectedComponentsWithStats(cand_rec, connectivity=8)
        recovered = np.zeros((h, w), dtype=np.uint8)
        max_rec_area = int(np.sum(mask_bi > 0) * 0.35)

        is_green_bg = is_green_screen_scene(src_bgr, alpha_bi)
        hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV)
        b_ch, g_ch, r_ch = (src_bgr[..., 0].astype(np.float32), src_bgr[..., 1].astype(np.float32), src_bgr[..., 2].astype(np.float32))
        g_exc = g_ch - np.maximum(b_ch, r_ch)
        w_lab_c = np.array([0.45, 1.25, 1.25], dtype=np.float32)
        k55 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
        k13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

        for cid in range(1, n_rc):
            area = int(st_rc[cid, cv2.CC_STAT_AREA])
            if area < 64 or area > max_rec_area:
                continue
            comp = l_rc == cid
            if not np.any(comp & touch_bi):
                continue

            # Block shadowed green-screen cloth leakage immediately on green-screen scenes
            if is_green_bg:
                comp_g_med = float(np.median(g_exc[comp]))
                green_ratio = float(np.mean((hsv[..., 0][comp] >= 32) & (hsv[..., 0][comp] <= 100) & (hsv[..., 1][comp] >= 28)))
                if comp_g_med > 10.0 and green_ratio > 0.35:
                    continue

            # Universal local background vs foreground Lab check (exclude candidate parts from bg sample)
            comp_u8 = comp.astype(np.uint8)
            local_bg = (cv2.dilate(comp_u8, k55) > 0) & (cand_rec == 0) & (alpha_bi <= 0.08)
            if int(np.sum(local_bg)) < 20:
                local_bg = ext_zone
            if np.any(local_bg):
                bg_vals = lab[local_bg]
                bg_q = np.percentile(bg_vals, [20, 50, 80], axis=0) if len(bg_vals) >= 6 else np.median(bg_vals, axis=0)[None, :]
                comp_med = np.median(lab[comp], axis=0)
                d_bg = float(np.min(np.linalg.norm((bg_q - comp_med[None, :]) * w_lab_c[None, :], axis=1)))
                local_fg = (cv2.dilate(comp_u8, k13) > 0) & (alpha_bi >= 0.75)
                d_fg = float(np.linalg.norm((np.median(lab[local_fg], axis=0) - comp_med) * w_lab_c)) if np.any(local_fg) else 1e9
                if d_bg < 22.0 or (d_bg < 38.0 and d_bg < d_fg * 1.05):
                    continue

            recovered[comp] = 1

        if np.any(recovered > 0):
            mask_out = ((mask_bi > 0) | (recovered > 0)).astype(np.uint8)
            mask_out = _fill_mask_holes(mask_out, max_hole_ratio=0.0015, bgr=src_bgr)
            local_a = refine_alpha_edge_controls(
                src_bgr,
                mask_out.astype(np.float32),
                erode_dilate=0.0,
                guided_radius=2,
                trimap_radius=6,
                edge_preserve=0.45,
                edge_feather=0.0,
            )
            splice_w = cv2.GaussianBlur(
                cv2.dilate(recovered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32),
                (9, 9),
                0,
            )
            alpha_out = np.clip(alpha_bi * (1.0 - splice_w) + local_a * splice_w, 0.0, 1.0).astype(np.float32)
    except Exception:
        pass
    finally:
        release_sam2_cache()

    return (mask_out > 0).astype(np.uint8), alpha_out.astype(np.float32)


auto_cut_hands_device_hybrid = auto_cut_universal_hybrid


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
                state = _init_sam2_video_state_bounded(
                    predictor,
                    frame_files,
                    Path(out_masks_dir).parent,
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
    *,
    prefer_best_checkpoint: bool = False,
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

    release_sam2_cache()
    try:
        from src.neural_matting import release_neural_models

        release_neural_models()
    except Exception:
        pass

    if prefer_best_checkpoint:
        ckpt_path, config_name = ensure_best_sam2_checkpoint(checkpoints_dir)
    else:
        ckpt_path, config_name = ensure_checkpoint(checkpoints_dir), SAM2_CONFIG

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(config_name, str(ckpt_path), device=device)

    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]
    video_segments: Dict[int, np.ndarray] = {
        f_idx: np.zeros((h, w), dtype=np.uint8) for f_idx in range(num_frames)
    }

    trusted_masks: Dict[int, np.ndarray] = {}
    try:
        with torch.inference_mode():
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else _null_ctx()
            with ctx:
                state = _init_sam2_video_state_bounded(
                    predictor,
                    frame_files,
                    Path(out_masks_dir).parent,
                )

                annotated_frames = sorted(confirmed_masks.keys())
                for f_idx in annotated_frames:
                    m = (confirmed_masks[f_idx] > 0).astype(np.uint8)
                    if m.shape[:2] != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    trusted_masks[f_idx] = m
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
        save_mask(trusted_masks.get(f_idx, video_segments[f_idx]), mp)
        out_paths.append(mp)
    return out_paths


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False

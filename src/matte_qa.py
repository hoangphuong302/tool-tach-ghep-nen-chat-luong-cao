from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.video_io import load_alpha, load_mask


def _encode_diff_thumb_b64(
    bgr: np.ndarray,
    repaired_mask: np.ndarray,
    removed_mask: np.ndarray,
    max_dim: int = 240,
) -> str:
    """Render a compact JPEG preview highlighting auto-healed missing (green) and pruned (red) pixels."""
    vis = (bgr.astype(np.float32) * 0.72).astype(np.uint8)
    if np.any(repaired_mask > 0):
        vis[repaired_mask > 0] = np.clip(
            vis[repaired_mask > 0].astype(np.float32) * 0.35 + np.array([40, 240, 80], dtype=np.float32) * 0.65,
            0,
            255,
        ).astype(np.uint8)
    if np.any(removed_mask > 0):
        vis[removed_mask > 0] = np.clip(
            vis[removed_mask > 0].astype(np.float32) * 0.35 + np.array([50, 60, 245], dtype=np.float32) * 0.65,
            0,
            255,
        ).astype(np.uint8)

    h, w = vis.shape[:2]
    if max(h, w) > max_dim:
        sc = float(max_dim) / float(max(h, w))
        vis = cv2.resize(vis, (max(32, int(round(w * sc))), max(32, int(round(h * sc)))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _detect_device_core_and_priors(bgr: np.ndarray, fg_mask_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Locate a rigid rectangular handheld device core inside `fg_mask_u8` alongside
    skin, green-screen, and bezel-hardware masks. Returns (dev_core, skin, is_green, hw_bezel).
    """
    h, w = bgr.shape[:2]
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    cr, cb = ycc[..., 1], ycc[..., 2]
    g_exc = g - np.maximum(r, b)

    is_green = (g_exc > 16.0) & (hue >= 28) & (hue <= 102) & (sat >= 30)
    skin = (cr >= 135) & (cr <= 178) & (cb >= 77) & (cb <= 129) & ((hue <= 17) | (hue >= 164)) & ((r - g) >= 12.0) & (val >= 65)
    hw_bezel = (
        (~is_green)
        & (~skin)
        & (
            ((sat <= 62) & (val >= 138) & (np.abs(r - b) <= 32.0))  # White / silver aluminum chassis
            | ((val < 78) & (sat <= 95))  # Black side volume rocker / bezel
            | ((r > g + 38.0) & (sat >= 82))  # Red/orange side power button
        )
    )

    blue_ui = (b > r + 18.0) & (sat > 42)
    dark_face = (val < 68) & (g_exc < 12.0)
    raw_dev = ((blue_ui | dark_face) & (fg_mask_u8 > 0)).astype(np.uint8)
    raw_dev = cv2.morphologyEx(raw_dev, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))

    num_cc, labels_cc, stats, _ = cv2.connectedComponentsWithStats(raw_dev, connectivity=8)
    dev_core = np.zeros((h, w), dtype=np.uint8)
    min_area = int(h * w * 0.008)
    max_area = int(h * w * 0.55)
    best_cid = 0
    best_area = 0
    for cid in range(1, num_cc):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area and area > best_area:
            best_area = area
            best_cid = cid

    if best_cid > 0:
        comp = (labels_cc == best_cid).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c_big = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c_big)
            rect_area = max(1.0, float(rect[1][0] * rect[1][1]))
            rectangularity = float(best_area) / rect_area
            if rectangularity >= 0.55:
                cv2.drawContours(dev_core, [cv2.convexHull(c_big)], -1, 1, thickness=cv2.FILLED)

    return dev_core, skin, is_green, hw_bezel


def heal_device_bezel_notches(
    bgr: np.ndarray,
    alpha: np.ndarray,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Per-frame rigid device bezel structural guard (~4–8 ms CPU):
    Prevents temporal drift from biting a concave notch into the straight metallic/white frame
    and side volume/power buttons of a rectangular handheld device (translator/phone).
    """
    a_f = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    h, w = a_f.shape[:2]
    if min(h, w) < 64 or not np.any(a_f >= 0.45):
        return a_f, 0, np.zeros((h, w), dtype=np.uint8)

    fg_u8 = (a_f >= 0.45).astype(np.uint8)
    ys, xs = np.where(fg_u8 > 0)
    pad = max(24, int(0.05 * min(h, w)))
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)

    bgr_roi = bgr[y0:y1, x0:x1]
    a_roi = a_f[y0:y1, x0:x1].copy()
    fg_roi = fg_u8[y0:y1, x0:x1]
    rh, rw = bgr_roi.shape[:2]

    dev_core, skin, is_green, hw_bezel = _detect_device_core_and_priors(bgr_roi, fg_roi)
    if not np.any(dev_core):
        return a_f, 0, np.zeros((h, w), dtype=np.uint8)

    bezel_pad = max(14, min(42, int(round(0.036 * min(h, w)))))
    k_bezel = bezel_pad * 2 + 1
    ker_bezel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_bezel, k_bezel))
    bezel_zone = cv2.dilate(dev_core, ker_bezel) > 0

    # Anchor points that define the true straight outer perimeter of the device chassis:
    # 1) Dev core itself
    # 2) Confirmed matte pixels along the bezel zone that are non-skin hardware
    # 3) Direct hardware rim (metallic/white frame or side buttons) connected to dev_core
    confirmed_rim = bezel_zone & (a_roi >= 0.50) & (~skin) & (~is_green)
    hw_seeds = (dev_core > 0) | confirmed_rim
    tight_hw = (cv2.dilate(dev_core, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))) > 0) & hw_bezel & (a_roi >= 0.15)
    hull_src = (hw_seeds | tight_hw).astype(np.uint8)

    cnts, _ = cv2.findContours(hull_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return a_f, 0, np.zeros((h, w), dtype=np.uint8)
    hull_pts = np.vstack(cnts)
    device_hull = np.zeros((rh, rw), dtype=np.uint8)
    cv2.drawContours(device_hull, [cv2.convexHull(hull_pts)], -1, 1, thickness=cv2.FILLED)

    # Also protect slightly beyond dev_core where true hardware bezel/buttons connect closely
    near_r = max(8, int(round(bezel_pad * 0.72)))
    near_core = cv2.dilate(dev_core, cv2.getStructuringElement(cv2.MORPH_RECT, (near_r * 2 + 1, near_r * 2 + 1))) > 0

    # Notch candidates inside the rigid convex chassis hull where alpha dropped below 0.85
    notch_cand = (
        (device_hull > 0)
        & bezel_zone
        & (a_roi < 0.85)
        & (~is_green)
        & (~skin)
        & (hw_bezel | near_core)
    )

    # Reject distant flat background if a skin gap/indent allowed the convex hull to bridge background
    # by keeping components touching/within 3px of confirmed foreground AND checking component size/aspect
    if not np.any(notch_cand):
        return a_f, 0, np.zeros((h, w), dtype=np.uint8)

    touch_fg = cv2.dilate(fg_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    num_nc, lbl_nc, stats_nc, _ = cv2.connectedComponentsWithStats(notch_cand.astype(np.uint8), connectivity=8)
    notch_u8 = np.zeros((rh, rw), dtype=np.uint8)
    min_notch_area = 12
    for cid in range(1, num_nc):
        area = int(stats_nc[cid, cv2.CC_STAT_AREA])
        if area < min_notch_area:
            continue
        comp = lbl_nc == cid
        if np.any(comp & touch_fg):
            notch_u8[comp] = 1

    healed_px = int(np.sum(notch_u8 > 0))
    if healed_px == 0:
        return a_f, 0, np.zeros((h, w), dtype=np.uint8)

    # Smooth 1px anti-aliased boundary around healed hardware bezel notch
    notch_soft = cv2.GaussianBlur(notch_u8.astype(np.float32), (3, 3), 0.7)
    a_roi = np.maximum(a_roi, np.clip(notch_soft, 0.0, 1.0))

    out = a_f.copy()
    out[y0:y1, x0:x1] = np.clip(a_roi, 0.0, 1.0)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = notch_u8
    return out.astype(np.float32), healed_px, full_mask


def _is_green_screen_exterior(bgr: np.ndarray, ext_mask: np.ndarray) -> bool:
    """Return True when the unmasked exterior background is predominantly chroma green."""
    if int(np.sum(ext_mask)) < 200:
        return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    g_exc = g - np.maximum(b, r)
    strong_green = (g_exc > 15.0) & (hsv[..., 0] >= 32) & (hsv[..., 0] <= 96) & (hsv[..., 1] >= 35)
    return float(np.mean(strong_green[ext_mask])) >= 0.22


def is_green_screen_scene(bgr: np.ndarray, alpha: np.ndarray) -> bool:
    """Check whether the exterior background outside foreground `alpha` is a green screen."""
    a_f = np.asarray(alpha, dtype=np.float32)
    return _is_green_screen_exterior(bgr, a_f <= 0.12)


def heal_universal_structural_defects(
    bgr: np.ndarray,
    alpha: np.ndarray,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Universal per-frame structural guard for ANY subject (book, person, product, gadget):
      1. Rigid Rectangular Device Guard (`heal_device_bezel_notches`).
      2. Color-Aware Enclosed Pinhole Guard: fills interior holes matching foreground tone,
         without filling real exterior background gaps (e.g. room wall or green-screen slit).
      3. Trapped Green-Screen Crevice Punch-Out: cleans shadowed green-screen gaps between
         fingers and subject while protecting glowing device UI screens.
    """
    a_f, bezel_px, bezel_mask = heal_device_bezel_notches(bgr, alpha)
    h, w = a_f.shape[:2]
    fg_u8 = (a_f >= 0.45).astype(np.uint8)
    if min(h, w) < 32 or not np.any(fg_u8 > 0):
        return a_f, bezel_px, bezel_mask

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b_ch = bgr[..., 0].astype(np.float32)
    g_ch = bgr[..., 1].astype(np.float32)
    r_ch = bgr[..., 2].astype(np.float32)
    g_exc = g_ch - np.maximum(b_ch, r_ch)
    is_green_strong = (g_exc > 18.0) & (hsv[..., 0] >= 28) & (hsv[..., 0] <= 102) & (hsv[..., 1] >= 35)
    is_green_soft = (g_exc > 8.0) & (hsv[..., 0] >= 28) & (hsv[..., 0] <= 104) & (hsv[..., 1] >= 18)

    ext_zone = a_f <= 0.08
    is_green_bg = _is_green_screen_exterior(bgr, ext_zone)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    w_lab = np.array([0.65, 1.15, 1.15], dtype=np.float32)
    ext_ring = (cv2.dilate(fg_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0) & (fg_u8 == 0)
    bg_sample = ext_ring if int(np.sum(ext_ring)) >= 40 else ext_zone
    if np.any(bg_sample):
        bg_vals = lab[bg_sample]
        bg_q = np.percentile(bg_vals, [20, 50, 80], axis=0) if len(bg_vals) >= 6 else np.median(bg_vals, axis=0)[None, :]
    else:
        bg_q = None

    inv = (1 - fg_u8).astype(np.uint8)
    max_hole_area = max(48, int(h * w * 0.006))
    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    pinhole_mask = np.zeros((h, w), dtype=np.uint8)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for cid in range(1, num_cc):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area > max_hole_area:
            continue
        x0 = int(stats[cid, cv2.CC_STAT_LEFT])
        y0 = int(stats[cid, cv2.CC_STAT_TOP])
        cw = int(stats[cid, cv2.CC_STAT_WIDTH])
        ch = int(stats[cid, cv2.CC_STAT_HEIGHT])
        if x0 <= 0 or y0 <= 0 or x0 + cw >= w or y0 + ch >= h:
            continue
        comp = labels == cid
        if is_green_bg and float(np.mean(is_green_soft[comp])) > 0.25:
            continue
        if not is_green_bg and float(np.mean(is_green_strong[comp])) > 0.35:
            continue
        ring = (cv2.dilate(comp.astype(np.uint8), k5) > 0) & (a_f >= 0.45)
        if not np.any(ring) or float(np.mean(a_f[ring])) < 0.72:
            continue
        if bg_q is not None:
            hole_med = np.median(lab[comp], axis=0)
            d_bg = float(np.min(np.linalg.norm((bg_q - hole_med[None, :]) * w_lab[None, :], axis=1)))
            fg_med = np.median(lab[ring], axis=0)
            d_fg = float(np.linalg.norm((fg_med - hole_med) * w_lab))
            if d_bg < d_fg * 0.95 or d_bg < 22.0:
                continue
        pinhole_mask[comp] = 1

    pin_px = int(np.sum(pinhole_mask > 0))
    if pin_px > 0:
        a_pin = cv2.GaussianBlur(pinhole_mask.astype(np.float32), (3, 3), 0.6)
        a_f = np.maximum(a_f, np.clip(a_pin, 0.0, 1.0))

    # Stage 3: Trapped Green-Screen Crevice Punch-Out (only on green-screen scenes)
    crevice_px = 0
    punch_mask = np.zeros((h, w), dtype=np.uint8)
    if is_green_bg:
        from src.relight_harmonizer import detect_emissive_screen_mask

        ext_strong_green = ext_zone & (g_exc > 15.0) & (hsv[..., 0] >= 32) & (hsv[..., 0] <= 96) & (hsv[..., 1] >= 35)
        if np.any(ext_strong_green):
            ext_g_med = float(np.median(g_exc[ext_strong_green]))
            ext_s_med = float(np.median(hsv[..., 1][ext_strong_green]))
            ext_a_vals = lab[..., 1][ext_strong_green]
            ext_b_vals = lab[..., 2][ext_strong_green]
            ext_a_p85 = float(np.percentile(ext_a_vals, 85))
            v_ext = np.array([float(np.median(ext_a_vals)) - 128.0, float(np.median(ext_b_vals)) - 128.0], dtype=np.float32)
        else:
            ext_g_med = 40.0
            ext_s_med = 80.0
            ext_a_p85 = 110.0
            v_ext = np.array([-30.0, 20.0], dtype=np.float32)
        v_ext_norm = max(float(np.linalg.norm(v_ext)), 1e-4)

        fg_bin_now = (a_f >= 0.45).astype(np.uint8)
        fg_area = max(1, int(np.sum(fg_bin_now > 0)))
        screen_raw = detect_emissive_screen_mask(bgr, fg_bin_now)
        screen_protect = cv2.dilate(screen_raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0

        min_g_exc = max(15.0, min(32.0, 0.25 * ext_g_med))
        min_sat = max(42.0, min(68.0, 0.35 * ext_s_med))
        max_lab_a = 115.0

        crevice_cand = (
            (a_f > 0.04)
            & (~screen_protect)
            & (g_exc > min_g_exc)
            & (hsv[..., 0] >= 36)
            & (hsv[..., 0] <= 90)
            & (hsv[..., 1] >= min_sat)
            & (lab[..., 1] <= max_lab_a)
        )
        if np.any(crevice_cand):
            ext_loose = (a_f <= 0.12).astype(np.uint8)
            near_ext = cv2.dilate(ext_loose, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (85, 85))) > 0
            k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            n_cr, l_cr, st_cr, _ = cv2.connectedComponentsWithStats(crevice_cand.astype(np.uint8), connectivity=8)
            max_cr_area = max(48, int(0.06 * fg_area))
            for cid in range(1, n_cr):
                c_area = int(st_cr[cid, cv2.CC_STAT_AREA])
                if c_area < 32:
                    continue
                comp = l_cr == cid
                comp_g = float(np.median(g_exc[comp]))
                comp_s = float(np.median(hsv[..., 1][comp]))
                comp_a = float(np.median(lab[..., 1][comp]))
                if comp_g < 18.0 or comp_s < 46.0 or comp_a > 114.0:
                    continue

                v_comp = np.array([comp_a - 128.0, float(np.median(lab[..., 2][comp])) - 128.0], dtype=np.float32)
                v_comp_norm = max(float(np.linalg.norm(v_comp)), 1e-4)
                cos_ext = float(np.dot(v_comp, v_ext) / (v_comp_norm * v_ext_norm))
                if cos_ext < 0.70:
                    continue

                fg_ring = (cv2.dilate(comp.astype(np.uint8), k9) > 0) & (a_f >= 0.5) & (~crevice_cand)
                if np.any(fg_ring):
                    fg_g_med = float(np.median(g_exc[fg_ring]))
                    if comp_g <= fg_g_med + 10.0:
                        continue

                if np.any(comp & near_ext) or c_area <= max_cr_area:
                    punch_mask[comp] = 1

            crevice_px = int(np.sum(punch_mask > 0))
            if crevice_px > 0:
                punch_soft = cv2.GaussianBlur(punch_mask.astype(np.float32), (3, 3), 0.7)
                a_f = a_f * (1.0 - np.clip(punch_soft * 1.15, 0.0, 1.0))

    total_healed = bezel_px + pin_px + crevice_px
    combined = ((bezel_mask > 0) | (pinhole_mask > 0) | (punch_mask > 0)).astype(np.uint8)
    return np.clip(a_f, 0.0, 1.0).astype(np.float32), total_healed, combined


def audit_and_heal_video_mattes(
    frame_paths: List[Path],
    matte_paths: List[Path],
    *,
    is_alpha_stream: bool,
    sample_ratio: float = 0.10,
    min_samples: int = 4,
    max_samples: int = 24,
    use_birefnet_crosscheck: bool = True,
    crosscheck_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    heal_repropagate_cb: Optional[Callable[[Dict[int, np.ndarray]], None]] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    confirmed_frame_indices: Optional[set[int]] = None,
) -> Dict[str, Any]:
    """
    Stratified 10% Multi-Model QA Audit & Closed-Loop Auto-Healing:
      Stage 1: 100% fast temporal continuity scan (downsampled) to catch frame-to-frame dropouts/spikes.
      Stage 2: Cross-audit ~10% stratified frames (plus temporal outliers) against BiRefNet + Bezel Guard.
      Stage 3: Automatically heal missing chunks / bezel notches / excess leaks and re-propagate repaired keyframes.
    """
    n_frames = min(len(frame_paths), len(matte_paths))
    if n_frames == 0:
        return {
            "audited_count": 0,
            "total_frames": 0,
            "sample_pct": 0.0,
            "healed_frames": [],
            "flagged_frames": 0,
            "integrity_score": 100.0,
            "summary": "Không có khung hình để kiểm tra QA.",
            "samples": [],
        }

    # Stage 1: 100% ultra-fast downsampled temporal area + IoU scan
    if progress_cb:
        progress_cb(0.05, "🛡️ QA Quét 100% đường cong diện tích theo thời gian...")
    areas: List[float] = []
    ious: List[float] = [1.0]
    prev_small: Optional[np.ndarray] = None
    for t in range(n_frames):
        p = matte_paths[t]
        m_u8 = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m_u8 is None:
            areas.append(0.0)
            ious.append(0.0 if t > 0 else 1.0)
            continue
        sh, sw = max(24, m_u8.shape[0] // 4), max(24, m_u8.shape[1] // 4)
        small = cv2.resize(m_u8, (sw, sh), interpolation=cv2.INTER_AREA) >= 115
        areas.append(float(np.mean(small)))
        if prev_small is not None:
            inter = float(np.sum(small & prev_small))
            union = max(1.0, float(np.sum(small | prev_small)))
            ious.append(inter / union)
        prev_small = small

    temporal_outliers: List[int] = []
    arr_a = np.asarray(areas, dtype=np.float32)
    for t in range(n_frames):
        w_lo, w_hi = max(0, t - 3), min(n_frames, t + 4)
        loc_med = float(np.median(arr_a[w_lo:w_hi]))
        if loc_med > 1e-4:
            if arr_a[t] < loc_med * 0.94 or arr_a[t] > loc_med * 1.07 or ious[t] < 0.93:
                temporal_outliers.append(t)

    # Stage 2: Stratified ~10% frames + up to 6 temporal outliers
    k_strat = int(round(n_frames * float(np.clip(sample_ratio, 0.02, 0.50))))
    k_strat = max(min(min_samples, n_frames), min(max_samples, k_strat))
    strat_indices = np.unique(np.linspace(0, n_frames - 1, num=k_strat, dtype=int)).tolist()
    outlier_extra = sorted(
        temporal_outliers,
        key=lambda t: (ious[t] if t < len(ious) else 1.0),
    )[:6]
    audit_indices = sorted(set(strat_indices) | set(outlier_extra))

    if crosscheck_fn is None and use_birefnet_crosscheck:
        from src.neural_matting import release_neural_models, run_birefnet_matting

        release_neural_models()
        crosscheck_fn = lambda bgr: run_birefnet_matting(bgr, max_size=1024)

    healed_keyframes: Dict[int, np.ndarray] = {}
    samples_report: List[Dict[str, Any]] = []
    healed_frame_ids: List[int] = []

    try:
        for i_step, t in enumerate(audit_indices):
            if progress_cb:
                progress_cb(
                    0.15 + 0.55 * ((i_step + 1) / max(1, len(audit_indices))),
                    f"🛡️ QA kiểm tra chéo frame {t} ({i_step + 1}/{len(audit_indices)})...",
                )
            bgr = cv2.imread(str(frame_paths[t]))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            alpha_t = load_alpha(matte_paths[t]) if is_alpha_stream else load_mask(matte_paths[t]).astype(np.float32)

            # A. Universal structural guard (interior pinholes + rigid device bezels if present)
            alpha_bezel, bezel_px, bezel_mask = heal_universal_structural_defects(bgr, alpha_t)

            # B. Independent BiRefNet cross-check for adjacent dropped subject parts or leaked background blobs
            # Guarded by size caps so custom boxed subjects never pull in unrelated nearby objects
            missing_mask = np.zeros((h, w), dtype=np.uint8)
            excess_mask = np.zeros((h, w), dtype=np.uint8)
            if crosscheck_fn is not None:
                alpha_ref = np.clip(crosscheck_fn(bgr).astype(np.float32), 0.0, 1.0)
                fg_bin = (alpha_t >= 0.45).astype(np.uint8)
                fg_area_cur = max(1, int(np.sum(fg_bin > 0)))
                fg_near = cv2.dilate(fg_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0

                # Missing chunk: BiRefNet confident (>= 0.68), tracker dropped (< 0.35), touching/near tracked subject, <= 35% of FG
                miss_cand = ((alpha_ref >= 0.68) & (alpha_t < 0.35) & fg_near).astype(np.uint8)
                min_miss_area = max(140, int(h * w * 0.00025))
                max_miss_area = max(min_miss_area + 100, int(fg_area_cur * 0.35))
                nm, lm, sm, _ = cv2.connectedComponentsWithStats(miss_cand, connectivity=8)
                for cid in range(1, nm):
                    area_c = int(sm[cid, cv2.CC_STAT_AREA])
                    if min_miss_area <= area_c <= max_miss_area:
                        missing_mask[lm == cid] = 1

                # Excess leak: skip erasing confirmed keyframe choices; only trim peripheral blobs <= 40% of FG
                if confirmed_frame_indices is None or int(t) not in confirmed_frame_indices:
                    dev_core, skin, _, _ = _detect_device_core_and_priors(bgr, fg_bin)
                    dev_protect = cv2.dilate(dev_core, cv2.getStructuringElement(cv2.MORPH_RECT, (37, 37))) > 0
                    exc_cand = ((alpha_t >= 0.60) & (alpha_ref < 0.12) & (~dev_protect) & (~skin)).astype(np.uint8)
                    min_exc_area = max(280, int(h * w * 0.0007))
                    max_exc_area = max(min_exc_area + 200, int(fg_area_cur * 0.40))
                    ne, le, se, _ = cv2.connectedComponentsWithStats(exc_cand, connectivity=8)
                    for cid in range(1, ne):
                        area_e = int(se[cid, cv2.CC_STAT_AREA])
                        if min_exc_area <= area_e <= max_exc_area:
                            excess_mask[le == cid] = 1

            repaired_total = ((bezel_mask > 0) | (missing_mask > 0)).astype(np.uint8)
            miss_px = int(np.sum(repaired_total > 0))
            exc_px = int(np.sum(excess_mask > 0))

            fg_area = max(1, int(np.sum((alpha_t >= 0.45) | (repaired_total > 0))))
            defect_ratio = (miss_px + exc_px) / float(fg_area)
            score = round(max(0.0, min(100.0, (1.0 - defect_ratio * 2.5) * 100.0)), 1)

            needs_heal = (int(np.sum(missing_mask)) >= 140) or (bezel_px >= 60) or (exc_px >= 280)
            if needs_heal:
                healed_a = alpha_bezel.copy()
                if np.any(missing_mask > 0):
                    miss_soft = cv2.GaussianBlur(missing_mask.astype(np.float32), (5, 5), 1.0)
                    healed_a = np.maximum(healed_a, miss_soft)
                if np.any(excess_mask > 0):
                    exc_soft = cv2.GaussianBlur(excess_mask.astype(np.float32), (5, 5), 1.0)
                    healed_a = np.clip(healed_a * (1.0 - exc_soft), 0.0, 1.0)

                healed_keyframes[t] = (healed_a >= 0.42).astype(np.uint8)
                healed_frame_ids.append(t)
                if is_alpha_stream:
                    cv2.imwrite(str(matte_paths[t]), np.round(np.clip(healed_a, 0.0, 1.0) * 255.0).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])
                else:
                    cv2.imwrite(str(matte_paths[t]), ((healed_a >= 0.42).astype(np.uint8) * 255))

            diff_b64 = ""
            if needs_heal or len(samples_report) < 4:
                diff_b64 = _encode_diff_thumb_b64(bgr, repaired_total, excess_mask)

            samples_report.append(
                {
                    "frame_idx": int(t),
                    "status": "healed" if needs_heal else "ok",
                    "missing_px": miss_px,
                    "excess_px": exc_px,
                    "bezel_notch_px": int(bezel_px),
                    "integrity_score": score,
                    "diff_b64": diff_b64,
                }
            )
    finally:
        if use_birefnet_crosscheck:
            try:
                from src.neural_matting import release_neural_models

                release_neural_models()
            except Exception:
                pass

    # Stage 3: Re-propagate repaired keyframes across neighboring frames
    if healed_keyframes and heal_repropagate_cb is not None:
        if progress_cb:
            progress_cb(0.75, f"🩹 Đang lan truyền sửa lỗi {len(healed_keyframes)} vùng khuyết sang các frame lân cận...")
        heal_repropagate_cb(healed_keyframes)

    mean_score = round(float(np.mean([s["integrity_score"] for s in samples_report])), 1) if samples_report else 100.0
    flagged = len(healed_frame_ids)
    # Show flagged frames first, then representative ok frames (max 8 thumbnails to keep JSON light)
    ordered_samples = sorted(samples_report, key=lambda s: (0 if s["status"] == "healed" else 1, s["frame_idx"]))[:8]
    summary = (
        f"QA đã kiểm tra {len(audit_indices)}/{n_frames} frame (~{round(len(audit_indices)/max(1,n_frames)*100)}%): "
        + (f"phát hiện & tự vá {flagged} khung hình khuyết/thừa (điểm toàn vẹn {mean_score}%)." if flagged else f"100% khung hình tròn vẹn ({mean_score}%).")
    )
    if progress_cb:
        progress_cb(1.0, summary)

    return {
        "audited_count": len(audit_indices),
        "total_frames": n_frames,
        "sample_pct": round((len(audit_indices) / max(1, n_frames)) * 100.0, 1),
        "healed_frames": healed_frame_ids,
        "flagged_frames": flagged,
        "integrity_score": mean_score,
        "summary": summary,
        "samples": ordered_samples,
    }

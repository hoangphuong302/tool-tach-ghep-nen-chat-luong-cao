from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MattingConfig:
    trimap_radius: int = 2              # Sub-pixel transition half-width in pixels (1..24)
    guided_radius: int = 2              # Joint Guided Filter radius (1..10)
    guided_eps: float = 5e-4            # Guided filter regularization
    decontaminate_strength: float = 0.90  # Edge color decontamination / despill strength (0..1)
    despill_green: bool = False         # Extra green/blue chroma spill suppression
    erode_dilate: int = 1               # Default 1 px outward margin; negative values still trim
    two_pass_verify: bool = True        # Enable Pass-2 Closed-Loop Pixel Verification & Compensation
    use_vitmatte_gpu: bool = False      # Optional ViTMatte neural transformer matting on trimap
    engine: Literal["classic", "vitmatte", "birefnet", "auto"] = "classic"
    use_pymatting_fg: bool = False      # Physical PyMatting multi-level foreground estimation (halo-free)


@dataclass
class MattingResult:
    alpha: np.ndarray                  # float32 (H, W) in [0.0, 1.0] with sub-pixel edge precision
    fg_clean_bgr: np.ndarray           # uint8 (H, W, 3) decontaminated foreground BGR
    rgba: np.ndarray                   # uint8 (H, W, 4) transparent RGBA for export
    trimap: np.ndarray                 # uint8 (H, W): 0=BG, 128=Unknown, 255=FG
    missing_restored_px: int = 0       # Pass-2 Audit: count of missing subject pixels restored
    excess_removed_px: int = 0         # Pass-2 Audit: count of excess background pixels removed
    audit_diff_rgb: Optional[np.ndarray] = None  # Visual map: Green=Restored, Red=Removed


def expand_mask_with_color_evidence(
    bgr: np.ndarray,
    mask: np.ndarray,
    pixels: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow a mask by a small margin only where source pixels fit the foreground edge."""
    grown = (mask > 0).astype(np.uint8)
    recovered_silver = np.zeros_like(grown, dtype=bool)
    if pixels <= 0 or not np.any(grown):
        return grown, recovered_silver

    ys, xs = np.where(grown > 0)
    pad = min(32, int(pixels) + 8)
    y0, y1 = max(0, int(ys.min()) - pad), min(grown.shape[0], int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(grown.shape[1], int(xs.max()) + pad + 1)
    grown_roi = grown[y0:y1, x0:x1].copy()
    silver_roi = np.zeros_like(grown_roi, dtype=bool)
    bgr_roi = bgr[y0:y1, x0:x1]

    lab = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    r_ch, g_ch, b_ch = (bgr_roi[..., c].astype(np.float32) for c in (2, 1, 0))
    skin = (
        (cr >= 135) & (cr <= 178) & (cb >= 77) & (cb <= 129)
        & ((hue <= 17) | (hue >= 164)) & ((r_ch - g_ch) >= 11.0)
    )
    neutral_silver = (
        (sat < 30) & (val > 130) & (np.abs(r_ch - b_ch) < 15.0)
    )
    lab_l = lab[..., 0]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(min(int(pixels), 6)):
        candidate = (cv2.dilate(grown_roi, ker) > 0) & (grown_roi == 0)
        if not np.any(candidate):
            break
        inner_seed = cv2.erode(grown_roi, ker) > 0
        if not np.any(inner_seed):
            inner_seed = grown_roi > 0
        outer_seed = cv2.dilate(grown_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) == 0
        if not np.any(outer_seed):
            break
        fg_est = _propagate_local_colors(lab / 255.0, inner_seed, sigma=4.0) * 255.0
        bg_est = _propagate_local_colors(lab / 255.0, outer_seed, sigma=4.0) * 255.0
        d_fg = np.linalg.norm(lab - fg_est, axis=-1)
        d_bg = np.linalg.norm(lab - bg_est, axis=-1)

        # A confident local foreground-color fit protects fingers and dark bezel edges.
        local_subject = d_fg < d_bg * 0.80
        local_skin = skin & (d_fg < d_bg * 1.10)
        # Neutral bright trim is recovered only when it is locally brighter than the
        # old background field, which rejects beige/gray background pixels.
        silver_edge = neutral_silver & (lab_l > bg_est[..., 0] + 15.0)
        bg_match = (d_bg < 9.0) & (d_fg > d_bg * 1.60 + 4.5)
        accept = candidate & (local_subject | local_skin | silver_edge) & ~bg_match
        if not np.any(accept):
            break
        silver_roi |= accept & silver_edge
        grown_roi[accept] = 1

    grown[y0:y1, x0:x1] = grown_roi
    recovered_silver[y0:y1, x0:x1] = silver_roi
    return grown, recovered_silver


def detect_active_frame_box(bgr: np.ndarray, threshold: int = 14) -> Tuple[int, int, int, int]:
    """
    Detect active content bounding box (x0, y0, x1, y1) inside a letterboxed or
    pillarboxed frame (e.g., 9:16 vertical video screenshot inside 16:9 black bars).
    """
    h, w = bgr.shape[:2]
    gray = np.max(bgr, axis=2)
    col_p95 = np.percentile(gray, 95, axis=0)
    row_p95 = np.percentile(gray, 95, axis=1)

    active_cols = np.where(col_p95 > threshold)[0]
    active_rows = np.where(row_p95 > threshold)[0]

    if len(active_cols) < 16 or len(active_rows) < 16:
        return (0, 0, w, h)

    x0, x1 = int(active_cols[0]), int(active_cols[-1] + 1)
    y0, y1 = int(active_rows[0]), int(active_rows[-1] + 1)

    if x0 <= int(w * 0.02) and x1 >= int(w * 0.98) and y0 <= int(h * 0.02) and y1 >= int(h * 0.98):
        return (0, 0, w, h)
    return (x0, y0, x1, y1)


def auto_seed_foreground_points(bgr: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Automatically place smart positive (+1) points on central handheld device & hands
    and negative (0) points on the background so 1-click segmentation works immediately.
    Returns list of (x, y, label).
    """
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = detect_active_frame_box(bgr)
    aw, ah = max(1, x1 - x0), max(1, y1 - y0)

    pts: list[tuple[int, int, int]] = []
    # 1. Central product / handheld device points
    cx = x0 + aw // 2
    pts.append((cx, y0 + int(ah * 0.50), 1))
    pts.append((cx, y0 + int(ah * 0.64), 1))

    # 2. Detect skin regions (hands holding the object) in the lower 75% of the active frame
    active_bgr = bgr[y0:y1, x0:x1]
    ycrcb = cv2.cvtColor(active_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    skin = (cr >= 135) & (cr <= 175) & (cb >= 77) & (cb <= 127)
    skin[: int(ah * 0.30), :] = False  # Ignore faces/upper background

    # Split left half and right half of active frame to seed left hand and right hand
    mid = aw // 2
    for x_offset, half_mask in [(0, skin[:, :mid]), (mid, skin[:, mid:])]:
        ys, xs = np.where(half_mask)
        if len(xs) > int(aw * ah * 0.008):
            mx = int(np.median(xs)) + x_offset + x0
            my = int(np.median(ys)) + y0
            pts.append((mx, my, 1))

    # 3. Negative background anchor points (upper corners, top-center, and side background)
    pts.append((x0 + int(aw * 0.12), y0 + int(ah * 0.12), 0))
    pts.append((x0 + int(aw * 0.50), y0 + int(ah * 0.10), 0))
    pts.append((x0 + int(aw * 0.88), y0 + int(ah * 0.12), 0))
    pts.append((x0 + int(aw * 0.10), y0 + int(ah * 0.42), 0))
    pts.append((x0 + int(aw * 0.90), y0 + int(ah * 0.42), 0))
    return pts


def auto_seed_roto_strokes(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate smart default Roto-Brush foreground (green) and background (red) stroke masks
    covering the central device from top bezel to bottom bezel plus both hands.
    Returns (fg_stroke_u8, bg_stroke_u8) of shape (H, W) in {0, 1}.
    """
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = detect_active_frame_box(bgr)
    aw, ah = max(1, x1 - x0), max(1, y1 - y0)

    fg_stroke = np.zeros((h, w), dtype=np.uint8)
    bg_stroke = np.zeros((h, w), dtype=np.uint8)

    # 1. Find central dark/electronic device bounding box strictly connected to the device center
    active_bgr = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(active_bgr, cv2.COLOR_BGR2HSV)
    b_ch = active_bgr[..., 0].astype(np.float32)
    r_ch = active_bgr[..., 2].astype(np.float32)
    # Device bezel & screen: dark chassis (V < 65) or glowing blue UI (B > R + 18)
    dev_cand = ((hsv[..., 2] < 65) | ((b_ch > r_ch + 18) & (hsv[..., 1] > 45))).astype(np.uint8)
    search_mask = np.zeros_like(dev_cand)
    search_mask[int(ah * 0.16): int(ah * 0.84), int(aw * 0.24): int(aw * 0.76)] = 1
    dev_cand &= search_mask
    # Open slightly to sever thin connections to blurred background signs, then keep ONLY the center component
    dev_cand = cv2.morphologyEx(dev_cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    dev_cand = cv2.morphologyEx(dev_cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))

    num_cc, labels_cc, stats, _ = cv2.connectedComponentsWithStats(dev_cand, connectivity=8)
    center_lbl = labels_cc[int(ah * 0.55), aw // 2]
    if center_lbl == 0 and num_cc > 1:
        # Pick largest component in center search zone
        center_lbl = max(range(1, num_cc), key=lambda c: stats[c, cv2.CC_STAT_AREA])
    if center_lbl > 0:
        dev_cand = (labels_cc == center_lbl).astype(np.uint8)

    device_core = dev_cand.copy()
    top_rim = np.zeros_like(dev_cand)
    if np.any(device_core):
        top_x0, top_x1 = np.where(device_core > 0)[1].min(), np.where(device_core > 0)[1].max() + 1
        for x in range(top_x0, top_x1):
            ys = np.where(device_core[:, x] > 0)[0]
            if not len(ys):
                continue
            top_y = int(ys[0])
            y_start = max(0, top_y - 12)
            strip = active_bgr[y_start:top_y, x]
            if not len(strip):
                continue
            strip_hsv = hsv[y_start:top_y, x]
            neutral_edge = (
                (strip_hsv[:, 2] >= 160) & (strip_hsv[:, 1] <= 85)
                & (np.max(strip, axis=1).astype(np.int16) - np.min(strip, axis=1).astype(np.int16) <= 80)
            )
            candidates = np.where(neutral_edge)[0]
            if len(candidates):
                top_rim[y_start + int(candidates[-1]), x] = 1

    # Expand dev_cand horizontally by up to 34px to include bright white/silver metallic side bezels
    # and red power buttons immediately adjacent to the dark front bezel (excluding green screen & skin)
    dil_side = cv2.dilate(dev_cand, cv2.getStructuringElement(cv2.MORPH_RECT, (65, 9)), iterations=1)
    g_ch_f = active_bgr[..., 1].astype(np.float32)
    is_green_px = (g_ch_f > np.maximum(r_ch, b_ch) + 6.0) & (hsv[..., 0] >= 30) & (hsv[..., 0] <= 98) & (hsv[..., 1] >= 25)
    # White/silver aluminum bezel (high V, low-medium S, not green) or red side button (R >> G)
    is_side_bezel = (
        (dil_side > 0)
        & (~is_green_px)
        & (
            ((hsv[..., 2] >= 125) & (hsv[..., 1] <= 65))  # White/silver side chassis
            | ((r_ch > g_ch_f + 45.0) & (hsv[..., 1] >= 90))  # Red/orange side button
        )
    ).astype(np.uint8)
    dev_cand = cv2.morphologyEx(dev_cand | is_side_bezel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))

    ys_d, xs_d = np.where(dev_cand > 0)
    cx = x0 + aw // 2
    if len(ys_d) > int(aw * ah * 0.01):
        dy0_rel = int(np.percentile(ys_d, 6))
        dy1_rel = int(np.percentile(ys_d, 94))
        dym_rel = (dy0_rel + dy1_rel) // 2

        def _row_span(y_rel: int) -> tuple[int, int, int, int]:
            y_lo = max(0, y_rel - 8)
            y_hi = min(ah, y_rel + 9)
            r_xs = np.where(np.any(dev_cand[y_lo:y_hi, :] > 0, axis=0))[0]
            if len(r_xs) >= 10:
                rx0 = x0 + int(np.percentile(r_xs, 14))
                rx1 = x0 + int(np.percentile(r_xs, 88))
                rx_edge_r = x0 + int(np.percentile(r_xs, 96))
            else:
                rx0 = x0 + int(np.percentile(xs_d, 18))
                rx1 = x0 + int(np.percentile(xs_d, 84))
                rx_edge_r = x0 + int(np.percentile(xs_d, 95))
            return rx0, rx1, (rx0 + rx1) // 2, rx_edge_r

        tx0, tx1, tcx, trx = _row_span(dy0_rel + 14)
        mx0, mx1, mcx, mrx = _row_span(dym_rel)
        bx0, bx1, bcx, brx = _row_span(dy1_rel - 14)

        dy0 = y0 + dy0_rel
        dy1 = y0 + dy1_rel
        dx0 = x0 + int(np.percentile(xs_d, 15))
        dx1 = x0 + int(np.percentile(xs_d, 88))

        # Draw tilted spine + row-local top/mid/bottom bars + side-bezel anchor line inside the device
        cv2.line(fg_stroke, (tcx, dy0 + 12), (bcx, dy1 - 12), 1, thickness=12)
        cv2.line(fg_stroke, (tx0, dy0 + 14), (tx1, dy0 + 14), 1, thickness=8)
        cv2.line(fg_stroke, (mx0, y0 + dym_rel), (mx1, y0 + dym_rel), 1, thickness=8)
        cv2.line(fg_stroke, (bx0, dy1 - 14), (bx1, dy1 - 14), 1, thickness=8)
        # Anchor stroke along the right side bezel so white/silver chassis is 100% included
        cv2.line(fg_stroke, (trx - 4, dy0 + 22), (mrx - 4, y0 + dym_rel), 1, thickness=6)
        rim_local = cv2.dilate(top_rim, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        fg_stroke[y0:y1, x0:x1] |= rim_local

        if dy0 - 22 > y0 + 10:
            cv2.line(bg_stroke, (tx0 - 15, dy0 - 22), (tx1 + 15, dy0 - 22), 1, thickness=8)
    else:
        cv2.line(fg_stroke, (cx, y0 + int(ah * 0.36)), (cx, y0 + int(ah * 0.72)), 1, thickness=14)

    device_fg = fg_stroke.copy()
    # 2. Detect true human skin hands (excluding yellow-hue wooden tables/desks!) in lower 72%
    ycrcb = cv2.cvtColor(active_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[..., 1].astype(np.int32), ycrcb[..., 2].astype(np.int32)
    hue = hsv[..., 0].astype(np.int32)
    g_ch = active_bgr[..., 1].astype(np.float32)

    # Human skin is pink/peach (Hue in 0..17 or 164..180, with R - G >= 11)
    # whereas yellow/brown wooden tables have Hue in 18..35 and G - B >> R - G
    skin = (
        (cr >= 135) & (cr <= 178) & (cb >= 77) & (cb <= 129)
        & ((hue <= 17) | (hue >= 164))
        & ((r_ch - g_ch) >= 11.0)
    ).astype(np.uint8)
    gray = cv2.cvtColor(active_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.boxFilter(gray, cv2.CV_32F, (17, 17))
    local_sq_mean = cv2.boxFilter(gray * gray, cv2.CV_32F, (17, 17))
    local_texture = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
    skin[local_texture < 2.0] = 0
    skin[: int(ah * 0.34), :] = 0
    skin = cv2.erode(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)

    # Detect wooden table/desk surface in lower half (Hue 18..35, G - B > 1.3 * (R - G))
    wood_table = (
        (cr >= 130) & (cb <= 116) & (hue >= 18) & (hue <= 35)
        & ((g_ch - b_ch) > (r_ch - g_ch) * 1.25)
    ).astype(np.uint8)
    wood_table[: int(ah * 0.55), :] = 0
    wood_table = cv2.erode(wood_table, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)

    mid = aw // 2
    for x_off, half in [(0, skin[:, :mid]), (mid, skin[:, mid:])]:
        ys_s, xs_s = np.where(half > 0)
        if len(xs_s) > int(aw * ah * 0.006):
            p_top = (x0 + x_off + int(np.median(xs_s[ys_s < np.median(ys_s)])), y0 + int(np.percentile(ys_s, 20)))
            p_mid = (x0 + x_off + int(np.median(xs_s)), y0 + int(np.percentile(ys_s, 55)))
            p_bot = (x0 + x_off + int(np.median(xs_s[ys_s >= np.median(ys_s)])), y0 + int(np.percentile(ys_s, 88)))
            cv2.line(fg_stroke, p_top, p_mid, 1, thickness=12)
            cv2.line(fg_stroke, p_mid, p_bot, 1, thickness=12)

    # Also check for fingers wrapping directly underneath the device bottom edge (dy1..dy1+110)
    if len(ys_d) > int(aw * ah * 0.01):
        under_y0 = min(ah - 1, max(0, (dy1 - y0) + 10))
        under_y1 = min(ah, under_y0 + int(ah * 0.12))
        under_x0 = max(0, (dx0 - x0))
        under_x1 = min(aw, (dx1 - x0))
        under_skin = skin[under_y0:under_y1, under_x0:under_x1]
        u_ys, u_xs = np.where(under_skin > 0)
        if len(u_xs) > int(aw * ah * 0.002):
            ux_l = x0 + under_x0 + int(np.percentile(u_xs, 20))
            uy_l = y0 + under_y0 + int(np.percentile(u_ys, 45))
            ux_r = x0 + under_x0 + int(np.percentile(u_xs, 80))
            uy_r = y0 + under_y0 + int(np.percentile(u_ys, 65))
            cv2.line(fg_stroke, (ux_l, uy_l), (ux_r, uy_r), 1, thickness=10)

    fg_stroke = device_fg | (fg_stroke & skin)

    # If a wooden table is present in the lower-left or lower-right background, mark it as negative background
    for x_off, w_half in [(0, wood_table[:, :mid]), (mid, wood_table[:, mid:])]:
        wys, wxs = np.where(w_half > 0)
        if len(wxs) > int(aw * ah * 0.015):
            wx0 = x0 + x_off + int(np.percentile(wxs, 25))
            wy0 = y0 + int(np.percentile(wys, 35))
            wx1 = x0 + x_off + int(np.percentile(wxs, 65))
            wy1 = y0 + int(np.percentile(wys, 85))
            cv2.line(bg_stroke, (wx0, wy0), (wx1, wy1), 1, thickness=12)

    # If green screen pixels exist (G > max(R, B) + 28), place negative anchors on any interior green gaps
    green_bg = ((g_ch > np.maximum(r_ch, b_ch) + 28.0) & (hsv[..., 1] > 45)).astype(np.uint8)
    green_bg = cv2.erode(green_bg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
    if np.sum(green_bg) > int(aw * ah * 0.05):
        # Sample negative background lines on strong green zones outside device/skin
        bg_stroke[y0:y1, x0:x1][green_bg[::1, ::1] > 0] = 0  # handled sparsely below
        for gy_frac in (0.55, 0.65, 0.72):
            gy = int(ah * gy_frac)
            g_xs = np.where(green_bg[gy, :] > 0)[0]
            if len(g_xs) > 10:
                cv2.circle(bg_stroke, (x0 + int(g_xs[len(g_xs) // 2]), y0 + gy), 8, 1, -1)
                cv2.circle(bg_stroke, (x0 + int(g_xs[-5]), y0 + gy), 8, 1, -1)

    # 3. Background exclusion stroke around upper & upper-side perimeter + bottom-center gap between hands
    cv2.line(bg_stroke, (x0 + int(aw * 0.08), y0 + int(ah * 0.08)), (x0 + int(aw * 0.92), y0 + int(ah * 0.08)), 1, thickness=12)
    cv2.line(bg_stroke, (x0 + int(aw * 0.06), y0 + int(ah * 0.08)), (x0 + int(aw * 0.06), y0 + int(ah * 0.48)), 1, thickness=10)
    cv2.line(bg_stroke, (x0 + int(aw * 0.94), y0 + int(ah * 0.08)), (x0 + int(aw * 0.94), y0 + int(ah * 0.48)), 1, thickness=10)
    # Check if lower-center bottom edge (under the device between the two hands) is non-skin background
    bot_y = int(ah * 0.92)
    if skin[bot_y, aw // 2] == 0:
        cv2.line(bg_stroke, (x0 + int(aw * 0.44), y0 + bot_y), (x0 + int(aw * 0.56), y0 + bot_y), 1, thickness=10)
    # Auto-generated exclusion strokes can cross a tilted display. Protect only
    # the detected device component; explicit editor strokes are handled later.
    if len(ys_d) > int(aw * ah * 0.01):
        device_interior = cv2.morphologyEx(
            dev_cand, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        )
        bg_stroke[y0:y1, x0:x1][device_interior > 0] = 0
    return fg_stroke, bg_stroke


def extract_strokes_from_editor(
    editor_val: Optional[dict],
    orig_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse Gradio ImageEditor output dict {"background", "layers", "composite"} into
    binary foreground stroke mask (green brush) and background stroke mask (red brush).
    """
    h, w = orig_rgb.shape[:2]
    fg_stroke = np.zeros((h, w), dtype=np.uint8)
    bg_stroke = np.zeros((h, w), dtype=np.uint8)
    if not editor_val or not isinstance(editor_val, dict):
        return fg_stroke, bg_stroke

    layers = editor_val.get("layers") or []
    for layer in layers:
        if layer is None:
            continue
        arr = np.asarray(layer)
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        if arr.ndim == 3 and arr.shape[2] == 4:
            r, g, b, a = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int), arr[..., 3].astype(int)
            painted = a > 25
            fg_stroke[(painted) & (g > r + 25) & (g > b - 10)] = 1
            bg_stroke[(painted) & (r > g + 25) & (r > b + 20)] = 1
        elif arr.ndim == 3 and arr.shape[2] == 3:
            diff = np.max(np.abs(arr.astype(int) - orig_rgb.astype(int)), axis=-1) > 25
            r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
            fg_stroke[diff & (g > r + 25)] = 1
            bg_stroke[diff & (r > g + 25)] = 1

    # Fallback: check composite vs orig_rgb if layers list was empty
    if "layers" not in editor_val and editor_val.get("composite") is not None:
        comp = np.asarray(editor_val["composite"])[..., :3]
        if comp.shape[:2] != (h, w):
            comp = cv2.resize(comp, (w, h), interpolation=cv2.INTER_NEAREST)
        diff = np.max(np.abs(comp.astype(int) - orig_rgb.astype(int)), axis=-1) > 28
        r, g, b = comp[..., 0].astype(int), comp[..., 1].astype(int), comp[..., 2].astype(int)
        fg_stroke[diff & (g > r + 25)] = 1
        bg_stroke[diff & (r > g + 25)] = 1

    return fg_stroke, bg_stroke


def sample_roto_anchor_points(
    fg_stroke: np.ndarray,
    bg_stroke: np.ndarray,
    bgr: np.ndarray,
    grid_step: int = 36,
    *,
    preserve_stroke_components: bool = False,
) -> list[tuple[int, int, int]]:
    """
    Convert Roto-Brush painted strokes into dense, spatially distributed positive (1)
    and negative (0) anchor points for SAM 2.1 multi-part snapping.
    """
    h, w = fg_stroke.shape[:2]
    pts: list[tuple[int, int, int]] = []

    def append_cell_samples(cell: np.ndarray, x0: int, y0: int, label: int) -> None:
        ys, xs = np.where(cell > 0)
        if len(xs) == 0:
            return
        if not preserve_stroke_components:
            # Keep the legacy auto sampler's exact output unless the caller
            # explicitly requests raster-valid manual brush prompts.
            pts.append((x0 + int(np.median(xs)), y0 + int(np.median(ys)), label))
            return

        # Each disconnected brush mark in this cell gets a prompt. Choose the
        # actual painted pixel nearest the component's coordinate-wise median.
        count, components = cv2.connectedComponents((cell > 0).astype(np.uint8), connectivity=8)
        for component_id in range(1, count):
            cy, cx = np.where(components == component_id)
            if len(cx) == 0:
                continue
            mx, my = float(np.median(cx)), float(np.median(cy))
            nearest = int(np.argmin((cx - mx) ** 2 + (cy - my) ** 2))
            pts.append((x0 + int(cx[nearest]), y0 + int(cy[nearest]), label))

    # Sample positive points from grid cells overlapping fg_stroke
    for y0 in range(0, h, grid_step):
        for x0 in range(0, w, grid_step):
            cell = fg_stroke[y0: min(h, y0 + grid_step), x0: min(w, x0 + grid_step)]
            append_cell_samples(cell, x0, y0, 1)

    # Sample negative points from bg_stroke if painted, else auto-place background perimeter anchors
    if np.any(bg_stroke > 0):
        for y0 in range(0, h, grid_step * 2):
            for x0 in range(0, w, grid_step * 2):
                cell = bg_stroke[y0: min(h, y0 + grid_step * 2), x0: min(w, x0 + grid_step * 2)]
                append_cell_samples(cell, x0, y0, 0)
    else:
        ax0, ay0, ax1, ay1 = detect_active_frame_box(bgr)
        aw, ah = max(1, ax1 - ax0), max(1, ay1 - ay0)
        pts.extend([
            (ax0 + int(aw * 0.10), ay0 + int(ah * 0.10), 0),
            (ax0 + int(aw * 0.50), ay0 + int(ah * 0.08), 0),
            (ax0 + int(aw * 0.90), ay0 + int(ah * 0.10), 0),
            (ax0 + int(aw * 0.08), ay0 + int(ah * 0.42), 0),
            (ax0 + int(aw * 0.92), ay0 + int(ah * 0.42), 0),
        ])

    return pts


def auto_chroma_key_mask(
    bgr: np.ndarray,
    key_color: Literal["green", "blue", "auto"] = "green",
    tolerance: float = 0.35,
) -> np.ndarray:
    """
    Automatic multi-space (RGB chroma difference + HSV) green/blue screen segmentation.
    Returns binary foreground mask uint8 in {0, 1} (1 = foreground subject, 0 = chroma background).
    """
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    if key_color == "auto":
        green_score = float(np.mean(np.maximum(0.0, g - np.maximum(r, b))))
        blue_score = float(np.mean(np.maximum(0.0, b - np.maximum(r, g))))
        mode = "green" if green_score >= blue_score else "blue"
    else:
        mode = key_color

    thresh_diff = max(8.0, 45.0 * (1.0 - float(np.clip(tolerance, 0.05, 0.95))))
    if mode == "green":
        chroma_excess = g - np.maximum(r, b)
        is_bg = (chroma_excess > thresh_diff) & (hue >= 30) & (hue <= 96) & (sat >= 35) & (val >= 25)
    else:
        chroma_excess = b - np.maximum(r, g)
        is_bg = (chroma_excess > thresh_diff) & (hue >= 92) & (hue <= 140) & (sat >= 35) & (val >= 25)

    fg_u8 = (~is_bg).astype(np.uint8) * 255
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, k_open)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, k_close)
    return (fg_u8 > 127).astype(np.uint8)


def _propagate_local_colors(
    img_f: np.ndarray,
    known_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Propagate known foreground or background colors smoothly into the unknown trimap band (half-res pyramid for speed)."""
    h, w = img_f.shape[:2]
    w_f = (known_mask > 0).astype(np.float32)
    scale = 0.5 if min(h, w) >= 480 else 1.0
    if scale < 1.0:
        sw, sh = max(32, int(w * scale)), max(32, int(h * scale))
        weighted = img_f * w_f[..., None]
        weighted_s = cv2.resize(weighted, (sw, sh), interpolation=cv2.INTER_AREA)
        w_s = cv2.resize(w_f, (sw, sh), interpolation=cv2.INTER_AREA)
        sig_s = max(1.2, sigma * scale)
    else:
        weighted, w_s, sig_s = img_f * w_f[..., None], w_f, sigma

    if scale < 1.0:
        weighted = weighted_s
    num1 = cv2.GaussianBlur(weighted, (0, 0), sigmaX=sig_s)
    den1 = cv2.GaussianBlur(w_s, (0, 0), sigmaX=sig_s)[..., None]

    num2 = cv2.GaussianBlur(weighted, (0, 0), sigmaX=sig_s * 2.5)
    den2 = cv2.GaussianBlur(w_s, (0, 0), sigmaX=sig_s * 2.5)[..., None]

    est2 = num2 / np.maximum(den2, 1e-5)
    est1 = np.where(den1 > 1e-3, num1 / np.maximum(den1, 1e-5), est2)
    if scale < 1.0:
        est1 = cv2.resize(est1, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(est1, 0.0, 1.0)


def decontaminate_foreground_edges(
    img_f: np.ndarray,
    alpha: np.ndarray,
    fg_est: np.ndarray,
    bg_est: Optional[np.ndarray] = None,
    strength: float = 0.90,
    despill_green: bool = False,
) -> np.ndarray:
    """
    Halo-free Edge Color Decontamination:
    Replaces background-contaminated boundary colors with the propagated true interior
    foreground color `fg_est` proportionally to `(1 - alpha)`, preventing any division-by-alpha
    ringing or double-contour artifacts while removing 100% of green/background spill.
    """
    fg = img_f.copy()
    if strength > 1e-3:
        a = alpha[..., None]
        edge_zone = (alpha > 0.005) & (alpha < 0.98)
        if np.any(edge_zone):
            w_replace = np.clip(strength * (1.0 - a) ** 0.75, 0.0, 1.0)
            blended = img_f * (1.0 - w_replace) + fg_est * w_replace
            if bg_est is not None:
                recovered = np.clip((img_f - (1.0 - a) * bg_est) / np.maximum(a, 0.08), 0.0, 1.0)
                use_recovered = edge_zone & (alpha >= 0.15) & (alpha < 0.90)
                blended = np.where(use_recovered[..., None], recovered, blended)
            fg = np.where(edge_zone[..., None], blended, img_f)

    if despill_green:
        # Suppress residual green chroma excess along boundary and skin/bezel reflections
        b, g, r = fg[..., 0], fg[..., 1], fg[..., 2]
        max_rb = np.maximum(r, b)
        avg_rb = 0.6 * r + 0.4 * b
        excess_g = np.maximum(0.0, g - max_rb)
        # Stronger despill near the edge, gentle despill inside
        edge_weight = np.clip(1.2 - 0.85 * alpha, 0.25, 1.0)
        g_clean = g - excess_g * edge_weight
        # Restore perceived luminance slightly so despilled edges don't turn dark
        lum_loss = (g - g_clean) * 0.45
        fg = np.stack(
            [
                np.clip(b + lum_loss * 0.4, 0.0, 1.0),
                np.clip(g_clean, 0.0, 1.0),
                np.clip(r + lum_loss * 0.6, 0.0, 1.0),
            ],
            axis=-1,
        )
        # For boundary pixels still leaning green over neutral/warm foreground, clamp G to avg_rb
        boundary = (alpha > 0.02) & (alpha < 0.96)
        if np.any(boundary):
            fg[boundary, 1] = np.minimum(fg[boundary, 1], np.maximum(avg_rb[boundary], max_rb[boundary]))

    return np.clip(fg, 0.0, 1.0)


_VITMATTE_CACHE: dict = {}


def _run_vitmatte_gpu(bgr: np.ndarray, trimap: np.ndarray) -> Optional[np.ndarray]:
    """
    Run ViTMatte (hustvl/vitmatte-base-composition-1k with Small fallback) on local CUDA GPU
    inside the active trimap bounding box for hair/sub-pixel neural alpha matting.
    """
    try:
        from src.neural_matting import run_vitmatte_neural

        return run_vitmatte_neural(bgr, trimap)
    except Exception:
        return None


def verify_and_compensate_pass2(
    bgr: np.ndarray,
    mask_pass1: np.ndarray,
    ref_bgr: Optional[np.ndarray] = None,
    ref_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, bool, int, int, np.ndarray]:
    """
    UNIVERSAL PASS-2 CLOSED-LOOP PIXEL VERIFICATION & COMPENSATION:
    Works for ANY object (any product, color, shape, hand) and ANY background
    (real-world indoor/outdoor, desk, street, or chroma green/blue screen).
    Compares Frame `bgr` + `mask_pass1` against:
      1. The user's Confirmed Reference Keyframe `(ref_bgr, ref_mask)` foreground/background distributions
      2. The current frame's confident interior foreground core vs. exterior background
      3. High-gradient rigid core contours inside the foreground
    Restores missing foreground pixels (False Negatives) and prunes excess background pixels (False Positives).
    """
    h, w = bgr.shape[:2]
    m1 = (mask_pass1 > 0).astype(np.uint8)
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    g_exc = g - np.maximum(r, b)

    # 1. Detect rigid interior core strictly inside eroded m1 (never overlapping the outer boundary ring or background!)
    eroded_m1 = cv2.erode(m1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
    blue_ui = ((b > g + 18.0) & (b > r + 22.0) & (sat > 45) & eroded_m1).astype(np.uint8)
    dark_face = ((val < 62) & (g_exc < 10.0) & eroded_m1).astype(np.uint8)
    dev_front = cv2.morphologyEx(
        (blue_ui | dark_face) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19)),
    )
    cnts, _ = cv2.findContours(dev_front, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    is_protected_core = np.zeros((h, w), dtype=bool)
    if cnts:
        largest_dev = max(cnts, key=cv2.contourArea)
        area_dev = cv2.contourArea(largest_dev)
        if int(h * w * 0.004) < area_dev < int(h * w * 0.50):
            hull_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(hull_mask, [cv2.convexHull(largest_dev)], -1, 255, thickness=cv2.FILLED)
            hull_mask = cv2.erode(hull_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            is_protected_core = (hull_mask > 0) & eroded_m1

    # Also protect deep eroded interior of m1 where pixel color is NOT green screen
    deep_m1 = cv2.erode(m1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))) > 0

    # 2. Check if the exterior background contains a chroma screen (green or blue)
    has_green_screen = bool(
        np.sum((g_exc > 18.0) & (hue >= 28) & (hue <= 102) & (sat >= 35) & (m1 == 0))
        > int(h * w * 0.03)
    )

    # 3. Universal Reference-Keyframe + Local Frame Lab Color Verification
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    inner = cv2.erode(m1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
    if not np.any(inner):
        inner = cv2.erode(m1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    outer = cv2.dilate(m1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) == 0

    # If reference keyframe (ref_bgr, ref_mask) is provided, use its ground-truth confirmed FG/BG regions
    ref_bg_match = np.zeros((h, w), dtype=bool)
    ref_fg_lab_est: Optional[np.ndarray] = None
    ref_bg_lab_est: Optional[np.ndarray] = None
    if ref_bgr is not None and ref_mask is not None and ref_bgr.shape[:2] == (h, w):
        ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_m = (ref_mask > 0).astype(np.uint8)
        ref_in = cv2.erode(ref_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
        ref_out = cv2.dilate(ref_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) == 0
        w_lab = np.array([0.4, 1.2, 1.2], dtype=np.float32)
        d_to_ref_pixel = np.linalg.norm((lab - ref_lab) * w_lab, axis=-1)
        is_human_skin = (
            (cr >= 133) & (cr <= 178) & (cb >= 77) & (cb <= 128)
            & ((hue <= 18) | (hue >= 162))
        )
        if np.any(ref_in) and np.any(ref_out):
            ref_fg_lab_est = _propagate_local_colors(ref_lab / 255.0, ref_in, sigma=12.0) * 255.0
            ref_bg_lab_est = _propagate_local_colors(ref_lab / 255.0, ref_out, sigma=12.0) * 255.0
            d_ref_fg = np.linalg.norm((lab - ref_fg_lab_est) * w_lab, axis=-1)
            d_ref_bg = np.linalg.norm((lab - ref_bg_lab_est) * w_lab, axis=-1)
            ref_bg_match = (ref_m == 0) & (d_to_ref_pixel < 9.0) & (d_ref_bg + 4.0 < d_ref_fg) & (~is_human_skin) & (~is_protected_core)
        else:
            ref_bg_match = (ref_m == 0) & (d_to_ref_pixel < 9.0) & (~is_human_skin) & (~is_protected_core)

    if has_green_screen:
        is_green_bg = (g_exc > 3.5) & (hue >= 28) & (hue <= 104) & (sat >= 22) & (~is_protected_core)
        is_secondary_bg = (
            (hue >= 19) & (hue <= 36) & (sat >= 42) & ((g - b) > (r - g) * 1.45)
            & (cr >= 128) & (cb <= 118) & (~is_protected_core)
        )
        is_secondary_bg[: int(h * 0.55), :] = False
        is_bg_sig = is_green_bg | is_secondary_bg

        dil_22 = cv2.dilate(m1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))) > 0
        missing_candidates = dil_22 & (m1 == 0) & (~is_bg_sig)
        if np.any(inner) and np.any(outer) and np.any(missing_candidates):
            fg_lab = _propagate_local_colors(lab / 255.0, inner, sigma=10.0) * 255.0
            bg_lab = _propagate_local_colors(lab / 255.0, outer, sigma=10.0) * 255.0
            w_lab = np.array([0.45, 1.2, 1.2], dtype=np.float32)
            d_fg = np.linalg.norm((lab - fg_lab) * w_lab, axis=-1)
            d_bg = np.linalg.norm((lab - bg_lab) * w_lab, axis=-1)
            local_bg_match = (d_bg < 9.0) & (d_fg > d_bg * 1.60 + 4.5)
            missing_candidates &= ~local_bg_match
        m2_raw = (((m1 > 0) | missing_candidates | is_protected_core) & (~is_bg_sig)).astype(np.uint8)
    else:
        # Universal Non-Green-Screen 2-Pass Verification (works on ANY object & ANY real-world background!)
        if np.any(inner) and np.any(outer):
            fg_lab = _propagate_local_colors(lab / 255.0, inner, sigma=10.0) * 255.0
            bg_lab = _propagate_local_colors(lab / 255.0, outer, sigma=10.0) * 255.0
            if ref_fg_lab_est is not None and ref_bg_lab_est is not None:
                fg_lab = 0.5 * fg_lab + 0.5 * ref_fg_lab_est
                bg_lab = 0.5 * bg_lab + 0.5 * ref_bg_lab_est

            w_lab = np.array([0.45, 1.2, 1.2], dtype=np.float32)
            d_fg = np.linalg.norm((lab - fg_lab) * w_lab, axis=-1)
            d_bg = np.linalg.norm((lab - bg_lab) * w_lab, axis=-1)

            # Outward band (up to 16px outside m1) & Inward band (up to 14px inside m1)
            dil_16 = cv2.dilate(m1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
            ero_14 = cv2.erode(m1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))) > 0
            out_ring = dil_16 & (m1 == 0)
            in_ring = (m1 > 0) & (~ero_14) & (~is_protected_core)

            recover_px = out_ring & (d_fg < 8.5) & (d_bg > d_fg * 1.85 + 5.5) & (~ref_bg_match)
            interior_bg_hole = (m1 > 0) & (~is_protected_core) & (d_bg < 6.0) & (d_fg > d_bg * 2.4 + 9.0)
            prune_px = (in_ring & (d_bg < 9.0) & (d_fg > d_bg * 1.60 + 4.5)) | interior_bg_hole | (in_ring & ref_bg_match)
            m2_raw = (((m1 > 0) | recover_px | is_protected_core) & (~prune_px)).astype(np.uint8)
        else:
            m2_raw = (m1 | is_protected_core.astype(np.uint8)).astype(np.uint8)

    n_cc, l_cc, stats, _ = cv2.connectedComponentsWithStats(m2_raw, connectivity=8)
    min_area = max(64, int(h * w * 0.002))
    clean_pass2 = np.zeros_like(m2_raw)
    for cid in range(1, n_cc):
        if stats[cid, cv2.CC_STAT_AREA] >= min_area:
            clean_pass2[l_cc == cid] = 1
    if not np.any(clean_pass2) and np.any(m2_raw):
        clean_pass2 = m2_raw

    actual_missing = (clean_pass2 == 1) & (m1 == 0)
    actual_excess = (clean_pass2 == 0) & (m1 == 1)

    diff_bgr = (bgr.astype(np.float32) * 0.55).astype(np.uint8)
    diff_bgr[actual_missing] = (0, 255, 0)
    diff_bgr[actual_excess] = (0, 0, 255)
    diff_rgb = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2RGB)

    return (
        clean_pass2,
        is_protected_core,
        has_green_screen,
        int(np.sum(actual_missing)),
        int(np.sum(actual_excess)),
        diff_rgb,
    )


def _pass2_recovered_rim_confidence(
    img_f: np.ndarray,
    fg_est: np.ndarray,
    bg_est: np.ndarray,
    alpha: np.ndarray,
    mask_pass1: np.ndarray,
    mask_pass2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sparse rim-preservation weights for strong Pass-2 FG recoveries."""
    candidate = (mask_pass2 > 0) & (mask_pass1 == 0) & (alpha >= 0.90)
    ys, xs = np.nonzero(candidate)
    if not len(xs):
        return ys, xs, np.empty(0, dtype=np.float32)

    observed = img_f[ys, xs]
    d_fg = np.linalg.norm(observed - fg_est[ys, xs], axis=1)
    d_bg = np.linalg.norm(observed - bg_est[ys, xs], axis=1)
    fg_support = np.clip((d_bg - 2.0 * d_fg - (8.0 / 255.0)) / 0.12, 0.0, 1.0)
    near_opaque = np.clip((alpha[ys, xs] - 0.90) / 0.10, 0.0, 1.0)
    confidence = fg_support * near_opaque
    supported = confidence > 0.0
    return ys[supported], xs[supported], confidence[supported].astype(np.float32)


def extract_pixel_accurate_matte(
    bgr: np.ndarray,
    coarse_mask: np.ndarray,
    cfg: Optional[MattingConfig] = None,
    ref_bgr: Optional[np.ndarray] = None,
    ref_mask: Optional[np.ndarray] = None,
) -> MattingResult:
    """
    2-Pass Closed-Loop Pixel-Accurate Alpha Matting & Zero-Blur Despill:
      - Pass 1: Coarse mask from SAM 2.1 / Roto-Brush / Chroma Key.
      - Pass 2: Closed-Loop Pixel Verification & Compensation against source frame `bgr`
        and Reference Keyframe `(ref_bgr, ref_mask)`, followed by ViTMatte GPU Neural Matting
        or Sub-Pixel Joint Guided Filtering + Continuous Chroma-Ratio Alpha.
    """
    if cfg is None:
        cfg = MattingConfig()

    h, w = bgr.shape[:2]
    m_in = (coarse_mask > 0).astype(np.uint8)

    if not np.any(m_in > 0):
        empty_a = np.zeros((h, w), dtype=np.float32)
        rgba = np.dstack([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.zeros((h, w), dtype=np.uint8)])
        return MattingResult(alpha=empty_a, fg_clean_bgr=bgr.copy(), rgba=rgba, trimap=m_in)

    # Optional Pass-2 Closed-Loop Verification & Compensation
    if cfg.two_pass_verify:
        clean_m, is_device_hull, has_green, n_miss, n_exc, diff_rgb = verify_and_compensate_pass2(
            bgr, m_in, ref_bgr=ref_bgr, ref_mask=ref_mask
        )
    else:
        clean_m = m_in
        is_device_hull = np.zeros((h, w), dtype=bool)
        has_green = False
        n_miss, n_exc, diff_rgb = 0, 0, None

    m_u8 = clean_m * 255
    recovered_silver = np.zeros((h, w), dtype=bool)

    if cfg.erode_dilate > 0:
        grown, recovered_silver = expand_mask_with_color_evidence(
            bgr, clean_m, pixels=int(cfg.erode_dilate)
        )
        m_u8 = grown * 255
    elif cfg.erode_dilate < 0 and not has_green:
        k_ed = abs(int(cfg.erode_dilate)) * 2 + 1
        ker_ed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_ed, k_ed))
        m_u8 = cv2.erode(m_u8, ker_ed, iterations=1)

    r = max(1, min(int(cfg.trimap_radius), 2 if has_green else 8))
    k_tri = r * 2 + 1
    ker_tri = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_tri, k_tri))
    fg_core = cv2.erode(m_u8, ker_tri, iterations=1) > 0
    bg_core = cv2.dilate(m_u8, ker_tri, iterations=1) == 0
    unknown = (~fg_core) & (~bg_core)

    trimap = np.zeros((h, w), dtype=np.uint8)
    trimap[unknown] = 128
    trimap[fg_core] = 255

    img_f = bgr.astype(np.float32) / 255.0

    # Fast Neural Matting Path (BiRefNet / ViTMatte-Base / Auto + PyMatting Foreground)
    engine = getattr(cfg, "engine", "classic")
    if engine in ("birefnet", "vitmatte", "auto"):
        neural_alpha: Optional[np.ndarray] = None
        if engine == "birefnet":
            try:
                from src.neural_matting import run_birefnet_guided_on_mask

                neural_alpha = run_birefnet_guided_on_mask(bgr, clean_m)
            except Exception:
                neural_alpha = None
        if neural_alpha is None:
            r_tri = max(3, min(int(cfg.trimap_radius), 24))
            k_n = r_tri * 2 + 1
            ker_n = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_n, k_n))
            fg_n = cv2.erode(m_u8, ker_n, iterations=1) > 0
            bg_n = cv2.dilate(m_u8, ker_n, iterations=1) == 0
            tri_n = np.full((h, w), 128, dtype=np.uint8)
            tri_n[fg_n] = 255
            tri_n[bg_n] = 0
            trimap = tri_n
            neural_alpha = _run_vitmatte_gpu(bgr, tri_n)

        if neural_alpha is not None:
            if cfg.use_pymatting_fg:
                from src.neural_matting import estimate_foreground_pymatting

                fg_clean_bgr = estimate_foreground_pymatting(
                    bgr,
                    neural_alpha,
                    strength=float(cfg.decontaminate_strength),
                    despill_green=bool(cfg.despill_green),
                )
            else:
                sigma_fg = float(max(3.0, r * 1.35))
                fg_est_n = _propagate_local_colors(img_f, neural_alpha >= 0.95, sigma=sigma_fg)
                bg_est_n = _propagate_local_colors(img_f, neural_alpha <= 0.05, sigma=sigma_fg)
                fg_clean_f_n = decontaminate_foreground_edges(
                    img_f,
                    neural_alpha,
                    fg_est=fg_est_n,
                    bg_est=bg_est_n,
                    strength=float(np.clip(cfg.decontaminate_strength, 0.0, 1.0)),
                    despill_green=bool(cfg.despill_green),
                )
                fg_clean_bgr = np.clip(fg_clean_f_n * 255.0, 0, 255).astype(np.uint8)
            if np.any(recovered_silver):
                fg_clean_bgr[recovered_silver] = bgr[recovered_silver]
            fg_clean_rgb = cv2.cvtColor(fg_clean_bgr, cv2.COLOR_BGR2RGB)
            alpha_u8 = np.clip(neural_alpha * 255.0, 0, 255).astype(np.uint8)
            rgba = np.dstack([fg_clean_rgb, alpha_u8])
            return MattingResult(
                alpha=neural_alpha.astype(np.float32),
                fg_clean_bgr=fg_clean_bgr,
                rgba=rgba,
                trimap=trimap,
                missing_restored_px=n_miss,
                excess_removed_px=n_exc,
                audit_diff_rgb=diff_rgb,
            )

    if has_green:
        # 1. Protect the full rigid device zone (protected core + adjacent white/silver/dark bezels within 25px)
        b_raw, g_raw, r_raw = bgr[..., 0].astype(np.float32), bgr[..., 1].astype(np.float32), bgr[..., 2].astype(np.float32)
        hsv_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hue_r, sat_r, val_r = hsv_raw[..., 0], hsv_raw[..., 1], hsv_raw[..., 2]
        dil_core = cv2.dilate(is_device_hull.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (55, 49))) > 0
        is_bezel_px = ((val_r >= 135) & (sat_r <= 52) & (np.abs(r_raw - b_raw) <= 30.0)) | (val_r < 68)
        is_bezel_px[int(h * 0.76):, :] = False
        is_device_zone = is_device_hull | (dil_core & is_bezel_px)

        # 2. Full-Resolution Continuous Soft Chroma-Ratio Alpha + Shaved SDF Curvature Smoothing (Zero Staircase)
        g_soft_exc = g_raw - (0.56 * np.maximum(r_raw, b_raw) + 0.44 * np.minimum(r_raw, b_raw))
        bg_chroma_ref = max(
            16.0,
            float(np.median(g_soft_exc[clean_m == 0])) if np.any(clean_m == 0) else 16.0,
        )
        alpha_chroma = 1.0 - np.clip((g_soft_exc - 2.0) / bg_chroma_ref, 0.0, 1.0)
        alpha_chroma[is_device_zone] = 1.0

        m_shaved = clean_m.copy()
        if min(h, w) >= 720:
            eroded_skin = cv2.erode(clean_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            m_shaved[~is_device_zone] = eroded_skin[~is_device_zone]
            m_shaved = cv2.morphologyEx(m_shaved, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            m_shaved = cv2.morphologyEx(m_shaved, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        sdf_sigma = 2.2 if min(h, w) >= 720 else 1.0

        d_in = cv2.distanceTransform(m_shaved, cv2.DIST_L2, 5)
        d_out = cv2.distanceTransform(1 - m_shaved, cv2.DIST_L2, 5)
        sdf = d_in - d_out
        sdf_smooth = cv2.GaussianBlur(sdf, (0, 0), sigmaX=sdf_sigma)
        alpha_sdf = np.clip((sdf_smooth + 0.45) / 2.1, 0.0, 1.0).astype(np.float32)
        alpha_init = np.where(
            is_device_zone,
            alpha_sdf,
            np.minimum(alpha_sdf, alpha_chroma * 0.55 + alpha_sdf * 0.45),
        ).astype(np.float32)

        # 3. Neutral Chroma Despill (Zero Red Bleed) + Highlight-Clamped White-Rim De-Glare Inpainting
        b_f, g_f, r_f = img_f[..., 0], img_f[..., 1], img_f[..., 2]
        limit_g = 0.58 * np.maximum(r_f, b_f) + 0.42 * np.minimum(r_f, b_f)
        spill = np.where(is_device_hull, 0.0, np.maximum(0.0, g_f - limit_g))
        fg_despilled = np.stack(
            [
                np.clip(b_f + spill * 0.08, 0.0, 1.0),
                np.clip(g_f - spill, 0.0, 1.0),
                np.clip(r_f + spill * 0.10, 0.0, 1.0),
            ],
            axis=-1,
        )

        # Propagate clean near-interior skin tone (4.5px..10.5px inside) outward onto outer 0..5px rim
        # and clamp any residual white studio rim glare without reddening or darkening the fingers
        near_skin_band = (sdf_smooth >= 4.5) & (sdf_smooth <= 10.5) & (~is_device_zone)
        if np.any(near_skin_band):
            near_albedo = _propagate_local_colors(fg_despilled, near_skin_band, sigma=7.0)
            rim_w = np.clip((5.0 - sdf_smooth) / 5.0, 0.0, 1.0)[..., None]
            rim_w[is_device_zone] = 0.0
            fg_rim_clean = np.minimum(fg_despilled, near_albedo + 0.02) * 0.65 + near_albedo * 0.35
            fg_clean_f = fg_despilled * (1.0 - rim_w) + fg_rim_clean * rim_w
        else:
            fg_clean_f = fg_despilled

        # 4. Full-Resolution Guided Filter on cleaned foreground + Sub-Pixel Anti-Aliasing
        gf_rad = 3 if min(h, w) >= 720 else 2
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            alpha = cv2.ximgproc.guidedFilter(guide=fg_clean_f, src=alpha_init, radius=gf_rad, eps=5e-4)
        else:
            alpha = cv2.GaussianBlur(alpha_init, (gf_rad * 2 + 1, gf_rad * 2 + 1), 0)
        alpha = np.clip((alpha - 0.08) / 0.88, 0.0, 1.0)
        edge_aa_band = (alpha > 0.005) & (alpha < 0.995)
        alpha_aa = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.85)
        alpha = np.where(edge_aa_band, alpha_aa, alpha).astype(np.float32)
        alpha[sdf_smooth < -2.2] = 0.0
        alpha[is_device_hull] = 1.0
    else:
        sigma = float(max(2.5, r * 1.35))
        fg_est = _propagate_local_colors(img_f, fg_core, sigma=sigma)
        bg_est = _propagate_local_colors(img_f, bg_core, sigma=sigma)

        lab_f = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        lab_f[..., 0] *= 1.0
        fg_lab_est = _propagate_local_colors(lab_f, fg_core, sigma=sigma)
        bg_lab_est = _propagate_local_colors(lab_f, bg_core, sigma=sigma)

        diff_fb = fg_lab_est - bg_lab_est
        dist_sq = np.sum(diff_fb * diff_fb, axis=-1)
        proj = np.sum((lab_f - bg_lab_est) * diff_fb, axis=-1)
        alpha_color = np.clip(proj / (dist_sq + 1e-4), 0.0, 1.0)
        diff_rgb = fg_est - bg_est
        rgb_dist_sq = np.sum(diff_rgb * diff_rgb, axis=-1)
        rgb_delta = img_f - bg_est
        rgb_proj = np.sum(rgb_delta * diff_rgb, axis=-1) / (rgb_dist_sq + 1e-5)
        rgb_alpha = np.clip(rgb_proj, 0.0, 1.0)
        rgb_conf = np.clip(rgb_dist_sq / 0.02, 0.0, 1.0)
        inside_mask = m_u8 > 0

        # Separated endpoint colors alone do not prove that the observed pixel
        # is their mixture. Trust geometry when an implausible fit disagrees with it.
        contradicts_mask = unknown & (rgb_conf > 0.85) & ((rgb_alpha >= 0.5) != inside_mask)
        fit_confidence = np.ones_like(rgb_conf)
        if np.any(contradicts_mask):
            projected = rgb_alpha[contradicts_mask]
            residual = rgb_delta[contradicts_mask] - projected[:, None] * diff_rgb[contradicts_mask]
            fit_error = np.linalg.norm(residual, axis=1)
            extrapolation = np.abs(rgb_proj[contradicts_mask] - projected)
            fit_confidence[contradicts_mask] = 1.0 - np.maximum(
                np.clip((fit_error - 0.07) / 0.06, 0.0, 1.0),
                np.clip((extrapolation - 0.08) / 0.10, 0.0, 1.0),
            )
            rgb_conf *= fit_confidence
        alpha_color = np.where(rgb_conf > 0.85, rgb_alpha, alpha_color)

        dist_to_bg = cv2.distanceTransform((~bg_core).astype(np.uint8), cv2.DIST_L2, 3)
        dist_to_fg = cv2.distanceTransform((~fg_core).astype(np.uint8), cv2.DIST_L2, 3)
        alpha_geom = np.clip(dist_to_bg / (dist_to_bg + dist_to_fg + 1e-5), 0.0, 1.0)

        color_conf = np.clip(dist_sq / 0.012, 0.0, 1.0)
        neutral_edge = np.linalg.norm(diff_fb[..., 1:], axis=-1) < 0.012
        neutral_luma_conf = np.clip((np.abs(diff_fb[..., 0]) - 0.008) / 0.018, 0.0, 1.0)
        color_conf = np.maximum(color_conf, neutral_luma_conf * neutral_edge) * fit_confidence
        alpha_blend = color_conf * alpha_color + (1.0 - color_conf) * alpha_geom
        alpha_blend = np.where(
            inside_mask,
            np.maximum(alpha_blend, alpha_geom * 0.88),
            np.minimum(alpha_blend, alpha_geom * 1.12),
        )
        color_conf = np.maximum(color_conf, rgb_conf)
        alpha_blend = np.where(color_conf > 0.85, alpha_color, alpha_blend)

        alpha_init = np.where(fg_core, 1.0, np.where(bg_core, 0.0, alpha_blend)).astype(np.float32)

        vit_alpha = _run_vitmatte_gpu(bgr, trimap) if cfg.use_vitmatte_gpu else None
        if vit_alpha is not None:
            alpha = vit_alpha
        else:
            gr_rad = max(1, int(cfg.guided_radius))
            gr_eps = float(max(1e-6, cfg.guided_eps))
            if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
                alpha_refined = cv2.ximgproc.guidedFilter(guide=img_f, src=alpha_init, radius=gr_rad, eps=gr_eps)
            else:
                alpha_refined = cv2.GaussianBlur(alpha_init, (gr_rad * 2 + 1, gr_rad * 2 + 1), 0)

            alpha = np.clip((alpha_refined - 0.06) / 0.88, 0.0, 1.0)
            k_far = (r + 2) * 2 + 1
            ker_far = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_far, k_far))
            deep_in = cv2.GaussianBlur((cv2.erode(m_u8, ker_far) > 0).astype(np.float32), (5, 5), 0)
            far_out = cv2.GaussianBlur((cv2.dilate(m_u8, ker_far) > 0).astype(np.float32), (5, 5), 0)
            alpha = np.clip((alpha * (1.0 - deep_in) + deep_in) * far_out, 0.0, 1.0)
            reliable_color = ((neutral_luma_conf > 0.85) & neutral_edge) | (rgb_conf > 0.85)
            reliable_color &= unknown & (fit_confidence > 0.85)
            alpha[reliable_color] = alpha_color[reliable_color]

        if cfg.use_pymatting_fg:
            from src.neural_matting import estimate_foreground_pymatting

            fg_clean_bgr = estimate_foreground_pymatting(
                bgr,
                alpha,
                strength=float(cfg.decontaminate_strength),
                despill_green=bool(cfg.despill_green),
            )
            fg_clean_f = fg_clean_bgr.astype(np.float32) / 255.0
        else:
            fg_clean_f = decontaminate_foreground_edges(
                img_f,
                alpha,
                fg_est=fg_est,
                bg_est=bg_est,
                strength=float(np.clip(cfg.decontaminate_strength, 0.0, 1.0)),
                despill_green=bool(cfg.despill_green),
            )
            # Apply Intrinsic Albedo + Micro-Texture Rim Normalization on non-green-screen edges too
            # so bright old-background rim glare is replaced by interior albedo + skin/object micro-texture
            if min(h, w) >= 480:
                d_in_ng = cv2.distanceTransform((m_u8 > 0).astype(np.uint8), cv2.DIST_L2, 3)
                deep_ng_core = (d_in_ng > 10.0) & (~is_device_hull)
                if np.any(deep_ng_core):
                    low_ng = cv2.GaussianBlur(fg_clean_f, (0, 0), sigmaX=2.2)
                    micro_ng = np.clip(fg_clean_f - low_ng, -0.045, 0.045)
                    deep_ng_albedo = _propagate_local_colors(low_ng, deep_ng_core, sigma=12.0)
                    rim_ng_w = np.clip((10.0 - d_in_ng) / 9.0, 0.0, 1.0)[..., None] * float(cfg.decontaminate_strength) * 0.75
                    rim_ng_w[is_device_hull] = 0.0
                    # Keep fully covered pixels from the original foreground mask
                    # true to their source; taper smoothly for partially covered edges.
                    opaque_conf = np.clip((alpha - 0.90) / 0.10, 0.0, 1.0) * m_in
                    # An unsupported color model should not repaint a foreground
                    # detail that still has strong coverage from the confirmed geometry.
                    unexplained_foreground = (fit_confidence < 1.0) & (m_in > 0) & (alpha > 0.5)
                    edge_conf = np.clip((alpha[unexplained_foreground] - 0.5) / 0.3, 0.0, 1.0)
                    opaque_conf[unexplained_foreground] = np.maximum(
                        opaque_conf[unexplained_foreground],
                        edge_conf * (1.0 - fit_confidence[unexplained_foreground]),
                    )
                    ys_p2, xs_p2, p2_conf = _pass2_recovered_rim_confidence(
                        img_f, fg_est, bg_est, alpha, m_in, clean_m
                    )
                    if len(xs_p2):
                        opaque_conf[ys_p2, xs_p2] = np.maximum(
                            opaque_conf[ys_p2, xs_p2], p2_conf
                        )
                    rim_ng_w *= (1.0 - opaque_conf[..., None])
                    fg_clean_f = fg_clean_f * (1.0 - rim_ng_w) + np.clip(deep_ng_albedo + micro_ng, 0.0, 1.0) * rim_ng_w

    # The matte's albedo cleanup can darken real silver trim recovered just beyond
    # the confirmed mask. Those accepted source pixels are already foreground evidence.
    fg_clean_f[recovered_silver] = img_f[recovered_silver]
    fg_clean_bgr = np.clip(fg_clean_f * 255.0, 0, 255).astype(np.uint8)
    fg_clean_rgb = cv2.cvtColor(fg_clean_bgr, cv2.COLOR_BGR2RGB)
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    rgba = np.dstack([fg_clean_rgb, alpha_u8])

    return MattingResult(
        alpha=alpha,
        fg_clean_bgr=fg_clean_bgr,
        rgba=rgba,
        trimap=trimap,
        missing_restored_px=n_miss,
        excess_removed_px=n_exc,
        audit_diff_rgb=diff_rgb,
    )


def refine_alpha_edge_controls(
    bgr: np.ndarray,
    alpha: np.ndarray,
    *,
    erode_dilate: float = 0.0,
    guided_radius: int = 2,
    trimap_radius: int = 8,
    edge_preserve: float = 0.45,
    edge_feather: float = 0.0,
) -> np.ndarray:
    """
    Fast, native-resolution, tight-ROI edge refinement driven directly by live UI edge sliders:
      1. Pixel-preserving color-evidence recovery (`edge_preserve`): recovers clipped hand/device rim pixels
         that fit local foreground Lab better than background while trimming outer background spill.
      2. Joint Guided Filter edge snap (`guided_radius`): aligns matte contour to native-res RGB edges.
      3. Continuous fractional shift (`erode_dilate` in [-6.0, +6.0]): shifts boundary smoothly in/out.
      4. Optical feather (`edge_feather` in [0.0, 4.0]): softens only the outer rim while keeping deep core = 1.0.
    """
    a_in = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    if not np.any(a_in > 0.004) or np.all(a_in >= 0.996):
        return a_in

    h, w = a_in.shape[:2]
    ys, xs = np.where(a_in > 0.004)
    shift_mag = min(6.0, abs(float(erode_dilate)))
    band_r = max(2, min(int(trimap_radius), 24))
    gr = max(0, min(int(guided_radius), 10))
    sigma_f = max(0.0, min(float(edge_feather), 6.0))
    pad = min(64, max(20, int(np.ceil(shift_mag)) + band_r + gr + int(np.ceil(sigma_f * 2.5)) + 10))

    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)

    a_roi = a_in[y0:y1, x0:x1].copy()
    bgr_roi = bgr[y0:y1, x0:x1]

    m_bin = (a_roi >= 0.45).astype(np.uint8)
    ker_band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_r * 2 + 1, band_r * 2 + 1))
    fg_deep = cv2.erode(m_bin, ker_band) > 0
    bg_deep = cv2.dilate(m_bin, ker_band) == 0
    edge_band = (~fg_deep) & (~bg_deep)

    # Stage A: Pixel-Preserving Color-Evidence Recovery (never shave subject pixels fitting FG Lab)
    ep = float(np.clip(edge_preserve, 0.0, 1.0))
    if ep > 0.01 and np.any(edge_band):
        fg_seed = a_roi >= 0.92
        if not np.any(fg_seed):
            fg_seed = m_bin > 0
        bg_seed = a_roi <= 0.05
        if not np.any(bg_seed):
            bg_seed = m_bin == 0

        if np.any(fg_seed) and np.any(bg_seed):
            rh_roi, rw_roi = bgr_roi.shape[:2]
            lab = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2LAB)
            prop_sigma = float(max(4.0, min(14.0, band_r * 0.9)))
            if max(rh_roi, rw_roi) >= 480:
                ds_w, ds_h = max(32, rw_roi // 2), max(32, rh_roi // 2)
                lab_ds = cv2.resize(lab, (ds_w, ds_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                fg_s_ds = cv2.resize(fg_seed.astype(np.uint8), (ds_w, ds_h), interpolation=cv2.INTER_NEAREST) > 0
                bg_s_ds = cv2.resize(bg_seed.astype(np.uint8), (ds_w, ds_h), interpolation=cv2.INTER_NEAREST) > 0
                fg_lab_ds = _propagate_local_colors(lab_ds, fg_s_ds, sigma=max(2.0, prop_sigma * 0.5)) * 255.0
                bg_lab_ds = _propagate_local_colors(lab_ds, bg_s_ds, sigma=max(2.0, prop_sigma * 0.5)) * 255.0
                fg_lab = cv2.resize(fg_lab_ds, (rw_roi, rh_roi), interpolation=cv2.INTER_LINEAR)
                bg_lab = cv2.resize(bg_lab_ds, (rw_roi, rh_roi), interpolation=cv2.INTER_LINEAR)
            else:
                lab_f = lab.astype(np.float32) / 255.0
                fg_lab = _propagate_local_colors(lab_f, fg_seed, sigma=prop_sigma) * 255.0
                bg_lab = _propagate_local_colors(lab_f, bg_seed, sigma=prop_sigma) * 255.0
            w_lab = np.array([0.55, 1.15, 1.15], dtype=np.float32)

            rec_r = max(2, min(band_r, 8))
            rec_dil = cv2.dilate(m_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rec_r * 2 + 1, rec_r * 2 + 1))) > 0
            recover_ring = edge_band & rec_dil & (a_roi >= 0.06) & (a_roi < 0.99)
            bleed_ring = edge_band & (a_roi > 0.01) & (a_roi < 0.72)
            a_rec = a_roi.copy()
            if np.any(recover_ring):
                lab_r = lab[recover_ring].astype(np.float32)
                df_r = np.linalg.norm((lab_r - fg_lab[recover_ring]) * w_lab, axis=-1)
                db_r = np.linalg.norm((lab_r - bg_lab[recover_ring]) * w_lab, axis=-1)
                fg_ev = np.clip((db_r - df_r * 1.12) / np.maximum(db_r + df_r, 12.0), 0.0, 1.0)
                a_rec[recover_ring] = np.clip(a_roi[recover_ring] + fg_ev * (1.0 - a_roi[recover_ring]) * ep * 0.92, 0.0, 1.0)

            if np.any(bleed_ring):
                lab_b = lab[bleed_ring].astype(np.float32)
                df_b = np.linalg.norm((lab_b - fg_lab[bleed_ring]) * w_lab, axis=-1)
                db_b = np.linalg.norm((lab_b - bg_lab[bleed_ring]) * w_lab, axis=-1)
                a_b = a_roi[bleed_ring]
                bad = np.where(
                    a_b < 0.48,
                    (db_b < 15.0) & (df_b > db_b * 1.35 + 3.0),
                    (db_b < 14.0) & (df_b > db_b * 1.55 + 5.0),
                )
                if np.any(bad):
                    by, bx = np.where(bleed_ring)
                    by, bx = by[bad], bx[bad]
                    bg_ev = np.clip((df_b[bad] - db_b[bad] * 1.25) / np.maximum(db_b[bad] + df_b[bad], 12.0), 0.0, 1.0)
                    a_rec[by, bx] *= 1.0 - bg_ev * ep * 0.75
            a_roi = a_rec

    # Stage B: Native-Resolution Joint Guided Filter Edge Snap
    guide_luma = None
    if gr >= 1 and np.any(edge_band):
        guide_luma = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            try:
                a_guided = cv2.ximgproc.guidedFilter(guide=guide_luma, src=a_roi.astype(np.float32), radius=gr, eps=3e-4)
            except cv2.error:
                a_guided = cv2.GaussianBlur(a_roi, (gr * 2 + 1, gr * 2 + 1), 0)
        else:
            a_guided = cv2.GaussianBlur(a_roi, (gr * 2 + 1, gr * 2 + 1), 0)
        band_w = cv2.GaussianBlur(edge_band.astype(np.float32), (5, 5), 0)
        deep_opaque = fg_deep & (a_roi >= 0.99)
        deep_empty = bg_deep & (a_roi <= 0.01)
        band_w[deep_opaque | deep_empty] = 0.0
        a_roi = np.clip(a_roi * (1.0 - band_w) + np.clip(a_guided, 0.0, 1.0) * band_w, 0.0, 1.0)

    # Stage C: Continuous Fractional In/Out Shift (-6.0 .. +6.0 px)
    if shift_mag >= 0.05:
        is_dilate = float(erode_dilate) > 0.0
        lo = int(np.floor(shift_mag))
        hi = int(np.ceil(shift_mag))
        frac = shift_mag - lo

        def _shift_step(arr: np.ndarray, rad: int) -> np.ndarray:
            if rad <= 0:
                return arr
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1, rad * 2 + 1))
            return cv2.dilate(arr, ker) if is_dilate else cv2.erode(arr, ker)

        a_shifted = (1.0 - frac) * _shift_step(a_roi, lo) + frac * _shift_step(a_roi, hi)
        if gr >= 1 and hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            try:
                if guide_luma is None:
                    guide_luma = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                a_g2 = cv2.ximgproc.guidedFilter(guide=guide_luma, src=a_shifted.astype(np.float32), radius=2, eps=5e-4)
                a_shifted = 0.80 * a_shifted + 0.20 * np.clip(a_g2, 0.0, 1.0)
            except cv2.error:
                pass
        a_roi = np.clip(a_shifted, 0.0, 1.0)

    # Stage D: Optical Edge Feather (preserves solid core while softening outer boundary)
    if sigma_f > 0.05:
        ring_r = max(3, int(np.ceil(sigma_f * 2.5)))
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m_edge = (a_roi >= 0.5).astype(np.uint8)
        contour = (cv2.dilate(m_edge, k3) ^ cv2.erode(m_edge, k3)) > 0
        soft_mask = ((a_roi > 0.01) & (a_roi < 0.99)) | contour
        feather_zone = cv2.dilate(
            soft_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_r * 2 + 1, ring_r * 2 + 1)),
        )
        feather_w = cv2.GaussianBlur(feather_zone.astype(np.float32), (0, 0), sigmaX=max(1.0, sigma_f))
        a_blur = cv2.GaussianBlur(a_roi, (0, 0), sigmaX=sigma_f)
        a_roi = np.clip(a_roi * (1.0 - feather_w) + a_blur * feather_w, 0.0, 1.0)

    out = a_in.copy()
    out[y0:y1, x0:x1] = a_roi
    return out.astype(np.float32)


def stabilize_alpha_temporal(
    curr_alpha: np.ndarray,
    prev_alpha: Optional[np.ndarray],
    curr_bgr: np.ndarray,
    prev_bgr: Optional[np.ndarray],
    motion_thresh: float = 14.0,
    temporal_weight: float = 0.45,
) -> np.ndarray:
    """
    Motion-adaptive temporal alpha smoothing across boundary pixels to eliminate
    frame-to-frame edge flicker ('nhap nhay') without ghosting fast finger motion.
    """
    if prev_alpha is None or prev_bgr is None:
        return curr_alpha

    curr_f = curr_alpha.astype(np.float32)
    curr_soft = (curr_f > 0.01) & (curr_f < 0.99)
    ys, xs = np.where(curr_soft)
    if len(xs) == 0:
        return curr_f

    h, w = curr_f.shape[:2]
    y0 = max(0, int(ys.min()) - 2)
    y1 = min(h, int(ys.max()) + 3)
    x0 = max(0, int(xs.min()) - 2)
    x1 = min(w, int(xs.max()) + 3)

    diff_u8 = np.max(cv2.absdiff(curr_bgr[y0:y1, x0:x1], prev_bgr[y0:y1, x0:x1]), axis=2)
    diff = cv2.dilate(diff_u8, np.ones((3, 3), np.uint8)).astype(np.float32)

    # Low motion -> blend toward prev_alpha to suppress flicker; high motion -> trust curr_alpha
    still_conf = 1.0 - np.clip((diff - 4.0) / max(motion_thresh, 1.0), 0.0, 1.0)
    weight = still_conf * float(np.clip(temporal_weight, 0.0, 0.85))
    soft_roi = curr_soft[y0:y1, x0:x1]
    c_roi = curr_f[y0:y1, x0:x1]
    p_roi = prev_alpha[y0:y1, x0:x1].astype(np.float32)

    out = curr_f.copy()
    out_roi = out[y0:y1, x0:x1]
    w_s = weight[soft_roi]
    out_roi[soft_roi] = c_roi[soft_roi] * (1.0 - w_s) + p_roi[soft_roi] * w_s
    return out

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RelightConfig:
    ambient_color_strength: float = 0.55      # CIE Lab color temperature & exposure match (0..1)
    directional_light_strength: float = 0.45  # 2D spatial illumination field transfer (0..1)
    light_wrap_Radius: int = 6                # Edge light-wrap width in pixels (0..20)
    light_wrap_strength: float = 0.35         # Edge light-wrap opacity (0..1)
    protect_emissive_screen: bool = True      # Keep glowing LCD/OLED screen UI untinted & sharp
    screen_glare_opacity: float = 0.12        # Subtle environmental glass reflection on screen (0..0.5)
    grain_match: float = 0.18                 # Sensor noise matching (0..1)
    bg_defocus_sigma: float = 0.0             # Optional extra bokeh blur on new background (0..15 px)
    clean_ai_ghost: bool = False              # Inpaint/erase old AI object halo on V2V background
    pass3_aa_and_restore: bool = True         # Pass-3 Post-Composite Tangential AA & Source-Pixel Audit/Restore
    linear_compositing: bool = False          # Composite & light-wrap in Linear RGB (eliminates dark sRGB fringes)
    edge_blur_match: float = 0.0              # Optional foreground edge softness match (0.0..2.0 px)
    black_point_match: float = 0.0            # Black-point & highlight roll-off match to background (0.0..1.0)
    # Background grading & reverse source-matching (BG -> Source)
    bg_match_source_strength: float = 0.0     # Match BG exposure/contrast/color-temp to source foreground (0.0..1.0)
    bg_brightness: float = 0.0                # Manual BG brightness shift (-0.5..+0.5)
    bg_contrast: float = 1.0                  # Manual BG contrast multiplier (0.5..1.6)
    bg_saturation: float = 1.0                # Manual BG saturation multiplier (0.0..2.0)
    bg_temperature: float = 0.0               # Manual BG color temperature cool(-1)..warm(+1)
    bg_tint: float = 0.0                      # Manual BG green(-1)..magenta(+1) tint


def transform_background_frame(
    bg_bgr: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """
    Aspect-preserving cover fit + user zoom (`scale` in [0.5, 3.0]) and pan
    (`offset_x`, `offset_y` as fractions of target frame size in [-0.5, +0.5]).
    Uses sub-pixel cubic warping with smooth border reflection (never stretches or adds black bars).
    """
    bh, bw = bg_bgr.shape[:2]
    tw, th = max(1, int(target_w)), max(1, int(target_h))
    s_user = float(np.clip(scale, 0.5, 3.0))
    ox_frac = float(np.clip(offset_x, -0.5, 0.5))
    oy_frac = float(np.clip(offset_y, -0.5, 0.5))

    cover_scale = max(tw / max(float(bw), 1.0), th / max(float(bh), 1.0))
    eff_scale = cover_scale * s_user

    # Fast exact path when default transform matches target aspect ratio
    if (
        abs(s_user - 1.0) < 1e-4
        and abs(ox_frac) < 1e-4
        and abs(oy_frac) < 1e-4
        and abs((bw / max(float(bh), 1.0)) - (tw / max(float(th), 1.0))) < 1e-3
    ):
        if (bw, bh) == (tw, th):
            return bg_bgr.copy()
        return cv2.resize(bg_bgr, (tw, th), interpolation=cv2.INTER_CUBIC)

    cx_src = bw * 0.5
    cy_src = bh * 0.5
    cx_dst = tw * 0.5 + ox_frac * tw
    cy_dst = th * 0.5 + oy_frac * th

    M = np.array(
        [
            [eff_scale, 0.0, cx_dst - eff_scale * cx_src],
            [0.0, eff_scale, cy_dst - eff_scale * cy_src],
        ],
        dtype=np.float32,
    )
    interp = cv2.INTER_AREA if eff_scale < 0.95 else cv2.INTER_CUBIC
    return cv2.warpAffine(
        bg_bgr,
        M,
        (tw, th),
        flags=interp,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def grade_and_match_background(
    bg_bgr: np.ndarray,
    fg_ref_bgr: Optional[np.ndarray],
    fg_mask_u8: Optional[np.ndarray],
    *,
    screen_mask: Optional[np.ndarray] = None,
    match_source_strength: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    temperature: float = 0.0,
    tint: float = 0.0,
) -> np.ndarray:
    """
    Fast background harmonization & color grading (~2–4ms at 1080p):
      1. Reverse Light/Tone Match (BG -> Source): aligns background exposure, contrast spread,
         and CIE Lab white-balance/warmth to the lighting falling on the source subject
         (excluding self-luminous UI screens and exterior chroma background).
      2. Manual Background Color Controls: brightness, contrast, saturation, temperature, tint.
    """
    ms = float(np.clip(match_source_strength, 0.0, 1.0))
    br = float(np.clip(brightness, -0.5, 0.5))
    ct = float(np.clip(contrast, 0.5, 1.6))
    sat = float(np.clip(saturation, 0.0, 2.0))
    temp = float(np.clip(temperature, -1.0, 1.0))
    tnt = float(np.clip(tint, -1.0, 1.0))

    if (
        ms <= 1e-3
        and abs(br) <= 1e-3
        and abs(ct - 1.0) <= 1e-3
        and abs(sat - 1.0) <= 1e-3
        and abs(temp) <= 1e-3
        and abs(tnt) <= 1e-3
    ):
        return bg_bgr

    out_lab = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 1. Auto-Match Background Lighting & Tone to Source Foreground
    if ms > 1e-3 and fg_ref_bgr is not None and fg_mask_u8 is not None and np.any(fg_mask_u8 > 0):
        h, w = bg_bgr.shape[:2]
        sample_mask = fg_mask_u8 > 0
        if screen_mask is not None and screen_mask.shape == (h, w):
            non_scr = sample_mask & (screen_mask == 0)
            if int(np.sum(non_scr)) >= 100:
                sample_mask = non_scr

        if min(h, w) >= 320:
            sw, sh = max(64, int(round(w * 0.25))), max(64, int(round(h * 0.25)))
            fg_s_lab = cv2.cvtColor(cv2.resize(fg_ref_bgr, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
            bg_s_lab = cv2.resize(out_lab, (sw, sh), interpolation=cv2.INTER_AREA)
            m_s = cv2.resize(sample_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
            fg_vals = fg_s_lab[m_s] if np.any(m_s) else fg_s_lab.reshape(-1, 3)
            bg_vals = bg_s_lab.reshape(-1, 3)
        else:
            fg_vals = cv2.cvtColor(fg_ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[sample_mask]
            bg_vals = out_lab.reshape(-1, 3)

        if len(fg_vals) >= 16 and len(bg_vals) >= 16:
            # Estimate ambient scene light from source subject (avoid crushed blacks & specular peaks)
            fg_l_med = float(np.median(fg_vals[:, 0]))
            fg_l_iqr = float(np.percentile(fg_vals[:, 0], 80) - np.percentile(fg_vals[:, 0], 20)) / 1.68 + 1e-4
            bg_l_med = float(np.median(bg_vals[:, 0]))
            bg_l_iqr = float(np.percentile(bg_vals[:, 0], 80) - np.percentile(bg_vals[:, 0], 20)) / 1.68 + 1e-4

            # Exposure & contrast alignment
            tgt_l = bg_l_med + (fg_l_med - bg_l_med) * (0.68 * ms)
            slope = float(np.clip(1.0 + (np.clip(fg_l_iqr / bg_l_iqr, 0.65, 1.55) - 1.0) * (0.55 * ms), 0.72, 1.38))
            out_lab[..., 0] = np.clip(tgt_l + (out_lab[..., 0] - bg_l_med) * slope, 0.0, 255.0)

            # Chromaticity (color temperature & tint) match referenced to source highlights/midtones
            # Prefer near-neutral / brighter subject pixels (white bezels, paper, neutral fabric) when available
            # so warm human skin alone doesn't over-orange the background.
            chroma_mag = np.hypot(fg_vals[:, 1] - 128.0, fg_vals[:, 2] - 128.0)
            neut = fg_vals[(fg_vals[:, 0] >= 70.0) & (chroma_mag <= 24.0)]
            if len(neut) >= 24:
                src_a = float(np.median(neut[:, 1]))
                src_b = float(np.median(neut[:, 2]))
            else:
                # Dampen skin warmth gently so background matches ambient light rather than turning peach
                src_a = 128.0 + (float(np.median(fg_vals[:, 1])) - 128.0) * 0.45
                src_b = 128.0 + (float(np.median(fg_vals[:, 2])) - 128.0) * 0.55

            bg_a = float(np.median(bg_vals[:, 1]))
            bg_b = float(np.median(bg_vals[:, 2]))
            da = float(np.clip(src_a - bg_a, -18.0, 18.0)) * (0.78 * ms)
            db = float(np.clip(src_b - bg_b, -28.0, 28.0)) * (0.85 * ms)
            out_lab[..., 1] = np.clip(out_lab[..., 1] + da, 0.0, 255.0)
            out_lab[..., 2] = np.clip(out_lab[..., 2] + db, 0.0, 255.0)

    # 2. Manual Background Color & Light Grading
    if abs(br) > 1e-3 or abs(ct - 1.0) > 1e-3:
        L = out_lab[..., 0]
        L = (L - 128.0) * ct + 128.0 + br * 110.0
        out_lab[..., 0] = np.clip(L, 0.0, 255.0)

    if abs(sat - 1.0) > 1e-3:
        out_lab[..., 1] = np.clip(128.0 + (out_lab[..., 1] - 128.0) * sat, 0.0, 255.0)
        out_lab[..., 2] = np.clip(128.0 + (out_lab[..., 2] - 128.0) * sat, 0.0, 255.0)

    if abs(temp) > 1e-3:
        out_lab[..., 2] = np.clip(out_lab[..., 2] + temp * 28.0, 0.0, 255.0)
        out_lab[..., 1] = np.clip(out_lab[..., 1] + temp * 4.0, 0.0, 255.0)

    if abs(tnt) > 1e-3:
        out_lab[..., 1] = np.clip(out_lab[..., 1] + tnt * 24.0, 0.0, 255.0)

    return cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


_SRGB_TO_LIN_LUT = np.power(np.linspace(0.0, 1.0, 4096, dtype=np.float32), 2.2).astype(np.float32)
_LIN_TO_SRGB_LUT = np.power(np.linspace(0.0, 1.0, 4096, dtype=np.float32), 1.0 / 2.2).astype(np.float32)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    idx = np.clip(x * 4095.0 + 0.5, 0.0, 4095.0).astype(np.int32)
    return _SRGB_TO_LIN_LUT[idx]


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    idx = np.clip(x * 4095.0 + 0.5, 0.0, 4095.0).astype(np.int32)
    return _LIN_TO_SRGB_LUT[idx]


def detect_emissive_screen_mask(bgr: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Automatically detect self-luminous screen region (e.g. blue/dark UI screen with crisp text)
    inside the foreground mask (hands + device) so screen colors & text are protected from
    ambient tint washout while hands and device bevel absorb the new background lighting.
    """
    fg = (fg_mask > 0).astype(np.uint8)
    if not np.any(fg > 0):
        return np.zeros_like(fg, dtype=np.uint8)

    h_full, w_full = fg.shape[:2]
    max_hw = max(h_full, w_full)
    scale = 1.0
    if max_hw > 640:
        scale = 640.0 / float(max_hw)
        sw = max(64, int(round(w_full * scale)))
        sh = max(64, int(round(h_full * scale)))
        work_bgr = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        work_fg = cv2.resize(fg, (sw, sh), interpolation=cv2.INTER_NEAREST)
    else:
        work_bgr = bgr
        work_fg = fg

    hsv = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV)
    b = work_bgr[..., 0].astype(np.float32)
    g = work_bgr[..., 1].astype(np.float32)
    r = work_bgr[..., 2].astype(np.float32)

    fg_area = float(np.sum(work_fg > 0))
    if fg_area <= 0:
        return np.zeros_like(fg, dtype=np.uint8)

    # Explicit skin-tone exclusion inside fg so convex/overlapping fingers (e.g. thumb over screen)
    # are never bridged over by convex hulls or morphological closing.
    is_skin_raw = (
        (work_fg > 0)
        & (r > g + 6.0)
        & (g >= b - 4.0)
        & (r > 70.0)
        & (hsv[..., 1] > 28)
        & ((hsv[..., 0] <= 24) | (hsv[..., 0] >= 160))
    ).astype(np.uint8) * 255
    n_skin, skin_labels, skin_stats, _ = cv2.connectedComponentsWithStats(is_skin_raw, connectivity=8)
    skin_clean = np.zeros_like(work_fg, dtype=np.uint8)
    min_skin_area = max(32 if scale < 1.0 else 120, int(fg_area * 0.0015))
    for lbl in range(1, n_skin):
        if skin_stats[lbl, cv2.CC_STAT_AREA] >= min_skin_area:
            skin_clean[skin_labels == lbl] = 255
    ks = max(3, int(round(9 * scale)) | 1)
    k_skin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    skin_protect = cv2.dilate(skin_clean, k_skin) > 0

    # Typical translator / phone screen: either non-skin blue/cyan/dark UI or high-contrast dark glass panel
    is_non_skin_screen = (
        (work_fg > 0)
        & (~skin_protect)
        & (
            ((b > r + 15) & (hsv[..., 1] > 40))  # Active blue/cyan UI screen
            | ((r < 55) & (g < 55) & (b < 75))   # Dark OLED/glass bezel & screen interior
        )
    ).astype(np.uint8) * 255

    kc = max(5, int(round(15 * scale)) | 1)
    ko = max(3, int(round(7 * scale)) | 1)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kc, kc))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (ko, ko))
    closed = cv2.morphologyEx(is_non_skin_screen, cv2.MORPH_CLOSE, k_close)
    closed[skin_protect] = 0
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

    # Keep the largest connected component inside the foreground and fill only
    # strictly interior enclosed holes (text / icons), never convexHull.
    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    work_out = np.zeros_like(work_fg, dtype=np.uint8)
    if n_cc > 1:
        areas = cc_stats[1:, cv2.CC_STAT_AREA]
        best_lbl = int(np.argmax(areas)) + 1
        if cc_stats[best_lbl, cv2.CC_STAT_AREA] > fg_area * 0.04:
            comp = (cc_labels == best_lbl).astype(np.uint8)
            inv = (1 - comp).astype(np.uint8)
            n_h, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
            h_img, w_img = comp.shape
            for h_lbl in range(1, n_h):
                x, y, bw, bh, _ = h_stats[h_lbl]
                touches_border = (x == 0) or (y == 0) or (x + bw >= w_img) or (y + bh >= h_img)
                if not touches_border:
                    hole_mask = h_labels == h_lbl
                    if float(np.mean(skin_protect[hole_mask])) < 0.25:
                        comp[hole_mask] = 1
            work_out = comp & work_fg & (~skin_protect).astype(np.uint8)

    if scale < 1.0:
        out = cv2.resize(work_out, (w_full, h_full), interpolation=cv2.INTER_NEAREST) & fg
        return out
    return work_out


def clean_ai_background_ghost(
    bg_bgr: np.ndarray,
    fg_mask: np.ndarray,
    dilate_px: int = 18,
) -> np.ndarray:
    """
    When the new background video comes from a V2V generator (so it still contains an AI copy
    of the device/hands in the middle), inpaint a safety ring around the foreground mask so
    no double-bezel or extra AI finger ever peeks out behind the real hands/device.
    """
    m_u8 = (fg_mask > 0).astype(np.uint8) * 255
    if not np.any(m_u8 > 0):
        return bg_bgr
    k = max(5, int(dilate_px) * 2 + 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(m_u8, ker, iterations=1)
    eroded = cv2.erode(m_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    inpaint_ring = cv2.subtract(dilated, eroded)

    # Fast downscaled Telea inpainting for smooth background continuation
    h, w = bg_bgr.shape[:2]
    scale = 0.5
    small_bg = cv2.resize(bg_bgr, (max(64, int(w * scale)), max(64, int(h * scale))), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(inpaint_ring, (small_bg.shape[1], small_bg.shape[0]), interpolation=cv2.INTER_NEAREST)
    inpainted_s = cv2.inpaint(small_bg, small_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    inpainted = cv2.resize(inpainted_s, (w, h), interpolation=cv2.INTER_LINEAR)

    w_ring = cv2.GaussianBlur((inpaint_ring > 0).astype(np.float32), (11, 11), 0)[..., None]
    return np.clip(bg_bgr.astype(np.float32) * (1.0 - w_ring) + inpainted.astype(np.float32) * w_ring, 0, 255).astype(np.uint8)


def _transfer_ambient_lab(
    fg_bgr: np.ndarray,
    new_bg_bgr: np.ndarray,
    src_full_bgr: Optional[np.ndarray],
    fg_mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Full Multi-Zone Studio-to-Ambient Harmonization in CIE Lab:
      1. Soft-knee specular highlight compression (tames harsh close-up studio lamp glare on skin).
      2. Mean luminance (exposure) & standard deviation (contrast slope) transfer from new background.
      3. Chromaticity (a, b) warmth/coolness & saturation harmonization referenced to neutral daylight.
    """
    if strength <= 1e-3:
        return fg_bgr

    fg_lab = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = fg_mask.shape[:2]
    fg_region_full = fg_mask > 0
    if not np.any(fg_region_full):
        return fg_bgr

    if min(h, w) >= 320:
        sw, sh = max(64, int(round(w * 0.25))), max(64, int(round(h * 0.25)))
        bg_small_lab = cv2.cvtColor(cv2.resize(new_bg_bgr, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
        fg_small_lab = cv2.cvtColor(cv2.resize(fg_bgr, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
        fg_s = cv2.resize((fg_region_full).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
        outside_s = (~fg_s).astype(np.uint8)
        if np.any(fg_s):
            distance = cv2.distanceTransform(outside_s, cv2.DIST_L2, 5)
            ring_width = max(3.0, min(sh, sw) * 0.08)
            bg_s = (~fg_s) & (distance <= ring_width)
        else:
            bg_s = np.zeros_like(fg_s)
        if np.sum(bg_s) < 60:
            bg_s = ~fg_s
        if np.sum(bg_s) < 60:
            bg_s = np.ones_like(fg_s, dtype=bool)
        if not np.any(fg_s):
            fg_s = fg_region_full
            fg_small_lab = fg_lab
        bg_samples = bg_small_lab[bg_s]
        fg_l_vals = fg_small_lab[fg_s, 0]
    else:
        bg_lab = cv2.cvtColor(new_bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        outside = (~fg_region_full).astype(np.uint8)
        distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
        ring_width = max(8.0, min(h, w) * 0.08)
        bg_region = (~fg_region_full) & (distance <= ring_width)
        if np.sum(bg_region) < 100:
            bg_region = ~fg_region_full
        if np.sum(bg_region) < 100:
            bg_region = np.ones_like(fg_region_full, dtype=bool)
        bg_samples = bg_lab[bg_region]
        fg_l_vals = fg_lab[fg_region_full, 0]

    new_bg_mean = np.median(bg_samples, axis=0)
    bg_l_std = float(np.percentile(bg_samples[:, 0], 75) - np.percentile(bg_samples[:, 0], 25)) / 1.349 + 1e-4

    fg_l_mean = float(np.mean(fg_l_vals))
    fg_l_std = float(np.std(fg_l_vals)) + 1e-4
    bg_l_mean = float(new_bg_mean[0])

    L = fg_lab[..., 0]

    # 1. Soft-knee studio highlight compression when foreground is brighter than background
    if fg_l_mean > bg_l_mean + 8.0:
        over_hi = np.maximum(0.0, L - fg_l_mean)
        L = L - (0.18 * strength) * over_hi

    # 2. Gentle exposure & contrast slope match toward the new background environment (preserves natural skin brightness)
    target_l_mean = (1.0 - 0.28 * strength) * fg_l_mean + (0.28 * strength) * (bg_l_mean + 28.0)
    target_l_std = (1.0 - 0.28 * strength) * fg_l_std + (0.28 * strength) * bg_l_std
    L_harm = target_l_mean + (L - fg_l_mean) * np.clip(target_l_std / fg_l_std, 0.72, 1.28)
    fg_lab[..., 0] = np.clip(L_harm, 0.0, 255.0)

    # 3. Chromatic warmth/coolness transfer: tightly bound 'a' channel so fingers NEVER turn red/magenta
    sat_scale = 1.0 - 0.08 * strength
    a_shift = np.clip(float(new_bg_mean[1]) - 128.0, -6.0, 4.0) * (strength * 0.75)
    b_shift = np.clip(float(new_bg_mean[2]) - 128.0, -24.0, 26.0) * (strength * 0.85)

    fg_lab[..., 1] = np.clip(128.0 + (fg_lab[..., 1] - 128.0) * sat_scale + a_shift, 0.0, 255.0)
    fg_lab[..., 2] = np.clip(128.0 + (fg_lab[..., 2] - 128.0) * sat_scale + b_shift, 0.0, 255.0)

    return cv2.cvtColor(fg_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _fast_lowpass(arr: np.ndarray, sigma_full: float, scale: float = 0.25) -> np.ndarray:
    """Exact low-frequency Gaussian blur via downsampled pyramid (50x-300x faster on 1080p/4K)."""
    if sigma_full <= 4.0:
        return cv2.GaussianBlur(arr, (0, 0), sigmaX=float(sigma_full))
    h, w = arr.shape[:2]
    sw, sh = max(16, int(w * scale)), max(16, int(h * scale))
    small = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
    blurred_s = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.0, float(sigma_full * scale)))
    return cv2.resize(blurred_s, (w, h), interpolation=cv2.INTER_LINEAR)


def _apply_2d_directional_lighting(
    fg_f: np.ndarray,
    new_bg_f: np.ndarray,
    src_full_f: Optional[np.ndarray],
    strength: float,
) -> np.ndarray:
    """
    Transfer the 2D low-frequency spatial LUMINANCE field (key light direction & brightness gradient)
    from the new background onto the foreground subject without transferring green-screen color tint.
    """
    if strength <= 1e-3:
        return fg_f

    h, w = fg_f.shape[:2]
    sigma = float(max(25.0, min(h, w) * 0.16))
    if min(h, w) < 320:
        bg_lum = 0.114 * new_bg_f[..., 0] + 0.587 * new_bg_f[..., 1] + 0.299 * new_bg_f[..., 2]
        lo, hi = np.percentile(bg_lum, (5.0, 95.0))
        low_new_lum = _fast_lowpass(np.clip(bg_lum, lo, hi), sigma_full=sigma, scale=0.125)
        mean_lum = float(np.median(low_new_lum)) + 1e-4
        ratio = np.clip((low_new_lum + 0.08) / (mean_lum + 0.08), 0.88, 1.12)[..., None]
    else:
        bg_sample = new_bg_f[::4, ::4]
        lum_sample = 0.114 * bg_sample[..., 0] + 0.587 * bg_sample[..., 1] + 0.299 * bg_sample[..., 2]
        lo, hi = np.percentile(lum_sample, (5.0, 95.0))
        sw, sh = max(16, int(w * 0.125)), max(16, int(h * 0.125))
        bg_small = cv2.resize(new_bg_f, (sw, sh), interpolation=cv2.INTER_AREA)
        bg_lum_s = np.clip(0.114 * bg_small[..., 0] + 0.587 * bg_small[..., 1] + 0.299 * bg_small[..., 2], lo, hi)
        low_s = cv2.GaussianBlur(bg_lum_s, (0, 0), sigmaX=max(1.0, sigma * 0.125))
        mean_lum = float(np.median(low_s)) + 1e-4
        ratio_s = np.clip((low_s + 0.08) / (mean_lum + 0.08), 0.88, 1.12).astype(np.float32)
        ratio = cv2.resize(ratio_s, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]

    scale_f = (1.0 - strength) + strength * ratio
    return np.clip(fg_f * scale_f, 0.0, 1.0)


def _apply_physical_light_wrap(
    fg_f: np.ndarray,
    bg_f: np.ndarray,
    alpha: np.ndarray,
    radius: int,
    strength: float,
) -> np.ndarray:
    """Add reflected ambient light with a distance falloff around the silhouette."""
    if radius <= 0 or strength <= 1e-3:
        return fg_f

    h, w = fg_f.shape[:2]
    m_bin = (alpha > 0.45).astype(np.uint8)
    if not np.any(m_bin > 0) or np.all(m_bin > 0):
        return fg_f

    bg_sample = bg_f[::4, ::4] if (h * w) > 100_000 else bg_f
    lo, hi = np.percentile(bg_sample, (5.0, 95.0), axis=(0, 1))
    low_bg_env = _fast_lowpass(np.clip(bg_f, lo, hi), sigma_full=12.0, scale=0.25)

    if min(h, w) >= 512:
        sw, sh = max(64, w // 2), max(64, h // 2)
        m_s = cv2.resize(m_bin, (sw, sh), interpolation=cv2.INTER_NEAREST)
        d_in = cv2.distanceTransform(m_s, cv2.DIST_L2, 5)
        d_out = cv2.distanceTransform(1 - m_s, cv2.DIST_L2, 5)
        sdf = (d_in - d_out) * 2.0

        gx = cv2.Sobel(sdf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sdf, cv2.CV_32F, 0, 1, ksize=3)
        gnorm = np.sqrt(gx * gx + gy * gy + 1e-5)
        nx = -gx / gnorm
        ny = -gy / gnorm

        grid_x, grid_y = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
        sample_x = np.clip(grid_x + nx * 5.0, 0.0, float(sw - 1))
        sample_y = np.clip(grid_y + ny * 5.0, 0.0, float(sh - 1))
        env_s = cv2.resize(low_bg_env, (sw, sh), interpolation=cv2.INTER_AREA)
        bg_env_s = cv2.remap(env_s, sample_x, sample_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        falloff_s = np.clip(1.0 - (np.abs(sdf) - 0.5) / radius, 0.0, 1.0).astype(np.float32)

        falloff = cv2.resize(falloff_s, (w, h), interpolation=cv2.INTER_LINEAR)
        bg_env_local = cv2.resize(bg_env_s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        d_in = cv2.distanceTransform(m_bin, cv2.DIST_L2, 5)
        d_out = cv2.distanceTransform(1 - m_bin, cv2.DIST_L2, 5)
        sdf = d_in - d_out

        gx = cv2.Sobel(sdf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sdf, cv2.CV_32F, 0, 1, ksize=3)
        gnorm = np.sqrt(gx * gx + gy * gy + 1e-5)
        nx = -gx / gnorm
        ny = -gy / gnorm

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        sample_x = np.clip(grid_x + nx * 10.0, 0.0, float(w - 1))
        sample_y = np.clip(grid_y + ny * 10.0, 0.0, float(h - 1))
        bg_env_local = cv2.remap(
            low_bg_env, sample_x, sample_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        contour_distance = np.abs(sdf) - 0.5
        falloff = np.clip(1.0 - contour_distance / radius, 0.0, 1.0)

    edge_mask = falloff > 1e-3
    if not np.any(edge_mask):
        return fg_f
    out = fg_f.copy()
    fg_e = fg_f[edge_mask]
    env_e = bg_env_local[edge_mask]
    f_e = falloff[edge_mask]
    headroom = 1.0 - np.max(fg_e, axis=-1)
    gain = f_e * f_e * headroom * (0.18 * float(np.clip(strength, 0.0, 1.0)))
    out[edge_mask] = fg_e * (1.0 + env_e * gain[..., None])
    return out


def _apply_screen_glass_glare(
    orig_fg_f: np.ndarray,
    bg_f: np.ndarray,
    glare_opacity: float,
) -> np.ndarray:
    """Keep emissive screen crisp & untinted, adding only subtle glass specular reflection."""
    if glare_opacity <= 1e-4:
        return orig_fg_f
    low_bg = _fast_lowpass(bg_f, sigma_full=18.0, scale=0.25)
    glare = np.clip((low_bg - 0.40) / 0.60, 0.0, 1.0)
    screened = 1.0 - (1.0 - orig_fg_f) * (1.0 - glare)
    return orig_fg_f * (1.0 - glare_opacity) + screened * glare_opacity


_GRAIN_TILE: Optional[np.ndarray] = None


def _match_sensor_grain(
    fg_f: np.ndarray,
    bg_f: np.ndarray,
    fg_mask_u8: np.ndarray,
    grain_strength: float,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    global _GRAIN_TILE
    if grain_strength <= 1e-4:
        return fg_f
    h, w = fg_f.shape[:2]
    # Fast noise estimation on a central 256x256 crop of bg_f
    cy, cx = h // 2, w // 2
    crop_bg = bg_f[max(0, cy - 128): min(h, cy + 128), max(0, cx - 128): min(w, cx + 128)]
    hp = crop_bg - cv2.GaussianBlur(crop_bg, (5, 5), 1.0)
    noise_std = min(float(np.std(hp)), 0.035)
    if _GRAIN_TILE is None or _GRAIN_TILE.shape[0] < (h + 256) or _GRAIN_TILE.shape[1] < (w + 256):
        rng = np.random.default_rng(42)
        small_tile = rng.standard_normal((256, 256, 3), dtype=np.float32)
        reps_y = ((h + 256) + 255) // 256
        reps_x = ((w + 256) + 255) // 256
        _GRAIN_TILE = np.tile(small_tile, (reps_y, reps_x, 1))[: h + 256, : w + 256].copy()
    sy = (int(frame_idx) * 73) % 256 if frame_idx is not None else 0
    sx = (int(frame_idx) * 137) % 256 if frame_idx is not None else 0
    tile = _GRAIN_TILE[sy : sy + h, sx : sx + w]
    return np.clip(fg_f + tile * (noise_std * grain_strength), 0.0, 1.0)


def verify_and_polish_composite_pass3(
    comp_f: np.ndarray,
    fg_f: np.ndarray,
    orig_fg_f: np.ndarray,
    bg_f: np.ndarray,
    alpha: np.ndarray,
    src_f: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    PASS-3 POST-COMPOSITE AUDIT, SOURCE-PIXEL COPY-BACK & TANGENTIAL ANTI-ALIASING:
      A. Interior Source-Pixel Fidelity (`alpha > 0.90`): restores 100% of micro-details
         (skin texture, fingernails, screen text) and reverts abnormal redness/darkness drift.
      B. Boundary Missing-Pixel Recovery (`0.05 < alpha < 0.88`): restores clean un-spilled
         foreground pixels from the source frame so delicate edges are never over-eroded.
      C. Tangential Sub-Pixel Anti-Aliasing: integrates 5-tap premultiplied
         Gaussian samples along the contour to smooth the soft boundary.
    """
    h, w = comp_f.shape[:2]
    alpha_f = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    fg_work = fg_f.astype(np.float32).copy()

    # A. Interior Source-Pixel Micro-Detail Restoration & Drift Copy-Back
    interior = alpha_f > 0.90
    detail_restored_px = 0
    drift_reverted_px = 0
    if np.any(interior):
        detail_delta = orig_fg_f - fg_work
        detail_weight = cv2.GaussianBlur(alpha_f, (0, 0), sigmaX=1.6)
        low_delta = cv2.GaussianBlur(detail_delta * alpha_f[..., None], (0, 0), sigmaX=1.6)
        hi_diff = detail_delta - low_delta / np.maximum(detail_weight[..., None], 1e-4)
        need_detail = interior & (np.max(np.abs(hi_diff), axis=-1) > 0.006)
        detail_restored_px = int(np.sum(need_detail))
        if detail_restored_px > 0:
            fg_work[need_detail] = np.clip(fg_work[need_detail] + hi_diff[need_detail], 0.0, 1.0)

        # Revert unnatural red-channel drift or heavy brightness deviation back toward source foreground
        red_drift = (fg_work[..., 2] - fg_work[..., 1]) - (orig_fg_f[..., 2] - orig_fg_f[..., 1])
        l2_drift = np.linalg.norm(fg_work - orig_fg_f, axis=-1)
        red_t = np.clip((red_drift - 0.065) / 0.04, 0.0, 1.0)
        l2_t = np.clip((l2_drift - 0.24) / 0.08, 0.0, 1.0)
        red_weight = red_t * red_t * (3.0 - 2.0 * red_t)
        l2_weight = l2_t * l2_t * (3.0 - 2.0 * l2_t)
        drift_weight = np.maximum(red_weight, l2_weight) * interior * 0.65
        drift_reverted_px = int(np.count_nonzero(drift_weight))
        if drift_reverted_px > 0:
            fg_work += (orig_fg_f - fg_work) * drift_weight[..., None]

    # B. Boundary Missing-Pixel Audit & Copy-Back from Source
    edge_restored_px = 0
    if src_f is not None and np.any(interior):
        core_mask_f = (alpha_f > 0.85).astype(np.float32)
        fg_num = cv2.GaussianBlur(orig_fg_f * core_mask_f[..., None], (0, 0), sigmaX=3.5)
        fg_den = np.maximum(cv2.GaussianBlur(core_mask_f, (0, 0), sigmaX=3.5)[..., None], 1e-4)
        near_orig_fg = fg_num / fg_den

        no_green_spill = src_f[..., 1] < (np.maximum(src_f[..., 0], src_f[..., 2]) - 0.02)
        close_to_fg = np.linalg.norm(src_f - near_orig_fg, axis=-1) < 0.055
        b_zone = (alpha_f > 0.05) & (alpha_f < 0.88) & no_green_spill & close_to_fg
        edge_restored_px = int(np.sum(b_zone))
        if edge_restored_px > 0:
            fg_work[b_zone] = fg_work[b_zone] * 0.25 + orig_fg_f[b_zone] * 0.75

    # C. Sub-Pixel Tangential Contour Anti-Aliasing (5-tap MLAA along contour tangent)
    edge_core = (alpha_f > 0.005) & (alpha_f < 0.995)
    aa_polished_px = int(np.sum(edge_core))

    if aa_polished_px > 0:
        gx = cv2.Sobel(alpha_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(alpha_f, cv2.CV_32F, 0, 1, ksize=3)
        gnorm = np.sqrt(gx * gx + gy * gy + 1e-6)
        # Unit tangent along the edge contour (-ny, nx)
        tx = gy / gnorm
        ty = -gx / gnorm

        lap_alpha = np.abs(cv2.Laplacian(alpha_f, cv2.CV_32F, ksize=3))
        jagged_w = np.where(edge_core, np.clip(lap_alpha * 3.2, 0.25, 1.0), 0.0)[..., None]

        edge_y, edge_x = np.nonzero(edge_core)
        if aa_polished_px <= alpha_f.size * 0.40:
            premultiplied = np.dstack((fg_work * alpha_f[..., None], alpha_f))
            premultiplied_core = premultiplied[edge_y, edge_x]
            gx_core = gx[edge_y, edge_x]
            gy_core = gy[edge_y, edge_x]
            gnorm_core = np.sqrt(gx_core * gx_core + gy_core * gy_core + 1e-6)
            tx_core = gy_core / gnorm_core
            ty_core = -gx_core / gnorm_core
            jagged_w_core = np.clip(lap_alpha[edge_y, edge_x] * 3.2, 0.25, 1.0)[:, None]

            accumulated_core = premultiplied_core * 0.30
            x_core = edge_x.astype(np.float32)
            y_core = edge_y.astype(np.float32)
            remap_chunk_size = 32000
            for k_step, k_wgt in ((-2, 0.12), (-1, 0.23), (1, 0.23), (2, 0.12)):
                sx = np.clip(x_core + tx_core * (k_step * 0.75), 0.0, float(w - 1))
                sy = np.clip(y_core + ty_core * (k_step * 0.75), 0.0, float(h - 1))
                samples = np.empty_like(premultiplied_core)
                for start in range(0, aa_polished_px, remap_chunk_size):
                    end = min(start + remap_chunk_size, aa_polished_px)
                    packed = cv2.remap(
                        premultiplied,
                        sx[start:end].reshape(1, -1),
                        sy[start:end].reshape(1, -1),
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    samples[start:end] = packed.reshape(-1, 4)
                accumulated_core += samples * k_wgt

            filtered_core = (
                premultiplied_core * (1.0 - jagged_w_core)
                + accumulated_core * jagged_w_core
            )
            comp_aa = bg_f * (1.0 - alpha_f[..., None]) + premultiplied[..., :3]
            comp_aa[edge_y, edge_x] = (
                bg_f[edge_y, edge_x] * (1.0 - filtered_core[:, 3:4])
                + filtered_core[:, :3]
            )
            return np.clip(comp_aa * 255.0, 0, 255).astype(np.uint8), {
                "detail_restored_px": detail_restored_px,
                "drift_reverted_px": drift_reverted_px,
                "edge_restored_px": edge_restored_px,
                "aa_polished_px": aa_polished_px,
            }

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        # Filter color and coverage together so pixels outside the cutout
        # contribute zero foreground color, even with bilinear sampling.
        premultiplied = np.dstack((fg_work * alpha_f[..., None], alpha_f))
        accumulated = np.zeros_like(premultiplied)
        taps = ((-2, 0.12), (-1, 0.23), (0, 0.30), (1, 0.23), (2, 0.12))
        for k_step, k_wgt in taps:
            if k_step == 0:
                accumulated += premultiplied * k_wgt
            else:
                sx = np.clip(grid_x + tx * (k_step * 0.75), 0.0, float(w - 1))
                sy = np.clip(grid_y + ty * (k_step * 0.75), 0.0, float(h - 1))
                accumulated += cv2.remap(
                    premultiplied, sx, sy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                ) * k_wgt

        filtered = premultiplied * (1.0 - jagged_w) + accumulated * jagged_w
        comp_aa = bg_f * (1.0 - filtered[..., 3:4]) + filtered[..., :3]
        return np.clip(comp_aa * 255.0, 0, 255).astype(np.uint8), {
            "detail_restored_px": detail_restored_px,
            "drift_reverted_px": drift_reverted_px,
            "edge_restored_px": edge_restored_px,
            "aa_polished_px": aa_polished_px,
        }

    # No soft boundary band (e.g. crisp hard binary mask), still apply interior micro-detail restore cleanly
    comp_rebuilt = bg_f * (1.0 - alpha_f[..., None]) + fg_work * alpha_f[..., None]
    return np.clip(comp_rebuilt * 255.0, 0, 255).astype(np.uint8), {
        "detail_restored_px": detail_restored_px,
        "drift_reverted_px": drift_reverted_px,
        "edge_restored_px": edge_restored_px,
        "aa_polished_px": 0,
    }


def _match_black_white_floor(
    fg_f: np.ndarray,
    bg_f: np.ndarray,
    fg_mask_u8: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Gently align foreground shadow black-point (P5) and highlight ceiling (P95)
    toward the target background so deep shadows never look crushed or washed out,
    while preserving original chromaticity ratios.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.01:
        return fg_f
    fg_sel = fg_mask_u8 > 0
    ys, xs = np.where(fg_sel)
    if len(xs) < 32:
        return fg_f

    h, w = fg_f.shape[:2]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    if (h * w) > 120_000:
        fg_sub = fg_f[y0:y1:4, x0:x1:4]
        sel_sub = fg_sel[y0:y1:4, x0:x1:4]
        y_fg_s = 0.114 * fg_sub[..., 0] + 0.587 * fg_sub[..., 1] + 0.299 * fg_sub[..., 2]
        fg_vals = y_fg_s[sel_sub] if np.any(sel_sub) else y_fg_s.reshape(-1)
        bg_sub = bg_f[::4, ::4]
        bg_vals = (0.114 * bg_sub[..., 0] + 0.587 * bg_sub[..., 1] + 0.299 * bg_sub[..., 2]).reshape(-1)
    else:
        y_fg_full = 0.114 * fg_f[..., 0] + 0.587 * fg_f[..., 1] + 0.299 * fg_f[..., 2]
        fg_vals = y_fg_full[fg_sel]
        bg_vals = (0.114 * bg_f[..., 0] + 0.587 * bg_f[..., 1] + 0.299 * bg_f[..., 2]).reshape(-1)

    fg_p5 = float(np.percentile(fg_vals, 5))
    fg_p95 = float(np.percentile(fg_vals, 95))
    fg_span = max(fg_p95 - fg_p5, 0.08)

    bg_p5 = float(np.clip(np.percentile(bg_vals, 5), 0.02, 0.28))
    bg_p95 = float(np.clip(np.percentile(bg_vals, 95), 0.65, 0.98))

    blend = s * 0.65
    tgt_p5 = fg_p5 * (1.0 - blend) + bg_p5 * blend
    tgt_p95 = fg_p95 * (1.0 - blend) + bg_p95 * blend
    tgt_span = max(tgt_p95 - tgt_p5, 0.08)

    roi_fg = fg_f[y0:y1, x0:x1]
    roi_sel = fg_sel[y0:y1, x0:x1]
    y_roi = 0.114 * roi_fg[..., 0] + 0.587 * roi_fg[..., 1] + 0.299 * roi_fg[..., 2]
    y_safe = np.maximum(y_roi, 1e-4)
    y_norm = np.clip((y_roi - fg_p5) / fg_span, -0.08, 1.15)
    y_mapped = np.clip(tgt_p5 + y_norm * tgt_span, 0.0, 1.0)
    ratio = np.clip(y_mapped / y_safe, 0.60, 1.60)[..., None]

    out = fg_f.copy()
    mapped_roi = np.clip(roi_fg * ratio, 0.0, 1.0)
    out_roi = out[y0:y1, x0:x1]
    out_roi[roi_sel] = mapped_roi[roi_sel]
    return out


def composite_with_relighting(
    fg_clean_bgr: np.ndarray,
    alpha: np.ndarray,
    new_bg_bgr: np.ndarray,
    src_full_bgr: Optional[np.ndarray] = None,
    screen_mask: Optional[np.ndarray] = None,
    cfg: Optional[RelightConfig] = None,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Composite extracted sub-pixel foreground (hands + product) onto a new background
    with Dual-Zone Relighting:
      - Hands & device body absorb ambient color temperature, 2D directional light, and edge light-wrap.
      - Emissive screen UI remains protected, crisp, and true-color with subtle glass reflection.
    """
    if cfg is None:
        cfg = RelightConfig()

    h, w = fg_clean_bgr.shape[:2]
    bg_bgr = new_bg_bgr.copy()
    if (bg_bgr.shape[1], bg_bgr.shape[0]) != (w, h):
        bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    fg_mask_u8 = (alpha > 0.15).astype(np.uint8)

    if cfg.clean_ai_ghost:
        bg_bgr = clean_ai_background_ghost(bg_bgr, fg_mask_u8)

    if cfg.bg_defocus_sigma > 0.1:
        bg_bgr = cv2.GaussianBlur(bg_bgr, (0, 0), sigmaX=float(cfg.bg_defocus_sigma))

    # Stage 0: Background Color Grading & Reverse Light/Tone Match (BG -> Source)
    if (
        float(getattr(cfg, "bg_match_source_strength", 0.0)) > 1e-3
        or abs(float(getattr(cfg, "bg_brightness", 0.0))) > 1e-3
        or abs(float(getattr(cfg, "bg_contrast", 1.0)) - 1.0) > 1e-3
        or abs(float(getattr(cfg, "bg_saturation", 1.0)) - 1.0) > 1e-3
        or abs(float(getattr(cfg, "bg_temperature", 0.0))) > 1e-3
        or abs(float(getattr(cfg, "bg_tint", 0.0))) > 1e-3
    ):
        scr_for_bg = screen_mask
        if scr_for_bg is None and cfg.protect_emissive_screen and float(getattr(cfg, "bg_match_source_strength", 0.0)) > 1e-3:
            scr_for_bg = detect_emissive_screen_mask(fg_clean_bgr, fg_mask_u8)
            screen_mask = scr_for_bg
        bg_bgr = grade_and_match_background(
            bg_bgr,
            fg_clean_bgr,
            fg_mask_u8,
            screen_mask=scr_for_bg,
            match_source_strength=float(getattr(cfg, "bg_match_source_strength", 0.0)),
            brightness=float(getattr(cfg, "bg_brightness", 0.0)),
            contrast=float(getattr(cfg, "bg_contrast", 1.0)),
            saturation=float(getattr(cfg, "bg_saturation", 1.0)),
            temperature=float(getattr(cfg, "bg_temperature", 0.0)),
            tint=float(getattr(cfg, "bg_tint", 0.0)),
        )

    # Stage 1: CIE Lab Ambient Color Temperature & Exposure Harmonization
    fg_harmonized_bgr = _transfer_ambient_lab(
        fg_clean_bgr,
        bg_bgr,
        src_full_bgr,
        fg_mask_u8,
        strength=float(np.clip(cfg.ambient_color_strength, 0.0, 1.0)),
    )

    fg_f = fg_harmonized_bgr.astype(np.float32) / 255.0
    orig_fg_f = fg_clean_bgr.astype(np.float32) / 255.0
    bg_f = bg_bgr.astype(np.float32) / 255.0
    src_f = src_full_bgr.astype(np.float32) / 255.0 if src_full_bgr is not None else None

    # Stage 1B: Black-point & highlight tone-curve alignment (before Stage 4 screen protect)
    bp_strength = float(getattr(cfg, "black_point_match", 0.0))
    if bp_strength > 0.01:
        fg_f = _match_black_white_floor(fg_f, bg_f, fg_mask_u8, bp_strength)

    # Stage 2: 2D Directional Illumination Field Transfer
    fg_f = _apply_2d_directional_lighting(
        fg_f,
        bg_f,
        src_f,
        strength=float(np.clip(cfg.directional_light_strength, 0.0, 1.0)),
    )

    # Stage 3: Physical Edge Light-Wrap around fingers and device perimeter
    if not cfg.linear_compositing:
        fg_f = _apply_physical_light_wrap(
            fg_f,
            bg_f,
            alpha,
            radius=int(cfg.light_wrap_Radius),
            strength=float(np.clip(cfg.light_wrap_strength, 0.0, 1.0)),
        )

    # Stage 4: Emissive Screen Protection (keep UI text/colors crisp & true, cropped to tight ROI)
    if cfg.protect_emissive_screen:
        if screen_mask is None:
            screen_mask = detect_emissive_screen_mask(fg_clean_bgr, fg_mask_u8)
        if screen_mask is not None and np.any(screen_mask > 0):
            sy_pts, sx_pts = np.where(screen_mask > 0)
            sy0 = max(0, int(sy_pts.min()) - 8)
            sy1 = min(h, int(sy_pts.max()) + 9)
            sx0 = max(0, int(sx_pts.min()) - 8)
            sx1 = min(w, int(sx_pts.max()) + 9)
            scr_u8 = (screen_mask[sy0:sy1, sx0:sx1] > 0).astype(np.uint8)
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            scr_eroded = cv2.erode(scr_u8, k3, iterations=1)
            scr_alpha = cv2.GaussianBlur(scr_eroded.astype(np.float32), (5, 5), 0)
            scr_outer = cv2.dilate(scr_u8, k3, iterations=1) > 0
            scr_alpha[~scr_outer] = 0.0
            scr_alpha = scr_alpha[..., None]
            screen_layer = _apply_screen_glass_glare(
                orig_fg_f[sy0:sy1, sx0:sx1],
                bg_f[sy0:sy1, sx0:sx1],
                float(np.clip(cfg.screen_glare_opacity, 0.0, 0.5)),
            )
            fg_roi = fg_f[sy0:sy1, sx0:sx1]
            fg_f[sy0:sy1, sx0:sx1] = fg_roi * (1.0 - scr_alpha) + screen_layer * scr_alpha

    # Stage 5: Sensor Grain Matching (dynamic per frame when frame_idx is given)
    fg_f = _match_sensor_grain(
        fg_f,
        bg_f,
        fg_mask_u8,
        float(np.clip(cfg.grain_match, 0.0, 1.0)),
        frame_idx=frame_idx,
    )

    # Optional subtle edge softness match to optical background
    edge_blur = float(getattr(cfg, "edge_blur_match", 0.0))
    if edge_blur > 0.05:
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_zone = cv2.dilate(((alpha > 0.01) & (alpha < 0.99)).astype(np.uint8), k3) > 0
        if np.any(edge_zone):
            fg_soft = cv2.GaussianBlur(fg_f, (0, 0), sigmaX=edge_blur)
            fg_f = np.where(edge_zone[..., None], fg_soft, fg_f)

    # Final Sub-Pixel Alpha Composite (Linear RGB or sRGB) + Optional Pass-3 Tangential AA
    a3 = np.clip(alpha, 0.0, 1.0)[..., None]
    if cfg.linear_compositing:
        fg_lin = _srgb_to_linear(fg_f)
        bg_lin = _srgb_to_linear(bg_f)
        if int(cfg.light_wrap_Radius) > 0 and float(cfg.light_wrap_strength) > 1e-3:
            fg_lin = _apply_physical_light_wrap(
                fg_lin,
                bg_lin,
                alpha,
                radius=int(cfg.light_wrap_Radius),
                strength=float(np.clip(cfg.light_wrap_strength, 0.0, 1.0)),
            )
        comp_lin = bg_lin * (1.0 - a3) + fg_lin * a3
        comp_f = _linear_to_srgb(comp_lin)
        if cfg.pass3_aa_and_restore:
            fg_f = _linear_to_srgb(fg_lin)
    else:
        comp_f = bg_f * (1.0 - a3) + fg_f * a3

    if cfg.pass3_aa_and_restore:
        comp_u8, _ = verify_and_polish_composite_pass3(
            comp_f=comp_f,
            fg_f=fg_f,
            orig_fg_f=orig_fg_f,
            bg_f=bg_f,
            alpha=alpha,
            src_f=src_f,
        )
        return comp_u8
    return np.clip(comp_f * 255.0, 0, 255).astype(np.uint8)

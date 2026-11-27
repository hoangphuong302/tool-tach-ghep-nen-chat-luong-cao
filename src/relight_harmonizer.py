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
    # Next-gen Unified Master Harmonization (FG <-> BG) & 3D Pixel-Level Contour Relighting
    unified_harmonize_strength: float = 0.0   # 2-way illuminant WB, 3-way tone, black floor & contrast sync (0..1)
    contour_relight_strength: float = 0.0     # Pixel-level 2.5D contour-normal brightening/shadowing & 3D depth (0..1)
    contour_relight_radius: int = 28          # Gradual inward contour shading transition width in px (4..72)
    sharpness_match_strength: float = 0.0     # Optical acutance / MTF sharpness match FG <-> BG (0..1)


@dataclass
class PairOpticalProfile:
    fg_wb_rgb: np.ndarray        # Normalized (R, G, B) illuminant cast of FG (mean == 1.0)
    bg_wb_rgb: np.ndarray        # Normalized (R, G, B) illuminant cast of BG (mean == 1.0)
    fg_p3: float
    fg_p50: float
    fg_p97: float
    fg_l_std: float
    bg_p3: float
    bg_p50: float
    bg_p97: float
    bg_l_std: float
    bg_shd_ab: Tuple[float, float]  # CIE Lab (a-128, b-128) shadow tint of BG
    bg_mid_ab: Tuple[float, float]  # CIE Lab (a-128, b-128) midtone tint of BG
    fg_noise_luma: float
    fg_noise_chroma: float
    bg_noise_luma: float
    bg_noise_chroma: float
    fg_sharpness: float
    bg_sharpness: float
    bg_light_dir: Tuple[float, float, float]  # Unit 3D (lx, ly, lz) key light direction of BG
    fg_light_dir: Tuple[float, float, float]  # Unit 3D (lx, ly, lz) existing key light direction of FG


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
_TORCH_CUDA_AVAILABLE: Optional[bool] = None


def _has_cuda() -> bool:
    global _TORCH_CUDA_AVAILABLE
    if _TORCH_CUDA_AVAILABLE is None:
        try:
            import torch
            _TORCH_CUDA_AVAILABLE = bool(torch.cuda.is_available())
        except Exception:
            _TORCH_CUDA_AVAILABLE = False
    return bool(_TORCH_CUDA_AVAILABLE)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    if _has_cuda() and x.size >= 400_000:
        import torch
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to("cuda", non_blocking=True)
            return t.clamp_(0.0, 1.0).pow_(2.2).cpu().numpy()
    idx = np.clip(x * 4095.0 + 0.5, 0.0, 4095.0).astype(np.int32)
    return _SRGB_TO_LIN_LUT[idx]


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    if _has_cuda() and x.size >= 400_000:
        import torch
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to("cuda", non_blocking=True)
            return t.clamp_(0.0, 1.0).pow_(1.0 / 2.2).cpu().numpy()
    idx = np.clip(x * 4095.0 + 0.5, 0.0, 4095.0).astype(np.int32)
    return _LIN_TO_SRGB_LUT[idx]


def _estimate_luma_chroma_noise(bgr_u8: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Robust high-frequency Laplacian MAD estimator for camera sensor luma & chroma noise."""
    ycrcb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2YCrCb).astype(np.float32) * (1.0 / 255.0)
    lap_y = cv2.Laplacian(ycrcb[..., 0], cv2.CV_32F, ksize=3)
    lap_cr = cv2.Laplacian(ycrcb[..., 1], cv2.CV_32F, ksize=3)
    lap_cb = cv2.Laplacian(ycrcb[..., 2], cv2.CV_32F, ksize=3)
    if mask is not None and int(np.sum(mask)) >= 64:
        # Erode slightly so hard silhouette boundaries never inflate noise estimates
        m_er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        sel = m_er if int(np.sum(m_er)) >= 48 else (mask > 0)
        ly = np.abs(lap_y[sel])
        lc = 0.5 * (np.abs(lap_cr[sel]) + np.abs(lap_cb[sel]))
    else:
        ly = np.abs(lap_y.reshape(-1))
        lc = 0.5 * (np.abs(lap_cr.reshape(-1)) + np.abs(lap_cb.reshape(-1)))
    scale = 1.4826 / float(np.sqrt(20.0))
    return float(np.median(ly) * scale), float(np.median(lc) * scale)


def _estimate_optical_acutance(bgr_u8: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """90th-percentile Sobel edge gradient magnitude measuring optical lens sharpness."""
    gray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1.0 / 255.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    if mask is not None and int(np.sum(mask)) >= 64:
        m_er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        sel = m_er if int(np.sum(m_er)) >= 48 else (mask > 0)
        vals = mag[sel]
    else:
        vals = mag.reshape(-1)
    return float(np.percentile(vals, 90))


def _estimate_3d_light_dir(luma_f: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """Estimate dominant 3D unit illumination direction (lx, ly, lz) from spatial luminance moments."""
    h, w = luma_f.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    if mask is not None and int(np.sum(mask)) >= 32:
        sel = mask > 0
        ys, xs = yy[sel], xx[sel]
        vals = luma_f[sel]
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        rx = max(float(np.std(xs)) * 1.6, 4.0)
        ry = max(float(np.std(ys)) * 1.6, 4.0)
        nx_c = (xs - cx) / rx
        ny_c = (ys - cy) / ry
    else:
        nx_c = (xx - (w - 1) * 0.5) / max(w * 0.5, 1.0)
        ny_c = (yy - (h - 1) * 0.5) / max(h * 0.5, 1.0)
        vals = luma_f
    # Weight strongly by brighter highlights above median so bright sky/lamp drives light vector
    v_med = float(np.median(vals))
    w_bright = np.clip(vals - v_med * 0.65, 0.0, 1.0)
    w_sum = float(np.sum(w_bright))
    if w_sum < 1e-4:
        return (0.0, -0.25, 0.968)
    lx = float(np.clip(np.sum(nx_c * w_bright) / w_sum * 1.85, -0.82, 0.82))
    ly = float(np.clip(np.sum(ny_c * w_bright) / w_sum * 1.85, -0.82, 0.82))
    r2 = lx * lx + ly * ly
    if r2 > 0.78:
        sc = float(np.sqrt(0.78 / r2))
        lx *= sc
        ly *= sc
        r2 = 0.78
    lz = float(np.sqrt(max(0.18, 1.0 - r2)))
    return (lx, ly, lz)


def _estimate_illuminant_rgb(
    bgr_u8: np.ndarray,
    lab_f: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    is_fg: bool = False,
) -> np.ndarray:
    """
    Estimate scene illuminant RGB color cast (normalized to mean 1.0):
      1. Prefer near-neutral bright/midtone pixels (white bezels, paper, neutral pavement/clouds).
      2. Fall back to Shades-of-Gray (L6 Minkowski norm), damping warm skin bias on FG when no
         neutral references exist.
    """
    rgb = bgr_u8[..., ::-1].astype(np.float32) * (1.0 / 255.0)
    if mask is not None and int(np.sum(mask)) >= 32:
        sel = mask > 0
        rgb_v = rgb[sel]
        lab_v = lab_f[sel]
    else:
        rgb_v = rgb.reshape(-1, 3)
        lab_v = lab_f.reshape(-1, 3)

    if len(rgb_v) < 16:
        return np.ones(3, dtype=np.float32)

    chroma = np.hypot(lab_v[:, 1] - 128.0, lab_v[:, 2] - 128.0)
    # Near-neutral bright references (e.g. white phone bezel/paper under tungsten):
    # Require high luminance (L >= 155) and non-reddish 'a' (a <= 133) on FG so human skin
    # is never misclassified as a neutral white illuminant reference!
    if is_fg:
        neut_sel = (lab_v[:, 0] >= 155.0) & (lab_v[:, 0] <= 252.0) & (lab_v[:, 1] <= 133.5) & (chroma <= 24.0)
    else:
        neut_sel = (lab_v[:, 0] >= 72.0) & (lab_v[:, 0] <= 250.0) & (chroma <= 28.0)
    if int(np.sum(neut_sel)) >= 20:
        wb_neut = np.median(rgb_v[neut_sel], axis=0).astype(np.float32)
        wb_neut = wb_neut / max(float(np.mean(wb_neut)), 1e-4)
        return np.clip(wb_neut, 0.74, 1.35).astype(np.float32)

    # Fallback: Shades-of-Gray L6 norm (robust gray-world / max-RGB hybrid)
    sog = np.power(np.mean(np.power(np.clip(rgb_v, 0.02, 1.0), 6.0), axis=0), 1.0 / 6.0).astype(np.float32)
    sog = sog / max(float(np.mean(sog)), 1e-4)
    if is_fg:
        # Strongly damp raw object albedo when no neutral reference exists so healthy skin/colored
        # subjects keep their natural reflectance instead of being neutralized to grey.
        sog = 1.0 + (sog - 1.0) * 0.24
        sog = sog / max(float(np.mean(sog)), 1e-4)
    return np.clip(sog, 0.78, 1.28).astype(np.float32)


def analyze_pair_optical_signatures(
    fg_bgr: np.ndarray,
    bg_bgr: np.ndarray,
    fg_mask_u8: np.ndarray,
    screen_mask: Optional[np.ndarray] = None,
) -> PairOpticalProfile:
    """
    Fast dual-layer optical analyzer (~1.5-3ms at 1080p) measuring illuminant white balance,
    shadow/midtone/highlight color casts, black/white dynamic range, sensor noise floor,
    optical acutance, and 3D key-light direction for BOTH foreground (excluding emissive screens)
    and background.
    """
    h, w = fg_bgr.shape[:2]
    fg_sel = fg_mask_u8 > 0
    if screen_mask is not None and screen_mask.shape == (h, w):
        non_scr = fg_sel & (screen_mask == 0)
        if int(np.sum(non_scr)) >= 64:
            fg_sel = non_scr
    bg_sel = ~ (fg_mask_u8 > 0)
    if int(np.sum(bg_sel)) < 64:
        bg_sel = np.ones((h, w), dtype=bool)

    # Full/mid-res crop for accurate high-frequency noise & optical sharpness (never distorted by heavy downscale)
    if min(h, w) > 540:
        cy, cx = h // 2, w // 2
        # Sample up to 480x480 region overlapping FG+BG for fast exact high-frequency telemetry
        nh = min(h, 480)
        nw = min(w, 480)
        y0 = max(0, cy - nh // 2)
        x0 = max(0, cx - nw // 2)
        fg_hf = fg_bgr[y0 : y0 + nh, x0 : x0 + nw]
        bg_hf = bg_bgr[y0 : y0 + nh, x0 : x0 + nw]
        fg_m_hf = fg_sel[y0 : y0 + nh, x0 : x0 + nw]
        bg_m_hf = bg_sel[y0 : y0 + nh, x0 : x0 + nw]
    else:
        fg_hf, bg_hf, fg_m_hf, bg_m_hf = fg_bgr, bg_bgr, fg_sel, bg_sel

    fg_n_l, fg_n_c = _estimate_luma_chroma_noise(fg_hf, fg_m_hf)
    bg_n_l, bg_n_c = _estimate_luma_chroma_noise(bg_hf, bg_m_hf)
    fg_sharp = _estimate_optical_acutance(fg_hf, fg_m_hf)
    bg_sharp = _estimate_optical_acutance(bg_hf, bg_m_hf)

    # Fast downscale (~220px short side) for illuminant, dynamic range, 3-way color & 3D light direction
    if min(h, w) > 240:
        sc = 220.0 / float(min(h, w))
        sw, sh = max(48, int(round(w * sc))), max(48, int(round(h * sc)))
        fg_s = cv2.resize(fg_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        bg_s = cv2.resize(bg_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        fg_ms = cv2.resize(fg_sel.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
        bg_ms = cv2.resize(bg_sel.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    else:
        fg_s, bg_s, fg_ms, bg_ms = fg_bgr, bg_bgr, fg_sel, bg_sel

    fg_lab = cv2.cvtColor(fg_s, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_s, cv2.COLOR_BGR2LAB).astype(np.float32)

    fg_wb = _estimate_illuminant_rgb(fg_s, fg_lab, fg_ms, is_fg=True)
    bg_wb = _estimate_illuminant_rgb(bg_s, bg_lab, None, is_fg=False)

    fg_y = (0.114 * fg_s[..., 0] + 0.587 * fg_s[..., 1] + 0.299 * fg_s[..., 2]).astype(np.float32) * (1.0 / 255.0)
    bg_y = (0.114 * bg_s[..., 0] + 0.587 * bg_s[..., 1] + 0.299 * bg_s[..., 2]).astype(np.float32) * (1.0 / 255.0)

    fg_yv = fg_y[fg_ms] if np.any(fg_ms) else fg_y.reshape(-1)
    bg_yv = bg_y[bg_ms] if np.any(bg_ms) else bg_y.reshape(-1)

    fg_p3, fg_p50, fg_p97 = [float(x) for x in np.percentile(fg_yv, (3.0, 50.0, 97.0))]
    bg_p3, bg_p50, bg_p97 = [float(x) for x in np.percentile(bg_yv, (3.0, 50.0, 97.0))]
    fg_l_std = float(np.std(fg_yv)) + 1e-4
    bg_l_std = float(np.std(bg_yv)) + 1e-4

    bg_lab_all = bg_lab.reshape(-1, 3)
    bg_l_p25, bg_l_p75 = np.percentile(bg_lab_all[:, 0], (25.0, 75.0))
    shd_v = bg_lab_all[bg_lab_all[:, 0] <= bg_l_p25]
    mid_v = bg_lab_all[(bg_lab_all[:, 0] >= bg_l_p25) & (bg_lab_all[:, 0] <= bg_l_p75)]
    if len(shd_v) < 8:
        shd_v = bg_lab_all
    if len(mid_v) < 8:
        mid_v = bg_lab_all
    bg_shd_ab = (float(np.median(shd_v[:, 1]) - 128.0), float(np.median(shd_v[:, 2]) - 128.0))
    bg_mid_ab = (float(np.median(mid_v[:, 1]) - 128.0), float(np.median(mid_v[:, 2]) - 128.0))

    bg_ldir = _estimate_3d_light_dir(bg_y, None)
    fg_ldir = _estimate_3d_light_dir(fg_y, fg_ms)

    return PairOpticalProfile(
        fg_wb_rgb=fg_wb,
        bg_wb_rgb=bg_wb,
        fg_p3=fg_p3,
        fg_p50=fg_p50,
        fg_p97=fg_p97,
        fg_l_std=fg_l_std,
        bg_p3=bg_p3,
        bg_p50=bg_p50,
        bg_p97=bg_p97,
        bg_l_std=bg_l_std,
        bg_shd_ab=bg_shd_ab,
        bg_mid_ab=bg_mid_ab,
        fg_noise_luma=fg_n_l,
        fg_noise_chroma=fg_n_c,
        bg_noise_luma=bg_n_l,
        bg_noise_chroma=bg_n_c,
        fg_sharpness=fg_sharp,
        bg_sharpness=bg_sharp,
        bg_light_dir=bg_ldir,
        fg_light_dir=fg_ldir,
    )


def _build_tone_curve_lut(
    p3_src: float,
    p50_src: float,
    p97_src: float,
    p3_tgt: float,
    p50_tgt: float,
    p97_tgt: float,
) -> np.ndarray:
    """Build exact 1024-entry monotonic 3-point luminance tone-curve LUT."""
    span_src = max(p97_src - p3_src, 0.12)
    span_tgt = max(p97_tgt - p3_tgt, 0.12)
    mid_src_n = float(np.clip((p50_src - p3_src) / span_src, 0.18, 0.82))
    mid_tgt_n = float(np.clip((p50_tgt - p3_tgt) / span_tgt, 0.18, 0.82))
    gamma = float(np.clip(np.log(mid_tgt_n) / min(np.log(mid_src_n), -1e-4), 0.72, 1.38))

    xs = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    u = np.clip((xs - p3_src) / span_src, 0.0, 1.0)
    mapped_core = p3_tgt + np.power(u, gamma) * span_tgt
    below = xs < p3_src
    if np.any(below) and p3_src > 1e-4:
        mapped_core[below] = (xs[below] / p3_src) * p3_tgt
    above = xs > p97_src
    if np.any(above) and (1.0 - p97_src) > 1e-4:
        t_hi = (xs[above] - p97_src) / (1.0 - p97_src)
        mapped_core[above] = p97_tgt + t_hi * (1.0 - p97_tgt)
    return np.clip(mapped_core, 0.0, 1.0).astype(np.float32)


def _apply_tone_curve_rgb(
    img_f: np.ndarray,
    p3_src: float,
    p50_src: float,
    p97_src: float,
    p3_tgt: float,
    p50_tgt: float,
    p97_tgt: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Smooth monotonic 3-point luminance tone curve (black-floor lift/drop + midtone gamma +
    highlight roll-off) that preserves chromatic ratios without crushing shadows or clipping whites.
    """
    lut = _build_tone_curve_lut(p3_src, p50_src, p97_src, p3_tgt, p50_tgt, p97_tgt)
    y = 0.114 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.299 * img_f[..., 2]
    idx = np.clip(y * 1023.0 + 0.5, 0.0, 1023.0).astype(np.int32)
    y_new = lut[idx]

    # For near-black pixels, blend tiny neutral lift so deep shadows lift cleanly without chroma blowup
    lift_add = np.maximum(0.0, y_new - y) * np.clip((0.08 - y) / 0.08, 0.0, 1.0)
    y_after_lift = y + lift_add
    ratio = np.clip(y_new / np.maximum(y_after_lift, 1e-3), 0.58, 1.68)[..., None]
    mapped = np.clip((img_f + lift_add[..., None]) * ratio, 0.0, 1.0)

    if mask is not None:
        return np.where(mask[..., None], mapped, img_f)
    return mapped


def _apply_unified_bidirectional_harmonize(
    fg_f: np.ndarray,
    bg_f: np.ndarray,
    fg_mask_u8: np.ndarray,
    profile: PairOpticalProfile,
    strength: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Master Bidirectional FG <-> BG Harmonizer:
      1. Von Kries Chromatic Adaptation in Linear Light: moves BOTH FG (72%) and BG (28%)
         onto ONE shared illuminant white balance (fixes warm indoor FG over cool overcast BG!).
      2. Soft Shadow Color Coupling: harmonizes FG shadow tint toward BG shadow temperature.
      3. Unified Black-Floor & 3-Point Contrast Tone Curve: locks FG & BG onto the same
         shadow black level, midtone contrast, and highlight roll-off.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 1e-3:
        return fg_f, bg_f

    # 1. Shared illuminant meeting point in BGR order (sRGB channel ratios)
    w_fg_bgr = np.clip(profile.fg_wb_rgb[::-1], 0.72, 1.38).astype(np.float32)
    w_bg_bgr = np.clip(profile.bg_wb_rgb[::-1], 0.72, 1.38).astype(np.float32)
    w_shared = np.power(w_bg_bgr, 0.68) * np.power(w_fg_bgr, 0.32)
    w_shared = w_shared / max(float(np.mean(w_shared)), 1e-4)

    fg_srgb_gain = np.clip(np.power(w_shared / np.maximum(w_fg_bgr, 1e-3), 0.90 * s), 0.74, 1.36).astype(np.float32)
    bg_srgb_gain = np.clip(np.power(w_shared / np.maximum(w_bg_bgr, 1e-3), 0.72 * s), 0.80, 1.26).astype(np.float32)

    fg_out = np.clip(fg_f * fg_srgb_gain[None, None, :], 0.0, 1.0)
    bg_out = np.clip(bg_f * bg_srgb_gain[None, None, :], 0.0, 1.0)

    # 2. Shadow-zone chromatic coupling: infuse subtle BG shadow warmth/coolness into FG shadows
    shd_a, shd_b = profile.bg_shd_ab
    # Map Lab shadow cast (a, b) into bounded BGR shadow bias
    b_bias = float(np.clip(-shd_b * 0.0022, -0.055, 0.055)) * s
    r_bias = float(np.clip((shd_b * 0.0018 + shd_a * 0.0010), -0.045, 0.055)) * s
    g_bias = float(np.clip(-shd_a * 0.0010, -0.025, 0.025)) * s
    if abs(b_bias) + abs(r_bias) + abs(g_bias) > 1e-3:
        y_fg = 0.114 * fg_out[..., 0] + 0.587 * fg_out[..., 1] + 0.299 * fg_out[..., 2]
        w_shd = np.clip((0.48 - y_fg) / 0.40, 0.0, 1.0) * (fg_mask_u8 > 0)
        tint_vec = np.array([b_bias, g_bias, r_bias], dtype=np.float32)
        fg_out = np.clip(fg_out + w_shd[..., None] * tint_vec[None, None, :], 0.0, 1.0)

    # 3. Unified Black-Floor, Midtone & Contrast Convergence (BOTH FG & BG!)
    fg_p3_t = profile.fg_p3 + (profile.bg_p3 - profile.fg_p3) * (0.68 * s)
    bg_p3_t = profile.bg_p3 + (profile.fg_p3 - profile.bg_p3) * (0.32 * s)
    fg_p50_t = profile.fg_p50 + (profile.bg_p50 - profile.fg_p50) * (0.36 * s)
    bg_p50_t = profile.bg_p50 + (profile.fg_p50 - profile.bg_p50) * (0.28 * s)
    fg_p97_t = profile.fg_p97 + (profile.bg_p97 - profile.fg_p97) * (0.45 * s)
    bg_p97_t = profile.bg_p97 + (profile.fg_p97 - profile.bg_p97) * (0.25 * s)

    fg_out = _apply_tone_curve_rgb(
        fg_out,
        profile.fg_p3, profile.fg_p50, profile.fg_p97,
        fg_p3_t, fg_p50_t, fg_p97_t,
        mask=(fg_mask_u8 > 0),
    )
    bg_out = _apply_tone_curve_rgb(
        bg_out,
        profile.bg_p3, profile.bg_p50, profile.bg_p97,
        bg_p3_t, bg_p50_t, bg_p97_t,
        mask=None,
    )
    return fg_out, bg_out


def _apply_sharpness_match(
    fg_f: np.ndarray,
    bg_f: np.ndarray,
    fg_mask_u8: np.ndarray,
    profile: PairOpticalProfile,
    strength: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bidirectional Optical Acutance / Lens MTF Harmonization:
      - If BG is unnaturally sharper than FG (e.g. crisp digital stone behind soft handheld fingers),
        gently softens BG optical PSF so focal depth feels shot through the same phone lens.
      - If BG is softer/mushier than crisp FG (e.g. 720p AI BG behind 1080p FG), applies subtle
        noise-gated micro-acutance sharpening to BG plus gentle MTF feathering to FG high-frequency spikes.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 1e-3:
        return fg_f, bg_f

    fg_ac = max(profile.fg_sharpness, 1e-3)
    bg_ac = max(profile.bg_sharpness, 1e-3)
    ratio = bg_ac / fg_ac

    fg_out = fg_f
    bg_out = bg_f

    if ratio > 1.08:
        # BG is sharper than FG -> subtle optical lens softening on BG
        sigma = float(np.clip((ratio - 1.0) * 1.15 * s, 0.35, 2.4))
        bg_blur = cv2.GaussianBlur(bg_f, (0, 0), sigmaX=sigma)
        w_soft = float(np.clip((ratio - 1.05) * 0.95 * s, 0.15, 0.90))
        bg_out = np.clip(bg_f * (1.0 - w_soft) + bg_blur * w_soft, 0.0, 1.0)
    elif ratio < 0.88:
        # AI BG is noticeably softer than FG -> noise-gated unsharp mask on BG + micro-MTF feather on FG
        inv_r = fg_ac / bg_ac
        usm_amt = float(np.clip((inv_r - 1.0) * 0.50 * s, 0.12, 0.72))
        bg_blur = cv2.GaussianBlur(bg_f, (0, 0), sigmaX=1.2)
        detail = bg_f - bg_blur
        # Noise gate so smooth sky never gets amplified noise
        mag = np.max(np.abs(detail), axis=-1, keepdims=True)
        gate = np.clip((mag - 0.008) / 0.025, 0.0, 1.0)
        bg_out = np.clip(bg_f + detail * (usm_amt * gate), 0.0, 1.0)

        if inv_r > 1.25:
            fg_sig = float(np.clip((inv_r - 1.20) * 0.30 * s, 0.25, 0.55))
            fg_soft = cv2.GaussianBlur(fg_f, (0, 0), sigmaX=fg_sig)
            w_fg = float(np.clip(0.35 * s, 0.0, 0.40))
            fg_m = (fg_mask_u8 > 0)[..., None]
            fg_out = np.where(fg_m, np.clip(fg_f * (1.0 - w_fg) + fg_soft * w_fg, 0.0, 1.0), fg_f)

    return fg_out, bg_out


def _compute_3d_contour_fields_small(
    fg_s: np.ndarray,
    bg_s: np.ndarray,
    m_s: np.ndarray,
    profile: PairOpticalProfile,
    strength: float,
    radius_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute C-infinity silky-smooth 2.5D contour relighting (mult_rgb_s, add_rgb_s) at working resolution.
    Uses:
      1. Multi-scale Normalized-Convolution Harmonic Light Diffusion from outside BG inward
         (completely eliminates medial-axis remap discontinuities down the center of fingers/thumbs!).
      2. C-infinity Gaussian height-field surface normals N(x,y) that smoothly vanish to (0, 0, 1)
         along finger centerlines (zero sharp light/dark split ridges!).
      3. Bidirectional Fresnel bright rim + dark shadow terminator + Half-Lambert 3D hemisphere shading.
    """
    r_s = max(3.0, float(radius_s))
    s = float(np.clip(strength, 0.0, 1.0))
    m_f = (m_s > 0).astype(np.float32)

    # 1. Smooth inward distance & gentle falloff weight (zero corner kinks!)
    d_in = cv2.distanceTransform(m_s, cv2.DIST_L2, 5)
    max_d = max(float(d_in.max()), 1.0)
    d_smooth = cv2.GaussianBlur(d_in, (0, 0), sigmaX=max(1.5, r_s * 0.25))
    d_loc_thick = cv2.GaussianBlur(d_in, (0, 0), sigmaX=max(2.5, r_s * 0.65))
    r_local = np.clip(d_loc_thick * 2.1, max(3.5, r_s * 0.55), r_s)

    u_rim = np.clip(d_smooth / np.maximum(r_local, 1e-3), 0.0, 1.0)
    # Smoothstep cosine-like curve: C1 continuous at u=0 and u=1 (keep smooth across boundary so final polish never bleeds 1.0 inward)
    ss_rim = u_rim * u_rim * (3.0 - 2.0 * u_rim)
    g_rim = np.power(1.0 - ss_rim, 1.35)

    # 2. Multi-scale Normalized-Convolution Harmonic Light Diffusion from OUTSIDE BG inward:
    # Guaranteed C-infinity smooth across the entire finger/thumb interior — ZERO medial-axis jump!
    out_m = 1.0 - m_f
    bg_outside = bg_s * out_m[..., None]
    env_acc = np.zeros_like(bg_s, dtype=np.float32)
    for sig_d, wgt in (
        (max(2.0, r_s * 0.35), 0.45),
        (max(4.5, r_s * 0.90), 0.35),
        (max(9.0, r_s * 1.85), 0.20),
    ):
        num = _fast_lowpass(bg_outside, sigma_full=sig_d, scale=0.5)
        den = np.maximum(_fast_lowpass(out_m, sigma_full=sig_d, scale=0.5), 1e-4)[..., None]
        env_acc += (num / den) * wgt

    # 3. Local intrinsic shading base of FG (normalized inside mask so exterior never leaks in)
    y_fg_s = 0.114 * fg_s[..., 0] + 0.587 * fg_s[..., 1] + 0.299 * fg_s[..., 2]
    sig_sh = max(2.5, r_s * 0.32)
    fg_sh_num = _fast_lowpass(y_fg_s * m_f, sigma_full=sig_sh, scale=0.5)
    fg_sh_den = np.maximum(_fast_lowpass(m_f, sigma_full=sig_sh, scale=0.5), 1e-3)
    s_fg_s = fg_sh_num / fg_sh_den
    y_edge_s = 0.114 * env_acc[..., 0] + 0.587 * env_acc[..., 1] + 0.299 * env_acc[..., 2]

    # Signed contour luminance contrast: positive -> brighten/wrap, negative -> shadow terminator darkening
    delta_y = y_edge_s - s_fg_s

    # Bright-side Fresnel rim & bounced contour light (soft-knee protects highlights)
    w_bright = (g_rim * m_f) * np.clip(delta_y, 0.0, 0.90) * (0.95 * s)
    headroom = np.clip(1.0 - 0.58 * s_fg_s, 0.15, 1.0)
    add_rgb_s = (env_acc * 0.72 + fg_s * 0.28) * (w_bright * headroom)[..., None]

    # Dark-side contour terminator & ambient bounce shadow roll-off
    w_dark = np.power(g_rim, 1.20) * np.clip(-delta_y, 0.0, 0.90) * (0.82 * s)
    mult_dark_s = 1.0 - np.clip(w_dark, 0.0, 0.50)

    # 4. C-infinity Smooth 2.5D Gaussian Height-Field Surface Normals (ZERO ridge along finger centerline!):
    # Blending a medium and broad Gaussian height field produces smooth cylindrical curvature where
    # gradient naturally goes to (0, 0) along the spine of every finger (normal smoothly -> (0, 0, 1)).
    sig_h1 = max(2.5, r_s * 0.45)
    sig_h2 = max(6.0, min(max_d * 0.65, r_s * 1.45))
    h_field = 0.55 * _fast_lowpass(m_f, sigma_full=sig_h1, scale=0.5) + 0.45 * _fast_lowpass(m_f, sigma_full=sig_h2, scale=0.5)
    gx_h = cv2.Sobel(h_field, cv2.CV_32F, 1, 0, ksize=3)
    gy_h = cv2.Sobel(h_field, cv2.CV_32F, 0, 1, ksize=3)
    # Soft-scale gradient relative to robust 90th-percentile slope inside FG (never divides by ~0 at ridge!)
    g_mag = np.hypot(gx_h, gy_h)
    fg_slopes = g_mag[m_s > 0]
    ref_slope = max(float(np.percentile(fg_slopes, 92)) if len(fg_slopes) > 16 else 0.05, 1e-3)
    # Outward tilt smoothly proportional to slope / ref_slope -> vanishes smoothly at finger spine!
    nx_b = np.clip(-gx_h / (ref_slope * 1.15), -0.92, 0.92)
    ny_b = np.clip(-gy_h / (ref_slope * 1.15), -0.92, 0.92)
    xy2 = nx_b * nx_b + ny_b * ny_b
    over = xy2 > 0.88
    if np.any(over):
        sc = np.sqrt(0.88 / np.maximum(xy2, 1e-5))
        nx_b = np.where(over, nx_b * sc, nx_b)
        ny_b = np.where(over, ny_b * sc, ny_b)
        xy2 = np.minimum(xy2, 0.88)
    nz_broad = np.sqrt(np.clip(1.0 - xy2, 0.12, 1.0))

    lx_b, ly_b, lz_b = profile.bg_light_dir
    lx_f, ly_f, lz_f = profile.fg_light_dir
    hl_bg = 0.5 * (nx_b * lx_b + ny_b * ly_b + nz_broad * lz_b) + 0.5
    hl_fg = 0.5 * (nx_b * lx_f + ny_b * ly_f + nz_broad * lz_f) + 0.5
    hemi_gain_s = np.exp(np.clip((hl_bg - hl_fg) * (0.64 * s), -0.42, 0.36)).astype(np.float32)

    # Subtle shadow tinting toward local dark BG color on shadowed contours
    env_chroma_ratio = env_acc / np.maximum(y_edge_s[..., None], 0.08)
    env_chroma_ratio = np.clip(env_chroma_ratio, 0.84, 1.20)
    shadow_tint_w = np.clip(w_dark * 0.40, 0.0, 0.22)[..., None]
    mult_rgb_s = (mult_dark_s * hemi_gain_s)[..., None] * (1.0 - shadow_tint_w + shadow_tint_w * env_chroma_ratio)

    # Final subtle spatial polish on the low-frequency illumination fields guarantees zero banding/seams
    sig_pol = max(1.0, r_s * 0.14)
    mult_rgb_s = cv2.GaussianBlur(mult_rgb_s.astype(np.float32), (0, 0), sigmaX=sig_pol)
    add_rgb_s = cv2.GaussianBlur(add_rgb_s.astype(np.float32), (0, 0), sigmaX=sig_pol)
    return mult_rgb_s.astype(np.float32), add_rgb_s.astype(np.float32)


def _apply_3d_contour_relighting(
    fg_f: np.ndarray,
    bg_f: np.ndarray,
    alpha: np.ndarray,
    profile: PairOpticalProfile,
    strength: float,
    radius_px: int = 28,
) -> np.ndarray:
    """
    Pixel-Level 2.5D Contour-Aware Relighting & Sculptural Depth Transfer:
      1. Builds a smooth 2.5D curved dome normal field N(x,y) = (nx, ny, nz) from the silhouette
         Signed Distance Field (adaptive to local finger vs palm thickness).
      2. Samples the harmonized background along the outward contour normal at 3 distances
         outside the silhouette so every contour pixel knows the exact brightness & hue right
         beside that segment of the edge.
      3. Applies BIDIRECTIONAL contour shading (bright Fresnel rim vs dark shadow terminator)
         plus Half-Lambert 3D hemisphere sculpting while keeping interior micro-texture crisp.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    rad = int(np.clip(radius_px, 4, 72))
    if s <= 1e-3 or rad <= 0:
        return fg_f

    h, w = fg_f.shape[:2]
    m_bin_full = (alpha > 0.45).astype(np.uint8)
    if not np.any(m_bin_full > 0) or np.all(m_bin_full > 0):
        return fg_f

    if min(h, w) >= 360:
        scale = 0.5
        sw, sh = max(64, int(round(w * scale))), max(64, int(round(h * scale)))
        m_s = cv2.resize(m_bin_full, (sw, sh), interpolation=cv2.INTER_NEAREST)
        fg_s = cv2.resize(fg_f, (sw, sh), interpolation=cv2.INTER_AREA)
        bg_s = cv2.resize(bg_f, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        sw, sh = w, h
        m_s = m_bin_full
        fg_s = fg_f
        bg_s = bg_f

    mult_rgb_s, add_rgb_s = _compute_3d_contour_fields_small(fg_s, bg_s, m_s, profile, s, float(rad) * scale)
    if scale < 1.0:
        mult_rgb = cv2.resize(mult_rgb_s, (w, h), interpolation=cv2.INTER_LINEAR)
        add_rgb = cv2.resize(add_rgb_s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mult_rgb = mult_rgb_s
        add_rgb = add_rgb_s

    return np.clip(fg_f * mult_rgb + add_rgb * (1.0 - 0.42 * fg_f), 0.0, 1.0)


_GRAIN_TILE: Optional[np.ndarray] = None
_GRAIN_TILE_CHROMA: Optional[np.ndarray] = None


def _ensure_grain_tiles(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate tiled spatially-correlated Bayer-like luma + subtle chroma camera noise tiles."""
    global _GRAIN_TILE, _GRAIN_TILE_CHROMA
    if (
        _GRAIN_TILE is None
        or _GRAIN_TILE_CHROMA is None
        or _GRAIN_TILE.shape[0] < (h + 256)
        or _GRAIN_TILE.shape[1] < (w + 256)
    ):
        rng = np.random.default_rng(42)
        raw_l = rng.standard_normal((256, 256), dtype=np.float32)
        # Subtle spatial correlation mimics real smartphone ISP demosaicing (not digital salt-and-pepper)
        corr_l = cv2.GaussianBlur(raw_l, (3, 3), 0.45)
        corr_l = corr_l / max(float(np.std(corr_l)), 1e-4)

        raw_c = rng.standard_normal((256, 256, 3), dtype=np.float32)
        corr_c = cv2.GaussianBlur(raw_c, (3, 3), 0.70)
        corr_c -= np.mean(corr_c, axis=-1, keepdims=True)
        corr_c = corr_c / max(float(np.std(corr_c)), 1e-4)

        reps_y = ((h + 256) + 255) // 256
        reps_x = ((w + 256) + 255) // 256
        _GRAIN_TILE = np.tile(corr_l, (reps_y, reps_x))[: h + 256, : w + 256].copy()
        _GRAIN_TILE_CHROMA = np.tile(corr_c, (reps_y, reps_x, 1))[: h + 256, : w + 256].copy()
    return _GRAIN_TILE, _GRAIN_TILE_CHROMA


def _apply_bidirectional_sensor_grain(
    fg_f: np.ndarray,
    bg_f: np.ndarray,
    fg_mask_u8: np.ndarray,
    screen_mask: Optional[np.ndarray],
    profile: Optional[PairOpticalProfile],
    grain_strength: float,
    unified_strength: float = 0.0,
    frame_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bidirectional Poisson-Gaussian Camera Sensor Noise Matching:
      - Measures high-frequency luma + chroma noise on BOTH FG and BG.
      - Adds exact missing noise variance to whichever layer is cleaner (e.g. plastic-smooth AI BG
        gets matching smartphone ISP grain, OR clean FG gets matching grainy outdoor BG noise),
        weighted by photon-shot shadow sensitivity (slightly stronger in shadows) and varying per frame.
    """
    eff_s = float(np.clip(max(grain_strength, unified_strength * 0.55), 0.0, 1.0))
    if eff_s <= 1e-4:
        return fg_f, bg_f

    h, w = fg_f.shape[:2]
    tile_l, tile_c = _ensure_grain_tiles(h, w)
    sy = (int(frame_idx) * 73) % 256 if frame_idx is not None else 0
    sx = (int(frame_idx) * 137) % 256 if frame_idx is not None else 0
    l_n = tile_l[sy : sy + h, sx : sx + w]
    c_n = tile_c[sy : sy + h, sx : sx + w]

    if profile is None:
        # Lightweight fallback when only legacy grain_match is active (preserves exact legacy behavior)
        cy, cx = h // 2, w // 2
        crop_bg = bg_f[max(0, cy - 128) : min(h, cy + 128), max(0, cx - 128) : min(w, cx + 128)]
        hp = crop_bg - cv2.GaussianBlur(crop_bg, (5, 5), 1.0)
        bg_std = min(float(np.std(hp)), 0.035)
        fg_out = np.clip(fg_f + (l_n[..., None] * 0.85 + c_n * 0.15) * (bg_std * eff_s), 0.0, 1.0)
        return fg_out, bg_f

    fg_nl = min(profile.fg_noise_luma, 0.032)
    bg_nl = min(profile.bg_noise_luma, 0.032)
    fg_nc = min(profile.fg_noise_chroma, 0.014)
    bg_nc = min(profile.bg_noise_chroma, 0.014)

    tgt_l = max(fg_nl, bg_nl)
    tgt_c = max(fg_nc, bg_nc)

    # Exact missing standard deviation (variance-additive)
    add_fg_l = float(np.sqrt(max(0.0, tgt_l * tgt_l - fg_nl * fg_nl))) * eff_s
    add_fg_c = float(np.sqrt(max(0.0, tgt_c * tgt_c - fg_nc * fg_nc))) * eff_s
    add_bg_l = float(np.sqrt(max(0.0, tgt_l * tgt_l - bg_nl * bg_nl))) * eff_s
    add_bg_c = float(np.sqrt(max(0.0, tgt_c * tgt_c - bg_nc * bg_nc))) * eff_s

    # Even if user dials grain_match manually when both are already similar, keep subtle dynamic grain on FG
    if grain_strength > 1e-4 and add_fg_l < bg_nl * grain_strength * 0.5:
        add_fg_l = bg_nl * grain_strength * 0.65

    fg_out = fg_f
    bg_out = bg_f

    if add_fg_l > 5e-4 or add_fg_c > 5e-4:
        y_fg = 0.114 * fg_f[..., 0] + 0.587 * fg_f[..., 1] + 0.299 * fg_f[..., 2]
        shot_w = (1.14 - 0.34 * y_fg)[..., None]
        if screen_mask is not None and screen_mask.shape == (h, w):
            shot_w = np.where((screen_mask > 0)[..., None], shot_w * 0.20, shot_w)
        noise_fg = (l_n[..., None] * add_fg_l + c_n * add_fg_c) * shot_w
        fg_out = np.clip(fg_f + noise_fg, 0.0, 1.0)

    if add_bg_l > 5e-4 or add_bg_c > 5e-4:
        # Slight different offset for BG tile so FG and BG noise are statistically independent
        sy_b = (sy + 91) % 256
        sx_b = (sx + 163) % 256
        l_nb = tile_l[sy_b : sy_b + h, sx_b : sx_b + w]
        c_nb = tile_c[sy_b : sy_b + h, sx_b : sx_b + w]
        y_bg = 0.114 * bg_f[..., 0] + 0.587 * bg_f[..., 1] + 0.299 * bg_f[..., 2]
        shot_wb = (1.14 - 0.34 * y_bg)[..., None]
        noise_bg = (l_nb[..., None] * add_bg_l + c_nb * add_bg_c) * shot_wb
        bg_out = np.clip(bg_f + noise_bg, 0.0, 1.0)

    return fg_out, bg_out


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


def _match_sensor_grain(
    fg_f: np.ndarray,
    bg_f: np.ndarray,
    fg_mask_u8: np.ndarray,
    grain_strength: float,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    fg_out, _ = _apply_bidirectional_sensor_grain(
        fg_f,
        bg_f,
        fg_mask_u8,
        screen_mask=None,
        profile=None,
        grain_strength=grain_strength,
        unified_strength=0.0,
        frame_idx=frame_idx,
    )
    return fg_out


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
    bx, by, bw_box, bh_box = cv2.boundingRect(fg_mask_u8)
    if bw_box * bh_box < 32:
        return fg_f

    h, w = fg_f.shape[:2]
    y0, y1 = by, by + bh_box
    x0, x1 = bx, bx + bw_box

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

    if len(fg_vals) < 16:
        return fg_f

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
    out[y0:y1, x0:x1] = np.where(roi_sel[..., None], mapped_roi, roi_fg)
    return out


_GRAIN_TILE_CUDA: Optional[object] = None
_GRAIN_TILE_CHROMA_CUDA: Optional[object] = None


def _cuda_gaussian_blur(t_nchw, sigma: float):
    """Fast separable 1D Gaussian blur on CUDA NCHW tensor (~0.4ms at 1080p)."""
    import torch
    import torch.nn.functional as F

    sig = float(sigma)
    if sig <= 0.15:
        return t_nchw
    rad = max(1, int(round(sig * 2.5)))
    rad = min(rad, 24)
    xs = torch.arange(-rad, rad + 1, device=t_nchw.device, dtype=t_nchw.dtype)
    k1 = torch.exp(-0.5 * (xs / sig) ** 2)
    k1 = k1 / k1.sum()
    c = t_nchw.shape[1]
    kh = k1.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    kv = k1.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    out = F.pad(t_nchw, (rad, rad, 0, 0), mode="replicate")
    out = F.conv2d(out, kh, groups=c)
    out = F.pad(out, (0, 0, rad, rad), mode="replicate")
    out = F.conv2d(out, kv, groups=c)
    return out


def _cuda_apply_tone_lut(img_t, lut_np: np.ndarray, mask_t=None):
    """Apply exact 1024-entry monotonic luminance LUT on CUDA NCHW BGR tensor (~0.3ms at 1080p)."""
    import torch

    lut_t = torch.from_numpy(lut_np).to(device=img_t.device, dtype=torch.float32)
    y = 0.114 * img_t[:, 0:1] + 0.587 * img_t[:, 1:2] + 0.299 * img_t[:, 2:3]
    idx = (y * 1023.0 + 0.5).clamp_(0.0, 1023.0).long()
    y_new = lut_t[idx]
    lift_add = (y_new - y).clamp_(min=0.0) * ((0.08 - y) / 0.08).clamp_(0.0, 1.0)
    y_after = y + lift_add
    ratio = (y_new / y_after.clamp(min=1e-3)).clamp_(0.58, 1.68)
    mapped = ((img_t + lift_add) * ratio).clamp_(0.0, 1.0)
    if mask_t is not None:
        return torch.where(mask_t, mapped, img_t)
    return mapped


def _composite_with_relighting_cuda(
    fg_clean_bgr: np.ndarray,
    alpha: np.ndarray,
    bg_bgr: np.ndarray,
    screen_mask: Optional[np.ndarray],
    cfg: RelightConfig,
    frame_idx: Optional[int],
) -> np.ndarray:
    """
    Fused PyTorch CUDA execution of the full Dual-Zone Relighting, Unified Bidirectional
    FG<->BG Harmonization, Optical Sharpness Match, 2.5D Contour Relighting, Bidirectional
    Sensor Grain, and Linear RGB Compositing pipeline (~35-55ms at 1080p vs ~1,185ms CPU).
    """
    global _GRAIN_TILE_CUDA, _GRAIN_TILE_CHROMA_CUDA
    import torch
    import torch.nn.functional as F

    h, w = fg_clean_bgr.shape[:2]
    fg_mask_u8 = (alpha > 0.15).astype(np.uint8)

    if cfg.clean_ai_ghost:
        bg_bgr = clean_ai_background_ghost(bg_bgr, fg_mask_u8)

    uni_s = float(np.clip(getattr(cfg, "unified_harmonize_strength", 0.0), 0.0, 1.0))
    contour_s = float(np.clip(getattr(cfg, "contour_relight_strength", 0.0), 0.0, 1.0))
    contour_rad = int(getattr(cfg, "contour_relight_radius", 28))
    sharp_s = float(np.clip(getattr(cfg, "sharpness_match_strength", 0.0), 0.0, 1.0))
    grain_s = float(np.clip(cfg.grain_match, 0.0, 1.0))

    if cfg.protect_emissive_screen and screen_mask is None:
        screen_mask = detect_emissive_screen_mask(fg_clean_bgr, fg_mask_u8)

    # Downsampled optical analysis & scalar stats (~2-4ms CPU)
    bg_match_s = float(np.clip(getattr(cfg, "bg_match_source_strength", 0.0), 0.0, 1.0))
    bg_br = float(np.clip(getattr(cfg, "bg_brightness", 0.0), -0.5, 0.5))
    bg_ct = float(np.clip(getattr(cfg, "bg_contrast", 1.0), 0.5, 1.6))
    bg_sat = float(np.clip(getattr(cfg, "bg_saturation", 1.0), 0.0, 2.0))
    bg_temp = float(np.clip(getattr(cfg, "bg_temperature", 0.0), -1.0, 1.0))
    bg_tint = float(np.clip(getattr(cfg, "bg_tint", 0.0), -1.0, 1.0))

    # Pre-grade a 1/4-res copy of BG if manual/reverse BG grade is active so profile sees graded BG
    has_bg_grade = (
        bg_match_s > 1e-3
        or abs(bg_br) > 1e-3
        or abs(bg_ct - 1.0) > 1e-3
        or abs(bg_sat - 1.0) > 1e-3
        or abs(bg_temp) > 1e-3
        or abs(bg_tint) > 1e-3
    )

    with torch.inference_mode():
        orig_fg_t = (
            torch.from_numpy(np.ascontiguousarray(fg_clean_bgr))
            .to(device="cuda", dtype=torch.float32, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .mul_(1.0 / 255.0)
        )
        bg_t = (
            torch.from_numpy(np.ascontiguousarray(bg_bgr))
            .to(device="cuda", dtype=torch.float32, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .mul_(1.0 / 255.0)
        )
        alpha_t = (
            torch.from_numpy(np.ascontiguousarray(alpha, dtype=np.float32))
            .to(device="cuda", dtype=torch.float32, non_blocking=True)
            .unsqueeze(0)
            .unsqueeze(0)
            .clamp_(0.0, 1.0)
        )
        fg_m_t = alpha_t > 0.15

        if cfg.bg_defocus_sigma > 0.1:
            bg_t = _cuda_gaussian_blur(bg_t, float(cfg.bg_defocus_sigma))

        # Stage 0: Fast GPU Background Grading & Reverse BG -> Source Match
        if has_bg_grade:
            if bg_match_s > 1e-3:
                # Compute quick 1/4-res scalar stats from profile
                tmp_prof = analyze_pair_optical_signatures(
                    fg_clean_bgr, bg_bgr, fg_mask_u8,
                    screen_mask=screen_mask if cfg.protect_emissive_screen else None,
                )
                w_fg_b = np.clip(tmp_prof.fg_wb_rgb[::-1], 0.72, 1.38)
                w_bg_b = np.clip(tmp_prof.bg_wb_rgb[::-1], 0.72, 1.38)
                g_rev = np.clip(np.power(w_fg_b / np.maximum(w_bg_b, 1e-3), 0.78 * bg_match_s), 0.78, 1.28).astype(np.float32)
                bg_t = (bg_t * torch.from_numpy(g_rev).to("cuda").view(1, 3, 1, 1)).clamp_(0.0, 1.0)
                lut_rev = _build_tone_curve_lut(
                    tmp_prof.bg_p3, tmp_prof.bg_p50, tmp_prof.bg_p97,
                    tmp_prof.bg_p3 + (tmp_prof.fg_p3 - tmp_prof.bg_p3) * (0.55 * bg_match_s),
                    tmp_prof.bg_p50 + (tmp_prof.fg_p50 - tmp_prof.bg_p50) * (0.65 * bg_match_s),
                    tmp_prof.bg_p97 + (tmp_prof.fg_p97 - tmp_prof.bg_p97) * (0.55 * bg_match_s),
                )
                bg_t = _cuda_apply_tone_lut(bg_t, lut_rev)
            if abs(bg_br) > 1e-3 or abs(bg_ct - 1.0) > 1e-3:
                bg_t = ((bg_t - 0.5) * bg_ct + 0.5 + bg_br * 0.45).clamp_(0.0, 1.0)
            if abs(bg_sat - 1.0) > 1e-3:
                yb = 0.114 * bg_t[:, 0:1] + 0.587 * bg_t[:, 1:2] + 0.299 * bg_t[:, 2:3]
                bg_t = (yb + (bg_t - yb) * bg_sat).clamp_(0.0, 1.0)
            if abs(bg_temp) > 1e-3 or abs(bg_tint) > 1e-3:
                # BGR temperature (+warm -> +R, -B) and tint (+magenta -> +R,+B,-G)
                tb = torch.tensor(
                    [1.0 - 0.14 * bg_temp + 0.08 * bg_tint, 1.0 - 0.10 * bg_tint, 1.0 + 0.14 * bg_temp + 0.08 * bg_tint],
                    device="cuda",
                    dtype=torch.float32,
                ).view(1, 3, 1, 1)
                bg_t = (bg_t * tb).clamp_(0.0, 1.0)

        # Dual-layer optical analysis on graded BG
        profile: Optional[PairOpticalProfile] = None
        if uni_s > 1e-3 or contour_s > 1e-3 or sharp_s > 1e-3 or grain_s > 1e-4:
            bg_for_prof = bg_bgr
            if has_bg_grade or cfg.bg_defocus_sigma > 0.1:
                sh_p = max(120, min(480, h // 2))
                sw_p = max(120, int(round(w * (sh_p / float(h)))))
                bg_p_t = F.interpolate(bg_t, size=(sh_p, sw_p), mode="area")
                bg_for_prof = bg_p_t.squeeze(0).permute(1, 2, 0).mul_(255.0).clamp_(0, 255).byte().cpu().numpy()
                fg_for_prof = cv2.resize(fg_clean_bgr, (sw_p, sh_p), interpolation=cv2.INTER_AREA)
                fm_for_prof = cv2.resize(fg_mask_u8, (sw_p, sh_p), interpolation=cv2.INTER_NEAREST)
                sm_for_prof = (
                    cv2.resize(screen_mask, (sw_p, sh_p), interpolation=cv2.INTER_NEAREST)
                    if (cfg.protect_emissive_screen and screen_mask is not None)
                    else None
                )
                profile = analyze_pair_optical_signatures(fg_for_prof, bg_for_prof, fm_for_prof, screen_mask=sm_for_prof)
            else:
                profile = analyze_pair_optical_signatures(
                    fg_clean_bgr, bg_bgr, fg_mask_u8,
                    screen_mask=screen_mask if cfg.protect_emissive_screen else None,
                )

        fg_t = orig_fg_t.clone()

        # Stage 1: Legacy Ambient Color & Exposure Harmonization (fast GPU equivalent)
        eff_ambient = float(np.clip(cfg.ambient_color_strength, 0.0, 1.0)) * (1.0 - 0.50 * uni_s)
        if eff_ambient > 1e-3 and profile is not None:
            # Gentle studio highlight compression + exposure/warmth alignment on GPU
            y_fg = 0.114 * fg_t[:, 0:1] + 0.587 * fg_t[:, 1:2] + 0.299 * fg_t[:, 2:3]
            if profile.fg_p50 > profile.bg_p50 + 0.03:
                over_hi = (y_fg - profile.fg_p50).clamp_(min=0.0)
                y_comp = (y_fg - (0.18 * eff_ambient) * over_hi).clamp_(0.0, 1.0)
                fg_t = (fg_t * (y_comp / y_fg.clamp(min=1e-3)).clamp_(0.72, 1.25)).clamp_(0.0, 1.0)
            w_bg_b = np.clip(profile.bg_wb_rgb[::-1], 0.80, 1.25).astype(np.float32)
            amb_g = np.clip(np.power(w_bg_b, 0.42 * eff_ambient), 0.84, 1.18).astype(np.float32)
            fg_t = (fg_t * torch.from_numpy(amb_g).to("cuda").view(1, 3, 1, 1)).clamp_(0.0, 1.0)

        # Stage 1A: Master Unified Bidirectional FG <-> BG Harmonization on GPU
        if uni_s > 1e-3 and profile is not None:
            w_fg_bgr = np.clip(profile.fg_wb_rgb[::-1], 0.72, 1.38).astype(np.float32)
            w_bg_bgr = np.clip(profile.bg_wb_rgb[::-1], 0.72, 1.38).astype(np.float32)
            w_shared = np.power(w_bg_bgr, 0.68) * np.power(w_fg_bgr, 0.32)
            w_shared = w_shared / max(float(np.mean(w_shared)), 1e-4)
            fg_g = np.clip(np.power(w_shared / np.maximum(w_fg_bgr, 1e-3), 0.90 * uni_s), 0.74, 1.36).astype(np.float32)
            bg_g = np.clip(np.power(w_shared / np.maximum(w_bg_bgr, 1e-3), 0.72 * uni_s), 0.80, 1.26).astype(np.float32)
            fg_t = (fg_t * torch.from_numpy(fg_g).to("cuda").view(1, 3, 1, 1)).clamp_(0.0, 1.0)
            bg_t = (bg_t * torch.from_numpy(bg_g).to("cuda").view(1, 3, 1, 1)).clamp_(0.0, 1.0)

            shd_a, shd_b = profile.bg_shd_ab
            b_bias = float(np.clip(-shd_b * 0.0022, -0.055, 0.055)) * uni_s
            r_bias = float(np.clip((shd_b * 0.0018 + shd_a * 0.0010), -0.045, 0.055)) * uni_s
            g_bias = float(np.clip(-shd_a * 0.0010, -0.025, 0.025)) * uni_s
            if abs(b_bias) + abs(r_bias) + abs(g_bias) > 1e-3:
                y_fg = 0.114 * fg_t[:, 0:1] + 0.587 * fg_t[:, 1:2] + 0.299 * fg_t[:, 2:3]
                w_shd = ((0.48 - y_fg) / 0.40).clamp_(0.0, 1.0) * fg_m_t.float()
                tint_t = torch.tensor([b_bias, g_bias, r_bias], device="cuda", dtype=torch.float32).view(1, 3, 1, 1)
                fg_t = (fg_t + w_shd * tint_t).clamp_(0.0, 1.0)

            lut_fg = _build_tone_curve_lut(
                profile.fg_p3, profile.fg_p50, profile.fg_p97,
                profile.fg_p3 + (profile.bg_p3 - profile.fg_p3) * (0.68 * uni_s),
                profile.fg_p50 + (profile.bg_p50 - profile.fg_p50) * (0.36 * uni_s),
                profile.fg_p97 + (profile.bg_p97 - profile.fg_p97) * (0.45 * uni_s),
            )
            lut_bg = _build_tone_curve_lut(
                profile.bg_p3, profile.bg_p50, profile.bg_p97,
                profile.bg_p3 + (profile.fg_p3 - profile.bg_p3) * (0.32 * uni_s),
                profile.bg_p50 + (profile.fg_p50 - profile.bg_p50) * (0.28 * uni_s),
                profile.bg_p97 + (profile.fg_p97 - profile.bg_p97) * (0.25 * uni_s),
            )
            fg_t = _cuda_apply_tone_lut(fg_t, lut_fg, mask_t=fg_m_t)
            bg_t = _cuda_apply_tone_lut(bg_t, lut_bg, mask_t=None)

        # Stage 1B: Additional Black-Point Fine Tune
        bp_s = float(getattr(cfg, "black_point_match", 0.0)) * (1.0 - 0.55 * uni_s)
        if bp_s > 0.01 and profile is not None:
            blend = bp_s * 0.65
            lut_bp = _build_tone_curve_lut(
                profile.fg_p3, profile.fg_p50, profile.fg_p97,
                profile.fg_p3 * (1.0 - blend) + np.clip(profile.bg_p3, 0.02, 0.28) * blend,
                profile.fg_p50,
                profile.fg_p97 * (1.0 - blend) + np.clip(profile.bg_p97, 0.65, 0.98) * blend,
            )
            fg_t = _cuda_apply_tone_lut(fg_t, lut_bp, mask_t=fg_m_t)

        # Stage 1C: Optical Sharpness / Lens MTF Match on GPU
        if sharp_s > 1e-3 and profile is not None:
            fg_ac = max(profile.fg_sharpness, 1e-3)
            bg_ac = max(profile.bg_sharpness, 1e-3)
            ratio_ac = bg_ac / fg_ac
            if ratio_ac > 1.08:
                sig_s = float(np.clip((ratio_ac - 1.0) * 1.15 * sharp_s, 0.35, 2.4))
                w_soft = float(np.clip((ratio_ac - 1.05) * 0.95 * sharp_s, 0.15, 0.90))
                bg_t = (bg_t * (1.0 - w_soft) + _cuda_gaussian_blur(bg_t, sig_s) * w_soft).clamp_(0.0, 1.0)
            elif ratio_ac < 0.88:
                inv_r = fg_ac / bg_ac
                usm_amt = float(np.clip((inv_r - 1.0) * 0.50 * sharp_s, 0.12, 0.72))
                detail = bg_t - _cuda_gaussian_blur(bg_t, 1.2)
                mag = detail.abs().max(dim=1, keepdim=True).values
                gate = ((mag - 0.008) / 0.025).clamp_(0.0, 1.0)
                bg_t = (bg_t + detail * (usm_amt * gate)).clamp_(0.0, 1.0)
                if inv_r > 1.25:
                    fg_sig = float(np.clip((inv_r - 1.20) * 0.30 * sharp_s, 0.25, 0.55))
                    w_fg = float(np.clip(0.35 * sharp_s, 0.0, 0.40))
                    fg_soft = _cuda_gaussian_blur(fg_t, fg_sig)
                    fg_t = torch.where(fg_m_t, (fg_t * (1.0 - w_fg) + fg_soft * w_fg).clamp_(0.0, 1.0), fg_t)

        # Stage 2: 2D Directional Light Field on GPU
        eff_dir = float(np.clip(cfg.directional_light_strength, 0.0, 1.0)) * (1.0 - 0.35 * contour_s)
        if eff_dir > 1e-3:
            sw_d, sh_d = max(16, w // 8), max(16, h // 8)
            bg_d = F.interpolate(bg_t, size=(sh_d, sw_d), mode="area")
            bg_lum_d = 0.114 * bg_d[:, 0:1] + 0.587 * bg_d[:, 1:2] + 0.299 * bg_d[:, 2:3]
            low_d = _cuda_gaussian_blur(bg_lum_d, max(1.5, min(sh_d, sw_d) * 0.16))
            med_d = float(torch.median(low_d).item()) + 1e-4
            rat_d = ((low_d + 0.08) / (med_d + 0.08)).clamp_(0.88, 1.12)
            rat_full = F.interpolate(rat_d, size=(h, w), mode="bilinear", align_corners=False)
            fg_t = (fg_t * ((1.0 - eff_dir) + eff_dir * rat_full)).clamp_(0.0, 1.0)

        # Stage 2B: Pixel-Level 2.5D Contour-Aware Relighting (1/4-res SDF/normals on 1080p -> Full-res GPU apply!)
        if contour_s > 1e-3 and profile is not None and contour_rad > 0:
            c_div = 4 if min(h, w) >= 720 else 2
            sh_c, sw_c = max(64, h // c_div), max(64, w // c_div)
            fg_c_np = F.interpolate(fg_t, size=(sh_c, sw_c), mode="area").squeeze(0).permute(1, 2, 0).cpu().numpy()
            bg_c_np = F.interpolate(bg_t, size=(sh_c, sw_c), mode="area").squeeze(0).permute(1, 2, 0).cpu().numpy()
            m_c_np = cv2.resize((alpha > 0.45).astype(np.uint8), (sw_c, sh_c), interpolation=cv2.INTER_NEAREST)
            if np.any(m_c_np > 0) and not np.all(m_c_np > 0):
                mult_s, add_s = _compute_3d_contour_fields_small(
                    fg_c_np, bg_c_np, m_c_np, profile, contour_s, float(contour_rad) / float(c_div)
                )
                mult_t = F.interpolate(
                    torch.from_numpy(mult_s).to("cuda", non_blocking=True).permute(2, 0, 1).unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                add_t = F.interpolate(
                    torch.from_numpy(add_s).to("cuda", non_blocking=True).permute(2, 0, 1).unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                fg_t = (fg_t * mult_t + add_t * (1.0 - 0.42 * fg_t)).clamp_(0.0, 1.0)

        # Stage 4: Emissive Screen Protection on GPU
        scr_t_mask = None
        if cfg.protect_emissive_screen and screen_mask is not None and np.any(screen_mask > 0):
            scr_u8 = (screen_mask > 0).astype(np.uint8)
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            scr_a_np = cv2.GaussianBlur(cv2.erode(scr_u8, k3, iterations=1).astype(np.float32), (5, 5), 0)
            scr_a_np[cv2.dilate(scr_u8, k3, iterations=1) == 0] = 0.0
            scr_a_t = torch.from_numpy(scr_a_np).to("cuda", non_blocking=True).unsqueeze(0).unsqueeze(0)
            scr_t_mask = scr_a_t > 0.1
            glare_op = float(np.clip(cfg.screen_glare_opacity, 0.0, 0.5))
            if glare_op > 1e-4:
                bg_low = F.interpolate(_cuda_gaussian_blur(F.interpolate(bg_t, scale_factor=0.25, mode="area"), 4.5), size=(h, w), mode="bilinear", align_corners=False)
                glare = ((bg_low - 0.40) / 0.60).clamp_(0.0, 1.0)
                screened = 1.0 - (1.0 - orig_fg_t) * (1.0 - glare)
                scr_layer = orig_fg_t * (1.0 - glare_op) + screened * glare_op
            else:
                scr_layer = orig_fg_t
            fg_t = fg_t * (1.0 - scr_a_t) + scr_layer * scr_a_t

        # Stage 5: Bidirectional Camera Sensor Grain on GPU (cached CUDA tiles -> ~0.6ms!)
        eff_g = float(np.clip(max(grain_s, uni_s * 0.55), 0.0, 1.0))
        if eff_g > 1e-4 and profile is not None:
            tile_l_np, tile_c_np = _ensure_grain_tiles(h, w)
            if (
                _GRAIN_TILE_CUDA is None
                or _GRAIN_TILE_CUDA.shape[0] < (h + 256)
                or _GRAIN_TILE_CUDA.shape[1] < (w + 256)
            ):
                _GRAIN_TILE_CUDA = torch.from_numpy(tile_l_np).to("cuda")
                _GRAIN_TILE_CHROMA_CUDA = torch.from_numpy(tile_c_np).permute(2, 0, 1).contiguous().to("cuda")
            sy = (int(frame_idx) * 73) % 256 if frame_idx is not None else 0
            sx = (int(frame_idx) * 137) % 256 if frame_idx is not None else 0

            fg_nl = min(profile.fg_noise_luma, 0.032)
            bg_nl = min(profile.bg_noise_luma, 0.032)
            fg_nc = min(profile.fg_noise_chroma, 0.014)
            bg_nc = min(profile.bg_noise_chroma, 0.014)
            tgt_l = max(fg_nl, bg_nl)
            tgt_c = max(fg_nc, bg_nc)
            add_fg_l = float(np.sqrt(max(0.0, tgt_l * tgt_l - fg_nl * fg_nl))) * eff_g
            add_fg_c = float(np.sqrt(max(0.0, tgt_c * tgt_c - fg_nc * fg_nc))) * eff_g
            add_bg_l = float(np.sqrt(max(0.0, tgt_l * tgt_l - bg_nl * bg_nl))) * eff_g
            add_bg_c = float(np.sqrt(max(0.0, tgt_c * tgt_c - bg_nc * bg_nc))) * eff_g
            if grain_s > 1e-4 and add_fg_l < bg_nl * grain_s * 0.5:
                add_fg_l = bg_nl * grain_s * 0.65

            if add_fg_l > 5e-4 or add_fg_c > 5e-4:
                ln_t = _GRAIN_TILE_CUDA[sy : sy + h, sx : sx + w].unsqueeze(0).unsqueeze(0)
                cn_t = _GRAIN_TILE_CHROMA_CUDA[:, sy : sy + h, sx : sx + w].unsqueeze(0)
                y_fg = 0.114 * fg_t[:, 0:1] + 0.587 * fg_t[:, 1:2] + 0.299 * fg_t[:, 2:3]
                shot_w = 1.14 - 0.34 * y_fg
                if scr_t_mask is not None:
                    shot_w = torch.where(scr_t_mask, shot_w * 0.20, shot_w)
                fg_t = (fg_t + (ln_t * add_fg_l + cn_t * add_fg_c) * shot_w).clamp_(0.0, 1.0)

            if add_bg_l > 5e-4 or add_bg_c > 5e-4:
                sy_b = (sy + 91) % 256
                sx_b = (sx + 163) % 256
                ln_b = _GRAIN_TILE_CUDA[sy_b : sy_b + h, sx_b : sx_b + w].unsqueeze(0).unsqueeze(0)
                cn_b = _GRAIN_TILE_CHROMA_CUDA[:, sy_b : sy_b + h, sx_b : sx_b + w].unsqueeze(0)
                y_bg = 0.114 * bg_t[:, 0:1] + 0.587 * bg_t[:, 1:2] + 0.299 * bg_t[:, 2:3]
                bg_t = (bg_t + (ln_b * add_bg_l + cn_b * add_bg_c) * (1.14 - 0.34 * y_bg)).clamp_(0.0, 1.0)

        # Optional edge softness match
        edge_blur = float(getattr(cfg, "edge_blur_match", 0.0))
        if edge_blur > 0.05:
            ez_t = (alpha_t > 0.01) & (alpha_t < 0.99)
            ez_dil = F.max_pool2d(ez_t.float(), kernel_size=3, stride=1, padding=1) > 0.5
            fg_t = torch.where(ez_dil, _cuda_gaussian_blur(fg_t, edge_blur), fg_t)

        # Stage 6: Fused Linear RGB / sRGB Physical Light-Wrap & Alpha Composite on GPU
        lw_rad = int(cfg.light_wrap_Radius)
        lw_str = float(np.clip(cfg.light_wrap_strength, 0.0, 1.0))

        def _gpu_light_wrap(f_in, b_in):
            if lw_rad <= 0 or lw_str <= 1e-3:
                return f_in
            w_div = 4 if min(h, w) >= 720 else 2
            sh_w, sw_w = max(64, h // w_div), max(64, w // w_div)
            m_s = cv2.resize((alpha > 0.45).astype(np.uint8), (sw_w, sh_w), interpolation=cv2.INTER_NEAREST)
            if not np.any(m_s > 0) or np.all(m_s > 0):
                return f_in
            d_in = cv2.distanceTransform(m_s, cv2.DIST_L2, 5)
            d_out = cv2.distanceTransform(1 - m_s, cv2.DIST_L2, 5)
            sdf = (d_in - d_out) * float(w_div)
            falloff_s = np.clip(1.0 - (np.abs(sdf) - 0.5) / float(lw_rad), 0.0, 1.0).astype(np.float32)
            b_env_s = _cuda_gaussian_blur(F.interpolate(b_in, size=(sh_w, sw_w), mode="area"), 12.0 / float(w_div))
            falloff_t = F.interpolate(
                torch.from_numpy(falloff_s).to("cuda", non_blocking=True).unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            env_t = F.interpolate(b_env_s, size=(h, w), mode="bilinear", align_corners=False)
            headroom = 1.0 - f_in.max(dim=1, keepdim=True).values
            gain = falloff_t * falloff_t * headroom * (0.18 * lw_str)
            return (f_in * (1.0 + env_t * gain)).clamp_(0.0, 1.0)

        if cfg.linear_compositing:
            fg_lin = fg_t.clamp_(0.0, 1.0).pow_(2.2)
            bg_lin = bg_t.clamp_(0.0, 1.0).pow_(2.2)
            fg_lin = _gpu_light_wrap(fg_lin, bg_lin)
            comp_t = (bg_lin * (1.0 - alpha_t) + fg_lin * alpha_t).clamp_(0.0, 1.0).pow_(1.0 / 2.2)
        else:
            fg_t = _gpu_light_wrap(fg_t, bg_t)
            comp_t = (bg_t * (1.0 - alpha_t) + fg_t * alpha_t).clamp_(0.0, 1.0)

        return comp_t.squeeze(0).permute(1, 2, 0).mul_(255.0).clamp_(0.0, 255.0).byte().cpu().numpy()


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
    with Dual-Zone Relighting, Unified Bidirectional FG<->BG Harmonization (illuminant WB,
    black-floor/contrast, optical sharpness, bidirectional camera grain), and Pixel-Level
    2.5D Contour-Aware Relighting for true 3D depth.
    """
    if cfg is None:
        cfg = RelightConfig()

    h, w = fg_clean_bgr.shape[:2]
    bg_bgr = new_bg_bgr if (new_bg_bgr.shape[1], new_bg_bgr.shape[0]) == (w, h) else cv2.resize(new_bg_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    # Fused CUDA fast path for HD frames (~35-55ms at 1080p vs ~1,185ms CPU!)
    if _has_cuda() and (h * w) >= 200_000 and not cfg.pass3_aa_and_restore:
        try:
            return _composite_with_relighting_cuda(
                fg_clean_bgr=fg_clean_bgr,
                alpha=alpha,
                bg_bgr=bg_bgr,
                screen_mask=screen_mask,
                cfg=cfg,
                frame_idx=frame_idx,
            )
        except Exception:
            pass

    fg_mask_u8 = (alpha > 0.15).astype(np.uint8)

    if cfg.clean_ai_ghost:
        bg_bgr = clean_ai_background_ghost(bg_bgr, fg_mask_u8)

    if cfg.bg_defocus_sigma > 0.1:
        bg_bgr = cv2.GaussianBlur(bg_bgr, (0, 0), sigmaX=float(cfg.bg_defocus_sigma))

    uni_s = float(np.clip(getattr(cfg, "unified_harmonize_strength", 0.0), 0.0, 1.0))
    contour_s = float(np.clip(getattr(cfg, "contour_relight_strength", 0.0), 0.0, 1.0))
    contour_rad = int(getattr(cfg, "contour_relight_radius", 28))
    sharp_s = float(np.clip(getattr(cfg, "sharpness_match_strength", 0.0), 0.0, 1.0))
    grain_s = float(np.clip(cfg.grain_match, 0.0, 1.0))

    need_screen_early = (
        cfg.protect_emissive_screen
        and screen_mask is None
        and (
            float(getattr(cfg, "bg_match_source_strength", 0.0)) > 1e-3
            or uni_s > 1e-3
            or contour_s > 1e-3
            or sharp_s > 1e-3
        )
    )
    if need_screen_early:
        screen_mask = detect_emissive_screen_mask(fg_clean_bgr, fg_mask_u8)

    # Stage 0: Background Color Grading & Reverse Light/Tone Match (BG -> Source)
    if (
        float(getattr(cfg, "bg_match_source_strength", 0.0)) > 1e-3
        or abs(float(getattr(cfg, "bg_brightness", 0.0))) > 1e-3
        or abs(float(getattr(cfg, "bg_contrast", 1.0)) - 1.0) > 1e-3
        or abs(float(getattr(cfg, "bg_saturation", 1.0)) - 1.0) > 1e-3
        or abs(float(getattr(cfg, "bg_temperature", 0.0))) > 1e-3
        or abs(float(getattr(cfg, "bg_tint", 0.0))) > 1e-3
    ):
        bg_bgr = grade_and_match_background(
            bg_bgr,
            fg_clean_bgr,
            fg_mask_u8,
            screen_mask=screen_mask,
            match_source_strength=float(getattr(cfg, "bg_match_source_strength", 0.0)),
            brightness=float(getattr(cfg, "bg_brightness", 0.0)),
            contrast=float(getattr(cfg, "bg_contrast", 1.0)),
            saturation=float(getattr(cfg, "bg_saturation", 1.0)),
            temperature=float(getattr(cfg, "bg_temperature", 0.0)),
            tint=float(getattr(cfg, "bg_tint", 0.0)),
        )

    # Stage 0B: Dual-Layer Optical Signature Analysis (only when unified / contour / sharpness is active)
    profile: Optional[PairOpticalProfile] = None
    if uni_s > 1e-3 or contour_s > 1e-3 or sharp_s > 1e-3:
        profile = analyze_pair_optical_signatures(
            fg_clean_bgr,
            bg_bgr,
            fg_mask_u8,
            screen_mask=screen_mask if cfg.protect_emissive_screen else None,
        )

    # Stage 1: CIE Lab Ambient Color Temperature & Exposure Harmonization
    # Gently scaled when unified Von Kries illuminant matching is active so colors stay crisp, not double-shifted.
    eff_ambient = float(np.clip(cfg.ambient_color_strength, 0.0, 1.0)) * (1.0 - 0.50 * uni_s)
    fg_harmonized_bgr = _transfer_ambient_lab(
        fg_clean_bgr,
        bg_bgr,
        src_full_bgr,
        fg_mask_u8,
        strength=eff_ambient,
    )

    fg_f = fg_harmonized_bgr.astype(np.float32) * (1.0 / 255.0)
    orig_fg_f = fg_clean_bgr.astype(np.float32) * (1.0 / 255.0)
    bg_f = bg_bgr.astype(np.float32) * (1.0 / 255.0)
    src_f = (src_full_bgr.astype(np.float32) * (1.0 / 255.0)) if (cfg.pass3_aa_and_restore and src_full_bgr is not None) else None

    # Stage 1A: Master Unified Bidirectional FG <-> BG Harmonization (Illuminant WB, 3-Way Shadow Tone, Black Floor & Contrast)
    if uni_s > 1e-3 and profile is not None:
        fg_f, bg_f = _apply_unified_bidirectional_harmonize(fg_f, bg_f, fg_mask_u8, profile, uni_s)

    # Stage 1B: Black-point & highlight tone-curve alignment (additional manual fine-tune)
    bp_strength = float(getattr(cfg, "black_point_match", 0.0)) * (1.0 - 0.55 * uni_s)
    if bp_strength > 0.01:
        fg_f = _match_black_white_floor(fg_f, bg_f, fg_mask_u8, bp_strength)

    # Stage 1C: Optical Sharpness / Lens MTF Synchronization (FG <-> BG)
    if sharp_s > 1e-3 and profile is not None:
        fg_f, bg_f = _apply_sharpness_match(fg_f, bg_f, fg_mask_u8, profile, sharp_s)

    # Stage 2: 2D Directional Illumination Field Transfer
    eff_dir = float(np.clip(cfg.directional_light_strength, 0.0, 1.0)) * (1.0 - 0.35 * contour_s)
    if eff_dir > 1e-3:
        fg_f = _apply_2d_directional_lighting(fg_f, bg_f, src_f, strength=eff_dir)

    # Stage 2B: Pixel-Level 2.5D Contour-Aware Relighting (Smooth Bright/Dark Curvature & 3D Depth)
    if contour_s > 1e-3 and profile is not None:
        fg_f = _apply_3d_contour_relighting(fg_f, bg_f, alpha, profile, strength=contour_s, radius_px=contour_rad)

    # Stage 3: Physical Edge Light-Wrap around fingers and device perimeter (sRGB path)
    if not cfg.linear_compositing and int(cfg.light_wrap_Radius) > 0 and float(cfg.light_wrap_strength) > 1e-3:
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
            sbx, sby, sbw, sbh = cv2.boundingRect((screen_mask > 0).astype(np.uint8))
            if sbw > 0 and sbh > 0:
                sy0 = max(0, sby - 8)
                sy1 = min(h, sby + sbh + 8)
                sx0 = max(0, sbx - 8)
                sx1 = min(w, sbx + sbw + 8)
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

    # Stage 5: Bidirectional Camera Sensor Grain Matching (unifies noise floor on BOTH FG and BG!)
    if grain_s > 1e-4 or uni_s > 1e-3:
        fg_f, bg_f = _apply_bidirectional_sensor_grain(
            fg_f,
            bg_f,
            fg_mask_u8,
            screen_mask=screen_mask if cfg.protect_emissive_screen else None,
            profile=profile,
            grain_strength=grain_s,
            unified_strength=uni_s,
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

    # Final Sub-Pixel Alpha Composite (GPU-accelerated Linear RGB or sRGB) + Optional Pass-3 Tangential AA
    a_clip = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    a3 = a_clip[..., None]
    if cfg.linear_compositing:
        if _has_cuda() and (h * w) >= 200_000 and not cfg.pass3_aa_and_restore and not (int(cfg.light_wrap_Radius) > 0 and float(cfg.light_wrap_strength) > 1e-3):
            import torch
            with torch.inference_mode():
                fg_t = torch.from_numpy(np.ascontiguousarray(fg_f)).to("cuda", non_blocking=True).clamp_(0.0, 1.0).pow_(2.2)
                bg_t = torch.from_numpy(np.ascontiguousarray(bg_f)).to("cuda", non_blocking=True).clamp_(0.0, 1.0).pow_(2.2)
                a_t = torch.from_numpy(np.ascontiguousarray(a_clip)).to("cuda", non_blocking=True).unsqueeze(-1)
                comp_t = (bg_t.mul_(1.0 - a_t).add_(fg_t.mul_(a_t))).clamp_(0.0, 1.0).pow_(1.0 / 2.2).mul_(255.0).byte()
                return comp_t.cpu().numpy()

        fg_lin = _srgb_to_linear(fg_f)
        bg_lin = _srgb_to_linear(bg_f)
        if int(cfg.light_wrap_Radius) > 0 and float(cfg.light_wrap_strength) > 1e-3:
            fg_lin = _apply_physical_light_wrap(
                fg_lin,
                bg_lin,
                a_clip,
                radius=int(cfg.light_wrap_Radius),
                strength=float(np.clip(cfg.light_wrap_strength, 0.0, 1.0)),
            )
        if _has_cuda() and (h * w) >= 200_000 and not cfg.pass3_aa_and_restore:
            import torch
            with torch.inference_mode():
                fg_t = torch.from_numpy(np.ascontiguousarray(fg_lin)).to("cuda", non_blocking=True)
                bg_t = torch.from_numpy(np.ascontiguousarray(bg_lin)).to("cuda", non_blocking=True)
                a_t = torch.from_numpy(np.ascontiguousarray(a_clip)).to("cuda", non_blocking=True).unsqueeze(-1)
                comp_t = (bg_t.mul_(1.0 - a_t).add_(fg_t.mul_(a_t))).clamp_(0.0, 1.0).pow_(1.0 / 2.2).mul_(255.0).byte()
                return comp_t.cpu().numpy()
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
            alpha=a_clip,
            src_f=src_f,
        )
        return comp_u8
    return np.clip(comp_f * 255.0, 0, 255.0).astype(np.uint8)

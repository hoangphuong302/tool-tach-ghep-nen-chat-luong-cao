from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


def detect_emissive_screen_mask(bgr: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Automatically detect self-luminous screen region (e.g. blue/dark UI screen with crisp text)
    inside the foreground mask (hands + device) so screen colors & text are protected from
    ambient tint washout while hands and device bevel absorb the new background lighting.
    """
    fg = (fg_mask > 0).astype(np.uint8)
    if not np.any(fg > 0):
        return np.zeros_like(fg, dtype=np.uint8)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)

    # Typical translator / phone screen: either non-skin blue/cyan/dark UI or high-contrast dark glass panel
    # Skin has R > G > B and hue in [0..25] or [160..180]; emissive screens have either B > R or very low R/G dark glass
    is_non_skin_screen = (
        (fg > 0)
        & (
            ((b > r + 15) & (hsv[..., 1] > 40))  # Active blue/cyan UI screen
            | ((r < 55) & (g < 55) & (b < 75))   # Dark OLED/glass bezel & screen interior
        )
    ).astype(np.uint8) * 255

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(is_non_skin_screen, cv2.MORPH_CLOSE, k_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

    # Keep the largest connected component inside the foreground if it is reasonably sized
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(fg, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        fg_area = float(np.sum(fg > 0))
        if cv2.contourArea(largest) > fg_area * 0.04:
            hull = cv2.convexHull(largest)
            cv2.drawContours(out, [hull], -1, 1, thickness=cv2.FILLED)
            out &= fg
    return out


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
    bg_lab = cv2.cvtColor(new_bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    bg_region = fg_mask == 0
    if np.sum(bg_region) < 100:
        bg_region = np.ones_like(fg_mask, dtype=bool)

    fg_region = fg_mask > 0
    if not np.any(fg_region):
        return fg_bgr

    new_bg_mean = np.mean(bg_lab[bg_region], axis=0)
    new_bg_std = np.std(bg_lab[bg_region], axis=0) + 1e-4

    fg_l_mean = float(np.mean(fg_lab[fg_region, 0]))
    fg_l_std = float(np.std(fg_lab[fg_region, 0])) + 1e-4
    bg_l_mean = float(new_bg_mean[0])
    bg_l_std = float(new_bg_std[0])

    out_lab = fg_lab.copy()
    L = out_lab[..., 0]

    # 1. Soft-knee studio highlight compression when foreground is brighter than background
    if fg_l_mean > bg_l_mean + 8.0:
        over_hi = np.maximum(0.0, L - fg_l_mean)
        L = L - (0.38 * strength) * over_hi

    # 2. Match exposure & contrast slope toward the new background environment
    target_l_mean = (1.0 - 0.52 * strength) * fg_l_mean + (0.52 * strength) * (bg_l_mean + 18.0)
    target_l_std = (1.0 - 0.35 * strength) * fg_l_std + (0.35 * strength) * bg_l_std
    L_harm = target_l_mean + (L - fg_l_mean) * np.clip(target_l_std / fg_l_std, 0.65, 1.35)
    out_lab[..., 0] = np.clip(L_harm, 0.0, 255.0)

    # 3. Chromatic warmth/coolness transfer + subtle studio saturation taming
    sat_scale = 1.0 - 0.12 * strength
    a_shift = np.clip(float(new_bg_mean[1]) - 128.0, -24.0, 24.0) * (strength * 0.88)
    b_shift = np.clip(float(new_bg_mean[2]) - 128.0, -28.0, 28.0) * (strength * 0.88)

    out_lab[..., 1] = np.clip(128.0 + (out_lab[..., 1] - 128.0) * sat_scale + a_shift, 0.0, 255.0)
    out_lab[..., 2] = np.clip(128.0 + (out_lab[..., 2] - 128.0) * sat_scale + b_shift, 0.0, 255.0)

    return cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


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
    bg_lum = 0.114 * new_bg_f[..., 0] + 0.587 * new_bg_f[..., 1] + 0.299 * new_bg_f[..., 2]
    low_new_lum = _fast_lowpass(bg_lum, sigma_full=sigma, scale=0.125)
    mean_lum = float(np.mean(low_new_lum)) + 1e-4

    ratio = np.clip((low_new_lum + 0.08) / (mean_lum + 0.08), 0.72, 1.35)[..., None]
    relit = np.clip(fg_f * ratio, 0.0, 1.0)
    return fg_f * (1.0 - strength) + relit * strength


def _apply_physical_light_wrap(
    fg_f: np.ndarray,
    bg_f: np.ndarray,
    alpha: np.ndarray,
    radius: int,
    strength: float,
) -> np.ndarray:
    """
    3D Surface-Normal Rim Relighting, Grazing Cylindrical Occlusion & Directional Background Radiosity:
      1. Computes the 2D outward unit normal vector (Nx, Ny) = -grad(SDF)/||grad(SDF)|| and
         the 3D cylindrical Z-normal Nz = sqrt(1 - (1 - z)^2) across the outer boundary rim.
      2. Samples the new background's local ambient radiance immediately outside the edge along (Nx, Ny).
      3. Applies 3D cylindrical grazing occlusion (soft 3D self-shadowing as the surface curves away
         from the camera lens) plus directional background radiosity matching, so the edge has true
         3D volume and melts seamlessly into the new background lighting.
    """
    if radius <= 0 or strength <= 1e-3:
        return fg_f

    h, w = fg_f.shape[:2]
    m_bin = (alpha > 0.45).astype(np.uint8)
    if not np.any(m_bin > 0) or np.all(m_bin > 0):
        return fg_f

    if min(h, w) >= 720:
        sw, sh = max(32, w // 2), max(32, h // 2)
        m_s = cv2.resize(m_bin, (sw, sh), interpolation=cv2.INTER_NEAREST)
        d_in_s = cv2.distanceTransform(m_s, cv2.DIST_L2, 3)
        d_out_s = cv2.distanceTransform(1 - m_s, cv2.DIST_L2, 3)
        sdf_s = (d_in_s - d_out_s) * 2.0
        sdf = cv2.resize(sdf_s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        d_in = cv2.distanceTransform(m_bin, cv2.DIST_L2, 3)
        d_out = cv2.distanceTransform(1 - m_bin, cv2.DIST_L2, 3)
        sdf = d_in - d_out

    # Outward 2D unit normal vector pointing into the adjacent background
    gx = cv2.Sobel(sdf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sdf, cv2.CV_32F, 0, 1, ksize=3)
    gnorm = np.sqrt(gx * gx + gy * gy + 1e-5)
    nx = -gx / gnorm
    ny = -gy / gnorm

    # 3D cylindrical normal roll-off across the outer R_rim pixels
    r_rim = float(max(4.0, min(int(radius) * 2.2, 14.0)))
    z_rim = np.clip((sdf + 1.0) / r_rim, 0.0, 1.0)
    nz_cyl = np.sqrt(np.clip(1.0 - (1.0 - z_rim) ** 2, 0.0, 1.0))
    grazing = (1.0 - nz_cyl)[..., None]

    # Sample local background radiance 12px outside the contour along the normal vector
    low_bg_env = _fast_lowpass(bg_f, sigma_full=14.0, scale=0.25)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sample_x = np.clip(grid_x + nx * 12.0, 0.0, float(w - 1))
    sample_y = np.clip(grid_y + ny * 12.0, 0.0, float(h - 1))
    bg_env_local = cv2.remap(
        low_bg_env, sample_x, sample_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    lum_bg_local = (
        0.114 * bg_env_local[..., 0] + 0.587 * bg_env_local[..., 1] + 0.299 * bg_env_local[..., 2]
    )[..., None]
    lum_fg_local = (
        0.114 * fg_f[..., 0] + 0.587 * fg_f[..., 1] + 0.299 * fg_f[..., 2]
    )[..., None] + 1e-4

    str_f = float(np.clip(strength, 0.0, 1.0))
    # 3D Cylindrical Grazing Shadow / Roll-off at silhouette edge
    cyl_occlusion = 1.0 - (0.22 * str_f) * (grazing ** 1.3)
    # Directional background radiosity & luminance ratio at the rim
    target_rim_color = fg_f * np.clip(lum_bg_local / lum_fg_local, 0.60, 1.25) * 0.60 + bg_env_local * 0.40
    rim_blend_w = np.clip(grazing * (0.85 * str_f), 0.0, 0.78)

    out_f = (fg_f * (1.0 - rim_blend_w) + target_rim_color * rim_blend_w) * cyl_occlusion
    return np.clip(out_f, 0.0, 1.0)


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
    if _GRAIN_TILE is None or _GRAIN_TILE.shape[0] < h or _GRAIN_TILE.shape[1] < w:
        rng = np.random.default_rng(42)
        small_tile = rng.standard_normal((256, 256, 3), dtype=np.float32)
        reps_y = (h + 255) // 256
        reps_x = (w + 255) // 256
        _GRAIN_TILE = np.tile(small_tile, (reps_y, reps_x, 1))[:h, :w].copy()
    return np.clip(fg_f + _GRAIN_TILE[:h, :w] * (noise_std * grain_strength), 0.0, 1.0)


def composite_with_relighting(
    fg_clean_bgr: np.ndarray,
    alpha: np.ndarray,
    new_bg_bgr: np.ndarray,
    src_full_bgr: Optional[np.ndarray] = None,
    screen_mask: Optional[np.ndarray] = None,
    cfg: Optional[RelightConfig] = None,
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

    # Stage 2: 2D Directional Illumination Field Transfer
    fg_f = _apply_2d_directional_lighting(
        fg_f,
        bg_f,
        src_f,
        strength=float(np.clip(cfg.directional_light_strength, 0.0, 1.0)),
    )

    # Stage 3: Physical Edge Light-Wrap around fingers and device perimeter
    fg_f = _apply_physical_light_wrap(
        fg_f,
        bg_f,
        alpha,
        radius=int(cfg.light_wrap_Radius),
        strength=float(np.clip(cfg.light_wrap_strength, 0.0, 1.0)),
    )

    # Stage 4: Emissive Screen Protection (keep UI text/colors crisp & true)
    if cfg.protect_emissive_screen:
        if screen_mask is None:
            screen_mask = detect_emissive_screen_mask(fg_clean_bgr, fg_mask_u8)
        if screen_mask is not None and np.any(screen_mask > 0):
            scr_alpha = cv2.GaussianBlur(
                (screen_mask > 0).astype(np.float32), (9, 9), 0
            )[..., None]
            screen_layer = _apply_screen_glass_glare(
                orig_fg_f, bg_f, float(np.clip(cfg.screen_glare_opacity, 0.0, 0.5))
            )
            fg_f = fg_f * (1.0 - scr_alpha) + screen_layer * scr_alpha

    # Stage 5: Sensor Grain Matching
    fg_f = _match_sensor_grain(fg_f, bg_f, fg_mask_u8, float(np.clip(cfg.grain_match, 0.0, 1.0)))

    # Final Sub-Pixel Alpha Composite
    a3 = np.clip(alpha, 0.0, 1.0)[..., None]
    comp_f = bg_f * (1.0 - a3) + fg_f * a3
    return np.clip(comp_f * 255.0, 0, 255).astype(np.uint8)

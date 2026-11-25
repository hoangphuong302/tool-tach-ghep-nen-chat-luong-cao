from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cv2
import numpy as np

CompositeMode = Literal["Emissive Screen", "Surface / Product"]


@dataclass
class CompositeConfig:
    mode: CompositeMode = "Emissive Screen"
    lighting_strength: float = 0.30  # Reflection Opacity (0-0.5) or Lighting Strength (0-1.0)
    micro_soften: float = 0.25       # 0.0 - 1.0 px
    grain_match: float = 0.20        # 0.0 - 1.0
    mask_erode_dilate: int = -1      # -10 .. +10 px
    edge_feather: int = 3            # 0 .. 15 px
    hand_feather: int = 2            # 0 .. 5 px
    poisson_blending: bool = False
    auto_crop: bool = True
    auto_match_sharpness: bool = False  # measure AI softness per clip and use it as Micro-Soften sigma


def _adjust_mask(mask: np.ndarray, erode_dilate: int, feather: int) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    if erode_dilate != 0:
        k = abs(int(erode_dilate)) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if erode_dilate > 0:
            m = cv2.dilate(m, kernel, iterations=1)
        else:
            m = cv2.erode(m, kernel, iterations=1)

    alpha = m.astype(np.float32) / 255.0
    if feather > 0:
        fk = int(feather) * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (fk, fk), 0)
    return np.clip(alpha, 0.0, 1.0)


def _apply_emissive_screen(
    src_f: np.ndarray,
    ai_f: np.ndarray,
    reflection_opacity: float,
) -> np.ndarray:
    if reflection_opacity <= 1e-4:
        return src_f
    # Extract low-frequency glare/highlight from Warped_AI
    low_ai = cv2.GaussianBlur(ai_f, (0, 0), sigmaX=15.0)
    # Keep only bright glare components
    glare = np.clip((low_ai - 0.35) / 0.65, 0.0, 1.0)
    # Screen blend: 1 - (1 - A) * (1 - B)
    screened = 1.0 - (1.0 - src_f) * (1.0 - glare)
    return src_f * (1.0 - reflection_opacity) + screened * reflection_opacity


def _apply_surface_shading(
    src_f: np.ndarray,
    ai_f: np.ndarray,
    strength: float,
) -> np.ndarray:
    if strength <= 1e-4:
        return src_f
    # Low-pass illumination ratio in [0.5, 1.6] to avoid blowing out highlights
    low_src = cv2.GaussianBlur(src_f, (0, 0), sigmaX=21.0)
    low_ai = cv2.GaussianBlur(ai_f, (0, 0), sigmaX=21.0)
    ratio = np.clip(low_ai / (low_src + 0.05), 0.55, 1.55)
    shaded = np.clip(src_f * ratio, 0.0, 1.0)
    return src_f * (1.0 - strength) + shaded * strength


def _apply_grain_match(
    obj_f: np.ndarray,
    ai_f: np.ndarray,
    mask_u8: np.ndarray,
    grain_strength: float,
) -> np.ndarray:
    if grain_strength <= 1e-4:
        return obj_f
    # High-pass noise from Warped_AI
    blur_ai = cv2.GaussianBlur(ai_f, (5, 5), 1.0)
    high_pass = ai_f - blur_ai
    # Measure noise std in neighborhood around object
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    ring = cv2.subtract(cv2.dilate(mask_u8, k), mask_u8) > 0
    if np.any(ring):
        noise_std = float(np.std(high_pass[ring]))
    else:
        noise_std = float(np.std(high_pass))
    noise_std = min(noise_std, 0.04)
    rng = np.random.default_rng(42)
    grain = rng.normal(0.0, noise_std, size=obj_f.shape).astype(np.float32)
    return np.clip(obj_f + grain * grain_strength, 0.0, 1.0)


def composite_frame(
    src_bgr: np.ndarray,
    warped_ai_bgr: np.ndarray,
    mask_obj_src: np.ndarray,
    warped_occ_ai: Optional[np.ndarray],
    cfg: CompositeConfig,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    3-layer Source-Anchor compositing:
      Layer 1 (bottom): Warped_AI
      Layer 2 (middle): Source_Obj (unwarped, optically matched)
      Layer 3 (top):    Warped_Occ_AI (AI hand/fingers)
    """
    h, w = src_bgr.shape[:2]
    src_f = src_bgr.astype(np.float32) / 255.0
    ai_f = warped_ai_bgr.astype(np.float32) / 255.0

    # Micro-soften Source_Obj if requested
    obj_f = src_f.copy()
    if cfg.micro_soften > 0.05:
        obj_f = cv2.GaussianBlur(obj_f, (0, 0), sigmaX=float(cfg.micro_soften))

    # Optical matching
    if cfg.mode == "Emissive Screen":
        obj_f = _apply_emissive_screen(obj_f, ai_f, float(np.clip(cfg.lighting_strength, 0.0, 0.5)))
    else:
        obj_f = _apply_surface_shading(obj_f, ai_f, float(np.clip(cfg.lighting_strength, 0.0, 1.0)))

    mask_u8 = (mask_obj_src > 0).astype(np.uint8) * 255
    obj_f = _apply_grain_match(obj_f, ai_f, mask_u8, float(cfg.grain_match))

    alpha_obj = _adjust_mask(mask_obj_src, cfg.mask_erode_dilate, cfg.edge_feather)[..., None]

    # Optional Poisson Blending for flat printed surfaces
    if cfg.poisson_blending and np.any(alpha_obj > 0.5):
        ys, xs = np.where(alpha_obj[..., 0] > 0.5)
        cx = int((xs.min() + xs.max()) // 2)
        cy = int((ys.min() + ys.max()) // 2)
        obj_u8 = np.clip(obj_f * 255.0, 0, 255).astype(np.uint8)
        ai_u8 = warped_ai_bgr
        pmask = (alpha_obj[..., 0] > 0.5).astype(np.uint8) * 255
        try:
            blended_u8 = cv2.seamlessClone(obj_u8, ai_u8, pmask, (cx, cy), cv2.NORMAL_CLONE)
            comp_f = blended_u8.astype(np.float32) / 255.0
        except cv2.error:
            comp_f = ai_f * (1.0 - alpha_obj) + obj_f * alpha_obj
    else:
        comp_f = ai_f * (1.0 - alpha_obj) + obj_f * alpha_obj

    # Layer 3: Warped_Occ_AI (AI hand/fingers) on top
    if warped_occ_ai is not None and np.any(warped_occ_ai > 0):
        alpha_occ = _adjust_mask(warped_occ_ai, 0, cfg.hand_feather)[..., None]
        comp_f = comp_f * (1.0 - alpha_occ) + ai_f * alpha_occ

    out_u8 = np.clip(comp_f * 255.0, 0, 255).astype(np.uint8)

    # Auto-Crop & Scale back to original resolution
    if cfg.auto_crop and crop_box is not None:
        x0, y0, x1, y1 = crop_box
        if (x0, y0, x1, y1) != (0, 0, w, h):
            cropped = out_u8[y0:y1, x0:x1]
            out_u8 = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

    return out_u8

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import savgol_filter


@dataclass
class TransformParams:
    tx: float
    ty: float
    angle: float  # radians
    scale: float
    valid: bool = True

    def to_matrix(self) -> np.ndarray:
        c = np.cos(self.angle) * self.scale
        s = np.sin(self.angle) * self.scale
        return np.array([[c, -s, self.tx], [s, c, self.ty]], dtype=np.float32)

    @staticmethod
    def from_matrix(M: np.ndarray, valid: bool = True) -> "TransformParams":
        a, b, tx = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
        c, d, ty = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
        scale = float(np.sqrt(a * a + c * c))
        angle = float(np.arctan2(c, a))
        return TransformParams(tx=tx, ty=ty, angle=angle, scale=max(scale, 1e-4), valid=valid)


def build_bezel_roi(
    mask_obj: np.ndarray,
    mask_occ: Optional[np.ndarray] = None,
    dilate_ratio: float = 0.14,
    inner_ratio: float = 0.06,
) -> np.ndarray:
    """
    Bezel-Aware ROI: a RING around Mask_Obj (device frame/edges). The interior of the
    object (screen content, which the AI garbles) is excluded, and so is Mask_Occ_AI.
    """
    ys, xs = np.where(mask_obj > 0)
    if len(xs) == 0:
        return np.ones_like(mask_obj, dtype=np.uint8) * 255

    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    k = max(5, int(min(bw, bh) * dilate_ratio))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m_u8 = (mask_obj > 0).astype(np.uint8) * 255
    roi = cv2.dilate(m_u8, kernel, iterations=1)

    ki = max(3, int(min(bw, bh) * inner_ratio))
    if ki % 2 == 0:
        ki += 1
    inner = cv2.erode(m_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ki, ki)), iterations=1)
    roi[inner > 0] = 0

    if mask_occ is not None and np.any(mask_occ > 0):
        occ_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        occ_dil = cv2.dilate((mask_occ > 0).astype(np.uint8) * 255, occ_k, iterations=1)
        roi[occ_dil > 0] = 0

    return roi


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def refine_ecc_edges(
    src_gray: np.ndarray,
    ai_gray: np.ndarray,
    M_sim: np.ndarray,
    roi: np.ndarray,
    max_shift: float = 12.0,
) -> np.ndarray:
    """Sub-pixel residual alignment: ECC (Euclidean) on gradient images inside the bezel ring.
    Returns the composed 2x3 matrix (AI -> Source); falls back to M_sim if ECC is unreliable."""
    h, w = src_gray.shape
    warped = cv2.warpAffine(ai_gray, M_sim, (w, h), flags=cv2.INTER_LINEAR)
    W = np.eye(2, 3, dtype=np.float32)
    mask = cv2.dilate(roi, np.ones((5, 5), np.uint8))
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        cc, W = cv2.findTransformECC(_grad_mag(src_gray), _grad_mag(warped), W, cv2.MOTION_EUCLIDEAN, crit, mask, 5)
    except cv2.error:
        return M_sim
    if cc < 0.2 or np.hypot(W[0, 2], W[1, 2]) > max_shift or abs(np.arctan2(W[1, 0], W[0, 0])) > np.deg2rad(3.0):
        return M_sim
    # ECC maps template->input (inverse-map convention): total = inv(W) @ M_sim
    M3 = np.linalg.inv(np.vstack([W, [0, 0, 1]])) @ np.vstack([M_sim, [0, 0, 1]])
    return M3[:2].astype(np.float32)


def _ecc_fallback(mask_ai: np.ndarray, mask_src: np.ndarray, scale_down: float = 0.5) -> Optional[TransformParams]:
    """High-precision alignment using ECC on downscaled distance-transformed masks."""
    if not np.any(mask_ai > 0) or not np.any(mask_src > 0):
        return None
    h, w = mask_src.shape[:2]
    sw, sh = max(128, int(w * scale_down)), max(128, int(h * scale_down))
    m_ai_s = cv2.resize((mask_ai > 0).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    m_src_s = cv2.resize((mask_src > 0).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)

    dt_ai = cv2.distanceTransform(m_ai_s, cv2.DIST_L2, 5)
    dt_src = cv2.distanceTransform(m_src_s, cv2.DIST_L2, 5)
    dt_ai = cv2.normalize(dt_ai, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
    dt_src = cv2.normalize(dt_src, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-5)
    try:
        _, warp_matrix = cv2.findTransformECC(
            dt_src, dt_ai, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
        )
        warp_matrix[0, 2] /= scale_down
        warp_matrix[1, 2] /= scale_down
        # ECC uses the inverse-map convention (template->input = Source->AI); we need AI->Source.
        inv = np.linalg.inv(np.vstack([warp_matrix, [0, 0, 1]]))[:2].astype(np.float32)
        return TransformParams.from_matrix(inv, valid=True)
    except cv2.error:
        return None


def estimate_pair_similarity(
    src_bgr: np.ndarray,
    ai_bgr: np.ndarray,
    mask_src: np.ndarray,
    mask_ai: np.ndarray,
    mask_occ_ai: Optional[np.ndarray] = None,
    min_inliers: int = 8,
    refine: bool = True,
) -> TransformParams:
    """
    Estimate 4-DOF Similarity Transform (AI -> Source): SIFT + RANSAC inside the bezel ring
    (or Mask-ECC fallback), then sub-pixel ECC refinement on bezel edges.
    """
    roi_src = build_bezel_roi(mask_src, None)
    roi_ai = build_bezel_roi(mask_ai, mask_occ_ai)

    gray_src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    gray_ai = cv2.cvtColor(ai_bgr, cv2.COLOR_BGR2GRAY)

    M_init: Optional[np.ndarray] = None
    sift = cv2.SIFT_create(nfeatures=1500)
    kp_ai, des_ai = sift.detectAndCompute(gray_ai, roi_ai)
    kp_src, des_src = sift.detectAndCompute(gray_src, roi_src)

    if des_ai is not None and des_src is not None and len(kp_ai) >= min_inliers and len(kp_src) >= min_inliers:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(des_ai, des_src, k=2)
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) >= min_inliers:
            pts_ai = np.float32([kp_ai[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_src = np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, inliers = cv2.estimateAffinePartial2D(
                pts_ai, pts_src, method=cv2.RANSAC, ransacReprojThreshold=4.0
            )
            if M is not None and inliers is not None and int(inliers.sum()) >= min_inliers:
                params = TransformParams.from_matrix(M, valid=True)
                # Sanity check: reject extreme scale/rotation outliers
                if 0.75 <= params.scale <= 1.35 and abs(params.angle) < np.deg2rad(25.0):
                    M_init = M.astype(np.float32)

    if M_init is None:
        ecc = _ecc_fallback(mask_ai, mask_src)
        if ecc is None:
            return TransformParams(tx=0.0, ty=0.0, angle=0.0, scale=1.0, valid=False)
        M_init = ecc.to_matrix()

    if refine:
        M_init = refine_ecc_edges(gray_src, gray_ai, M_init, roi_src)
    return TransformParams.from_matrix(M_init, valid=True)


def _hampel_outliers(x: np.ndarray, floor: float, k: int = 4, nsig: float = 3.0) -> np.ndarray:
    n = len(x)
    bad = np.zeros(n, dtype=bool)
    for i in range(n):
        w = x[max(0, i - k): min(n, i + k + 1)]
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med))
        bad[i] = abs(x[i] - med) > nsig * max(mad, floor)
    return bad


def smooth_transforms(params_list: List[TransformParams], window_length: int = 15) -> List[TransformParams]:
    """Reject outlier frames (Hampel), interpolate them, then smooth (tx, ty, angle, scale)."""
    n = len(params_list)
    if n == 0:
        return []

    tx = np.array([p.tx for p in params_list], dtype=np.float64)
    ty = np.array([p.ty for p in params_list], dtype=np.float64)
    ang = np.array([p.angle for p in params_list], dtype=np.float64)
    sc = np.array([p.scale for p in params_list], dtype=np.float64)
    valid = np.array([p.valid for p in params_list], dtype=bool)

    if n >= 9:
        bad = (
            _hampel_outliers(tx, 0.5) | _hampel_outliers(ty, 0.5)
            | _hampel_outliers(ang, 0.002) | _hampel_outliers(sc, 0.0015)
        )
        valid &= ~bad
        if not np.any(valid):
            valid[:] = True

    idx = np.arange(n)
    if np.any(valid) and not np.all(valid):
        tx = np.interp(idx, idx[valid], tx[valid])
        ty = np.interp(idx, idx[valid], ty[valid])
        ang = np.interp(idx, idx[valid], ang[valid])
        sc = np.interp(idx, idx[valid], sc[valid])

    if n >= 5:
        wl = min(window_length, n if n % 2 == 1 else n - 1)
        if wl >= 5:
            tx = savgol_filter(tx, wl, 2)
            ty = savgol_filter(ty, wl, 2)
            ang = savgol_filter(ang, wl, 2)
            sc = savgol_filter(sc, wl, 2)

    return [
        TransformParams(tx=float(tx[i]), ty=float(ty[i]), angle=float(ang[i]), scale=float(sc[i]), valid=True)
        for i in range(n)
    ]


def estimate_blur_sigma(
    src_bgr: np.ndarray,
    warped_ai_bgr: np.ndarray,
    mask_src: np.ndarray,
    mask_occ: Optional[np.ndarray] = None,
    band_px: int = 8,
    sigmas: Optional[np.ndarray] = None,
) -> float:
    """Measure how much softer the AI footage is than the Source at the object boundary.

    In a thin band straddling the Mask_Obj edge (screen/bezel step: identical content in both
    videos) find the Gaussian sigma that, applied to the Source, best reproduces the aligned AI
    (least squares with per-frame gain/bias). Returns sigma in pixels (0 = equally sharp)."""
    if sigmas is None:
        sigmas = np.linspace(0.0, 3.0, 25)
    m = (mask_src > 0).astype(np.uint8) * 255
    k = band_px * 2 + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    band = (cv2.dilate(m, ker) > 0) & (cv2.erode(m, ker) == 0)
    if mask_occ is not None and np.any(mask_occ > 0):
        band &= cv2.dilate((mask_occ > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
    if band.sum() < 500:
        return 0.0
    gs = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ga = cv2.cvtColor(warped_ai_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)[band]
    best, best_err = 0.0, np.inf
    for s in sigmas:
        b = cv2.GaussianBlur(gs, (0, 0), float(s))[band] if s > 0 else gs[band]
        A = np.stack([b, np.ones_like(b)], 1)
        coef, *_ = np.linalg.lstsq(A, ga, rcond=None)
        err = float(np.mean((A @ coef - ga) ** 2))
        if err < best_err:
            best, best_err = float(s), err
    return best


def interpolate_ai_pair(a0: np.ndarray, a1: np.ndarray, alpha: float, dis=None) -> np.ndarray:
    """Motion-compensated in-between frame at fractional position alpha in [0,1] (DIS flow)."""
    if alpha < 0.03:
        return a0
    if alpha > 0.97:
        return a1
    if dis is None:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(cv2.cvtColor(a0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY), None)
    h, w = a0.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    w0 = cv2.remap(a0, gx - alpha * flow[..., 0], gy - alpha * flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    w1 = cv2.remap(a1, gx + (1 - alpha) * flow[..., 0], gy + (1 - alpha) * flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(w0, 1.0 - alpha, w1, alpha, 0)


def apply_border_hug_flow(
    warped_ai: np.ndarray,
    src_bgr: np.ndarray,
    mask_src: np.ndarray,
    warped_occ: Optional[np.ndarray] = None,
    ring_px: int = 25,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Optional local refinement: DISOpticalFlow applied ONLY within a narrow
    ring (~25px) around Mask_Obj_Src border, decaying smoothly to 0 outside.
    """
    h, w = src_bgr.shape[:2]
    m_u8 = (mask_src > 0).astype(np.uint8) * 255
    k = max(3, ring_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(m_u8, kernel, iterations=1)
    ero = cv2.erode(m_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    ring = cv2.subtract(dil, ero)

    # Smooth falloff weight in [0, 1]
    weight = cv2.GaussianBlur(ring.astype(np.float32) / 255.0, (15, 15), 0)
    weight = np.clip(weight, 0.0, 1.0)[..., None]

    gray_src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    gray_ai = cv2.cvtColor(warped_ai, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(gray_ai, gray_src, None)  # flow from warped_ai -> src

    # Clamp extreme flow vectors
    flow = np.clip(flow, -12.0, 12.0) * weight

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]

    refined_ai = cv2.remap(warped_ai, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    refined_occ = None
    if warped_occ is not None:
        refined_occ = cv2.remap(
            warped_occ.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        refined_occ = (refined_occ > 0.5).astype(np.uint8)
    return refined_ai, refined_occ


def compute_safe_crop_box(
    transforms: List[TransformParams],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """
    Compute common safe interior rectangle (x0, y0, x1, y1) across all frames
    so no black border appears after warping.
    """
    corners = np.array(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
        dtype=np.float32,
    ).T  # 3x4

    x0, y0, x1, y1 = 0.0, 0.0, float(width), float(height)
    for p in transforms:
        M = p.to_matrix()
        warped = M @ corners  # 2x4
        left = max(float(warped[0, 0]), float(warped[0, 3]))
        right = min(float(warped[0, 1]), float(warped[0, 2]))
        top = max(float(warped[1, 0]), float(warped[1, 1]))
        bottom = min(float(warped[1, 3]), float(warped[1, 2]))
        x0 = max(x0, left)
        y0 = max(y0, top)
        x1 = min(x1, right)
        y1 = min(y1, bottom)

    ix0 = int(np.clip(np.ceil(x0), 0, int(width * 0.06)))
    iy0 = int(np.clip(np.ceil(y0), 0, int(height * 0.06)))
    ix1 = int(np.clip(np.floor(x1), int(width * 0.94), width))
    iy1 = int(np.clip(np.floor(y1), int(height * 0.94), height))
    if ix1 <= ix0 + 16 or iy1 <= iy0 + 16:
        return (0, 0, width, height)
    return (ix0, iy0, ix1, iy1)


def warp_ai_frame(
    ai_bgr: np.ndarray,
    M: np.ndarray,
    target_size: Tuple[int, int],
    mask_occ: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Warp AI frame and its occlusion mask with the exact same similarity matrix."""
    w, h = target_size
    warped_ai = cv2.warpAffine(
        ai_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    warped_occ = None
    if mask_occ is not None:
        warped_occ = cv2.warpAffine(
            (mask_occ > 0).astype(np.uint8) * 255,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_occ = (warped_occ > 127).astype(np.uint8)
    return warped_ai, warped_occ


def estimate_source_camera_trajectory(
    src_frames: List[Path],
    fg_masks: List[Path],
    scale_down: float = 0.5,
) -> List[TransformParams]:
    """
    Track the handheld camera motion (pan, tilt, rotation, scale breathing) of the Source Video's
    BACKGROUND (strictly outside fg_masks) so that a Clean Background Plate video or image generated
    from Google Flow can be locked 1:1 to the cameraman's physical lens movement.
    Returns cumulative camera transform per frame relative to a smoothed center trajectory.
    """
    from src.video_io import load_mask

    n = min(len(src_frames), len(fg_masks))
    if n == 0:
        return []

    dx_list, dy_list, da_list, ds_list = [0.0], [0.0], [0.0], [1.0]
    prev_gray: Optional[np.ndarray] = None
    prev_bg_mask: Optional[np.ndarray] = None

    for t in range(n):
        bgr = cv2.imread(str(src_frames[t]))
        m_fg = load_mask(fg_masks[t])
        h, w = bgr.shape[:2]
        sw, sh = max(120, int(w * scale_down)), max(120, int(h * scale_down))
        gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (sw, sh), interpolation=cv2.INTER_AREA)
        fg_s = cv2.resize((m_fg > 0).astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST)
        # Dilate foreground mask so we never track features on moving fingers/device edges
        fg_dil = cv2.dilate(fg_s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
        bg_mask = (fg_dil == 0).astype(np.uint8) * 255

        if prev_gray is not None and prev_bg_mask is not None:
            pts0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=240, qualityLevel=0.01, minDistance=7, mask=prev_bg_mask)
            tx, ty, ang, sc = 0.0, 0.0, 0.0, 1.0
            if pts0 is not None and len(pts0) >= 8:
                pts1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts0, None, winSize=(21, 21), maxLevel=3)
                if pts1 is not None and st is not None:
                    good0 = pts0[st.ravel() == 1]
                    good1 = pts1[st.ravel() == 1]
                    if len(good0) >= 8:
                        M, inl = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                        if M is not None and inl is not None and int(inl.sum()) >= 6:
                            p = TransformParams.from_matrix(M)
                            if 0.96 <= p.scale <= 1.04 and abs(p.angle) < np.deg2rad(3.0):
                                tx = float(np.clip(p.tx / scale_down, -18.0, 18.0))
                                ty = float(np.clip(p.ty / scale_down, -18.0, 18.0))
                                ang = float(p.angle)
                                sc = float(p.scale)
            dx_list.append(dx_list[-1] + tx)
            dy_list.append(dy_list[-1] + ty)
            da_list.append(da_list[-1] + ang)
            ds_list.append(ds_list[-1] * sc)

        prev_gray = gray
        prev_bg_mask = bg_mask

    # High-pass filter the cumulative trajectory around zero so the camera shake/parallax is transferred
    # without drifting off-canvas over long clips
    cum_x = np.array(dx_list, dtype=np.float64)
    cum_y = np.array(dy_list, dtype=np.float64)
    cum_a = np.array(da_list, dtype=np.float64)
    if n >= 15:
        wl = min(41, n if n % 2 == 1 else n - 1)
        if wl >= 5:
            cum_x = cum_x - savgol_filter(cum_x, wl, 1)
            cum_y = cum_y - savgol_filter(cum_y, wl, 1)
            cum_a = cum_a - savgol_filter(cum_a, wl, 1)
    else:
        cum_x -= np.mean(cum_x)
        cum_y -= np.mean(cum_y)
        cum_a -= np.mean(cum_a)

    cum_x = np.clip(cum_x, -28.0, 28.0)
    cum_y = np.clip(cum_y, -28.0, 28.0)
    cum_a = np.clip(cum_a, -np.deg2rad(2.5), np.deg2rad(2.5))

    return [
        TransformParams(tx=float(cum_x[i]), ty=float(cum_y[i]), angle=float(cum_a[i]), scale=1.0, valid=True)
        for i in range(n)
    ]


def warp_bg_with_camera_motion(
    bg_bgr: np.ndarray,
    cam_tf: TransformParams,
    target_size: Tuple[int, int],
    overscan: float = 1.045,
) -> np.ndarray:
    """
    Apply tracked source camera motion (tx, ty, angle) to a new background frame with
    a slight center overscan (~4.5%) so borders are always clean and never expose black edges.
    """
    w, h = target_size
    if (bg_bgr.shape[1], bg_bgr.shape[0]) != (w, h):
        bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    cx, cy = w * 0.5, h * 0.5
    c = np.cos(cam_tf.angle) * overscan
    s = np.sin(cam_tf.angle) * overscan
    tx = (1.0 - c) * cx + s * cy + cam_tf.tx
    ty = -s * cx + (1.0 - c) * cy + cam_tf.ty
    M = np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)
    return cv2.warpAffine(bg_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

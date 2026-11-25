from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import cv2
import numpy as np

from src.video_io import load_mask

SyncMode = Literal["Auto DTW", "Manual Frame Offset", "Lock 1:1 (No Sync)"]


def _extract_feature_sequence(
    frame_paths: List[Path],
    mask_paths: List[Path],
) -> np.ndarray:
    """
    Extract normalized [cx, cy, bw, bh, aspect_ratio, motion_energy] per frame.
    """
    n = min(len(frame_paths), len(mask_paths))
    feats = np.zeros((n, 6), dtype=np.float32)
    prev_gray_roi: Optional[np.ndarray] = None

    for i in range(n):
        mask = load_mask(mask_paths[i])
        h, w = mask.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            cx = (x0 + x1) / (2.0 * max(w, 1))
            cy = (y0 + y1) / (2.0 * max(h, 1))
            bw = (x1 - x0 + 1) / max(w, 1)
            bh = (y1 - y0 + 1) / max(h, 1)
            ar = (x1 - x0 + 1) / max(y1 - y0 + 1, 1)
        else:
            cx, cy, bw, bh, ar = 0.5, 0.5, 0.0, 0.0, 1.0

        # Local motion energy around object
        img = cv2.imread(str(frame_paths[i]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
            small_m = cv2.resize(mask, (160, 90), interpolation=cv2.INTER_NEAREST)
            kernel = np.ones((7, 7), np.uint8)
            roi = cv2.dilate(small_m, kernel, iterations=1) > 0
            if prev_gray_roi is not None and np.any(roi):
                diff = np.abs(small.astype(np.float32) - prev_gray_roi.astype(np.float32))
                energy = float(np.mean(diff[roi])) / 255.0
            else:
                energy = 0.0
            prev_gray_roi = small
        else:
            energy = 0.0

        feats[i] = [cx, cy, bw, bh, ar * 0.1, energy * 2.0]

    # Z-score normalize each column safely
    std = np.std(feats, axis=0)
    mean = np.mean(feats, axis=0)
    std[std < 1e-6] = 1.0
    return (feats - mean) / std


def _dtw_sakoe_chiba(src_feats: np.ndarray, ai_feats: np.ndarray, window: int = 15) -> List[int]:
    """
    Compute DTW mapping from each src frame t (0..N-1) to ai frame t' (0..M-1)
    with Sakoe-Chiba band constraint.
    """
    n = len(src_feats)
    m = len(ai_feats)
    if n == 0 or m == 0:
        return []

    # Band is centred on the proportional (real-time) line j = i * (m-1)/(n-1),
    # not on the index diagonal, so different fps / lengths do not skew the band.
    w = max(window, 2)
    ratio = (m - 1) / max(n - 1, 1)
    inf = 1e12
    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        c = (i - 1) * ratio
        j_start = max(1, int(np.floor(c)) + 1 - w)
        j_end = min(m, int(np.ceil(c)) + 1 + w)
        for j in range(j_start, j_end + 1):
            cost = float(np.linalg.norm(src_feats[i - 1] - ai_feats[j - 1]))
            # Slight penalty for non-diagonal steps to keep path smooth
            dp[i, j] = cost + min(
                dp[i - 1, j - 1],
                dp[i - 1, j] + 0.05,
                dp[i, j - 1] + 0.05,
            )

    # Backtrack
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (dp[i - 1, j - 1], i - 1, j - 1),
            (dp[i - 1, j], i - 1, j),
            (dp[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda x: x[0])

    path.reverse()

    # Map each src frame t to median/nearest matched ai frame
    mapping = [0] * n
    buckets: List[List[int]] = [[] for _ in range(n)]
    for si, aj in path:
        buckets[si].append(aj)
    for t in range(n):
        if buckets[t]:
            mapping[t] = int(round(float(np.median(buckets[t]))))
        elif t > 0:
            mapping[t] = min(mapping[t - 1] + 1, m - 1)
    return mapping


def smooth_fractional_mapping(mapping: List[int], n_ai: int, window: int = 21) -> np.ndarray:
    """Integer DTW mapping -> smooth, monotone, fractional AI position per source frame.

    Fractional positions let the renderer interpolate between AI frames instead of
    holding/skipping them (which causes stutter when fps or pacing differ)."""
    from scipy.signal import savgol_filter

    x = np.asarray(mapping, dtype=np.float64)
    wl = min(window, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if wl >= 5:
        x = savgol_filter(x, wl, 2)
    return np.maximum.accumulate(np.clip(x, 0, max(n_ai - 1, 0)))


def compute_temporal_mapping(
    src_frames: List[Path],
    ai_frames: List[Path],
    src_masks: List[Path],
    ai_masks: List[Path],
    mode: SyncMode = "Auto DTW",
    sakoe_chiba_window: int = 15,
    manual_offset: int = 0,
) -> List[int]:
    """
    Return list `mapping` of length len(src_frames) where mapping[t] is the
    corresponding AI frame index t' in [0, len(ai_frames)-1].
    """
    n_src = len(src_frames)
    n_ai = len(ai_frames)
    if n_src == 0 or n_ai == 0:
        return []

    if mode == "Lock 1:1 (No Sync)":
        return [min(t, n_ai - 1) for t in range(n_src)]

    if mode == "Manual Frame Offset":
        return [int(np.clip(t + manual_offset, 0, n_ai - 1)) for t in range(n_src)]

    # Auto DTW
    src_feats = _extract_feature_sequence(src_frames, src_masks)
    ai_feats = _extract_feature_sequence(ai_frames, ai_masks)
    return _dtw_sakoe_chiba(src_feats, ai_feats, window=sakoe_chiba_window)

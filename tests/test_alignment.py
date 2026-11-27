"""Quantitative validation of the drift/jitter fixes. Run: python tests/test_alignment.py"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.spatial_align import (
    TransformParams,
    build_bezel_roi,
    estimate_pair_similarity,
    interpolate_ai_pair,
    smooth_transforms,
)
from src.temporal_sync import _dtw_sakoe_chiba, smooth_fractional_mapping


def _load_or_make_alignment_frame() -> np.ndarray:
    p = ROOT / "workspace/taxi_test/src_f0.jpg"
    if p.is_file():
        img = cv2.imread(str(p))
        if img is not None:
            return img
    # Self-contained synthetic natural-camera 1080x1920 frame with unique non-repeating landmarks
    rng = np.random.default_rng(42)
    h, w = 1920, 1080
    base = cv2.resize(rng.integers(50, 200, (96, 54, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    img = base.copy()
    for i in range(180):
        cx, cy = int(rng.integers(40, w - 40)), int(rng.integers(40, h - 40))
        rad = int(rng.integers(14, 58))
        col = tuple(int(c) for c in rng.integers(25, 240, 3))
        if i % 3 == 0:
            cv2.circle(img, (cx, cy), rad, col, -1)
        elif i % 3 == 1:
            pts = rng.integers(-rad, rad + 1, (4, 2)) + np.array([cx, cy])
            cv2.fillConvexPoly(img, pts.astype(np.int32), col)
        else:
            cv2.putText(img, f"M{i}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7 + (i % 4) * 0.25, col, 2)
    # Distinct bezel & screen interior with sharp step right at mask boundary [370:700, 760:1330]
    img[760:1331, 370:701] = (232, 232, 232)
    img[778:1313, 388:683] = (30, 45, 95)
    for k in range(8):
        cv2.putText(img, f"ITEM-{k*17}", (415, 845 + k * 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
    return img


def test_ecc_recovers_known_motion():
    img = _load_or_make_alignment_frame()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (370, 760), (700, 1330), 1, -1)
    # AI frame = source shifted/rotated/scaled, interior (screen) garbled, plus noise
    P = cv2.getRotationMatrix2D((540, 960), 0.9, 1.012)
    P[0, 2] += 17.3
    P[1, 2] += -11.6
    ai = cv2.warpAffine(img, P, (w, h), borderMode=cv2.BORDER_REFLECT)
    pm = cv2.warpAffine(mask, P, (w, h), flags=cv2.INTER_NEAREST)
    inner = cv2.erode(pm, np.ones((41, 41), np.uint8)) > 0
    ai[inner] = cv2.GaussianBlur(ai, (31, 31), 12)[inner]
    ai = np.clip(ai.astype(np.float32) + np.random.default_rng(0).normal(0, 4, ai.shape), 0, 255).astype(np.uint8)

    pts = np.array([[370, 760], [700, 760], [700, 1330], [370, 1330], [535, 1045]], np.float64)
    ai_pts = (P[:, :2] @ pts.T).T + P[:, 2]

    def corner_err(est):
        M = est.to_matrix()
        return np.linalg.norm((M[:, :2] @ ai_pts.T).T + M[:, 2] - pts, axis=1)

    e_old = corner_err(estimate_pair_similarity(img, ai, mask, pm, refine=False))
    e_new = corner_err(estimate_pair_similarity(img, ai, mask, pm, refine=True))
    print(f"[ECC] corner error px  SIFT-only: max={e_old.max():.3f} mean={e_old.mean():.3f} | +ECC refine: max={e_new.max():.3f} mean={e_new.mean():.3f}")
    # NOTE: this synthetic case (AI == shifted source) is ideal for SIFT, so ECC is not expected to
    # beat it here; we only require sub-pixel accuracy. Real-footage benefit is measured in
    # workspace/diag_refine.py (edge chamfer 0.88px -> 0.08px, jitter down) .
    assert e_new.max() < 1.0, (e_old, e_new)


def test_dtw_band_follows_real_time():
    n, m = 300, 240  # 30fps vs 24fps, same duration
    rng = np.random.default_rng(1)
    sig = np.cumsum(rng.normal(0, 1, (400, 4)), axis=0)
    ts = np.linspace(0, 380, n)
    ta = np.linspace(0, 380, m)
    f = lambda tt: np.stack([np.interp(tt, np.arange(400), sig[:, k]) for k in range(4)], 1).astype(np.float32)
    mp = _dtw_sakoe_chiba(f(ts), f(ta), window=15)
    ideal = np.arange(n) * (m - 1) / (n - 1)
    dev = np.abs(np.array(mp) - ideal)
    print(f"[DTW] max |map - ideal| = {dev.max():.2f} frames")
    assert dev.max() <= 2.0
    pos = smooth_fractional_mapping(mp, m)
    st = np.diff(pos)
    print(f"[DTW] fractional step mean={st.mean():.3f} min={st.min():.3f} max={st.max():.3f}")
    assert st.min() >= 0 and st.max() <= 1.0


def test_interpolation_places_object_midway():
    tex = np.random.default_rng(5).integers(150, 255, (30, 30, 3)).astype(np.uint8)
    tex[..., 1] = 255

    def frame(x):
        f = np.zeros((200, 300, 3), np.uint8)
        f[80:110, x:x + 30] = tex
        return f
    a0, a1 = frame(100), frame(120)
    mid = interpolate_ai_pair(cv2.GaussianBlur(a0, (5, 5), 1), cv2.GaussianBlur(a1, (5, 5), 1), 0.5)
    cols = np.where(mid[95, :, 1] > 128)[0]
    cx = (cols.min() + cols.max()) / 2
    print(f"[interp] object centre at alpha=.5: {cx:.1f} (expect 125.0 ; plain blend would show two half-bright boxes)")
    assert abs(cx - 125.0) < 2.0 and mid[95, 115:135, 1].min() > 128


def test_blur_sigma_estimation():
    from src.spatial_align import estimate_blur_sigma
    img = _load_or_make_alignment_frame()
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.rectangle(mask, (370, 760), (700, 1330), 1, -1)
    out = []
    for true_s in (0.0, 0.8, 1.5, 2.2):
        ai = cv2.GaussianBlur(img, (0, 0), true_s) if true_s > 0 else img.copy()
        ai = np.clip(ai.astype(np.float32) * 0.93 + 7 + np.random.default_rng(2).normal(0, 2, ai.shape), 0, 255).astype(np.uint8)
        est = estimate_blur_sigma(img, ai, mask)
        out.append((true_s, est))
        assert abs(est - true_s) <= 0.2, (true_s, est)
    print("[sigma] true vs estimated:", [(a, round(b, 2)) for a, b in out])


def test_outlier_frame_does_not_pull_trajectory():
    n = 120
    t = np.arange(n)
    true_tx = 0.05 * t
    tp = [TransformParams(tx=float(true_tx[i]), ty=0.0, angle=0.0, scale=1.0) for i in range(n)]
    for i in (30, 31, 77):
        tp[i] = TransformParams(tx=float(true_tx[i] + 25), ty=-20.0, angle=0.0, scale=1.0)
    sm = smooth_transforms(tp)
    err = np.abs(np.array([p.tx for p in sm]) - true_tx)
    print(f"[outlier] max abs error after smoothing = {err.max():.3f}px")
    assert err.max() < 0.6


if __name__ == "__main__":
    for fn in (test_outlier_frame_does_not_pull_trajectory, test_interpolation_places_object_midway,
               test_dtw_band_follows_real_time, test_ecc_recovers_known_motion, test_blur_sigma_estimation):
        fn()
        print("PASS", fn.__name__)

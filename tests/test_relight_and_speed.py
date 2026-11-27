from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import studio_server
from src.neural_matting import estimate_foreground_pymatting
from src.relight_harmonizer import (
    RelightConfig,
    analyze_pair_optical_signatures,
    composite_with_relighting,
)


def _measure_luma_noise(bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    vals = np.abs(lap[mask]) if (mask is not None and np.any(mask)) else np.abs(lap.reshape(-1))
    return float(1.4826 * np.median(vals) / np.sqrt(20.0))


def _measure_acutance(bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    vals = mag[mask] if (mask is not None and np.any(mask)) else mag.reshape(-1)
    return float(np.percentile(vals, 90))


def test_unified_harmonization_converges_warm_fg_and_cool_bg_to_shared_tone_and_contrast() -> None:
    """
    Reproduce the real-world indoor-warm FG (3500K tungsten) over overcast cool-grey BG (6200K):
    Both FG and BG must converge toward a shared white balance and shadow black-floor,
    while the emissive blue UI screen stays crisp and untinted.
    """
    h, w = 240, 240
    # Warm indoor subject: warm-tinted white bezel + warm skin + deep shadow + bright blue UI screen
    fg = np.full((h, w, 3), (120, 148, 185), dtype=np.uint8)
    # Warm-yellowish white device bezel (B=192, G=224, R=244 -> strong warm tungsten cast!)
    fg[40:200, 60:180] = (192, 224, 244)
    # Deep shadow grip region (black floor ~14)
    fg[165:200, 65:175] = (12, 16, 20)
    # Emissive blue UI screen
    fg[60:155, 80:160] = (205, 55, 25)

    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[40:200, 60:180] = 1.0

    # Cool overcast outdoor sidewalk BG (B=172, G=164, R=150; lifted dark shadows ~58)
    bg = np.full((h, w, 3), (172, 164, 150), dtype=np.uint8)
    bg[180:, :] = (66, 62, 56)

    scr = np.zeros((h, w), dtype=np.uint8)
    scr[60:155, 80:160] = 1

    # Baseline (no unified harmonization)
    cfg_base = RelightConfig(
        ambient_color_strength=0.0,
        directional_light_strength=0.0,
        light_wrap_strength=0.0,
        grain_match=0.0,
        black_point_match=0.0,
        unified_harmonize_strength=0.0,
        contour_relight_strength=0.0,
        sharpness_match_strength=0.0,
        protect_emissive_screen=True,
        pass3_aa_and_restore=False,
    )
    cfg_uni = RelightConfig(
        ambient_color_strength=0.55,
        directional_light_strength=0.0,
        light_wrap_strength=0.0,
        grain_match=0.0,
        black_point_match=0.35,
        unified_harmonize_strength=0.90,
        contour_relight_strength=0.0,
        sharpness_match_strength=0.0,
        protect_emissive_screen=True,
        pass3_aa_and_restore=False,
    )

    out_base = composite_with_relighting(fg, alpha, bg, src_full_bgr=fg, screen_mask=scr, cfg=cfg_base)
    out_uni = composite_with_relighting(fg, alpha, bg, src_full_bgr=fg, screen_mask=scr, cfg=cfg_uni)

    # 1. Warm bezel cast (R - B) must cool down significantly to harmonize with cool overcast BG
    bezel_rb_base = float(np.mean(out_base[44:56, 70:170, 2].astype(np.float32) - out_base[44:56, 70:170, 0].astype(np.float32)))
    bezel_rb_uni = float(np.mean(out_uni[44:56, 70:170, 2].astype(np.float32) - out_uni[44:56, 70:170, 0].astype(np.float32)))
    assert bezel_rb_uni < bezel_rb_base - 14.0, f"Expected warm bezel cast to neutralize: base={bezel_rb_base:.1f}, uni={bezel_rb_uni:.1f}"

    # 2. BG must ALSO gently warm toward the shared meeting tone (bidirectional convergence!)
    bg_rb_base = float(np.mean(out_base[10:30, 10:50, 2].astype(np.float32) - out_base[10:30, 10:50, 0].astype(np.float32)))
    bg_rb_uni = float(np.mean(out_uni[10:30, 10:50, 2].astype(np.float32) - out_uni[10:30, 10:50, 0].astype(np.float32)))
    assert bg_rb_uni > bg_rb_base + 3.0, f"Expected BG to gently converge toward FG tone: base={bg_rb_base:.1f}, uni={bg_rb_uni:.1f}"

    # 3. Shadow black-floor gap between FG deep shadow and BG shadow must narrow cleanly
    fg_dark_base = float(np.mean(out_base[172:192, 80:160]))
    bg_dark_base = float(np.mean(out_base[205:230, 10:50]))
    fg_dark_uni = float(np.mean(out_uni[172:192, 80:160]))
    bg_dark_uni = float(np.mean(out_uni[205:230, 10:50]))
    assert abs(fg_dark_uni - bg_dark_uni) < abs(fg_dark_base - bg_dark_base) - 8.0

    # 4. Protected emissive blue screen stays crisp and untinted
    scr_b_minus_r = float(np.mean(out_uni[80:135, 95:145, 0].astype(np.float32) - out_uni[80:135, 95:145, 2].astype(np.float32)))
    assert scr_b_minus_r > 130.0


def test_3d_contour_relighting_gradually_brightens_lit_edge_and_darkens_shadow_edge_with_depth() -> None:
    """
    Pixel-level 2.5D contour relighting must smoothly brighten the contour facing bright BG light
    and smoothly darken the contour facing dark BG shadow, decaying gradually inward along the
    3D curvature while preserving interior skin/surface micro-texture.
    """
    h, w = 240, 240
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - 120.0, yy - 120.0)
    alpha = np.clip((72.0 - dist) * 2.0 + 0.5, 0.0, 1.0).astype(np.float32)

    # Uniform mid-tone subject with fine high-frequency micro-texture (simulating skin pores)
    rng = np.random.default_rng(7)
    tex = rng.normal(0.0, 6.0, (h, w, 1)).astype(np.float32)
    fg = np.clip(118.0 + tex, 0, 255).astype(np.uint8)
    fg = np.repeat(fg, 3, axis=2)

    # Split BG: bright warm daylight on LEFT (x < 120), deep cool shadow on RIGHT (x >= 120)
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :120] = (215, 232, 248)  # bright warm-white light on left
    bg[:, 120:] = (38, 30, 24)     # deep cool shadow on right

    cfg = RelightConfig(
        ambient_color_strength=0.0,
        directional_light_strength=0.0,
        light_wrap_strength=0.0,
        grain_match=0.0,
        black_point_match=0.0,
        unified_harmonize_strength=0.0,
        contour_relight_strength=0.90,
        contour_relight_radius=34,
        sharpness_match_strength=0.0,
        protect_emissive_screen=False,
        linear_compositing=True,
        pass3_aa_and_restore=False,
    )
    out = composite_with_relighting(fg, alpha, bg, src_full_bgr=fg, cfg=cfg)
    out_y = (0.114 * out[..., 0] + 0.587 * out[..., 1] + 0.299 * out[..., 2]).astype(np.float32)

    # Sample horizontal strip along center row y=114..126 from left edge (x=48) to center (x=120) to right edge (x=192)
    left_rim = float(np.mean(out_y[114:126, 50:55]))    # ~2-6px inside bright-facing left contour
    left_mid = float(np.mean(out_y[114:126, 66:72]))    # ~18-24px inside left contour
    center = float(np.mean(out_y[114:126, 114:126]))    # deep interior center
    right_mid = float(np.mean(out_y[114:126, 168:174])) # ~18-24px inside shadow-facing right contour
    right_rim = float(np.mean(out_y[114:126, 185:190])) # ~2-6px inside dark-facing right contour

    # Smooth monotonic 3D brightness gradient: bright left rim > left mid > center > right mid > dark right rim
    assert left_rim > left_mid + 4.0, f"Left contour must fade smoothly inward: rim={left_rim:.1f}, mid={left_mid:.1f}"
    assert left_mid > center + 2.0, f"Left mid must stay above center: mid={left_mid:.1f}, center={center:.1f}"
    assert center > right_mid + 2.0, f"Right mid must roll off below center: center={center:.1f}, mid={right_mid:.1f}"
    assert right_mid > right_rim + 4.0, f"Right contour must darken smoothly toward shadow BG: mid={right_mid:.1f}, rim={right_rim:.1f}"

    # Micro-texture preservation inside interior (intrinsic reflectance preserved, not blurred/flat)
    def _hp_std(img_u8: np.ndarray) -> float:
        g = img_u8[100:140, 100:140, 1].astype(np.float32)
        return float(np.std(g - cv2.GaussianBlur(g, (5, 5), 1.0)))

    assert _hp_std(out) >= _hp_std(fg) * 0.80


def test_3d_contour_relighting_has_zero_medial_axis_split_ridge_on_narrow_finger() -> None:
    """
    Reproduce the exact thumb/finger scenario: a narrow cylindrical finger (38px thick) with
    bright floor BG above and dark shadow BG below. Pixel-to-pixel brightness across the
    finger's centerline MUST transition silk-smoothly with ZERO abrupt bright/dark ridge step!
    """
    h, w = 200, 240
    alpha = np.zeros((h, w), dtype=np.float32)
    # Horizontal rounded finger bar centered at y=100, thickness 38px (y=81..119)
    alpha[81:119, 40:200] = 1.0
    fg = np.full((h, w, 3), (125, 148, 182), dtype=np.uint8)  # uniform smooth skin
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:100, :] = (228, 228, 226)  # bright floor/wall above finger
    bg[100:, :] = (28, 26, 24)     # dark chair/shadow below finger

    cfg = RelightConfig(
        ambient_color_strength=0.0,
        directional_light_strength=0.0,
        light_wrap_strength=0.0,
        grain_match=0.0,
        black_point_match=0.0,
        unified_harmonize_strength=0.0,
        contour_relight_strength=0.85,
        contour_relight_radius=28,
        sharpness_match_strength=0.0,
        protect_emissive_screen=False,
        linear_compositing=True,
        pass3_aa_and_restore=False,
    )
    out = composite_with_relighting(fg, alpha, bg, src_full_bgr=fg, cfg=cfg)
    out_y = (0.114 * out[..., 0] + 0.587 * out[..., 1] + 0.299 * out[..., 2]).astype(np.float32)

    # Vertical luminance profile straight through the finger interior (y=84..116 at x=120)
    col_profile = out_y[84:116, 120]
    max_step = float(np.max(np.abs(np.diff(col_profile))))
    # Top near bright BG must still be smoothly brighter than bottom near dark BG
    assert float(col_profile[0]) > float(col_profile[-1]) + 15.0
    # Zero sharp medial-axis jump! Smooth pixel-to-pixel transition <= 3.5 luma levels per pixel
    assert max_step <= 3.5, f"Abrupt light/dark ridge detected down finger center: max_step={max_step:.2f}, profile={np.round(col_profile, 1)}"


def test_bidirectional_grain_and_sharpness_synchronization() -> None:
    """
    1. When smartphone FG has natural ISO noise and AI BG is plastic-smooth, BG receives
       matching temporally-varying camera grain.
    2. When BG is razor-sharp vs softer FG (or softer AI BG vs crisp FG), optical sharpness
       matching narrows the acutance gap cleanly.
    """
    h, w = 220, 220
    rng = np.random.default_rng(99)
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[45:175, 55:165] = 1.0

    # Noisy smartphone FG + smooth AI BG
    fg_noisy = np.clip(128.0 + rng.normal(0.0, 9.5, (h, w, 3)), 0, 255).astype(np.uint8)
    bg_smooth = np.full((h, w, 3), (126, 128, 130), dtype=np.uint8)
    # Subtle smooth gradient so BG has structure but zero sensor grain
    bg_smooth[:, :] = np.clip(
        bg_smooth.astype(np.float32) + np.linspace(-18, 18, w, dtype=np.float32)[None, :, None],
        0,
        255,
    ).astype(np.uint8)

    cfg_grain = RelightConfig(
        ambient_color_strength=0.0,
        directional_light_strength=0.0,
        light_wrap_strength=0.0,
        black_point_match=0.0,
        unified_harmonize_strength=0.80,
        contour_relight_strength=0.0,
        sharpness_match_strength=0.0,
        grain_match=0.65,
        protect_emissive_screen=False,
        pass3_aa_and_restore=False,
    )
    out0 = composite_with_relighting(fg_noisy, alpha, bg_smooth, cfg=cfg_grain, frame_idx=0)
    out1 = composite_with_relighting(fg_noisy, alpha, bg_smooth, cfg=cfg_grain, frame_idx=1)

    bg_region = alpha == 0
    noise_bg_before = _measure_luma_noise(bg_smooth, bg_region)
    noise_bg_after = _measure_luma_noise(out0, bg_region)
    assert noise_bg_after > max(noise_bg_before * 4.0, 0.008), (
        f"Smooth BG should inherit matched FG sensor grain: before={noise_bg_before:.5f}, after={noise_bg_after:.5f}"
    )
    assert not np.array_equal(out0[bg_region], out1[bg_region]), "BG grain must vary temporally per frame"

    # Optical Sharpness Match: razor-sharp checkerboard pavement BG vs naturally softer FG
    bg_sharp = np.full((h, w, 3), 120, dtype=np.uint8)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if ((x // 8) + (y // 8)) % 2 == 0:
                bg_sharp[y : y + 8, x : x + 8] = 175
    fg_soft = cv2.GaussianBlur(bg_sharp, (9, 9), 2.2)

    cfg_sharp = RelightConfig(
        ambient_color_strength=0.0,
        directional_light_strength=0.0,
        light_wrap_strength=0.0,
        black_point_match=0.0,
        unified_harmonize_strength=0.0,
        contour_relight_strength=0.0,
        sharpness_match_strength=0.90,
        grain_match=0.0,
        protect_emissive_screen=False,
        pass3_aa_and_restore=False,
    )
    out_sharp = composite_with_relighting(fg_soft, alpha, bg_sharp, cfg=cfg_sharp)
    ac_fg = _measure_acutance(fg_soft, alpha > 0.5)
    ac_bg_before = _measure_acutance(bg_sharp, bg_region)
    ac_bg_after = _measure_acutance(out_sharp, bg_region)
    assert abs(ac_bg_after - ac_fg) < abs(ac_bg_before - ac_fg) * 0.78, (
        f"Expected sharpness gap to narrow: fg={ac_fg:.3f}, bg_before={ac_bg_before:.3f}, bg_after={ac_bg_after:.3f}"
    )


def test_optical_signature_analyzer_returns_valid_metrics() -> None:
    h, w = 160, 160
    fg = np.full((h, w, 3), (110, 150, 195), dtype=np.uint8)
    bg = np.full((h, w, 3), (170, 160, 145), dtype=np.uint8)
    bg[:, :80] = (220, 215, 205)  # brighter on left
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:130, 40:120] = 1

    prof = analyze_pair_optical_signatures(fg, bg, mask)
    assert prof.bg_light_dir[0] < -0.15  # key light clearly comes from left (-x)
    assert prof.fg_wb_rgb.shape == (3,)
    assert prof.bg_wb_rgb.shape == (3,)


def test_gpu_pymatting_and_relighting_quality_and_speed_budget() -> None:
    """
    Verify GPU-accelerated PyMatting keeps exact halo removal + thin dark button preservation
    AND meets strict 1080p speed budgets on CUDA GPUs.
    """
    # 1. Accuracy check: thin 4px dark button next to white bezel on green screen
    h, w = 160, 160
    fg_true = np.full((h, w, 3), (236, 238, 240), dtype=np.float32)
    fg_true[45:115, 86:91] = (28, 30, 32)  # 5px thin black side button
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[20:140, 35:90] = 1.0
    alpha[45:115, 86:91] = 0.94
    alpha[20:140, 91:95] = np.array([0.75, 0.45, 0.22, 0.08], dtype=np.float32)[None, :]
    bg_green = np.full((h, w, 3), (35, 210, 45), dtype=np.float32)
    observed = np.clip(fg_true * alpha[..., None] + bg_green * (1.0 - alpha[..., None]), 0, 255).astype(np.uint8)

    fg_clean = estimate_foreground_pymatting(observed, alpha, strength=0.95, despill_green=True, fast_preview=False)
    button_luma = float(np.mean(fg_clean[50:110, 86:91]))
    assert button_luma < 52.0, f"Thin dark button washed out: luma={button_luma:.1f}"
    halo_g_excess = float(np.mean(fg_clean[30:130, 91:94, 1].astype(np.float32) - np.maximum(fg_clean[30:130, 91:94, 0], fg_clean[30:130, 91:94, 2]).astype(np.float32)))
    assert halo_g_excess < 12.0, f"Green halo remained on edge: excess_g={halo_g_excess:.1f}"

    # 2. 1080p Speed check when CUDA is available
    if torch.cuda.is_available():
        src_hd = cv2.resize(observed, (1080, 1920), interpolation=cv2.INTER_LINEAR)
        a_hd = cv2.resize(alpha, (1080, 1920), interpolation=cv2.INTER_LINEAR)
        bg_hd = np.full((1920, 1080, 3), (150, 145, 140), dtype=np.uint8)
        # Warmup once on identical inputs so one-time CUDA/cuDNN kernel JIT is excluded
        fg_warm = estimate_foreground_pymatting(src_hd, a_hd, strength=0.95, despill_green=True, fast_preview=False)
        rcfg = RelightConfig(
            ambient_color_strength=0.65,
            directional_light_strength=0.45,
            black_point_match=0.35,
            unified_harmonize_strength=0.75,
            contour_relight_strength=0.72,
            contour_relight_radius=28,
            sharpness_match_strength=0.55,
            light_wrap_Radius=4,
            light_wrap_strength=0.55,
            grain_match=0.40,
            linear_compositing=True,
            pass3_aa_and_restore=False,
        )
        _ = composite_with_relighting(fg_warm, a_hd, bg_hd, src_full_bgr=src_hd, cfg=rcfg, frame_idx=0)
        torch.cuda.synchronize()

        n_runs = 3
        t0 = time.perf_counter()
        for _ in range(n_runs):
            fg_hd = estimate_foreground_pymatting(src_hd, a_hd, strength=0.95, despill_green=True, fast_preview=False)
        torch.cuda.synchronize()
        pym_ms = (time.perf_counter() - t0) * 1000.0 / n_runs

        t1 = time.perf_counter()
        for k in range(n_runs):
            _ = composite_with_relighting(fg_hd, a_hd, bg_hd, src_full_bgr=src_hd, cfg=rcfg, frame_idx=k + 1)
        torch.cuda.synchronize()
        rel_ms = (time.perf_counter() - t1) * 1000.0 / n_runs

        assert pym_ms < 160.0, f"GPU PyMatting too slow at 1080p: {pym_ms:.1f} ms"
        assert rel_ms < 240.0, f"Unified 3D Relighting too slow at 1080p: {rel_ms:.1f} ms"


def test_studio_server_wires_unified_harmonize_and_contour_relight_in_preview(tmp_path: Path, monkeypatch) -> None:
    store = studio_server.SessionStore(tmp_path / "sessions")
    monkeypatch.setattr(studio_server, "STORE", store)
    app_inst = studio_server.create_app()
    client = TestClient(app_inst)

    sid = client.post("/api/session").json()["session_id"]
    sess = store.get(sid)

    h, w = 140, 140
    src = np.full((h, w, 3), (110, 145, 195), dtype=np.uint8)
    src[25:115, 35:105] = (195, 224, 245)
    bg = np.full((h, w, 3), (175, 165, 150), dtype=np.uint8)
    bg[:, 70:] = (38, 32, 26)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[25:115, 35:105] = 1

    src_p = sess.work_dir / "src_frames" / "00000.jpg"
    bg_p = sess.work_dir / "bg_frames" / "00000.jpg"
    src_p.parent.mkdir(parents=True, exist_ok=True)
    bg_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(src_p), src)
    cv2.imwrite(str(bg_p), bg)
    sess.src_frames = [src_p]
    sess.bg_frames = [bg_p]
    sess.width, sess.height, sess.fps = w, h, 30.0
    studio_server._save_confirmed_mask(sess, 0, mask, push_undo=False)
    sess.birefnet_alpha_by_frame[0] = mask.astype(np.float32)

    r_off = client.post(
        f"/api/session/{sid}/preview",
        json={
            "frame_idx": 0,
            "matting_engine": "birefnet",
            "use_pymatting_fg": False,
            "ambient_color_strength": 0.0,
            "directional_light_strength": 0.0,
            "black_point_match": 0.0,
            "light_wrap_strength": 0.0,
            "grain_match": 0.0,
            "bg_defocus": 0.0,
            "unified_harmonize": 0.0,
            "contour_relight": 0.0,
            "sharpness_match": 0.0,
        },
    )
    r_on = client.post(
        f"/api/session/{sid}/preview",
        json={
            "frame_idx": 0,
            "matting_engine": "birefnet",
            "use_pymatting_fg": False,
            "ambient_color_strength": 0.0,
            "directional_light_strength": 0.0,
            "black_point_match": 0.0,
            "light_wrap_strength": 0.0,
            "grain_match": 0.0,
            "bg_defocus": 0.0,
            "unified_harmonize": 0.85,
            "contour_relight": 0.85,
            "contour_relight_radius": 28,
            "sharpness_match": 0.60,
        },
    )
    assert r_off.status_code == 200
    assert r_on.status_code == 200
    assert r_off.json()["right_b64"] != r_on.json()["right_b64"]


def test_qa_audit_recovers_whole_second_hand_entering_from_edge_not_clipped_20px(tmp_path: Path) -> None:
    """
    When a second hand enters from the right border mid-video (touching the phone OR just reaching
    in from the border) and MatAnyone 2 drops it, QA audit must recover the ENTIRE connected
    second hand all the way to the border instead of chopping it off at a 20px ribbon.
    """
    from src.matte_qa import audit_and_heal_video_mattes

    h, w = 220, 220
    frames_dir = tmp_path / "src"
    mattes_dir = tmp_path / "alpha"
    frames_dir.mkdir()
    mattes_dir.mkdir()

    frame_paths = []
    matte_paths = []
    for t in range(6):
        img = np.full((h, w, 3), (145, 160, 175), dtype=np.uint8)
        # Left hand + phone (present in all frames)
        img[50:210, 30:105] = (120, 145, 190)
        # Tracker matte: always has left hand + phone, but DROPS the second hand!
        m = np.zeros((h, w), dtype=np.uint8)
        m[50:210, 30:105] = 255
        if t in (2, 3):
            # Frame 2: second hand enters and touches phone (width 115px, area > 45% of primary FG!)
            # Frame 3: second hand slightly detached (12px gap) entering from right border
            x_start = 100 if t == 2 else 120
            img[95:215, x_start:w] = (115, 148, 198)
            if t == 2:
                # Tracker only kept a tiny fingertip at x=100..112, dropped the entire lower palm x=112..220!
                m[95:125, 100:112] = 255
        fp = frames_dir / f"{t:05d}.jpg"
        mp = mattes_dir / f"{t:05d}.png"
        cv2.imwrite(str(fp), img)
        cv2.imwrite(str(mp), m)
        frame_paths.append(fp)
        matte_paths.append(mp)

    def _fake_birefnet(bgr: np.ndarray) -> np.ndarray:
        # Confidently detects BOTH hands wherever warm hand color (R > 180) is present
        return (bgr[..., 2] >= 182).astype(np.float32)

    report = audit_and_heal_video_mattes(
        frame_paths,
        matte_paths,
        is_alpha_stream=True,
        sample_ratio=1.0,
        min_samples=6,
        max_samples=6,
        use_birefnet_crosscheck=False,
        crosscheck_fn=_fake_birefnet,
    )

    assert 2 in report["healed_frames"], f"Touching second hand not healed: {report}"
    assert 3 in report["healed_frames"], f"Border-entering second hand not healed: {report}"

    a2 = cv2.imread(str(matte_paths[2]), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    a3 = cv2.imread(str(matte_paths[3]), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    # Whole second hand all the way to the far right edge (x=165..215, >60px from primary FG!) must be intact
    assert float(np.mean(a2[130:205, 165:215])) > 0.85, "Touching second hand lower palm was clipped!"
    assert float(np.mean(a3[130:205, 165:215])) > 0.85, "Detached border-entering second hand was dropped!"


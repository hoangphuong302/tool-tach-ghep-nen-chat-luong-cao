from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app
from src import segmentation
from src.neural_matting import (
    estimate_foreground_pymatting,
    release_neural_models,
    run_birefnet_matting,
    run_matanyone2_video,
    run_vitmatte_neural,
)
from src.relight_harmonizer import RelightConfig, composite_with_relighting


def test_pymatting_foreground_removes_halo_and_preserves_opaque_interior() -> None:
    h, w = 160, 160
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - 80.0) ** 2 + (yy - 80.0) ** 2)
    alpha = np.clip((42.0 - dist) / 4.0 + 0.5, 0.0, 1.0).astype(np.float32)

    fg_true = np.full((h, w, 3), (65, 95, 165), dtype=np.float32)
    # Add fine interior detail that must remain 100% lossless
    fg_true[70:90, 70:90] = (230, 210, 40)

    for bg_color in ((255.0, 255.0, 255.0), (30.0, 225.0, 40.0)):
        bg = np.full((h, w, 3), bg_color, dtype=np.float32)
        observed = np.clip(fg_true * alpha[..., None] + bg * (1.0 - alpha[..., None]), 0, 255).astype(np.uint8)
        fg_clean = estimate_foreground_pymatting(observed, alpha, strength=1.0, despill_green=False)

        # Strictly opaque interior is 100% preserved
        opaque = alpha >= 0.996
        np.testing.assert_array_equal(fg_clean[opaque], observed[opaque])

        # Edge composited over pure black & pure white must match true foreground without halo
        edge = (alpha > 0.10) & (alpha < 0.85)
        comp_black = fg_clean.astype(np.float32) * alpha[..., None]
        expected_black = fg_true * alpha[..., None]
        mae_black = float(np.mean(np.abs(comp_black[edge] - expected_black[edge])))
        assert mae_black < 4.0, f"Halo MAE too high on black bg for {bg_color}: {mae_black:.2f}"


def test_linear_rgb_compositing_avoids_srgb_dark_fringe() -> None:
    h, w = 48, 48
    alpha = np.full((h, w), 0.5, dtype=np.float32)
    fg = np.full((h, w, 3), 255, dtype=np.uint8)
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    cfg_base = dict(
        ambient_color_strength=0.0,
        directional_light_strength=0.0,
        light_wrap_strength=0.0,
        protect_emissive_screen=False,
        grain_match=0.0,
        pass3_aa_and_restore=False,
    )
    out_srgb = composite_with_relighting(fg, alpha, bg, cfg=RelightConfig(linear_compositing=False, **cfg_base))
    out_lin = composite_with_relighting(fg, alpha, bg, cfg=RelightConfig(linear_compositing=True, **cfg_base))

    # Physical 50% mix of 1.0 and 0.0 linear light = 0.5^(1/2.2) * 255 ~= 186, vs 128 in sRGB
    assert float(np.mean(out_srgb)) == pytest.approx(127.5, abs=1.5)
    assert float(np.mean(out_lin)) > 180.0


def test_dynamic_grain_varies_across_frames() -> None:
    h, w = 96, 96
    rng = np.random.default_rng(123)
    bg = np.clip(120 + rng.normal(0, 12, (h, w, 3)), 0, 255).astype(np.uint8)
    fg = np.full((h, w, 3), 130, dtype=np.uint8)
    alpha = np.ones((h, w), dtype=np.float32)
    cfg = RelightConfig(
        ambient_color_strength=0.0,
        directional_light_strength=0.0,
        light_wrap_strength=0.0,
        protect_emissive_screen=False,
        grain_match=0.35,
        pass3_aa_and_restore=False,
        linear_compositing=True,
    )
    f0 = composite_with_relighting(fg, alpha, bg, cfg=cfg, frame_idx=0)
    f1 = composite_with_relighting(fg, alpha, bg, cfg=cfg, frame_idx=1)
    assert not np.array_equal(f0, f1)


def test_sam2_predictor_and_embedding_cache_reused_across_strokes(monkeypatch) -> None:
    segmentation.release_sam2_cache()
    build_calls = []
    set_image_calls = []

    class FakePredictor:
        def __init__(self, model):
            pass

        def set_image(self, image):
            set_image_calls.append(image.shape)

        def predict(self, point_coords, point_labels, multimask_output=True):
            h, w = 64, 64
            m = np.zeros((1, h, w), dtype=bool)
            m[0, 10:54, 10:54] = True
            return m, np.array([0.95]), None

    pkg = types.ModuleType("sam2")
    pkg.__path__ = []
    build = types.ModuleType("sam2.build_sam")
    build.build_sam2 = lambda *args, **kwargs: (build_calls.append(args), object())[1]
    pred = types.ModuleType("sam2.sam2_image_predictor")
    pred.SAM2ImagePredictor = FakePredictor
    for name, module in [("sam2", pkg), ("sam2.build_sam", build), ("sam2.sam2_image_predictor", pred)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(segmentation, "ensure_checkpoint", lambda _: "fake.pt")
    monkeypatch.setattr(segmentation.torch.cuda, "is_available", lambda: False)

    img1 = np.full((64, 64, 3), 100, dtype=np.uint8)
    img2 = np.full((64, 64, 3), 180, dtype=np.uint8)

    segmentation.preview_single_frame_mask(img1, [(20, 20, 1)], manual_prompts=True, snap_boundary=False, use_cache=True)
    segmentation.preview_single_frame_mask(img1, [(30, 30, 1)], manual_prompts=True, snap_boundary=False, use_cache=True)
    segmentation.preview_single_frame_mask(img1, [(40, 40, 1)], manual_prompts=True, snap_boundary=False, use_cache=True)
    assert len(build_calls) == 1
    assert len(set_image_calls) == 1

    segmentation.preview_single_frame_mask(img2, [(25, 25, 1)], manual_prompts=True, snap_boundary=False, use_cache=True)
    assert len(build_calls) == 1
    assert len(set_image_calls) == 2
    segmentation.release_sam2_cache()


def test_smart_red_brush_subtracts_clicked_region_via_sam2(monkeypatch, tmp_path) -> None:
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    prev = np.zeros((64, 64), dtype=np.uint8)
    prev[10:50, 10:50] = 1
    state = {
        "work_dir": str(tmp_path),
        "confirmed_masks": {},
        "roto_strokes": {},
        "active_mask_by_frame": {0: prev.copy()},
    }
    calls = []

    def fake_refine(rgb, prev_mask, fg_stroke, bg_delta, **kwargs):
        calls.append((prev_mask.copy(), bg_delta.copy()))
        out = prev_mask.copy()
        out[30:50, 30:50] = 0
        return out

    monkeypatch.setattr(app, "refine_mask_with_sam2_negative", fake_refine)
    monkeypatch.setattr(app, "_run_confirm_and_preview", lambda idx, bgr, fg, bg, *a, **kw: (
        state["active_mask_by_frame"].__setitem__(idx, kw["direct_mask"]),
        (None, None, None, None, state, "ok"),
    )[1])

    red_layer = np.zeros((64, 64, 4), dtype=np.uint8)
    red_layer[36:40, 36:40] = [255, 51, 51, 255]
    app._apply_manual_editor_strokes(
        0,
        image,
        {"layers": [red_layer]},
        8,
        2,
        0.95,
        False,
        0,
        state,
        smart_red_subtract=True,
        matting_engine=app.MATTING_ENGINE_VITMATTE,
        use_pymatting_fg=True,
        two_pass_verify=False,
    )
    assert len(calls) == 1
    updated = state["active_mask_by_frame"][0]
    assert updated[15, 15] == 1
    assert updated[40, 40] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_neural_vitmatte_base_gpu_smoke() -> None:
    try:
        h, w = 256, 256
        yy, xx = np.mgrid[:h, :w]
        dist = np.sqrt((xx - 128.0) ** 2 + (yy - 128.0) ** 2)
        true_alpha = np.clip((64.0 - dist) / 3.0 + 0.5, 0.0, 1.0).astype(np.float32)

        fg = np.full((h, w, 3), (120, 160, 210), dtype=np.float32)
        bg = np.full((h, w, 3), (45, 70, 50), dtype=np.float32)
        img = np.clip(fg * true_alpha[..., None] + bg * (1.0 - true_alpha[..., None]), 0, 255).astype(np.uint8)

        trimap = np.full((h, w), 128, dtype=np.uint8)
        trimap[dist <= 54.0] = 255
        trimap[dist >= 74.0] = 0

        alpha = run_vitmatte_neural(img, trimap)
        assert alpha is not None
        assert alpha.shape == (h, w)
        mae = float(np.mean(np.abs(alpha - true_alpha)))
        assert mae < 0.05, f"ViTMatte-Base MAE too high: {mae:.4f}"
    finally:
        release_neural_models()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_neural_birefnet_hr_gpu_smoke() -> None:
    try:
        sample_path = ROOT / "outputs" / "tam7_source_full.png"
        if sample_path.is_file():
            bgr = cv2.imread(str(sample_path))
        else:
            bgr = np.full((384, 384, 3), (40, 190, 50), dtype=np.uint8)
            cv2.circle(bgr, (192, 192), 90, (120, 155, 200), -1, lineType=cv2.LINE_AA)

        alpha = run_birefnet_matting(bgr, max_size=768)
        assert alpha.shape == bgr.shape[:2]
        assert 0.0 <= float(alpha.min()) <= float(alpha.max()) <= 1.0
        assert float(np.mean(alpha > 0.5)) > 0.05
        assert np.any((alpha > 0.05) & (alpha < 0.95))
    finally:
        release_neural_models()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_neural_matanyone2_gpu_smoke(tmp_path) -> None:
    try:
        frames = []
        h, w = 360, 480
        for idx in range(4):
            img = np.full((h, w, 3), (45, 70, 95), dtype=np.uint8)
            cx = 190 + idx * 16
            cv2.circle(img, (cx, 180), 72, (150, 185, 225), -1, lineType=cv2.LINE_AA)
            fp = tmp_path / f"{idx:05d}.jpg"
            cv2.imwrite(str(fp), img)
            frames.append(fp)

        seed_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(seed_mask, (190, 180), 72, 1, -1)

        out_dir = tmp_path / "m2_alpha"
        # Exercise max_internal_size < min(h, w) downscaling path
        alpha_paths = run_matanyone2_video(frames, {0: seed_mask}, out_dir, max_internal_size=240, n_warmup=4)
        assert len(alpha_paths) == 4
        for p in alpha_paths:
            assert p.is_file()
            a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            assert a is not None and a.shape == (h, w)
            assert int(a.max()) > 200
    finally:
        release_neural_models()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_neural_matanyone2_bidirectional_multi_keyframe(tmp_path) -> None:
    try:
        frames = []
        h, w = 240, 320
        for idx in range(4):
            img = np.full((h, w, 3), (45, 70, 95), dtype=np.uint8)
            cv2.circle(img, (180, 120), 40, (150, 185, 225), -1, lineType=cv2.LINE_AA)
            if idx >= 2:
                # Second entering object appears on the left at frame 2..3
                cv2.rectangle(img, (20, 80), (80, 160), (140, 175, 215), -1)
            fp = tmp_path / f"bi_{idx:05d}.jpg"
            cv2.imwrite(str(fp), img)
            frames.append(fp)

        m0 = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(m0, (180, 120), 40, 1, -1)
        m3 = m0.copy()
        m3[80:160, 20:80] = 1

        out_dir = tmp_path / "m2_bi"
        alpha_paths = run_matanyone2_video(frames, {0: m0, 3: m3}, out_dir, max_internal_size=240, n_warmup=4)
        assert len(alpha_paths) == 4
        # Frame 2 (before keyframe 3) should capture the entering left object via backward pass
        a2 = cv2.imread(str(alpha_paths[2]), cv2.IMREAD_GRAYSCALE)
        assert a2 is not None and float(a2[100:140, 30:70].mean()) > 150.0
    finally:
        release_neural_models()


def test_emissive_screen_mask_excludes_overlapping_thumb() -> None:
    from src.relight_harmonizer import detect_emissive_screen_mask

    h, w = 240, 240
    bgr = np.full((h, w, 3), (35, 45, 50), dtype=np.uint8)
    fg_mask = np.zeros((h, w), dtype=np.uint8)
    # Device silhouette (40:200, 60:180)
    fg_mask[40:200, 60:180] = 255
    # Active blue screen (50:190, 70:170) with dark bottom bezel
    bgr[50:155, 70:170] = (175, 55, 25)
    bgr[155:190, 70:170] = (25, 25, 30)
    # Overlapping warm skin thumb protruding upward into bottom of screen (130:220, 95:145)
    fg_mask[130:220, 95:145] = 255
    bgr[130:220, 95:145] = (115, 145, 200)  # BGR skin tone (R=200 > G=145 > B=115)

    scr = detect_emissive_screen_mask(bgr, fg_mask)
    assert np.sum(scr > 0) > 2500
    # Upper screen center must be protected
    assert float(np.mean(scr[65:110, 90:150] > 0)) > 0.90
    # Overlapping thumb area must NOT be included in screen_mask (convexHull used to bridge across it)
    assert float(np.mean(scr[140:185, 102:138] > 0)) < 0.05


def test_preview_zoom_and_diagnostic_modes() -> None:
    import app

    h, w = 200, 160
    src_rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    comp_rgb = np.full((h, w, 3), 90, dtype=np.uint8)
    fg_rgb = np.full((h, w, 3), (200, 140, 110), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[70:130, 60:100] = 1.0
    scr_mask = np.zeros((h, w), dtype=np.uint8)
    scr_mask[50:100, 55:105] = 1

    l1, r1 = app._render_preview_view(src_rgb, comp_rgb, fg_rgb, alpha, scr_mask, diag_mode=app.PREVIEW_DIAG_SPLIT, zoom=1.0)
    assert l1.shape == (h, w, 3) and r1.shape == (h, w, 3)

    l2, r2 = app._render_preview_view(src_rgb, comp_rgb, fg_rgb, alpha, scr_mask, diag_mode=app.PREVIEW_DIAG_BLACK, zoom=2.0, pan_x=0.5, pan_y=0.5)
    assert l2.shape == (100, 80, 3) and r2.shape == (100, 80, 3)
    # Background corners inside crop must be pure black (0) on right panel
    assert int(r2[0, 0].max()) == 0

    l3, r3 = app._render_preview_view(src_rgb, comp_rgb, fg_rgb, alpha, scr_mask, diag_mode=app.PREVIEW_DIAG_WHITE, zoom=2.0, pan_x=0.5, pan_y=0.5)
    assert int(r3[0, 0].min()) == 255

    l4, r4 = app._render_preview_view(src_rgb, comp_rgb, fg_rgb, alpha, scr_mask, diag_mode=app.PREVIEW_DIAG_SCREEN, zoom=1.0)
    assert l4.shape == (h, w, 3) and not np.array_equal(l4, src_rgb)

    demo = app.build_ui()
    assert demo is not None

from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import app
from sam2 import build_sam
from src import segmentation
from src.video_io import load_mask, save_mask


def test_confirmed_reference_is_causal_after_first_anchor():
    assert app._confirmed_reference_frame({}, 9) is None
    assert app._confirmed_reference_frame({4: "one"}, 20) == 4
    assert app._confirmed_reference_frame({368: "later", "2": "earlier"}, 0) == "2"
    assert app._confirmed_reference_frame({368: "later", "2": "earlier"}, 367) == "2"
    assert app._confirmed_reference_frame({368: "later", "2": "earlier"}, 368) == 368


def test_no_confirmed_anchor_returns_no_tracking(tmp_path):
    assert segmentation.track_roto_video_masks(tmp_path, tmp_path / "tracked", {}) == []


@pytest.mark.parametrize("indices", [(0,), (1, 3)])
def test_tracking_saves_confirmed_masks_without_filling_user_holes(tmp_path, monkeypatch, indices):
    frames = tmp_path / "frames"
    frames.mkdir()
    for idx in range(5):
        cv2.imwrite(str(frames / f"{idx:05d}.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
    confirmed = {}
    for idx in indices:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[8:55, 10:53] = 1
        mask[24, 22 + idx] = 0
        confirmed[idx] = mask

    class FakePredictor:
        def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
            inference_state[frame_idx] = mask.copy()

        def propagate_in_video(self, state, start_frame_idx, reverse):
            indices = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, 5)
            for idx in indices:
                mask = confirmed.get(idx, confirmed[next(iter(confirmed))])
                yield idx, [1], torch.from_numpy(mask.astype(np.float32)[None, None] * 2 - 1)

    monkeypatch.setattr(segmentation, "ensure_checkpoint", lambda directory: tmp_path / "fake.pt")
    monkeypatch.setattr(segmentation.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(segmentation, "_init_sam2_video_state_bounded", lambda *args: {})
    monkeypatch.setattr(build_sam, "build_sam2_video_predictor", lambda *args, **kwargs: FakePredictor())

    paths = segmentation.track_roto_video_masks(frames, tmp_path / "tracked", confirmed)
    assert len(paths) == 5
    for idx, expected in confirmed.items():
        np.testing.assert_array_equal(load_mask(paths[idx]), expected)
    assert load_mask(paths[4])[24, 22 + indices[0]] == 1


def test_preview_uses_causal_reference_and_bypasses_temporal_vote_at_anchor(tmp_path, monkeypatch):
    frames = []
    for idx in range(5):
        frame = tmp_path / f"{idx:05d}.jpg"
        cv2.imwrite(str(frame), np.full((32, 32, 3), idx * 10, dtype=np.uint8))
        frames.append(str(frame))
    first = np.zeros((32, 32), dtype=np.uint8)
    last = np.ones((32, 32), dtype=np.uint8)
    first_mask = tmp_path / "first.png"
    last_mask = tmp_path / "last.png"
    save_mask(first, first_mask)
    save_mask(last, last_mask)
    state = dict(
        src_frames=frames,
        tracked_masks=[str(first_mask)] * 5,
        confirmed_masks={0: str(first_mask), "4": str(last_mask)},
        work_dir=str(tmp_path),
    )
    observed = []
    voted_frames = []

    def vote(paths, idx, **kwargs):
        voted_frames.append(idx)
        return first

    def matte(bgr, mask, cfg, *, ref_bgr, ref_mask):
        observed.append((int(ref_bgr[0, 0, 0]), int(mask.sum()), int(ref_mask.sum())))
        return SimpleNamespace(fg_clean_bgr=bgr, alpha=mask.astype(np.float32))

    monkeypatch.setattr(app, "temporal_smooth_mask", vote)
    monkeypatch.setattr(app, "extract_pixel_accurate_matte", matte)
    monkeypatch.setattr(app, "composite_with_relighting", lambda **kwargs: kwargs["fg_clean_bgr"])
    for idx in (1, 3, 4):
        app._prepare_single_preview_composite(
            idx, frames[0], 2, 2, .95, False, 0, .68, .45, 4, .65, True, .06, .12, .6, False, state,
        )
    assert voted_frames == [1, 3]
    assert observed == [(0, 0, 0), (0, 0, 0), (40, 1024, 1024)]


def test_preview_clears_non_anchor_without_current_mask(tmp_path, monkeypatch):
    frames = []
    for idx in range(74):
        frame = tmp_path / f"{idx:05d}.jpg"
        cv2.imwrite(str(frame), np.full((16, 16, 3), 80, dtype=np.uint8))
        frames.append(str(frame))
    mask_path = tmp_path / "anchor.png"
    save_mask(np.ones((16, 16), dtype=np.uint8), mask_path)
    state = dict(src_frames=frames, tracked_masks=None, confirmed_masks={0: str(mask_path)}, work_dir=str(tmp_path))

    def unexpected(*args, **kwargs):
        pytest.fail("Preview must not matte or track a frame without its own mask")

    monkeypatch.setattr(app, "extract_pixel_accurate_matte", unexpected)
    monkeypatch.setattr(app, "temporal_smooth_mask", unexpected)
    preview, message = app.on_step5_preview_composite(
        73, None, 2, 2, .95, False, 0, .68, .45, 4, .65, True, .06, .12, .6, False, state,
    )

    assert preview is None
    assert "Khung 73 chưa có mask" in message
    assert "Tự Động Tách Tay + Sản Phẩm" in message
    assert "Chạy Tách Sub-Pixel Toàn Bộ Video (SAM 2.1)" in message


@pytest.mark.parametrize("selected_frame,tracked", [(0, False), (73, True)])
def test_preview_runs_with_anchor_or_current_tracked_mask(tmp_path, monkeypatch, selected_frame, tracked):
    frames = []
    for idx in range(74):
        frame = tmp_path / f"{idx:05d}.jpg"
        cv2.imwrite(str(frame), np.full((16, 16, 3), idx, dtype=np.uint8))
        frames.append(str(frame))
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:12, 3:13] = 1
    mask_path = tmp_path / "mask.png"
    save_mask(mask, mask_path)
    state = dict(
        src_frames=frames,
        tracked_masks=[str(mask_path)] * 74 if tracked else None,
        confirmed_masks={0: str(mask_path)},
        work_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        app, "extract_pixel_accurate_matte",
        lambda bgr, *_args, **_kwargs: SimpleNamespace(fg_clean_bgr=bgr, alpha=mask.astype(np.float32)),
    )
    monkeypatch.setattr(app, "composite_with_relighting", lambda **kwargs: kwargs["fg_clean_bgr"])

    (before, after), _message = app.on_step5_preview_composite(
        selected_frame, None, 2, 2, .95, False, 0, .68, .45, 4, .65, True, .06, .12, .6, False, state,
    )

    assert before.shape == after.shape == (16, 16, 3)


def test_changed_or_removed_confirmation_invalidates_stale_tracking(tmp_path, monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    source = tmp_path / "source.jpg"
    cv2.imwrite(str(source), image)
    state = dict(src_frames=[str(source)], work_dir=str(tmp_path), confirmed_masks={}, roto_strokes={})
    mask = np.ones(image.shape[:2], dtype=np.uint8)
    strokes = np.zeros_like(mask)
    monkeypatch.setattr(app, "analyze_scene_telemetry", lambda *args: {})
    monkeypatch.setattr(
        app, "extract_pixel_accurate_matte",
        lambda *args: SimpleNamespace(
            rgba=np.zeros((*mask.shape, 4), dtype=np.uint8), fg_clean_bgr=image,
            alpha=mask.astype(np.float32), missing_restored_px=0, excess_removed_px=0,
        ),
    )

    def confirm():
        app._run_confirm_and_preview(0, image, strokes, strokes, 2, 2, .95, False, 0, state, direct_mask=mask)

    confirm()
    state["tracked_masks"] = ["previous"]
    confirm()
    assert state["tracked_masks"] == ["previous"]
    mask[1, 1] = 0
    confirm()
    assert state["tracked_masks"] is None
    state["tracked_masks"] = ["previous"]
    app.on_step2_reset_selection(0, state)
    assert state["tracked_masks"] is None



def test_final_render_switches_reference_at_anchor_and_resets_temporal_alpha(tmp_path, monkeypatch):
    frames = []
    for idx in range(5):
        frame = tmp_path / f"{idx:05d}.jpg"
        cv2.imwrite(str(frame), np.full((32, 32, 3), idx * 10, dtype=np.uint8))
        frames.append(str(frame))
    masks = []
    for idx, area in enumerate((0, 8, 16)):
        path = tmp_path / f"mask{idx}.png"
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[:, :area] = 1
        save_mask(mask, path)
        masks.append(str(path))

    state = dict(
        src_frames=frames, tracked_masks=[masks[0]] * 5,
        confirmed_masks={1: masks[1], 3: masks[2]}, work_dir=str(tmp_path), width=32, height=32, fps=30,
    )
    voted = []
    refs = []
    reset_at = []
    rendered = []
    red = np.zeros((32, 32), dtype=np.uint8); red[0, 0] = 1
    state["manual_strokes"] = {1: {"bg": red}}
    alpha_at_red = []

    def vote(paths, idx, **kwargs):
        voted.append(idx)
        return np.zeros((32, 32), dtype=np.uint8)

    def matte(bgr, mask, cfg, *, ref_bgr, ref_mask, **kwargs):
        refs.append((int(ref_bgr[0, 0, 0]), int(mask.sum()), int(ref_mask.sum())))
        return SimpleNamespace(
            fg_clean_bgr=bgr, alpha=mask.astype(np.float32),
            missing_restored_px=0, excess_removed_px=0,
        )

    def stabilize(alpha, previous, bgr, prev_bgr):
        if previous is None or prev_bgr is None:
            reset_at.append(int(bgr[0, 0, 0]) // 10)
        return alpha

    def composite(**kwargs):
        rendered.append(int(kwargs["src_full_bgr"][0, 0, 0]) // 10)
        alpha_at_red.append(float(kwargs["alpha"][0, 0]))
        return kwargs["fg_clean_bgr"]

    monkeypatch.setattr(app, "temporal_smooth_mask", vote)
    monkeypatch.setattr(app, "extract_pixel_accurate_matte", matte)
    monkeypatch.setattr(app, "stabilize_alpha_temporal", stabilize)
    monkeypatch.setattr(app, "composite_with_relighting", composite)
    monkeypatch.setattr(app, "_extract_media_frames", lambda *args, **kwargs: ([frames[0]], 30, 32, 32, False))
    monkeypatch.setattr(app, "encode_video", lambda directory, path, **kwargs: path)

    app.on_step5_render_final_video(
        frames[0], "Lock 1:1", "None", 2, 2, .95, False, 0, .68, .45, 4, .65,
        True, .06, .12, .6, False, state, progress=lambda *args, **kwargs: None,
    )

    assert voted == [0, 2, 4]
    assert refs == [(10, 0, 256), (10, 256, 256), (10, 0, 256), (30, 512, 512), (30, 0, 512)]
    assert reset_at == [0, 1, 3]
    assert rendered == [0, 1, 2, 3, 4]
    assert alpha_at_red == [0, 0, 0, 1, 0]


def test_fast_tiny_skips_cpu_matte_and_reports_no_pass2(tmp_path, monkeypatch):
    frames = []
    for idx in range(4):
        frame = tmp_path / f"fast_{idx:05d}.jpg"
        cv2.imwrite(str(frame), np.full((32, 32, 3), idx * 10, dtype=np.uint8))
        frames.append(str(frame))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    mask_path = tmp_path / "fast_mask.png"
    save_mask(mask, mask_path)
    alpha_paths = []
    for idx in range(4):
        alpha_path = tmp_path / f"tiny_alpha_{idx:05d}.png"
        cv2.imwrite(str(alpha_path), mask * 255)
        alpha_paths.append(alpha_path)
    red = np.zeros_like(mask)
    red[12, 12] = 1
    state = dict(
        src_frames=frames, tracked_masks=[str(mask_path)] * 4,
        confirmed_masks={0: str(mask_path)}, manual_strokes={0: {"bg": red}},
        work_dir=str(tmp_path), width=32, height=32, fps=30,
    )
    rendered_alpha = []

    monkeypatch.setattr(app, "_run_sam2matting_tiny", lambda *args, **kwargs: alpha_paths)
    monkeypatch.setattr(app, "temporal_smooth_mask", lambda *args, **kwargs: mask.copy())
    monkeypatch.setattr(app, "extract_pixel_accurate_matte", lambda *args, **kwargs: pytest.fail("Fast Tiny must skip CPU matte"))
    monkeypatch.setattr(app, "_extract_media_frames", lambda *args, **kwargs: ([frames[0]], 30, 32, 32, False))
    monkeypatch.setattr(app, "encode_video", lambda directory, path, **kwargs: path)
    monkeypatch.setattr(
        app, "composite_with_relighting",
        lambda **kwargs: rendered_alpha.append(kwargs["alpha"][12, 12]) or kwargs["fg_clean_bgr"],
    )

    _, status = app.on_step5_render_final_video(
        frames[0], "Lock 1:1", "None", 2, 2, .95, False, 0, .68, .45, 4, .65,
        True, .06, .12, .6, False, state, progress=lambda *args, **kwargs: None,
        use_sam2matting_tiny=True, fast_tiny=True,
    )

    assert rendered_alpha == [0.0, 1.0, 1.0, 1.0]
    assert "Fast Tiny" in status
    assert "không chạy kiểm định Vòng 2" in status
    assert "Báo cáo Kiểm Định Vòng 2" not in status


def test_tiny_runner_receives_nonempty_255_seed_mask(tmp_path, monkeypatch):
    frames = []
    for idx in range(2):
        frame = tmp_path / f"seed_frame_{idx:05d}.jpg"
        cv2.imwrite(str(frame), np.zeros((32, 32, 3), dtype=np.uint8))
        frames.append(frame)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 11:21] = 1
    mask_path = tmp_path / "confirmed_seed.png"
    save_mask(mask, mask_path)
    runner = tmp_path / "runner.py"
    python = tmp_path / "python.exe"
    runner.touch()
    python.touch()
    monkeypatch.setenv("SAM2MATTING_RUNNER", str(runner))
    monkeypatch.setenv("SAM2MATTING_PYTHON", str(python))
    seen = {}

    def fake_run(args, **kwargs):
        manifest_path = Path(args[args.index("--segments-json") + 1])
        output_dir = Path(args[args.index("--output-dir") + 1])
        manifest = app.json.loads(manifest_path.read_text(encoding="utf-8"))
        staged_mask = cv2.imread(manifest[0]["mask"], cv2.IMREAD_GRAYSCALE)
        seen["mask"] = staged_mask
        segment_dir = output_dir / manifest[0]["output"]
        segment_dir.mkdir(parents=True)
        for idx in range(len(frames)):
            cv2.imwrite(str(segment_dir / f"{idx:05d}.png"), np.full((32, 32), 255, dtype=np.uint8))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(app.subprocess, "run", fake_run)
    alpha_paths = app._run_sam2matting_tiny(
        frames, [mask_path] * len(frames), {0: mask_path}, tmp_path / "tiny", (32, 32),
        lambda *args, **kwargs: None,
    )

    staged_mask = seen["mask"]
    assert staged_mask[16, 16] == 255
    assert set(np.unique(staged_mask)) == {0, 255}
    assert len(alpha_paths) == len(frames)
    assert all(path.is_file() for path in alpha_paths)


def test_auto_and_manual_grid_and_background_constraints(tmp_path, monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), image)
    fg = np.zeros((32, 32), dtype=np.uint8)
    bg = np.zeros_like(fg)
    fg[12, 12] = 1
    bg[4, 4] = 1
    seen = []

    def sample(fg_stroke, bg_stroke, bgr, grid_step, **kwargs):
        seen.append(grid_step)
        return [(12, 12, 1), (4, 4, 0)]

    monkeypatch.setattr(app, "auto_seed_roto_strokes", lambda bgr: (fg, bg))
    monkeypatch.setattr(app, "sample_roto_anchor_points", sample)
    monkeypatch.setattr(app, "preview_single_frame_mask", lambda rgb, pts, **kwargs: np.ones((32, 32), dtype=np.uint8))
    monkeypatch.setattr(app, "analyze_scene_telemetry", lambda *args: {})
    monkeypatch.setattr(app, "extract_pixel_accurate_matte", lambda *args: SimpleNamespace(
        rgba=np.zeros((32, 32, 4), dtype=np.uint8), fg_clean_bgr=image,
        alpha=np.ones((32, 32), dtype=np.float32), missing_restored_px=0, excess_removed_px=0,
    ))
    def state():
        return dict(src_frames=[str(source)], work_dir=str(tmp_path), confirmed_masks={}, roto_strokes={})
    s1, s2 = state(), state()
    auto_result = app.on_step2_auto_roto(0, 2, 2, .95, False, 0, s1)
    app.on_step2_confirm_roto_brush(None, 0, 2, 2, .95, False, 0, s2)
    fallback = app.apply_photoshop_stroke_to_mask(image, None, fg, bg, grid_step=app.AUTO_ROTO_GRID_STEP)
    np.testing.assert_array_equal(auto_result[4]["active_mask_by_frame"][0], fallback)
    np.testing.assert_array_equal(s2["active_mask_by_frame"][0], fallback)
    assert fallback[4, 4] == 0 and fallback[12, 12] == 1
    app.apply_photoshop_stroke_to_mask(image, None, fg, bg, manual_prompts=True)
    assert seen == [32, 32, 32, 28]

import sys, types
import numpy as np
import pytest
from src import segmentation
from src.alpha_matting import auto_seed_roto_strokes, sample_roto_anchor_points


def test_manual_sam_candidate_is_not_clipped_to_short_stroke(monkeypatch):
    image = np.full((200, 200, 3), 220, np.uint8)
    image[30:170, 30:170] = (20, 20, 60)
    candidate = np.zeros((200, 200), dtype=bool)
    candidate[30:170, 30:170] = True
    candidate[75:125, 145:170] = False
    candidate[10:20, 175:185] = True  # Unrelated dark component has no positive prompt.
    image[10:20, 175:185] = (15, 15, 15)
    class Predictor:
        def __init__(self, model): pass
        def set_image(self, image): pass
        def predict(self, **kwargs): return np.array([candidate]), np.array([.95]), None
    pkg = types.ModuleType('sam2'); pkg.__path__ = []
    build = types.ModuleType('sam2.build_sam'); build.build_sam2 = lambda *args, **kwargs: object()
    pred = types.ModuleType('sam2.sam2_image_predictor'); pred.SAM2ImagePredictor = Predictor
    for name, module in [('sam2',pkg), ('sam2.build_sam',build), ('sam2.sam2_image_predictor',pred)]:
        monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(segmentation, 'ensure_checkpoint', lambda _: 'fake')
    monkeypatch.setattr(segmentation.torch.cuda, 'is_available', lambda: False)
    mask = segmentation.preview_single_frame_mask(image, [(100,100,1),(60,60,0),(20,20,0)], manual_prompts=True)
    assert mask[40,40] == 1
    assert mask[20,20] == 0
    assert mask[60,60] == 0
    assert mask[100,100] == 1
    assert mask[15,180] == 0
    auto = segmentation.preview_single_frame_mask(image, [(100,100,1),(20,20,0)])
    assert auto[100,150] == 0  # Concave background gap stays out without a negative point.


def test_auto_exclusions_do_not_cross_detected_device():
    import cv2
    image = np.full((800, 500, 3), (170, 190, 205), np.uint8)
    cv2.rectangle(image, (146, 216), (354, 654), (235, 235, 235), -1)
    cv2.rectangle(image, (150, 220), (350, 650), (20, 20, 50), -1)
    cv2.rectangle(image, (180, 250), (320, 550), (90, 20, 15), -1)
    fg,bg = auto_seed_roto_strokes(image)
    assert not bg[250:550,180:320].any()
    assert bg[:180].any()
    assert fg[216:220,160:340].any()


def test_auto_skin_seed_rejects_flat_beige_background():
    import cv2

    image = np.full((800, 500, 3), (107, 123, 139), np.uint8)
    image[500:760, 320:440] = (112, 130, 151)
    yy, xx = np.mgrid[500:760, 320:440]
    texture = ((xx * 7 + yy * 11) % 9 - 4).astype(np.int16)
    hand = image[500:760, 320:440].astype(np.int16)
    hand += texture[..., None]
    image[500:760, 320:440] = np.clip(hand, 0, 255).astype(np.uint8)

    fg, _ = auto_seed_roto_strokes(image)
    assert not fg[350:480, 300:470].any()
    assert fg[540:720, 340:420].any()


def test_auto_roto_recomputes_instead_of_unioning_previous_auto_mask(monkeypatch):
    import app

    bgr = np.zeros((32, 32, 3), np.uint8)
    monkeypatch.setattr(app, "_load_bgr", lambda _: bgr)
    monkeypatch.setattr(app, "auto_seed_roto_strokes", lambda _: (np.ones((32, 32), np.uint8), np.zeros((32, 32), np.uint8)))
    captured = {}

    def capture(*args, **kwargs):
        captured["state"] = args[9]
        return (None, None, None, None, args[9], "")

    monkeypatch.setattr(app, "_run_confirm_and_preview", capture)
    old_mask = np.ones((32, 32), np.uint8)
    state = {"src_frames": ["frame.jpg"], "active_mask_by_frame": {0: old_mask}}
    result = app.on_step2_auto_roto(0, 2, 2, 0.9, False, 1, state)

    assert 0 not in captured["state"]["active_mask_by_frame"]
    assert state["active_mask_by_frame"][0] is old_mask
    assert result[4] is captured["state"]


def test_manual_anchor_sampling_uses_pixels_on_curved_stroke():
    import cv2

    fg = np.zeros((64, 64), np.uint8)
    bg = np.zeros_like(fg)
    cv2.ellipse(fg, (16, 16), (12, 12), 0, 30, 330, 1, 2)
    bgr = np.zeros((64, 64, 3), np.uint8)

    # The legacy coordinate-wise median falls in the unpainted center of this C.
    legacy = [p for p in sample_roto_anchor_points(fg, bg, bgr, grid_step=32) if p[2] == 1]
    assert legacy == [(13, 16, 1)]
    assert fg[legacy[0][1], legacy[0][0]] == 0

    manual = [p for p in sample_roto_anchor_points(
        fg, bg, bgr, grid_step=32, preserve_stroke_components=True
    ) if p[2] == 1]
    assert manual
    assert all(fg[y, x] > 0 for x, y, _ in manual)


def test_manual_anchor_sampling_keeps_disconnected_marks_in_one_cell():
    fg = np.zeros((64, 64), np.uint8)
    bg = np.zeros_like(fg)
    for x, y in ((5, 5), (25, 5), (5, 25)):
        fg[y, x] = 1
    bgr = np.zeros((64, 64, 3), np.uint8)

    legacy = [p for p in sample_roto_anchor_points(fg, bg, bgr, grid_step=32) if p[2] == 1]
    manual = [p for p in sample_roto_anchor_points(
        fg, bg, bgr, grid_step=32, preserve_stroke_components=True
    ) if p[2] == 1]

    assert len(legacy) == 1
    assert len(manual) == 3
    assert all(fg[y, x] > 0 for x, y, _ in manual)


def test_manual_negative_anchor_sampling_uses_red_stroke_pixels():
    bg = np.zeros((64, 64), np.uint8)
    bg[3, 3] = 1
    bg[25, 25] = 1
    fg = np.zeros_like(bg)
    bgr = np.zeros((64, 64, 3), np.uint8)

    points = sample_roto_anchor_points(
        fg, bg, bgr, grid_step=32, preserve_stroke_components=True
    )
    negatives = [p for p in points if p[2] == 0]
    assert len(negatives) == 2
    assert all(bg[y, x] > 0 for x, y, _ in negatives)


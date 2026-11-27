import numpy as np

from src.alpha_matting import extract_strokes_from_editor


def test_empty_layers_ignore_lossy_composite():
    original = np.full((20, 20, 3), 100, dtype=np.uint8)
    composite = original.copy()
    composite[3:8, 3:8] = [30, 130, 30]
    fg, bg = extract_strokes_from_editor({"layers": [], "composite": composite}, original)
    assert not fg.any() and not bg.any()


def test_transparent_brush_layers_detect_only_painted_pixels():
    original = np.zeros((20, 20, 3), dtype=np.uint8)
    layer = np.zeros((20, 20, 4), dtype=np.uint8)
    layer[3:8, 3:8] = [0, 255, 102, 255]
    layer[10:13, 10:13] = [255, 51, 51, 255]
    layer[0, 0] = [0, 255, 102, 0]
    fg, bg = extract_strokes_from_editor({"layers": [layer], "composite": original}, original)
    assert fg.sum() == 25 and bg.sum() == 9


def test_legacy_composite_still_detects_brush():
    original = np.zeros((20, 20, 3), dtype=np.uint8)
    composite = original.copy()
    composite[3:8, 3:8] = [0, 255, 102]
    fg, bg = extract_strokes_from_editor({"composite": composite}, original)
    assert fg.sum() == 25 and not bg.any()


def test_manual_strokes_accumulate_and_red_overrides():
    from unittest.mock import patch
    import app
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    state = {}
    calls = []
    def fake_run(idx, bgr, fg, bg, *args, **kwargs):
        calls.append((fg.copy(), bg.copy(), kwargs))
        state.setdefault('active_mask_by_frame', {})[idx] = kwargs['direct_mask']
        return (None, None, None, None, state, 'ok')
    def layer(green=(), red=()):
        a = np.zeros((40, 40, 4), dtype=np.uint8)
        for y,x in green: a[y,x] = [0,255,102,255]
        for y,x in red: a[y,x] = [255,51,51,255]
        return {'layers':[a]}
    with patch.object(app, '_run_confirm_and_preview', fake_run), patch.object(app, 'apply_photoshop_stroke_to_mask', side_effect=lambda rgb, prev, fg, bg, **kw: fg):
        run = lambda e: app._apply_manual_editor_strokes(0,image,e,2,2,.95,False,0,state)
        first = layer([(4,4)])
        run(first)
        assert run(first) is None
        run(layer([(4,4),(20,20)]))
        run(layer([(20,20)],[(4,4)]))
    assert len(calls) == 3
    fg,bg,kw = calls[-1]
    assert fg[20,20] == 1 and fg[4,4] == 0 and bg[4,4] == 1
    assert kw['direct_mask'][20,20] == 1 and kw['direct_mask'][4,4] == 0
    assert state['manual_strokes'][0]['bg'][4,4] == 1


def test_manual_red_first_requires_green():
    import app
    image = np.zeros((20,20,3), dtype=np.uint8)
    layer = np.zeros((20,20,4),dtype=np.uint8)
    layer[5,5] = [255,51,51,255]
    try:
        app._apply_manual_editor_strokes(0,image,{'layers':[layer]},2,2,.95,False,0,{})
    except app.gr.Error as exc:
        assert 'nét xanh' in str(exc)
    else:
        raise AssertionError('red-only input must be rejected')


def test_manual_sam_receives_only_explicit_prompts():
    from unittest.mock import patch
    import app
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    fg = np.zeros((100, 100), dtype=np.uint8); fg[10, 10] = 1
    bg = np.zeros_like(fg); bg[90, 90] = 1
    seen = []
    def fake(img, points, **kwargs):
        seen.append((points, kwargs))
        return np.ones((100, 100), dtype=np.uint8)
    with patch.object(app, 'preview_single_frame_mask', fake):
        app.apply_photoshop_stroke_to_mask(rgb, None, fg, bg, manual_prompts=True)
        app.apply_photoshop_stroke_to_mask(rgb, None, fg, np.zeros_like(bg), manual_prompts=True)
    assert seen[0][1]['manual_prompts'] and {p[2] for p in seen[0][0]} == {0, 1}
    assert seen[1][1]['manual_prompts'] and all(p[2] == 1 for p in seen[1][0])


def test_manual_accumulates_across_reset_and_retained_layers_and_respects_red(monkeypatch):
    import app
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    calls = []
    def candidate(rgb, prev, fg, bg, **kwargs):
        calls.append(fg.copy())
        result = np.zeros((40, 40), dtype=np.uint8)
        for y, x in np.argwhere(fg):
            result[y, x] = 1
        return result
    def run_confirm(idx, bgr, fg, bg, *args, **kwargs):
        state = args[-1]
        state.setdefault('active_mask_by_frame', {})[idx] = kwargs['direct_mask']
        state['tracked_masks'] = None
        return (None, None, None, None, state, 'ok')
    monkeypatch.setattr(app, 'apply_photoshop_stroke_to_mask', candidate)
    monkeypatch.setattr(app, '_run_confirm_and_preview', run_confirm)
    def layer(green=(), red=()):
        a = np.zeros((40, 40, 4), dtype=np.uint8)
        for y, x in green: a[y, x] = [0, 255, 102, 255]
        for y, x in red: a[y, x] = [255, 51, 51, 255]
        return {'layers': [a]}
    a, b, c = (4, 4), (12, 12), (20, 20)
    state = {'tracked_masks': ['old']}
    def run(e, idx=0): return app._apply_manual_editor_strokes(idx, image, e, 2, 2, .95, False, 0, state)
    run(layer([a])); run(layer([b]))
    mask = state['active_mask_by_frame'][0]
    assert mask[a] and mask[b] and state['tracked_masks'] is None
    assert run(layer([b])) is None and len(calls) == 2
    run(layer(red=[b])); assert mask[a] and not state['active_mask_by_frame'][0][b]
    run(layer([c])); mask = state['active_mask_by_frame'][0]
    assert mask[a] and mask[c] and not mask[b]
    run(layer([b])); assert state['active_mask_by_frame'][0][b]
    run(layer([a]), 1); assert not state['active_mask_by_frame'][1][b]

    retained = {}
    state = retained
    run(layer([a])); run(layer([a, b]))
    assert state['active_mask_by_frame'][0][a] and state['active_mask_by_frame'][0][b]


def test_red_after_auto_and_green_preserves_auto_mask(monkeypatch):
    import app
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    auto = np.zeros((30, 30), dtype=np.uint8); auto[4, 4] = auto[8, 8] = 1
    state = {'active_mask_by_frame': {0: auto}}
    monkeypatch.setattr(app, 'apply_photoshop_stroke_to_mask', lambda rgb, prev, fg, bg, **kwargs: fg.copy())
    def confirm(idx, bgr, fg, bg, *args, **kwargs):
        state['active_mask_by_frame'][idx] = kwargs['direct_mask']; return (None,) * 6
    monkeypatch.setattr(app, '_run_confirm_and_preview', confirm)
    def layer(y, x, color):
        a = np.zeros((30, 30, 4), dtype=np.uint8); a[y, x] = color; return {'layers': [a]}
    run = lambda e: app._apply_manual_editor_strokes(0, image, e, 2, 2, .95, False, 0, state)
    run(layer(8, 8, [255, 51, 51, 255])); run(layer(12, 12, [0, 255, 102, 255]))
    result = state['active_mask_by_frame'][0]
    assert result[4, 4] and result[12, 12] and not result[8, 8]


def test_explicit_red_survives_real_matting_in_saved_cutout(tmp_path):
    import app
    import cv2
    image = np.full((128, 128, 3), 180, np.uint8)
    image[25:105, 25:105] = [60, 100, 160]
    mask = np.zeros((128, 128), np.uint8); mask[25:105, 25:105] = 1
    red = np.zeros_like(mask); red[60:65, 60:65] = 1
    mask[red > 0] = 0
    state = dict(work_dir=str(tmp_path), confirmed_masks={}, roto_strokes={})
    result = app._run_confirm_and_preview(0, image, np.zeros_like(mask), red, 2, 2, .95, False, 0, state, direct_mask=mask, explicit_bg=red)
    rgba = cv2.imread(result[3], cv2.IMREAD_UNCHANGED)
    assert np.all(rgba[60:65, 60:65, 3] == 0)
    assert rgba[40, 40, 3] == 255


def test_background_guard_is_only_on_requested_anchor():
    import app
    alpha = np.ones((8, 8), dtype=np.float32)
    red = np.zeros((8, 8), dtype=np.uint8); red[3, 3] = 1
    state = {'manual_strokes': {2: {'bg': red}}}
    assert app._protect_manual_background(alpha, state, 2)[3, 3] == 0
    assert app._protect_manual_background(alpha, state, 3)[3, 3] == 1
    assert alpha[3, 3] == 1


def test_selection_overlay_uses_real_fractional_alpha_without_changing_mask():
    import app
    rgb = np.full((32, 32, 3), 100, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8); mask[4:28, 4:28] = 1
    original = mask.copy()
    alpha = np.zeros((32, 32), dtype=np.float32); alpha[4:28, 4:28] = .3
    soft = app._render_photoshop_green_selection(rgb, mask, alpha)
    binary = app._render_photoshop_green_selection(rgb, mask)
    assert np.array_equal(soft[16, 16], [89, 114, 98])
    assert not np.array_equal(soft[16, 16], binary[16, 16])
    assert np.array_equal(mask, original)


def test_selection_navigation_loads_matching_saved_alpha_only(tmp_path):
    import app
    import cv2
    from src.video_io import save_mask
    mask = np.zeros((16, 16), dtype=np.uint8); mask[3:13, 3:13] = 1
    mask_file = tmp_path / 'mask.png'; save_mask(mask, mask_file)
    rgba = np.zeros((16, 16, 4), dtype=np.uint8); rgba[:, :, 3] = 76
    cv2.imwrite(str(tmp_path / 'subpixel_cutout_f00000.png'), rgba)
    state = {'work_dir': str(tmp_path), 'confirmed_masks': {0: str(mask_file)}}
    alpha = app._saved_selection_alpha(state, 0, mask)
    assert alpha is not None and abs(float(alpha[5, 5]) - 76 / 255) < 1e-6
    different = mask.copy(); different[5, 5] = 0
    assert app._saved_selection_alpha(state, 0, different) is None

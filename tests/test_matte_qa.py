from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.flow_prompt_engine import (
    analyze_video_optical_profile,
    generate_matched_background_prompts,
)
from src.matte_qa import (
    audit_and_heal_video_mattes,
    heal_device_bezel_notches,
    heal_universal_structural_defects,
    is_green_screen_scene,
)
from src.neural_matting import estimate_foreground_pymatting
from src.segmentation import _fill_mask_holes


def test_heal_device_bezel_notches_repairs_volume_rocker_bite() -> None:
    """Reproduce a concave bite dropped along the white side bezel & black volume rocker."""
    h, w = 360, 240
    bgr = np.full((h, w, 3), (175, 190, 200), dtype=np.uint8)  # light studio wall BG

    # Tilted/vertical handheld translator: dark glass bezel + blue UI + white side frame + black & red side buttons
    dev_y0, dev_y1, dev_x0, dev_x1 = 60, 290, 65, 175
    bgr[dev_y0:dev_y1, dev_x0:dev_x1] = (25, 25, 30)          # black front chassis
    bgr[dev_y0 + 18:dev_y1 - 35, dev_x0 + 8:dev_x1 - 8] = (145, 55, 25)  # blue UI screen
    # Crisp white/silver right metallic chassis rim + upper volume rocker (black) + lower power button (red)
    bgr[dev_y0:dev_y1, dev_x1 - 6:dev_x1 + 3] = (238, 240, 242)
    bgr[90:145, dev_x1 + 1:dev_x1 + 5] = (32, 32, 36)       # black volume rocker
    bgr[152:176, dev_x1 + 1:dev_x1 + 5] = (38, 52, 225)     # red power button

    # Ground-truth full mask
    alpha_gt = np.zeros((h, w), dtype=np.float32)
    alpha_gt[dev_y0:dev_y1, dev_x0:dev_x1 + 5] = 1.0

    # Simulate MatAnyone 2 temporal dropout: big concave bite above red button (y=85..150, x=162..180)
    alpha_bug = alpha_gt.copy()
    alpha_bug[85:150, 162:dev_x1 + 6] = 0.0

    healed, healed_px, notch_mask = heal_device_bezel_notches(bgr, alpha_bug)
    assert healed_px > 400
    # >= 85% of the missing white bezel + black volume button chunk is restored automatically
    restored_frac = float(np.mean(healed[88:148, 164:dev_x1 + 2] >= 0.70))
    assert restored_frac >= 0.85, f"Only restored {restored_frac:.2%}"
    # Far background outside device remains 0.0
    assert float(np.max(healed[:40, :])) < 0.01
    assert float(np.max(healed[:, 195:])) < 0.01
    assert np.any(notch_mask > 0)


def test_heal_device_bezel_notches_noop_on_non_device_scene() -> None:
    """Organic non-rectangular subject must never trigger device convex hull filling."""
    h, w = 220, 220
    bgr = np.full((h, w, 3), (120, 130, 140), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)
    cv2.circle(bgr, (110, 110), 50, (110, 150, 200), -1)  # warm skin-like disk
    cv2.circle(alpha, (110, 110), 50, 1.0, -1)
    # Cut an organic crescent bay into the circle
    cv2.circle(alpha, (150, 110), 22, 0.0, -1)

    healed, healed_px, _ = heal_device_bezel_notches(bgr, alpha)
    assert healed_px == 0
    assert float(np.max(np.abs(healed - alpha))) < 1e-6


def test_audit_and_heal_video_mattes_samples_10pct_and_repairs_defects(tmp_path: Path) -> None:
    n_frames = 20
    h, w = 160, 140
    frames_dir = tmp_path / "frames"
    matte_dir = tmp_path / "mattes"
    frames_dir.mkdir()
    matte_dir.mkdir()

    frame_paths = []
    matte_paths = []
    gt_masks = []
    for t in range(n_frames):
        bgr = np.full((h, w, 3), (50, 60, 70), dtype=np.uint8)
        bgr[35:135, 35:105] = (120, 155, 210)
        fp = frames_dir / f"{t:05d}.jpg"
        cv2.imwrite(str(fp), bgr)
        frame_paths.append(fp)

        gt = np.zeros((h, w), dtype=np.uint8)
        gt[35:135, 35:105] = 255
        gt_masks.append(gt)

        broken = gt.copy()
        if t == 8:
            # Inject large missing hole (> 140px)
            broken[40:75, 45:75] = 0
        if t == 14:
            # Inject large false background leak blob (> 300px)
            broken[5:32, 5:32] = 255

        mp = matte_dir / f"{t:05d}.png"
        cv2.imwrite(str(mp), broken)
        matte_paths.append(mp)

    reprop_calls = []

    def _mock_crosscheck(bgr: np.ndarray) -> np.ndarray:
        ref = np.zeros((h, w), dtype=np.float32)
        ref[35:135, 35:105] = 1.0
        return ref

    report = audit_and_heal_video_mattes(
        frame_paths,
        matte_paths,
        is_alpha_stream=True,
        sample_ratio=0.10,
        min_samples=4,
        max_samples=10,
        use_birefnet_crosscheck=False,
        crosscheck_fn=_mock_crosscheck,
        heal_repropagate_cb=lambda kfs: reprop_calls.append(dict(kfs)),
    )

    assert report["audited_count"] >= 4
    assert 8 in report["healed_frames"]
    assert 14 in report["healed_frames"]
    assert report["flagged_frames"] >= 2
    assert len(reprop_calls) == 1
    assert 8 in reprop_calls[0] and 14 in reprop_calls[0]
    assert any(s["frame_idx"] == 8 and s["diff_b64"].startswith("data:image/jpeg;base64,") for s in report["samples"])

    # Verify the written mattes for frames 8 and 14 were healed on disk!
    healed8 = cv2.imread(str(matte_paths[8]), cv2.IMREAD_GRAYSCALE)
    healed14 = cv2.imread(str(matte_paths[14]), cv2.IMREAD_GRAYSCALE)
    assert float(np.mean(healed8[45:70, 50:70] >= 180)) > 0.90
    assert float(np.mean(healed14[8:28, 8:28] < 40)) > 0.90


def test_heal_universal_structural_defects_fills_book_page_pinhole_without_filling_outer_bay() -> None:
    h, w = 220, 220
    bgr = np.full((h, w, 3), (75, 95, 80), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)
    # Person/arms holding an open bright book (not an electronic screen!)
    bgr[40:190, 45:175] = (230, 232, 236)  # bright white book pages + light shirt
    alpha[40:190, 45:175] = 1.0
    # Inject a bright glare pinhole inside the book page (must be healed)
    alpha[95:110, 95:110] = 0.0
    # Outer concavity on the right perimeter (must NOT be bridged/filled!)
    alpha[80:140, 155:175] = 0.0

    healed, px, _ = heal_universal_structural_defects(bgr, alpha)
    assert px > 150
    assert float(np.mean(healed[97:108, 97:108])) > 0.85
    assert float(np.mean(healed[90:130, 160:175])) < 0.05


def test_box_select_extracts_arbitrary_subject_like_book_or_person(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import studio_server
    from src import segmentation

    store = studio_server.SessionStore(base_dir=tmp_path / "sessions", ttl_seconds=600)
    monkeypatch.setattr(studio_server, "STORE", store)

    captured_boxes = []

    class _FakeBoxPredictor:
        def __init__(self):
            self.h = 180
            self.w = 180

        def set_image(self, img):
            self.h, self.w = img.shape[:2]

        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
            captured_boxes.append(box.tolist() if box is not None else None)
            m0 = np.zeros((1, self.h, self.w), dtype=bool)
            m0[0, 35:145, 38:135] = True  # book + person inside box
            m1 = np.ones((1, self.h, self.w), dtype=bool)
            m2 = np.zeros((1, self.h, self.w), dtype=bool)
            m2[0, 60:90, 60:100] = True
            return np.concatenate([m0, m1, m2], axis=0), np.array([0.93, 0.72, 0.80], dtype=np.float32), None

    fake_pred = _FakeBoxPredictor()
    monkeypatch.setattr(
        segmentation,
        "_get_cached_sam2_predictor",
        lambda *a, **kw: (None, fake_pred, "cpu", True, ("sig",)),
    )

    client = TestClient(studio_server.app)
    sid = client.post("/matting/api/session").json()["session_id"]

    src = np.full((180, 180, 3), (45, 50, 55), dtype=np.uint8)
    src[35:145, 38:135] = (225, 225, 230)  # open white book / shirt
    _, src_b = cv2.imencode(".png", src)
    ui = client.post(f"/matting/api/session/{sid}/upload/init", json={"role": "source", "filename": "book.png", "total_size": len(src_b)})
    uid = ui.json()["upload_id"]
    client.put(f"/matting/api/session/{sid}/upload/{uid}/chunk?index=0", content=src_b.tobytes())
    client.post(f"/matting/api/session/{sid}/upload/{uid}/complete")

    # Box-select with NO brush strokes at all
    sr = client.post(
        f"/matting/api/session/{sid}/strokes",
        json={"frame_idx": 0, "box_xyxy": [32, 30, 142, 152], "engine": "matanyone2"},
    )
    assert sr.status_code == 200, sr.text
    assert captured_boxes and captured_boxes[0] == [32.0, 30.0, 142.0, 152.0]
    assert sr.json()["fg_pct"] > 30.0


def test_universal_optical_prompt_handles_person_reading_book_without_device_jargon() -> None:
    h, w = 240, 180
    frames = []
    masks = []
    for k in range(3):
        bgr = np.full((h, w, 3), (45, 50, 55), dtype=np.uint8)
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        # Seated person wearing blue-gray sweater holding open white book (0 skin dominant!)
        fg_mask[25:235, 30:150] = 255
        for x in range(30, 150):
            val = int(round(215 - ((x - 30) / 120.0) * 80))
            bgr[25:235, x] = (val, val - 5, val - 10)
        frames.append(bgr)
        masks.append(fg_mask)

    profile = analyze_video_optical_profile(frames, fg_masks=masks, sample_count=3, use_vision_ai=False)
    assert profile["shot_kind"] == "portrait_medium"
    assert "TRÁI" in profile["lighting_direction"]

    prompts = generate_matched_background_prompts(
        profile,
        target_background_vi="Phòng đọc sách thư viện ấm áp có ánh nắng qua cửa sổ",
        subject_description_vi="Người phụ nữ ngồi đọc cuốn sách bìa cứng",
        use_local_llm=False,
    )
    v2v = prompts["v2v_prompt"].lower()
    assert "cuốn sách" in v2v or "người phụ nữ" in v2v
    assert "electronic device" not in v2v
    assert "morphing" not in v2v
    assert "máy phiên dịch" not in v2v
    assert "library" in prompts["clean_plate_prompt"].lower()


def test_universal_hole_guard_preserves_bg_gap_and_cleans_green_slit() -> None:
    # 1) Green-screen scene: hand + device + shadowed green slit between finger & device + body pinhole + screen green icon
    h, w = 240, 240
    bgr = np.full((h, w, 3), (45, 175, 42), dtype=np.uint8)  # chroma green background
    alpha = np.zeros((h, w), dtype=np.float32)

    # Dark handheld chassis + blue emissive screen
    bgr[35:200, 65:165] = (24, 24, 28)
    bgr[50:175, 76:154] = (145, 55, 25)
    alpha[35:200, 65:165] = 1.0

    # Small green UI button INSIDE emissive screen must be protected
    bgr[125:140, 108:124] = (55, 215, 65)

    # Interior dark pinhole in lower bezel body (not green, not BG-colored) -> should heal
    alpha[183:192, 108:122] = 0.0

    # Thumb + palm gripping around right side, creating a trapped shadowed green slit between finger & bezel
    bgr[130:230, 165:205] = (120, 155, 210)
    bgr[195:235, 95:195] = (120, 155, 210)
    alpha[130:230, 165:205] = 1.0
    alpha[195:235, 95:195] = 1.0

    # Shadowed green slit at x=[156..167], y=[135..185] where BiRefNet mistakenly left alpha=0.90
    bgr[135:185, 156:167] = (42, 86, 44)
    alpha[135:185, 156:167] = 0.90

    assert is_green_screen_scene(bgr, alpha) is True
    # Default mode leaves exterior/slit geometry untouched (never eats shadowed skin/green subjects by surprise)
    healed_default, _, _ = heal_universal_structural_defects(bgr, alpha)
    assert float(np.mean(healed_default[184:191, 110:120])) > 0.70  # interior body pinhole healed
    assert float(np.mean(healed_default[127:138, 110:122])) > 0.85  # green UI dot inside emissive screen preserved
    # Explicit green-screen crevice mode punches out confirmed chroma-green cloth slit
    healed, _, _ = heal_universal_structural_defects(bgr, alpha, clean_green_crevices=True)
    assert float(np.mean(healed[142:178, 158:165])) < 0.15  # trapped shadowed green crevice punched out
    assert float(np.mean(healed[184:191, 110:120])) > 0.70  # interior device body pinhole healed
    assert float(np.mean(healed[127:138, 110:122])) > 0.85  # green UI dot inside emissive screen preserved

    # 2) Non-green room background (warm beige wall) with arm/coat loop enclosing small beige wall gap
    h2, w2 = 200, 200
    bgr_room = np.full((h2, w2, 3), (185, 200, 215), dtype=np.uint8)
    mask_room = np.zeros((h2, w2), dtype=np.uint8)
    bgr_room[30:170, 40:160] = (48, 44, 42)  # dark coat/book loop
    mask_room[30:170, 40:160] = 1
    # Small enclosed beige room wall gap (e.g. between elbow and torso)
    bgr_room[85:105, 88:104] = (185, 200, 215)
    mask_room[85:105, 88:104] = 0
    # True foreground-colored missing pinhole spot on coat (6x6 = 36 px <= max_hole_area 48)
    mask_room[126:132, 90:96] = 0

    assert is_green_screen_scene(bgr_room, mask_room.astype(np.float32)) is False
    healed_room, _, _ = heal_universal_structural_defects(bgr_room, mask_room.astype(np.float32))
    assert float(np.mean(healed_room[88:102, 91:101])) < 0.10
    assert float(np.mean(healed_room[126:132, 90:96])) > 0.70

    filled_mask = _fill_mask_holes(mask_room, bgr=bgr_room)
    assert int(np.sum(filled_mask[88:102, 91:101])) == 0
    assert float(np.mean(filled_mask[126:132, 90:96])) == 1.0


def test_pymatting_fast_preview_preserves_core() -> None:
    bgr = np.full((240, 240, 3), (45, 175, 42), dtype=np.uint8)
    bgr[60:180, 60:180] = (90, 120, 190)
    alpha = np.zeros((240, 240), dtype=np.float32)
    alpha[60:180, 60:180] = 1.0
    alpha[59:181, 59] = 0.5
    alpha[59:181, 180] = 0.5

    out = estimate_foreground_pymatting(bgr, alpha, strength=0.8, despill_green=True, fast_preview=True)
    assert out.dtype == np.uint8 and out.shape == bgr.shape
    core = alpha >= 0.996
    # Wait: core has despill check disabled on pure opaque interior (edge_weight is 0 when alpha >= 0.995)
    assert np.array_equal(out[core], bgr[core])


def test_green_bounce_on_shadowed_skin_and_white_bezel_never_eaten() -> None:
    """Green bounce light on shadowed skin or white bezels must NEVER be punched out; real green cloth slits still clean."""
    h, w = 240, 240
    bgr = np.full((h, w, 3), (45, 175, 42), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)

    # Device chassis + blue screen + reflective white rim
    bgr[35:200, 65:158] = (24, 24, 28)
    bgr[48:170, 74:148] = (145, 55, 25)
    bgr[35:200, 152:159] = (198, 212, 200)  # white metal bezel with subtle green spill
    alpha[35:200, 65:159] = 1.0

    # Hand + shadowed thumb edge with strong green bounce spill
    bgr[125:230, 170:215] = (120, 155, 210)
    bgr[192:235, 90:200] = (120, 155, 210)
    bgr[140:175, 175:195] = (65, 80, 72)  # shadowed skin with green bounce
    alpha[125:230, 170:215] = 1.0
    alpha[192:235, 90:200] = 1.0

    # Real shadowed green-screen cloth slit between white bezel and finger (42 x 10 = 420 px)
    bgr[135:185, 159:170] = (42, 86, 44)
    alpha[135:185, 159:170] = 0.90

    healed, _, _ = heal_universal_structural_defects(bgr, alpha, clean_green_crevices=True)
    assert float(np.mean(healed[142:173, 176:194])) > 0.95  # shadowed skin preserved 100%
    assert float(np.mean(healed[60:120, 153:158])) > 0.95   # tinted white bezel preserved 100%
    assert float(np.mean(healed[142:178, 161:168])) < 0.15  # real green cloth crevice punched out


def test_auto_cut_hybrid_rejects_shadowed_green_triangle_and_keeps_real_part(monkeypatch) -> None:
    from src import neural_matting, segmentation

    h, w = 240, 240
    bgr = np.full((h, w, 3), (45, 175, 42), dtype=np.uint8)
    # Main dark device core
    bgr[50:190, 90:155] = (28, 30, 34)
    bgr[65:165, 98:148] = (145, 55, 25)
    # Genuine dropped white bezel strip on right side
    bgr[55:185, 155:166] = (232, 234, 238)
    # Shadowed green-screen cloth triangle touching left side of device
    bgr[120:185, 65:90] = (35, 110, 38)

    def _fake_birefnet(img_bgr, max_size=1024):
        a = np.zeros((h, w), dtype=np.float32)
        a[50:190, 90:155] = 0.98
        return a

    monkeypatch.setattr(neural_matting, "run_birefnet_matting", _fake_birefnet)
    monkeypatch.setattr(neural_matting, "release_neural_models", lambda keep=None: None)

    class _FakeHybridPredictor:
        def set_image(self, img):
            pass

        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
            m = np.zeros((1, h, w), dtype=bool)
            m[0, 50:190, 90:155] = True      # core
            m[0, 55:185, 155:166] = True     # genuine white bezel strip on right
            m[0, 120:185, 65:90] = True      # erroneous shadowed green triangle on left
            return m, np.array([0.92], dtype=np.float32), None

    monkeypatch.setattr(
        segmentation,
        "_get_cached_sam2_predictor",
        lambda *a, **kw: (None, _FakeHybridPredictor(), "cpu", True, ("sig",)),
    )

    mask_out, alpha_out = segmentation.auto_cut_universal_hybrid(bgr, prefer_best_checkpoint=False)
    assert float(np.mean(alpha_out[125:180, 68:87])) < 0.15
    assert float(np.mean(mask_out[125:180, 68:87])) < 0.15
    assert float(np.mean(alpha_out[60:180, 156:164])) > 0.70


def test_tilted_device_over_shadowed_green_and_wood_never_adds_background_triangles(tmp_path: Path, monkeypatch) -> None:
    """
    Regression test for real-world tilted handheld device over shadowed green screen + wooden desk:
    Pure AI matte must pass through `heal_universal_structural_defects`, `audit_and_heal_video_mattes`,
    and `/auto-cut` (1-Click & Chroma AI) with ZERO convex-hull background triangles added.
    """
    from fastapi.testclient import TestClient

    import studio_server

    h, w = 320, 220
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:195, :] = (55, 118, 48)    # shadowed dark teal-green screen cloth
    bgr[195:, :] = (115, 168, 205)  # warm wooden desk in lower third

    # Tilted rectangular device (dark glass + blue UI + white side rim) held by hand
    pts_dev = np.array([[72, 48], [156, 62], [134, 228], [50, 214]], dtype=np.int32)
    pts_ui = np.array([[80, 64], [146, 75], [130, 194], [64, 183]], dtype=np.int32)
    cv2.fillConvexPoly(bgr, pts_dev, (24, 24, 28))
    cv2.fillConvexPoly(bgr, pts_ui, (150, 52, 24))
    cv2.line(bgr, (156, 62), (134, 228), (235, 238, 240), thickness=4)
    # Hand fingers wrapping right & bottom (creating concave bays next to tilted chassis)
    cv2.circle(bgr, (162, 165), 26, (125, 155, 205), -1)
    cv2.ellipse(bgr, (120, 248), (64, 30), 12, 0, 360, (125, 155, 205), -1)

    alpha_clean = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(alpha_clean, pts_dev, 1.0)
    cv2.circle(alpha_clean, (162, 165), 26, 1.0, -1)
    cv2.ellipse(alpha_clean, (120, 248), (64, 30), 12, 0, 360, 1.0, -1)

    # 1. heal_universal_structural_defects must add ZERO background pixels around tilted flanks/wood
    healed, healed_px, _ = heal_universal_structural_defects(bgr, alpha_clean)
    added_bg = int(np.sum((healed >= 0.5) & (alpha_clean < 0.5)))
    removed_fg = int(np.sum((healed < 0.5) & (alpha_clean >= 0.5)))
    assert healed_px == 0
    assert added_bg == 0, f"Unexpectedly added {added_bg} background pixels!"
    assert removed_fg == 0, f"Unexpectedly removed {removed_fg} foreground pixels!"

    # 2. audit_and_heal_video_mattes must score 100% with 0 false heals when tracker matches BiRefNet
    fdir = tmp_path / "f"
    mdir = tmp_path / "m"
    fdir.mkdir()
    mdir.mkdir()
    fpaths, mpaths = [], []
    for t in range(6):
        fp = fdir / f"{t:05d}.jpg"
        mp = mdir / f"{t:05d}.png"
        cv2.imwrite(str(fp), bgr)
        cv2.imwrite(str(mp), (alpha_clean * 255).astype(np.uint8))
        fpaths.append(fp)
        mpaths.append(mp)
    rep = audit_and_heal_video_mattes(
        fpaths,
        mpaths,
        is_alpha_stream=True,
        sample_ratio=0.5,
        min_samples=3,
        use_birefnet_crosscheck=False,
        crosscheck_fn=lambda _: alpha_clean,
    )
    assert rep["flagged_frames"] == 0
    assert rep["integrity_score"] == 100.0

    # 3. Studio server 1-Click & Chroma AI endpoints return exact clean alpha without desk or triangle leaks
    store = studio_server.SessionStore(base_dir=tmp_path / "sessions", ttl_seconds=600)
    monkeypatch.setattr(studio_server, "STORE", store)
    monkeypatch.setattr(studio_server, "run_birefnet_matting", lambda img, max_size=1024: alpha_clean.copy())

    client = TestClient(studio_server.app)
    sid = client.post("/matting/api/session").json()["session_id"]
    _, enc = cv2.imencode(".png", bgr)
    ui = client.post(f"/matting/api/session/{sid}/upload/init", json={"role": "source", "filename": "tilted.png", "total_size": len(enc)})
    uid = ui.json()["upload_id"]
    client.put(f"/matting/api/session/{sid}/upload/{uid}/chunk?index=0", content=enc.tobytes())
    client.post(f"/matting/api/session/{sid}/upload/{uid}/complete")

    for mode in ("birefnet", "chroma"):
        res = client.post(f"/matting/api/session/{sid}/auto-cut", json={"frame_idx": 0, "mode": mode, "engine": "matanyone2"})
        assert res.status_code == 200, res.text
        sess = store.get(sid)
        saved_alpha = sess.birefnet_alpha_by_frame[0]
        assert int(np.sum((saved_alpha >= 0.5) & (alpha_clean < 0.5))) == 0
        assert int(np.sum((saved_alpha < 0.5) & (alpha_clean >= 0.5))) == 0


def test_pymatting_preserves_thin_dark_side_button_while_despilling_green_halo() -> None:
    """Thin dark side button at alpha ~0.92 next to a bright white bezel must NOT be bleached white by PyMatting."""
    h, w = 260, 220
    bgr = np.full((h, w, 3), (45, 180, 42), dtype=np.uint8)  # chroma green background
    alpha = np.zeros((h, w), dtype=np.float32)

    # Device body + bright white right bezel
    bgr[30:230, 50:155] = (25, 25, 30)
    bgr[30:230, 140:158] = (242, 244, 246)
    alpha[30:230, 50:158] = 1.0

    # Thin 4px black volume rocker right outside white bezel with near-opaque neural alpha (~0.90)
    bgr[75:145, 158:163] = (24, 24, 28)
    alpha[75:145, 158:163] = 0.90

    # 2px translucent green-contaminated edge fringe outside the button (alpha=0.35)
    bgr[75:145, 163:165] = (48, 138, 46)
    alpha[75:145, 163:165] = 0.35

    out = estimate_foreground_pymatting(bgr, alpha, strength=0.95, despill_green=True, fast_preview=True)
    # Dark button must stay dark (< 75 mean luma), never bleached white by neighboring bezel diffusion
    btn_luma = float(np.mean(out[80:140, 159:162]))
    assert btn_luma < 75.0, f"Dark side button was washed out to luma={btn_luma:.1f}!"
    # Translucent outer fringe must have its green spill suppressed
    fringe = out[80:140, 163:165].astype(np.float32)
    fringe_g_exc = float(np.mean(fringe[..., 1] - np.maximum(fringe[..., 0], fringe[..., 2])))
    assert fringe_g_exc < 12.0, f"Green fringe remained: {fringe_g_exc:.1f}"


def test_background_transform_grade_and_reverse_source_match_in_preview(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import studio_server
    from src.relight_harmonizer import grade_and_match_background, transform_background_frame

    # 1. Transform scale + offset shifts a landmark cleanly with zero black border artifacts
    bg = np.full((180, 120, 3), (60, 75, 90), dtype=np.uint8)
    cv2.circle(bg, (60, 90), 16, (40, 220, 240), -1)  # bright yellow center marker
    tr_default = transform_background_frame(bg, 120, 180, scale=1.0, offset_x=0.0, offset_y=0.0)
    tr_zoom_pan = transform_background_frame(bg, 120, 180, scale=1.6, offset_x=0.22, offset_y=-0.18)
    assert tr_zoom_pan.shape == (180, 120, 3)
    assert float(np.mean(np.abs(tr_zoom_pan.astype(np.float32) - tr_default.astype(np.float32)))) > 5.0
    # Center marker shifted up-right into quadrant (x > 65, y < 80)
    assert float(np.max(tr_zoom_pan[:85, 65:, 2])) > 180.0

    # 2. Reverse match (BG -> Source) pulls dark cool BG toward warm bright source subject
    fg_warm = np.full((120, 120, 3), (120, 165, 215), dtype=np.uint8)
    fg_mask = np.zeros((120, 120), dtype=np.uint8)
    fg_mask[20:100, 20:100] = 1
    bg_dark_cool = np.full((120, 120, 3), (95, 70, 50), dtype=np.uint8)
    bg_matched = grade_and_match_background(bg_dark_cool, fg_warm, fg_mask, match_source_strength=0.9)
    assert float(np.mean(bg_matched)) > float(np.mean(bg_dark_cool)) + 20.0
    # Warmth (R - B) moves positively toward warm source
    assert float(np.mean(bg_matched[..., 2].astype(float) - bg_matched[..., 0].astype(float))) > float(
        np.mean(bg_dark_cool[..., 2].astype(float) - bg_dark_cool[..., 0].astype(float))
    ) + 12.0

    # 3. Preview endpoint accepts all new BG params and visibly changes output
    store = studio_server.SessionStore(base_dir=tmp_path / "sessions", ttl_seconds=600)
    monkeypatch.setattr(studio_server, "STORE", store)
    client = TestClient(studio_server.app)
    sid = client.post("/matting/api/session").json()["session_id"]
    sess = store.get(sid)

    src = np.full((180, 120, 3), (45, 175, 42), dtype=np.uint8)
    src[60:150, 30:90] = (125, 160, 210)
    _, sb = cv2.imencode(".png", src)
    _, bb = cv2.imencode(".png", bg)
    for role, raw, fn in (("source", sb.tobytes(), "s.png"), ("background", bb.tobytes(), "b.png")):
        ui = client.post(f"/matting/api/session/{sid}/upload/init", json={"role": role, "filename": fn, "total_size": len(raw)})
        uid = ui.json()["upload_id"]
        client.put(f"/matting/api/session/{sid}/upload/{uid}/chunk?index=0", content=raw)
        client.post(f"/matting/api/session/{sid}/upload/{uid}/complete")

    a = np.zeros((180, 120), dtype=np.float32)
    a[60:150, 30:90] = 1.0
    studio_server._save_confirmed_mask(sess, 0, (a >= 0.5).astype(np.uint8), push_undo=False)
    sess.birefnet_alpha_by_frame[0] = a

    import base64

    def _dec(url: str) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), dtype=np.uint8), cv2.IMREAD_COLOR)

    r0 = client.post(f"/matting/api/session/{sid}/preview", json={"frame_idx": 0, "use_pymatting_fg": False})
    r1 = client.post(
        f"/matting/api/session/{sid}/preview",
        json={
            "frame_idx": 0,
            "use_pymatting_fg": False,
            "bg_scale": 1.65,
            "bg_offset_x": 0.20,
            "bg_offset_y": -0.15,
            "bg_match_source": 0.85,
            "bg_brightness": 0.15,
            "bg_temperature": 0.4,
        },
    )
    assert r0.status_code == 200 and r1.status_code == 200
    diff_bg = float(np.mean(np.abs(_dec(r1.json()["right_b64"])[:50, :].astype(np.float32) - _dec(r0.json()["right_b64"])[:50, :].astype(np.float32))))
    assert diff_bg > 15.0


def test_optical_prompt_measures_noise_sharpness_tone_and_orientation_on_green_screen() -> None:
    """Green-screen video telemetry must measure real foreground sensor noise, tone/Kelvin, sharpness, and tilt."""
    rng = np.random.default_rng(7)
    h, w = 320, 200
    frames = []
    masks = []
    for _ in range(3):
        # Perfectly flat compressed green screen exterior (zero noise on BG!)
        bgr = np.full((h, w, 3), (45, 178, 42), dtype=np.uint8)
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        # Tilted right (~14 deg) warm subject with authentic camera sensor noise
        pts = np.array([[78, 70], [142, 86], [116, 260], [52, 244]], dtype=np.int32)
        cv2.fillConvexPoly(fg_mask, pts, 255)
        subj_patch = np.full((h, w, 3), (118, 156, 208), dtype=np.float32)
        subj_patch += rng.normal(0.0, 5.5, size=(h, w, 3)).astype(np.float32)
        bgr[fg_mask > 0] = np.clip(subj_patch[fg_mask > 0], 0, 255).astype(np.uint8)
        frames.append(bgr)
        masks.append(fg_mask)

    profile = analyze_video_optical_profile(frames, fg_masks=masks, sample_count=3, use_vision_ai=False)
    assert profile["is_green_screen"] is True
    assert profile["grain_sigma"] > 0.006, f"Expected real FG sensor noise, got {profile['grain_sigma']}"
    assert profile["subject_orientation_deg"] > 5.0
    assert profile["iso_est"] >= 400
    card_ids = {c["id"] for c in profile["telemetry_cards"]}
    assert {"lens_scale", "camera_angle", "key_light", "color_tone", "sharp_dof", "mode"} == card_ids

    prompts = generate_matched_background_prompts(
        profile,
        target_background_vi="Sảnh sân bay quốc tế hiện đại có ánh sáng tự nhiên",
        subject_description_vi="Bàn tay cầm máy phiên dịch",
        use_local_llm=False,
    )
    cp = prompts["clean_plate_prompt"].lower()
    v2v = prompts["v2v_prompt"].lower()
    assert "iso" in cp and "k" in cp and "deep front-to-back sharp focus" in cp
    assert "tilted" in v2v


def test_stationary_camera_lock_and_prompt_regeneration_variations(tmp_path: Path, monkeypatch) -> None:
    """
    Clean background prompts MUST be strictly positive-only (ZERO pink-elephant tokens like
    'hand', 'holding', '30cm', 'bokeh', 'blur', 'dslr', 'cinematic' that trick Veo 3 / Flow
    into hallucinating a foreground smartphone in hand or blurry portrait bokeh!), enforce
    stationary standing camera with gentle organic sway, hide Windows FFmpeg consoles, and
    regenerate fresh variations using cached profile.
    """
    import os
    from fastapi.testclient import TestClient

    import studio_server
    from src.video_io import _hidden_subprocess_kwargs

    if os.name == "nt":
        kw = _hidden_subprocess_kwargs()
        assert kw.get("creationflags", 0) != 0
        assert kw.get("startupinfo") is not None

    # Even with fast foreground hand motion between frames, camera motion MUST stay stationary standing + gentle sway!
    h, w = 220, 160
    frames = []
    for k in range(3):
        img = np.full((h, w, 3), (45, 175, 42), dtype=np.uint8)
        cv2.rectangle(img, (25 + k * 24, 60), (95 + k * 24, 180), (120, 155, 210), -1)
        frames.append(img)

    profile = analyze_video_optical_profile(frames, sample_count=3, use_vision_ai=False)
    assert "stationary standing camera" in profile["camera_motion_en"].lower()
    assert "gentle organic sway" in profile["camera_motion_en"].lower()
    assert "strictly no walking" in profile["camera_motion_en"].lower()
    assert "walk/sway" not in profile["camera_motion_en"].lower()

    p0 = generate_matched_background_prompts(profile, "Sảnh sân bay quốc tế hiện đại", use_local_llm=False, variation_seed=0)
    p1 = generate_matched_background_prompts(profile, "Sảnh sân bay quốc tế hiện đại", use_local_llm=False, variation_seed=1)
    cp0 = p0["clean_plate_prompt"].lower()
    sp0 = p0["still_bg_prompt"].lower()
    assert "stationary standing camera with gentle organic sway" in cp0
    assert "strictly no walking" in cp0
    assert "raw unedited everyday smartphone" in cp0
    assert "deep front-to-back sharp focus" in cp0
    # Anti-pink-elephant guard: clean background plate & still photo MUST NEVER mention hands/phones-in-hand/bokeh/blur/cinematic!
    forbidden = ("hand", "holding", "device", "30cm", "close-up", "foreground subject", "cinematic", "dslr", "bokeh", "f/1.", "unsharpness", "blur", "hdr", "green-screen")
    for bad in forbidden:
        assert bad not in cp0, f"Forbidden pink-elephant token '{bad}' leaked into clean_plate_prompt!"
        assert bad not in sp0, f"Forbidden pink-elephant token '{bad}' leaked into still_bg_prompt!"
    assert "strictly no walking" in p0["v2v_prompt"].lower()
    assert p0["clean_plate_prompt"] != p1["clean_plate_prompt"]

    # Studio server endpoint: first call analyzes, regenerate call reuses cached profile & changes prompt
    store = studio_server.SessionStore(base_dir=tmp_path / "sessions", ttl_seconds=600)
    monkeypatch.setattr(studio_server, "STORE", store)
    client = TestClient(studio_server.app)
    sid = client.post("/matting/api/session").json()["session_id"]

    _, buf = cv2.imencode(".png", frames[0])
    ui = client.post(f"/matting/api/session/{sid}/upload/init", json={"role": "source", "filename": "f.png", "total_size": len(buf)})
    uid = ui.json()["upload_id"]
    client.put(f"/matting/api/session/{sid}/upload/{uid}/chunk?index=0", content=buf.tobytes())
    client.post(f"/matting/api/session/{sid}/upload/{uid}/complete")

    r1 = client.post(
        f"/matting/api/session/{sid}/analyze-prompts",
        json={"target_background_vi": "Quán cà phê kính hiện đại", "use_local_llm": False, "use_vision_ai": False, "variation_seed": 0, "force_reanalyze": True},
    )
    assert r1.status_code == 200
    j1 = r1.json()
    assert j1["reused_profile"] is False
    assert "strictly no walking" in j1["clean_plate_prompt"].lower()

    r2 = client.post(
        f"/matting/api/session/{sid}/analyze-prompts",
        json={"target_background_vi": "Quán cà phê kính hiện đại", "use_local_llm": False, "use_vision_ai": False, "variation_seed": 1, "force_reanalyze": False},
    )
    assert r2.status_code == 200
    j2 = r2.json()
    assert j2["reused_profile"] is True
    assert j2["variation_seed"] == 1
    assert j2["clean_plate_prompt"] != j1["clean_plate_prompt"]

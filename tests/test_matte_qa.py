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
)


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

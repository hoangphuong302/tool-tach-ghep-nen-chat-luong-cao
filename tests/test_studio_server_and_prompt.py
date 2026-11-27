from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

import studio_server
from src.flow_prompt_engine import (
    analyze_video_optical_profile,
    generate_matched_background_prompts,
)
from src.relight_harmonizer import RelightConfig, composite_with_relighting
from src.video_io import VideoPipeEncoder, get_video_metadata


def test_black_point_match_harmonizes_shadow_floor_without_touching_screen() -> None:
    h, w = 180, 180
    fg = np.full((h, w, 3), (45, 55, 70), dtype=np.uint8)
    fg[40:150, 50:130] = (32, 48, 72)  # shadowed warm skin (R > 55 so distinct from OLED bezel)
    fg[55:115, 65:115] = (180, 45, 25)  # emissive blue screen

    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[40:150, 50:130] = 1.0

    # Lifted bright-shadow haze background (shadow floor ~85, bright upper ~175)
    bg = np.full((h, w, 3), (85, 95, 105), dtype=np.uint8)
    bg[:90, :] = (165, 175, 185)

    cfg0 = RelightConfig(
        ambient_color_strength=0.0,
        directional_light_strength=0.0,
        light_wrap_strength=0.0,
        grain_match=0.0,
        black_point_match=0.0,
        pass3_aa_and_restore=False,
    )
    cfg1 = RelightConfig(
        ambient_color_strength=0.0,
        directional_light_strength=0.0,
        light_wrap_strength=0.0,
        grain_match=0.0,
        black_point_match=1.0,
        pass3_aa_and_restore=False,
    )

    out0 = composite_with_relighting(fg, alpha, bg, src_full_bgr=fg, cfg=cfg0)
    out1 = composite_with_relighting(fg, alpha, bg, src_full_bgr=fg, cfg=cfg1)

    shadow_luma0 = float(np.mean(out0[125:145, 60:120]))
    shadow_luma1 = float(np.mean(out1[125:145, 60:120]))
    assert shadow_luma1 > shadow_luma0 + 5.0

    # Emissive blue screen center remains protected and faithful
    scr_b_diff = float(np.mean(out1[70:100, 75:105, 0] - out1[70:100, 75:105, 2]))
    assert scr_b_diff > 100.0


def test_multiframe_optical_profile_and_matched_prompts() -> None:
    h, w = 240, 180
    frames = []
    masks = []
    for k in range(4):
        bgr = np.full((h, w, 3), (40, 45, 50), dtype=np.uint8)
        # Mildly soft distant pattern shifted slightly per frame -> camera motion
        cv2.rectangle(bgr, (20 + k * 2, 20), (160 + k * 2, 110), (130, 145, 160), -1)
        bgr = cv2.GaussianBlur(bgr, (5, 5), 1.4)

        # Crisp hand with directional skin light brighter on LEFT side (x=45 -> x=135 fades darker)
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        fg_mask[115:238, 45:135] = 255
        for x in range(45, 135):
            t = (x - 45) / 90.0
            r_val = int(round(225 - t * 85))
            g_val = int(round(165 - t * 60))
            b_val = int(round(125 - t * 45))
            bgr[115:238, x] = (b_val, g_val, r_val)
        frames.append(bgr)
        masks.append(fg_mask)

    profile = analyze_video_optical_profile(frames, fg_masks=masks, sample_count=4, use_vision_ai=False)
    assert len(profile["sampled_indices"]) == 4
    assert "TRÁI" in profile["lighting_direction"]
    assert "PHẢI" in profile["shadow_direction_vi"]
    assert len(profile["telemetry_cards"]) == 6
    assert profile["camera_motion_px"] >= 0.0

    prompts = generate_matched_background_prompts(
        profile,
        target_background_vi="Quầy bán linh kiện điện tử sáng tự nhiên",
        use_local_llm=False,
    )
    assert "clean_plate_prompt" in prompts
    cp_low = prompts["clean_plate_prompt"].lower()
    assert "open empty floor" in cp_low and "hand" not in cp_low and "bokeh" not in cp_low
    assert "v2v_prompt" in prompts and "KEEP" in prompts["v2v_prompt"]
    assert "still_bg_prompt" in prompts and str(profile["focal_length_mm"]) in prompts["still_bg_prompt"]


def test_studio_server_chunk_upload_preview_and_cleanup(tmp_path, monkeypatch) -> None:
    store = studio_server.SessionStore(base_dir=tmp_path / "sessions", ttl_seconds=600)
    monkeypatch.setattr(studio_server, "STORE", store)

    client = TestClient(studio_server.app)

    # Create isolated session
    r = client.post("/matting/api/session")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    sess_dir = store.base_dir / sid
    assert sess_dir.is_dir()

    # Build tiny PNG source image (hand skin patch on wall) and BG image
    src = np.full((128, 128, 3), (60, 70, 80), dtype=np.uint8)
    src[40:120, 35:95] = (120, 155, 210)
    ok, src_bytes = cv2.imencode(".png", src)
    assert ok
    raw_src = src_bytes.tobytes()

    bg = np.full((128, 128, 3), (90, 120, 150), dtype=np.uint8)
    ok, bg_bytes = cv2.imencode(".png", bg)
    assert ok
    raw_bg = bg_bytes.tobytes()

    # 2-chunk upload for source
    ui = client.post(f"/matting/api/session/{sid}/upload/init", json={"role": "source", "filename": "hand.png", "total_size": len(raw_src)})
    assert ui.status_code == 200
    uid = ui.json()["upload_id"]
    mid = len(raw_src) // 2
    assert client.put(f"/matting/api/session/{sid}/upload/{uid}/chunk?index=0", content=raw_src[:mid]).status_code == 200
    assert client.put(f"/matting/api/session/{sid}/upload/{uid}/chunk?index=1", content=raw_src[mid:]).status_code == 200
    uc = client.post(f"/matting/api/session/{sid}/upload/{uid}/complete")
    assert uc.status_code == 200
    assert uc.json()["width"] == 128 and uc.json()["frame_count"] == 1

    # 1-chunk upload for background
    bi = client.post(f"/matting/api/session/{sid}/upload/init", json={"role": "background", "filename": "bg.png", "total_size": len(raw_bg)})
    buid = bi.json()["upload_id"]
    assert client.put(f"/matting/api/session/{sid}/upload/{buid}/chunk?index=0", content=raw_bg).status_code == 200
    assert client.post(f"/matting/api/session/{sid}/upload/{buid}/complete").status_code == 200

    # Paint green stroke (bypassing heavy SAM/BiRefNet downloads with classic engine stroke test)
    fg_png = np.zeros((128, 128, 4), dtype=np.uint8)
    fg_png[50:110, 45:85] = [34, 197, 94, 255]
    ok, st_bytes = cv2.imencode(".png", fg_png)
    assert ok
    st_b64 = "data:image/png;base64," + base64.b64encode(st_bytes.tobytes()).decode("ascii")
    sr = client.post(f"/matting/api/session/{sid}/strokes", json={"frame_idx": 0, "fg_mask_b64": st_b64, "engine": "classic"})
    assert sr.status_code == 200
    assert sr.json()["fg_pct"] > 5.0

    # Multi-frame prompt analysis
    ap = client.post(
        f"/matting/api/session/{sid}/analyze-prompts",
        json={"target_background_vi": "Sảnh sân bay hiện đại", "use_local_llm": False, "use_vision_ai": False},
    )
    assert ap.status_code == 200
    assert len(ap.json()["telemetry_cards"]) == 6
    assert ap.json()["clean_plate_prompt"]

    # Fast live preview across all 4 diagnostic modes
    for mode in ("split", "halo_black", "halo_white", "screen_mask"):
        pr = client.post(
            f"/matting/api/session/{sid}/preview",
            json={"frame_idx": 0, "diag_mode": mode, "matting_engine": "classic", "black_point_match": 0.4},
        )
        assert pr.status_code == 200
        pj = pr.json()
        assert pj["left_b64"].startswith("data:image/jpeg;base64,")
        assert pj["right_b64"].startswith("data:image/jpeg;base64,")

    # Download with cleanup=1 MUST stream file cleanly AND wipe server directory immediately!
    dl = client.get(f"/matting/api/session/{sid}/download?kind=cutout&frame_idx=0&cleanup=1")
    assert dl.status_code == 200
    assert dl.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert not sess_dir.exists()


def test_linear_lut_accuracy_and_video_pipe_encoder(tmp_path) -> None:
    from src.relight_harmonizer import _linear_to_srgb, _srgb_to_linear
    from src.video_io import VideoPipeEncoder, get_video_metadata

    vals = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16, 1)
    lin_lut = _srgb_to_linear(vals)
    lin_ref = np.power(vals, 2.2)
    assert float(np.max(np.abs(lin_lut - lin_ref))) < 5e-4

    srgb_lut = _linear_to_srgb(vals)
    srgb_ref = np.power(vals, 1.0 / 2.2)
    assert float(np.max(np.abs(srgb_lut - srgb_ref))) < 5e-4

    out_mp4 = tmp_path / "pipe_test.mp4"
    with VideoPipeEncoder(out_mp4, width=128, height=128, fps=24.0, prefer_nvenc=True) as enc:
        for i in range(6):
            frame = np.full((128, 128, 3), (30 + i * 20, 90, 180), dtype=np.uint8)
            enc.write(frame)
    assert out_mp4.is_file() and out_mp4.stat().st_size > 1000
    meta = get_video_metadata(out_mp4)
    assert meta.width == 128 and meta.height == 128 and meta.frame_count == 6
    assert abs(meta.duration_sec - 0.25) < 0.05


def test_studio_server_real_mp4_full_pipeline_and_cleanup(tmp_path, monkeypatch) -> None:
    store = studio_server.SessionStore(base_dir=tmp_path / "sessions", ttl_seconds=600)
    monkeypatch.setattr(studio_server, "STORE", store)
    client = TestClient(studio_server.app)

    # Trailing-slash redirect check on /matting
    redir = client.get("/matting", follow_redirects=False)
    assert redir.status_code == 302
    assert redir.headers["location"] == "/matting/"

    # Build real MP4 source & background videos (12 frames, 160x200)
    src_mp4 = tmp_path / "src.mp4"
    bg_mp4 = tmp_path / "bg.mp4"
    with VideoPipeEncoder(src_mp4, width=160, height=200, fps=24.0, prefer_nvenc=False) as enc:
        for i in range(12):
            f = np.full((200, 160, 3), (35, 190, 45), dtype=np.uint8)  # chroma green BG
            f[50:170, 45:115] = (120, 155, 210)  # hand + device block
            enc.write(f)
    with VideoPipeEncoder(bg_mp4, width=160, height=200, fps=24.0, prefer_nvenc=False) as enc:
        for i in range(12):
            f = np.full((200, 160, 3), (80 + i * 3, 110, 145), dtype=np.uint8)
            enc.write(f)

    sid = client.post("/matting/api/session").json()["session_id"]
    sess_dir = store.base_dir / sid

    # 2-chunk source MP4 upload
    raw_src = src_mp4.read_bytes()
    ui = client.post(
        f"/matting/api/session/{sid}/upload/init",
        json={"role": "source", "filename": "clip.mp4", "total_size": len(raw_src)},
    )
    assert ui.status_code == 200
    uid = ui.json()["upload_id"]
    mid = len(raw_src) // 2
    assert client.put(f"/matting/api/session/{sid}/upload/{uid}/chunk?index=0", content=raw_src[:mid]).status_code == 200
    assert client.put(f"/matting/api/session/{sid}/upload/{uid}/chunk?index=1", content=raw_src[mid:]).status_code == 200
    uc = client.post(f"/matting/api/session/{sid}/upload/{uid}/complete")
    assert uc.status_code == 200, uc.text
    uc_json = uc.json()
    assert uc_json["frame_count"] == 12
    assert uc_json["width"] == 160 and uc_json["height"] == 200
    assert uc_json["is_image"] is False
    assert uc_json["duration"] > 0

    # Background MP4 upload
    raw_bg = bg_mp4.read_bytes()
    bi = client.post(
        f"/matting/api/session/{sid}/upload/init",
        json={"role": "background", "filename": "bg.mp4", "total_size": len(raw_bg)},
    )
    buid = bi.json()["upload_id"]
    assert client.put(f"/matting/api/session/{sid}/upload/{buid}/chunk?index=0", content=raw_bg).status_code == 200
    bc = client.post(f"/matting/api/session/{sid}/upload/{buid}/complete")
    assert bc.status_code == 200 and bc.json()["frame_count"] == 12

    # 0-byte upload returns clean 400 instead of 500
    zi = client.post(f"/matting/api/session/{sid}/upload/init", json={"role": "background", "filename": "empty.mp4", "total_size": 0})
    zuid = zi.json()["upload_id"]
    assert client.post(f"/matting/api/session/{sid}/upload/{zuid}/complete").status_code == 400

    # Red-only stroke on empty frame returns clean 400
    red_png = np.zeros((200, 160, 4), dtype=np.uint8)
    red_png[10:30, 10:30] = [239, 68, 68, 255]
    _, red_bytes = cv2.imencode(".png", red_png)
    red_b64 = "data:image/png;base64," + base64.b64encode(red_bytes.tobytes()).decode("ascii")
    bad_stroke = client.post(f"/matting/api/session/{sid}/strokes", json={"frame_idx": 0, "bg_mask_b64": red_b64, "engine": "classic"})
    assert bad_stroke.status_code == 400

    # Auto chroma cut + stroke + undo + prompt analysis + live preview
    ac = client.post(f"/matting/api/session/{sid}/auto-cut", json={"frame_idx": 0, "mode": "chroma", "engine": "classic"})
    assert ac.status_code == 200 and ac.json()["fg_pct"] > 5.0

    fg_png = np.zeros((200, 160, 4), dtype=np.uint8)
    fg_png[55:160, 50:110] = [34, 197, 94, 255]
    _, fg_bytes = cv2.imencode(".png", fg_png)
    fg_b64 = "data:image/png;base64," + base64.b64encode(fg_bytes.tobytes()).decode("ascii")
    assert client.post(f"/matting/api/session/{sid}/strokes", json={"frame_idx": 0, "fg_mask_b64": fg_b64, "bg_mask_b64": red_b64, "engine": "classic"}).status_code == 200
    assert client.post(f"/matting/api/session/{sid}/undo", json={"frame_idx": 0}).json()["ok"] is True

    ap = client.post(f"/matting/api/session/{sid}/analyze-prompts", json={"use_local_llm": False, "use_vision_ai": False})
    assert ap.status_code == 200 and len(ap.json()["telemetry_cards"]) == 6

    for mode in ("split", "halo_black", "halo_white", "screen_mask"):
        assert client.post(f"/matting/api/session/{sid}/preview", json={"frame_idx": 0, "diag_mode": mode, "matting_engine": "classic"}).status_code == 200

    dl_cut = client.get(f"/matting/api/session/{sid}/download?kind=cutout&frame_idx=0&cleanup=0&engine=classic")
    assert dl_cut.status_code == 200 and dl_cut.content[:8] == b"\x89PNG\r\n\x1a\n"

    # Full MP4 video render + poll + download with cleanup=1
    rj = client.post(
        f"/matting/api/session/{sid}/render",
        json={"matting_engine": "classic", "tracking_mode": "Lock 1:1", "audio_choice": "none"},
    )
    assert rj.status_code == 200
    jid = rj.json()["job_id"]
    deadline = time.time() + 30.0
    st = {}
    while time.time() < deadline:
        st = client.get(f"/matting/api/session/{sid}/render/{jid}").json()
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert st.get("status") == "done", st

    dl_vid = client.get(f"/matting/api/session/{sid}/download?kind=video&job_id={jid}&cleanup=1")
    assert dl_vid.status_code == 200
    assert b"ftyp" in dl_vid.content[:64]
    assert not sess_dir.exists()
    assert client.post(f"/matting/api/session/{sid}/upload/init", json={"role": "source", "filename": "x.mp4", "total_size": 10}).status_code == 404


def _decode_b64_img(data_url: str) -> np.ndarray:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None
    return img


def test_refine_alpha_edge_controls_preserves_opaque_interior_and_recovers_clipped_rim() -> None:
    from src.alpha_matting import refine_alpha_edge_controls

    h, w = 160, 160
    bgr = np.full((h, w, 3), (45, 55, 65), dtype=np.uint8)
    bgr[35:130, 45:115] = (120, 155, 215)  # warm skin / subject

    # Neural alpha clipped 2px inward at bottom/right rim (0.28 instead of 1.0 on true foreground rim)
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[37:128, 47:113] = 1.0
    alpha[128:130, 47:113] = 0.28
    alpha[37:128, 113:115] = 0.28

    refined = refine_alpha_edge_controls(
        bgr,
        alpha,
        erode_dilate=0.0,
        guided_radius=2,
        trimap_radius=6,
        edge_preserve=0.85,
        edge_feather=0.0,
    )
    assert float(np.mean(refined[55:110, 65:95])) > 0.995
    assert float(np.mean(refined[5:25, 5:25])) < 0.02
    assert float(np.mean(refined[128:130, 55:105])) > 0.55
    assert float(np.mean(refined[50:110, 113:115])) > 0.55


def test_edge_sliders_visibly_change_cached_birefnet_preview(tmp_path, monkeypatch) -> None:
    store = studio_server.SessionStore(base_dir=tmp_path / "sessions", ttl_seconds=600)
    monkeypatch.setattr(studio_server, "STORE", store)
    client = TestClient(studio_server.app)

    sid = client.post("/matting/api/session").json()["session_id"]
    sess = store.get(sid)

    src = np.full((180, 180, 3), (40, 45, 50), dtype=np.uint8)
    src[40:145, 45:135] = (115, 155, 215)
    bg = np.full((180, 180, 3), (200, 185, 160), dtype=np.uint8)
    _, src_b = cv2.imencode(".png", src)
    _, bg_b = cv2.imencode(".png", bg)

    for role, raw, fn in (("source", src_b.tobytes(), "s.png"), ("background", bg_b.tobytes(), "b.png")):
        ui = client.post(f"/matting/api/session/{sid}/upload/init", json={"role": role, "filename": fn, "total_size": len(raw)})
        uid = ui.json()["upload_id"]
        assert client.put(f"/matting/api/session/{sid}/upload/{uid}/chunk?index=0", content=raw).status_code == 200
        assert client.post(f"/matting/api/session/{sid}/upload/{uid}/complete").status_code == 200

    # Simulate BiRefNet 1-Click cutout slightly clipped inward on skin edge
    bi_alpha = np.zeros((180, 180), dtype=np.float32)
    bi_alpha[42:143, 47:133] = 1.0
    bi_alpha[40:42, 47:133] = 0.25
    bi_alpha[143:145, 47:133] = 0.25
    studio_server._save_confirmed_mask(sess, 0, (bi_alpha >= 0.42).astype(np.uint8), push_undo=False)
    sess.birefnet_alpha_by_frame[0] = bi_alpha

    def _prev(**kw):
        payload = {"frame_idx": 0, "diag_mode": "split", "matting_engine": "matanyone2", "use_pymatting_fg": False}
        payload.update(kw)
        r = client.post(f"/matting/api/session/{sid}/preview", json=payload)
        assert r.status_code == 200, r.text
        return _decode_b64_img(r.json()["right_b64"])

    img_e_neg = _prev(erode_dilate=-3.0)
    img_e_zero = _prev(erode_dilate=0.0)
    img_e_pos = _prev(erode_dilate=3.0)
    assert float(np.mean(np.abs(img_e_neg[36:46, 55:125].astype(np.float32) - img_e_zero[36:46, 55:125].astype(np.float32)))) > 12.0
    assert float(np.mean(np.abs(img_e_pos[36:46, 55:125].astype(np.float32) - img_e_zero[36:46, 55:125].astype(np.float32)))) > 12.0
    assert float(np.mean(np.abs(img_e_neg.astype(np.float32) - img_e_pos.astype(np.float32)))) > 3.0

    # Verify edge_preserve recovers clipped top rim rows toward source skin vs 0.0
    img_ep0 = _prev(edge_preserve=0.0, guided_radius=0, trimap_radius=6)
    img_ep1 = _prev(edge_preserve=1.0, guided_radius=0, trimap_radius=6)
    assert float(np.mean(np.abs(img_ep1[40:42, 60:120].astype(np.float32) - src[40:42, 60:120].astype(np.float32)))) < float(
        np.mean(np.abs(img_ep0[40:42, 60:120].astype(np.float32) - src[40:42, 60:120].astype(np.float32)))
    )

    # Verify edge_blur_match widens soft transition visibly
    img_f0 = _prev(edge_blur_match=0.0)
    img_f3 = _prev(edge_blur_match=3.0)
    assert float(np.mean(np.abs(img_f3.astype(np.float32) - img_f0.astype(np.float32)))) > 0.5


def test_smart_roto_brush_preserves_birefnet_alpha_and_undo_restores_it(tmp_path, monkeypatch) -> None:
    from src import segmentation

    store = studio_server.SessionStore(base_dir=tmp_path / "sessions", ttl_seconds=600)
    monkeypatch.setattr(studio_server, "STORE", store)

    class _FakePredictor:
        def __init__(self, h: int = 180, w: int = 180):
            self.h = h
            self.w = w

        def set_image(self, img):
            self.h, self.w = img.shape[:2]

        def predict(self, point_coords, point_labels, multimask_output=True):
            # Produce 3 masks around positive anchor (finger extension on right)
            m0 = np.zeros((1, self.h, self.w), dtype=bool)
            m0[0, 60:95, 110:150] = True
            m1 = m0.copy()
            m1[0, 55:100, 105:155] = True
            m2 = np.ones((1, self.h, self.w), dtype=bool)
            return np.concatenate([m0, m1, m2], axis=0), np.array([0.92, 0.85, 0.70], dtype=np.float32), None

    fake_pred = _FakePredictor()
    monkeypatch.setattr(
        segmentation,
        "_get_cached_sam2_predictor",
        lambda *a, **kw: (None, fake_pred, "cpu", True, ("sig",)),
    )

    client = TestClient(studio_server.app)
    sid = client.post("/matting/api/session").json()["session_id"]
    sess = store.get(sid)

    src = np.full((180, 180, 3), (35, 40, 45), dtype=np.uint8)
    src[40:140, 35:110] = (120, 155, 210)
    src[60:95, 110:150] = (120, 155, 210)
    _, src_b = cv2.imencode(".png", src)
    ui = client.post(f"/matting/api/session/{sid}/upload/init", json={"role": "source", "filename": "s.png", "total_size": len(src_b)})
    uid = ui.json()["upload_id"]
    client.put(f"/matting/api/session/{sid}/upload/{uid}/chunk?index=0", content=src_b.tobytes())
    client.post(f"/matting/api/session/{sid}/upload/{uid}/complete")

    # Initial BiRefNet fractional alpha (0.63 feather on left edge) must be untouched outside the right-hand stroke edit zone!
    init_alpha = np.zeros((180, 180), dtype=np.float32)
    init_alpha[40:140, 36:110] = 1.0
    init_alpha[40:140, 35] = 0.63
    studio_server._save_confirmed_mask(sess, 0, (init_alpha >= 0.42).astype(np.uint8), push_undo=False)
    sess.birefnet_alpha_by_frame[0] = init_alpha.copy()

    fg_png = np.zeros((180, 180, 4), dtype=np.uint8)
    fg_png[70:85, 125:140] = [34, 197, 94, 255]
    _, st_b = cv2.imencode(".png", fg_png)
    st_b64 = "data:image/png;base64," + base64.b64encode(st_b.tobytes()).decode("ascii")

    sr = client.post(f"/matting/api/session/{sid}/strokes", json={"frame_idx": 0, "fg_mask_b64": st_b64, "engine": "matanyone2"})
    assert sr.status_code == 200, sr.text
    after_alpha = sess.birefnet_alpha_by_frame[0]
    assert float(np.mean(after_alpha[65:90, 120:145])) > 0.85
    # Untouched soft feather edge on far left (x=35) is preserved!
    assert abs(float(after_alpha[80, 35]) - 0.63) < 0.05

    # Undo restores BOTH mask and exact initial soft alpha
    ur = client.post(f"/matting/api/session/{sid}/undo", json={"frame_idx": 0})
    assert ur.status_code == 200 and ur.json()["ok"] is True
    assert float(np.max(np.abs(sess.birefnet_alpha_by_frame[0] - init_alpha))) < 1e-5


def test_preview_any_unconfirmed_frame_and_slider_updates(tmp_path, monkeypatch) -> None:
    store = studio_server.SessionStore(base_dir=tmp_path / "sessions", ttl_seconds=600)
    monkeypatch.setattr(studio_server, "STORE", store)
    client = TestClient(studio_server.app)

    # 1. Cache-busting on index HTML & static JS
    idx_res = client.get("/matting/")
    assert idx_res.status_code == 200
    assert "./static/styles.css?v=" in idx_res.text
    assert "./static/app.js?v=" in idx_res.text
    js_res = client.get("/matting/static/app.js")
    assert js_res.status_code == 200
    assert "no-store" in js_res.headers.get("cache-control", "").lower()

    # Create a 6-frame MP4 clip with a moving blue-ish foreground rectangle
    src_mp4 = tmp_path / "scrub_src.mp4"
    writer = cv2.VideoWriter(str(src_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (160, 160))
    for i in range(6):
        fr = np.full((160, 160, 3), (30, 35, 40), dtype=np.uint8)
        x0 = 20 + i * 8
        cv2.rectangle(fr, (x0, 35), (x0 + 50, 125), (210, 140, 70), -1)
        writer.write(fr)
    writer.release()

    sid = client.post("/matting/api/session").json()["session_id"]
    sess = store.get(sid)
    raw_bytes = src_mp4.read_bytes()
    ui = client.post(
        f"/matting/api/session/{sid}/upload/init",
        json={"role": "source", "filename": "scrub_src.mp4", "total_size": len(raw_bytes)},
    )
    uid = ui.json()["upload_id"]
    client.put(f"/matting/api/session/{sid}/upload/{uid}/chunk?index=0", content=raw_bytes)
    client.post(f"/matting/api/session/{sid}/upload/{uid}/complete")

    # Empty session without ANY confirmed mask must return 400
    r_empty = client.post(
        f"/matting/api/session/{sid}/preview",
        json={"frame_idx": 4, "matting_engine": "classic"},
    )
    assert r_empty.status_code == 400

    # Confirm ONLY frame 0
    mask0 = np.zeros((160, 160), dtype=np.uint8)
    mask0[35:125, 20:70] = 1
    studio_server._save_confirmed_mask(sess, 0, mask0, push_undo=False)

    # Classic engine preview on unconfirmed frame 4 -> 200, auto_tracked=True, not added to confirmed_masks
    r_cls = client.post(
        f"/matting/api/session/{sid}/preview",
        json={"frame_idx": 4, "matting_engine": "classic"},
    )
    assert r_cls.status_code == 200, r_cls.text
    d_cls = r_cls.json()
    assert d_cls["auto_tracked"] is True
    assert d_cls["left_b64"].startswith("data:image/jpeg;base64,")
    assert d_cls["right_b64"].startswith("data:image/jpeg;base64,")
    assert 4 not in sess.confirmed_masks

    # Mock BiRefNet returning frame 4 subject + decoy blob to test disambiguation + slider reactivity
    sess.preview_auto_cache.clear()
    biref_calls = {"n": 0}

    def _fake_birefnet(bgr, max_size=768):
        biref_calls["n"] += 1
        a = np.zeros((160, 160), dtype=np.float32)
        # Moving subject at frame 4 (x0 = 52..102) matching reference color
        a[35:125, 52:102] = 0.98
        return a

    monkeypatch.setattr(studio_server, "run_birefnet_matting", _fake_birefnet)

    base_payload = {
        "frame_idx": 4,
        "matting_engine": "matanyone2",
        "ambient_color_strength": 0.0,
        "bg_defocus": 0.0,
        "erode_dilate": 0.0,
    }
    r1 = client.post(f"/matting/api/session/{sid}/preview", json=base_payload)
    assert r1.status_code == 200, r1.text
    d1 = r1.json()
    assert d1["auto_tracked"] is True
    assert biref_calls["n"] == 1
    assert 4 not in sess.confirmed_masks

    # Slider change on unconfirmed frame 4 hits cache & visibly updates right_b64
    r2 = client.post(
        f"/matting/api/session/{sid}/preview",
        json={
            "frame_idx": 4,
            "matting_engine": "matanyone2",
            "ambient_color_strength": 1.0,
            "bg_defocus": 10.0,
            "erode_dilate": 3.0,
        },
    )
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    assert d2["auto_tracked"] is True
    assert biref_calls["n"] == 1

    def _decode_b64_img(uri: str) -> np.ndarray:
        raw = base64.b64decode(uri.split(",", 1)[1])
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    img1 = _decode_b64_img(d1["right_b64"])
    img2 = _decode_b64_img(d2["right_b64"])
    assert float(np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))) > 1.0

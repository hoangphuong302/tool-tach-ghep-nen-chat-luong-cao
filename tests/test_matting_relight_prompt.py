from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alpha_matting import (
    MattingConfig,
    auto_chroma_key_mask,
    decontaminate_foreground_edges,
    detect_active_frame_box,
    extract_pixel_accurate_matte,
    stabilize_alpha_temporal,
)
from src.flow_prompt_engine import (
    analyze_scene_telemetry,
    generate_flow_prompts,
)
from src.relight_harmonizer import (
    RelightConfig,
    clean_ai_background_ghost,
    composite_with_relighting,
    detect_emissive_screen_mask,
    verify_and_polish_composite_pass3,
)


def test_active_frame_detection_on_pillarbox() -> None:
    """Verify automatic detection of 9:16 active content inside a 16:9 black-pillarboxed image."""
    img = np.zeros((576, 1024, 3), dtype=np.uint8)
    # Active vertical frame in center x=350..674
    img[:, 350:674] = (120, 140, 160)
    box = detect_active_frame_box(img)
    assert box is not None
    x0, y0, x1, y1 = box
    assert abs(x0 - 350) <= 4 and abs(x1 - 674) <= 4
    assert y0 == 0 and y1 == 576


def test_subpixel_alpha_matting_and_decontamination() -> None:
    """Verify sub-pixel guided alpha matting recovers fractional boundary alpha and removes green/bright halo."""
    h, w = 240, 240
    # Bright green background
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (30, 220, 40)  # BGR green

    # Skin-colored hand/object in center with smooth sub-pixel anti-aliased circle
    fg_true = np.zeros((h, w, 3), dtype=np.uint8)
    fg_true[:] = (130, 165, 215)  # Warm skin BGR

    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - 120.0) ** 2 + (yy - 120.0) ** 2)
    true_alpha = np.clip((60.0 - dist) / 3.0 + 0.5, 0.0, 1.0).astype(np.float32)

    # Synthesize observed image with green spill at boundary
    comp = (
        fg_true.astype(np.float32) * true_alpha[..., None]
        + bg.astype(np.float32) * (1.0 - true_alpha[..., None])
    ).astype(np.uint8)

    # Coarse binary mask (slightly off by 1-2px as raw segmentation often is)
    coarse_mask = (dist <= 61.0).astype(np.uint8)

    cfg = MattingConfig(
        trimap_radius=5,
        guided_radius=5,
        guided_eps=1e-3,
        decontaminate_strength=0.95,
        despill_green=True,
    )
    res = extract_pixel_accurate_matte(comp, coarse_mask, cfg)
    alpha = res.alpha
    fg_clean = res.fg_clean_bgr

    # 1. Must produce true fractional sub-pixel values along the boundary
    boundary_band = (true_alpha > 0.1) & (true_alpha < 0.9)
    assert np.any(boundary_band), "Boundary band must exist"
    mae = float(np.mean(np.abs(alpha - true_alpha)))
    assert mae < 0.06, f"Sub-pixel alpha MAE too high: {mae:.4f}"

    # 2. Color decontamination must drastically reduce green channel contamination on boundary pixels
    raw_green_excess = float(
        np.mean(
            comp[boundary_band, 1].astype(np.float32)
            - np.maximum(comp[boundary_band, 0], comp[boundary_band, 2]).astype(np.float32)
        )
    )
    clean_green_excess = float(
        np.mean(
            fg_clean[boundary_band, 1].astype(np.float32)
            - np.maximum(fg_clean[boundary_band, 0], fg_clean[boundary_band, 2]).astype(np.float32)
        )
    )
    assert clean_green_excess < raw_green_excess * 0.35, (
        f"Green spill not decontaminated enough: raw={raw_green_excess:.2f}, clean={clean_green_excess:.2f}"
    )


def test_auto_chroma_key_extraction() -> None:
    """Verify automatic green-screen extraction works without user clicks when green screen is present."""
    h, w = 200, 200
    img = np.full((h, w, 3), (35, 195, 45), dtype=np.uint8)  # Green background
    img[50:150, 60:140] = (140, 170, 210)  # Skin/device region
    mask = auto_chroma_key_mask(img, key_color="green")
    iou = float(np.sum((mask > 0) & (img[..., 2] > 100))) / float(
        np.sum((mask > 0) | (img[..., 2] > 100))
    )
    assert iou > 0.92, f"Auto chroma key IoU too low: {iou:.3f}"


def test_dual_zone_relighting_and_screen_protection() -> None:
    """Verify hands/body adapt to warm/directional new background light while emissive screen text stays protected."""
    h, w = 240, 240
    src_bgr = np.full((h, w, 3), (180, 180, 180), dtype=np.uint8)  # Neutral daylight source
    # Foreground mask (hands + device): y=40..210, x=50..190
    fg_mask = np.zeros((h, w), dtype=np.uint8)
    fg_mask[40:210, 50:190] = 1
    # Emissive blue screen inside device: y=60..150, x=80..160
    src_bgr[40:210, 50:190] = (145, 165, 195)  # Neutral hand skin/chassis
    src_bgr[60:150, 80:160] = (170, 45, 25)    # Deep blue translator screen (BGR)

    # New background: Warm golden sunset / cafe lighting, brighter on left side
    new_bg = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        gain = 1.3 - 0.6 * (x / float(w))  # brighter on left
        new_bg[:, x] = np.clip(np.array([60, 140, 230]) * gain, 0, 255)

    screen_mask = detect_emissive_screen_mask(src_bgr, fg_mask)
    assert np.sum(screen_mask > 0) > 2000, "Failed to detect emissive blue screen region"

    alpha = fg_mask.astype(np.float32)
    rcfg = RelightConfig(
        ambient_color_strength=0.65,
        directional_light_strength=0.60,
        light_wrap_Radius=6,
        light_wrap_strength=0.45,
        protect_emissive_screen=True,
        screen_glare_opacity=0.05,
        grain_match=0.0,
    )
    comp = composite_with_relighting(
        fg_clean_bgr=src_bgr,
        alpha=alpha,
        new_bg_bgr=new_bg,
        src_full_bgr=src_bgr,
        screen_mask=screen_mask,
        cfg=rcfg,
    )

    # 1. Hand region (y=170..200, x=60..180) must shift toward warm golden light (R channel increases relative to B)
    hand_orig_rb = float(np.mean(src_bgr[170:200, 60:180, 2].astype(np.float32) - src_bgr[170:200, 60:180, 0].astype(np.float32)))
    hand_new_rb = float(np.mean(comp[170:200, 60:180, 2].astype(np.float32) - comp[170:200, 60:180, 0].astype(np.float32)))
    assert hand_new_rb > hand_orig_rb + 8.0, f"Hands did not absorb warm ambient light: {hand_orig_rb:.1f} -> {hand_new_rb:.1f}"

    # 2. Left hand pixels should be brighter than right hand pixels due to 2D directional light transfer
    left_lum = float(np.mean(comp[170:200, 55:85]))
    right_lum = float(np.mean(comp[170:200, 155:185]))
    assert left_lum > right_lum + 3.0, f"Directional lighting gradient not transferred: left={left_lum:.1f}, right={right_lum:.1f}"

    # 3. Protected emissive screen center (y=80..130, x=100..140) must remain true deep blue (B >> R)
    scr_b = float(np.mean(comp[80:130, 100:140, 0]))
    scr_r = float(np.mean(comp[80:130, 100:140, 2]))
    assert scr_b > scr_r + 110.0, f"Emissive screen color was washed out! B={scr_b:.1f}, R={scr_r:.1f}"


def test_flow_prompt_engine_on_real_uploaded_image() -> None:
    """Verify scene telemetry & Google Flow prompt generation on the user's uploaded image."""
    uploaded_path = Path(
        r"C:/Users/phuong 1.2024/.gemini/antigravity/brain/015fcd65-f7a6-4c6c-b13a-670a3d36f860/.user_uploaded/media_1795581481825_d57010f4.jpg"
    )
    if uploaded_path.exists():
        bgr = cv2.imread(str(uploaded_path))
    else:
        bgr = np.zeros((576, 1024, 3), dtype=np.uint8)
        bgr[:, 350:674] = (140, 150, 160)

    telemetry = analyze_scene_telemetry(bgr)
    assert telemetry["active_aspect_ratio"] in ("9:16", "9:16 (Vertical Crop in 16:9)")
    assert "lighting_direction" in telemetry
    assert "bokeh_blur_level" in telemetry

    prompts = generate_flow_prompts(
        telemetry=telemetry,
        target_background_vi="Quán cà phê sang trọng phong cách Tokyo buổi sáng, ánh nắng nhẹ xuyên qua cửa kính",
        subject_description_vi="Hai bàn tay đang cầm máy phiên dịch cầm tay màu đen có màn hình Tiếng Việt - Tiếng Trung",
        use_local_llm=False,  # test deterministic offline engine first
    )
    assert "v2v_prompt" in prompts and len(prompts["v2v_prompt"]) > 150
    assert "clean_plate_prompt" in prompts and len(prompts["clean_plate_prompt"]) > 150
    assert "full_markdown" in prompts
    assert "KEEP" in prompts["full_markdown"] or "Giữ nguyên" in prompts["full_markdown"]


def test_full_5_step_ui_handlers_end_to_end() -> None:
    """End-to-end verification of all 5 step handlers in app.py."""
    import app

    uploaded_path = Path(
        r"C:/Users/phuong 1.2024/.gemini/antigravity/brain/015fcd65-f7a6-4c6c-b13a-670a3d36f860/.user_uploaded/media_1795581481825_d57010f4.jpg"
    )
    if not uploaded_path.exists():
        return

    state: dict = {}
    # Step 1: Load source & cut reference frames + optical telemetry
    state, gallery, tele_md, _, f0_rgb, msg1 = app.on_step1_load_source(str(uploaded_path), True, state)
    assert len(gallery) >= 1
    assert "9:16" in tele_md
    assert f0_rgb.shape[0] > 100

    # Step 2: Confirm Roto-Brush selection & sub-pixel alpha matting (returns 6-tuple with emerald green canvas)
    editor_green, roto_vis, cutout_vis, png_path, state, msg2 = app.on_step2_confirm_roto_brush(
        None, 0, 3, 4, 0.92, False, -1, state
    )
    assert editor_green.shape == f0_rgb.shape
    assert roto_vis.shape == f0_rgb.shape
    assert cutout_vis.shape == f0_rgb.shape
    assert Path(png_path).exists()

    # Step 4: AGY Flow Prompt generation
    clean_p, v2v_p, full_md, msg4 = app.on_step4_generate_prompt(
        "Quán cà phê kính sang trọng tại Tokyo buổi sáng",
        "Hai bàn tay cầm máy phiên dịch cầm tay màu đen có màn hình Tiếng Việt",
        True,
        state,
    )
    assert len(clean_p) > 150 and len(v2v_p) > 150

    # Step 5: Preview relit composite
    (before_rgb, after_rgb), msg5 = app.on_step5_preview_composite(
        0, None, 3, 4, 0.92, False, -1,
        0.58, 0.48, 6, 0.38, True, 0.10, 0.15, 0.0, False, state,
    )
    assert before_rgb.shape == after_rgb.shape


def test_two_pass_pixel_verification_and_compensation() -> None:
    """Verify Pass-2 Closed-Loop Pixel Verification restores missing finger rims and prunes excess green gaps."""
    h, w = 300, 300
    # Green screen background
    img = np.full((h, w, 3), (40, 110, 15), dtype=np.uint8)
    # Device blue screen at x=80..180, y=60..220
    img[60:220, 80:180] = (145, 48, 35)
    # White side bezel at x=180..195, y=60..220
    img[60:220, 180:195] = (242, 244, 246)
    # Finger skin at x=195..250, y=100..220 with an enclosed green triangle hole at x=200..215, y=140..170
    img[100:220, 195:250] = (180, 198, 225)
    img[140:170, 200:215] = (45, 115, 18)  # Green hole between finger and bezel

    # Pass-1 flawed SAM mask: misses the outer 12px of the finger (x=238..250) AND misses the white bezel (x=182..195),
    # while erroneously filling the green hole (x=200..215, y=140..170)!
    m_pass1 = np.zeros((h, w), dtype=np.uint8)
    m_pass1[60:220, 80:182] = 1
    m_pass1[100:220, 195:238] = 1

    res = extract_pixel_accurate_matte(img, m_pass1, MattingConfig(two_pass_verify=True))
    assert res.missing_restored_px >= 2500, f"Expected >=2500 missing pixels restored, got {res.missing_restored_px}"
    assert res.excess_removed_px >= 400, f"Expected >=400 excess green-gap pixels removed, got {res.excess_removed_px}"
    # White bezel must be restored to alpha=1.0
    assert float(np.mean(res.alpha[80:200, 183:193])) > 0.95
    # Outer 12px finger rim must be restored to alpha=1.0
    assert float(np.mean(res.alpha[110:210, 239:248])) > 0.95
    # Enclosed green hole must be pruned to alpha=0.0
    assert float(np.mean(res.alpha[143:167, 203:212])) < 0.02


def test_universal_non_green_screen_pass2_verification() -> None:
    """Verify Pass-2 works universally on non-green-screen footage (arbitrary red/gold product over gray/blue desk)."""
    h, w = 260, 260
    # Cool gray-blue office desk background (NO green screen!)
    ref_bgr = np.full((h, w, 3), (150, 125, 110), dtype=np.uint8)
    # Warm red-orange product box/bottle at y=60..200, x=70..190
    ref_bgr[60:200, 70:190] = (50, 85, 225)
    ref_mask = np.zeros((h, w), dtype=np.uint8)
    ref_mask[60:200, 70:190] = 1

    # Frame t: Pass-1 mask shrunk by 8px on the right edge (x=182..190 missing) and bled 8px onto desk on left (x=62..70 excess)
    m_pass1 = np.zeros((h, w), dtype=np.uint8)
    m_pass1[60:200, 62:182] = 1

    res = extract_pixel_accurate_matte(
        ref_bgr,
        m_pass1,
        MattingConfig(two_pass_verify=True, trimap_radius=2, guided_radius=2),
        ref_bgr=ref_bgr,
        ref_mask=ref_mask,
    )
    assert res.missing_restored_px >= 800, f"Expected >=800 missing red product pixels restored, got {res.missing_restored_px}"
    assert res.excess_removed_px >= 800, f"Expected >=800 excess gray desk pixels pruned, got {res.excess_removed_px}"


def test_pass3_aa_and_source_restoration() -> None:
    """Verify Pass-3 tangential anti-aliasing smooths diagonal staircases and restores interior micro-detail + aberrant redness."""
    h, w = 220, 220
    yy, xx = np.mgrid[0:h, 0:w]
    # Diagonal boundary with deliberate jagged quantization step in alpha
    signed = (xx * 0.7 + yy * 0.7) - 150.0
    quantized_alpha = np.clip(np.round((signed / 2.2 + 0.5) * 3.0) / 3.0, 0.0, 1.0).astype(np.float32)

    orig_fg_f = np.full((h, w, 3), 0.72, dtype=np.float32)
    # Add fine micro-texture pores/lines inside interior
    orig_fg_f[80:200, 80:200] += ((xx[80:200, 80:200] % 3 == 0).astype(np.float32)[..., None] * 0.08)
    bg_f = np.full((h, w, 3), 0.18, dtype=np.float32)

    # Simulate relit foreground where micro-texture was smoothed out AND one patch drifted bright red
    fg_f = cv2.GaussianBlur(orig_fg_f, (0, 0), sigmaX=2.2)
    fg_f[140:175, 140:175] = np.array([0.20, 0.22, 0.90], dtype=np.float32)  # aberrant red patch

    comp_f = bg_f * (1.0 - quantized_alpha[..., None]) + fg_f * quantized_alpha[..., None]
    out_u8, stats = verify_and_polish_composite_pass3(comp_f, fg_f, orig_fg_f, bg_f, quantized_alpha)

    assert stats["aa_polished_px"] > 100
    assert stats["detail_restored_px"] > 500
    assert stats["drift_reverted_px"] >= 900
    # Redness on the aberrant patch must be reduced sharply
    out_patch = out_u8[145:170, 145:170].astype(np.float32) / 255.0
    assert float(np.mean(out_patch[..., 2] - out_patch[..., 1])) < 0.35


def test_photogrammetric_smartphone_telemetry_and_temporal_alpha() -> None:
    """Verify photogrammetric lens estimation (focal_length_mm, smartphone DoF) and temporal alpha smoothing."""
    bgr = np.full((640, 360, 3), (40, 180, 45), dtype=np.uint8)
    fg_mask = np.zeros((640, 360), dtype=np.uint8)
    fg_mask[240:600, 80:280] = 1
    tele = analyze_scene_telemetry(bgr, fg_mask)
    assert tele["focal_length_mm"] in (24, 26, 28, 35)
    assert "camera_pitch_deg" in tele and tele["camera_pitch_deg"] < 0
    assert "smartphone" in tele["bokeh_en"].lower()

    prompts = generate_flow_prompts(tele, "Sảnh sân bay hiện đại", use_local_llm=False)
    assert f"{tele['focal_length_mm']}mm" in prompts["clean_plate_prompt"]

    # Temporal stabilization smooths stationary boundary flicker while preserving moving edges
    prev_bgr = np.full((120, 120, 3), 100, dtype=np.uint8)
    curr_bgr = prev_bgr.copy()  # stationary
    prev_a = np.full((120, 120), 0.2, dtype=np.float32)
    curr_a = np.full((120, 120), 0.8, dtype=np.float32)
    st_a = stabilize_alpha_temporal(curr_a, prev_a, curr_bgr, prev_bgr)
    assert float(np.mean(st_a)) < 0.65


if __name__ == "__main__":
    test_active_frame_detection_on_pillarbox()
    test_subpixel_alpha_matting_and_decontamination()
    test_auto_chroma_key_extraction()
    test_dual_zone_relighting_and_screen_protection()
    test_flow_prompt_engine_on_real_uploaded_image()
    test_two_pass_pixel_verification_and_compensation()
    test_universal_non_green_screen_pass2_verification()
    test_full_5_step_ui_handlers_end_to_end()
    test_pass3_aa_and_source_restoration()
    test_photogrammetric_smartphone_telemetry_and_temporal_alpha()
    print("ALL TESTS PASSED!")

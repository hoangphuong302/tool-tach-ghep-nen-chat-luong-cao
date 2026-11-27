import cv2
import numpy as np

from src.alpha_matting import MattingConfig, extract_pixel_accurate_matte, stabilize_alpha_temporal, verify_and_compensate_pass2


def test_opaque_dark_attachment_keeps_source_color_at_edge() -> None:
    source = np.full((480, 480, 3), (210, 200, 185), dtype=np.uint8)
    source[80:400, 100:320] = (170, 175, 180)
    source[140:300, 317:324] = (25, 30, 35)
    mask = np.zeros((480, 480), dtype=np.uint8)
    mask[80:400, 100:324] = 1

    result = extract_pixel_accurate_matte(
        source, mask, MattingConfig(two_pass_verify=False, decontaminate_strength=0.95)
    )

    assert result.alpha[200, 320] > 0.99
    np.testing.assert_array_equal(result.fg_clean_bgr[200, 320], source[200, 320])
    assert result.alpha[200, 325] == 0
from src.segmentation import temporal_smooth_mask
from src.video_io import save_mask


def test_low_contrast_neutral_boundary_can_correct_biased_coarse_mask():
    bgr = np.full((96, 112, 3), 220, dtype=np.uint8)
    bgr[20:76, 25:82] = 205
    coarse = np.zeros(bgr.shape[:2], dtype=np.uint8)
    coarse[20:76, 27:84] = 1

    result = extract_pixel_accurate_matte(
        bgr, coarse, MattingConfig(trimap_radius=4, two_pass_verify=False),
    )

    assert result.alpha[28:68, 25:27].mean() > 0.9
    assert result.alpha[28:68, 82:84].mean() < 0.10


def test_white_spill_is_reduced_in_black_composite_and_opaque_highlight_survives():
    bgr = np.full((144, 160, 3), 250, dtype=np.uint8)
    skin = np.array([126, 165, 211], dtype=np.float32)
    bgr[25:120, 27:130] = skin
    bgr[32:46, 70:84] = (244, 246, 248)
    bgr[50:110, 129] = 0.65 * skin + 0.35 * 250.0
    bgr[50:110, 130] = 0.30 * skin + 0.70 * 250.0
    coarse = np.zeros(bgr.shape[:2], dtype=np.uint8)
    coarse[25:120, 27:130] = 1

    result = extract_pixel_accurate_matte(bgr, coarse, MattingConfig(two_pass_verify=False))
    edge_slice = np.s_[55:105, 129:131]
    alpha = result.alpha[edge_slice][..., None]
    expected_black = np.array([0.65, 0.30])[None, :, None] * skin
    contaminated_black = alpha * bgr[edge_slice]
    cleaned_black = alpha * result.fg_clean_bgr[edge_slice]

    assert np.abs(cleaned_black - expected_black).mean() < 5.0
    assert np.abs(cleaned_black - expected_black).mean() < 0.55 * np.abs(contaminated_black - expected_black).mean()
    assert np.array_equal(result.fg_clean_bgr[32:46, 70:84], bgr[32:46, 70:84])


def test_temporal_alpha_rejects_equal_luma_color_motion_and_keeps_current_support():
    previous_bgr = np.full((24, 36, 3), (125, 100, 91), dtype=np.uint8)
    current_bgr = previous_bgr.copy()
    current_bgr[:, 18:] = (100, 100, 100)
    previous_alpha = np.full((24, 36), 0.65, dtype=np.float32)
    current_alpha = np.full((24, 36), 0.35, dtype=np.float32)
    current_alpha[:, 24:] = 0.0

    assert abs(float(cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)[0, 0]) - 100.0) <= 1.0
    result = stabilize_alpha_temporal(current_alpha, previous_alpha, current_bgr, previous_bgr)

    assert result[:, :12].mean() > 0.45
    assert np.array_equal(result[:, 18:], current_alpha[:, 18:])


def test_temporal_mask_votes_only_on_still_nearby_edges(tmp_path):
    current_mask = np.zeros((28, 36), dtype=np.uint8)
    current_mask[8:20, 5:25] = 1
    neighbor_mask = current_mask.copy()
    neighbor_mask[8:20, 5] = 0
    neighbor_mask[8:20, 25] = 1
    current = np.full((28, 36, 3), (100, 100, 100), dtype=np.uint8)
    neighbor = current.copy()
    neighbor[8:20, 25] = (116, 100, 94)
    paths = [tmp_path / f"mask-{i}.png" for i in range(3)]
    frames = [tmp_path / f"frame-{i}.png" for i in range(3)]
    for i, (mask, frame) in enumerate(zip((neighbor_mask, current_mask, neighbor_mask), (neighbor, current, neighbor))):
        save_mask(mask, paths[i])
        assert cv2.imwrite(str(frames[i]), frame)

    result = temporal_smooth_mask(paths, 1, frame_paths=frames, current_bgr=current)

    assert np.all(result[10:18, 5] == 0)
    assert np.all(result[10:18, 25] == 0)


def test_green_pass2_does_not_restore_local_wood_but_keeps_true_finger_and_bezel():
    bgr = np.empty((220, 220, 3), dtype=np.uint8)
    bgr[:] = (40, 180, 40)
    bgr[136:, :] = (107, 160, 197)
    bgr[40:135, 60:160] = (126, 165, 211)
    bgr[75:101, 154:159] = (235, 235, 235)
    coarse = np.zeros(bgr.shape[:2], dtype=np.uint8)
    coarse[40:130, 60:160] = 1

    result, _, has_green, _, _, _ = verify_and_compensate_pass2(bgr, coarse)

    assert has_green
    assert result[140, 110] == 0
    assert result[132, 110] == 1
    assert result[90, 156] == 1



def test_rgb_mixture_rejects_extrapolation_over_a_thin_dark_foreground_edge():
    background = (108, 131, 165)
    source = np.full((480, 480, 3), background, dtype=np.uint8)
    source[60:420, 80:342] = (191, 194, 199)
    source[140:340, 342:344] = (10, 15, 23)
    tracked_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    tracked_mask[60:420, 80:344] = 1

    result = extract_pixel_accurate_matte(source, tracked_mask, MattingConfig(two_pass_verify=False))

    assert np.all(result.alpha[180:280, 342] > 0.65)
    assert np.all(result.alpha[180:280, 343] > 0.45)
    assert np.max(result.fg_clean_bgr[180:280, 342]) < 55
    assert np.max(result.alpha[80:125, 342:344]) < 0.05
    assert np.max(result.alpha[180:280, 344:]) < 0.05

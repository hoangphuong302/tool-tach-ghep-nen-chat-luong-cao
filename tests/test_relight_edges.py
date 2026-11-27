import numpy as np

from src.relight_harmonizer import (
    RelightConfig,
    _apply_2d_directional_lighting,
    _apply_physical_light_wrap,
    composite_with_relighting,
    verify_and_polish_composite_pass3,
)


def test_soft_diagonal_aa_ignores_hidden_foreground_rgb() -> None:
    yy, xx = np.mgrid[:44, :44]
    distance = xx + yy - 41
    alpha = np.clip(distance * 0.48 + 0.50, 0, 1).astype(np.float32)
    plate = np.zeros((*alpha.shape, 3), dtype=np.float32)
    foreground = np.full_like(plate, (0.35, 0.52, 0.72))
    background_rgb_leak = foreground.copy()
    background_rgb_leak[alpha == 0] = 1.0
    comp = foreground * alpha[..., None]

    clean, stats = verify_and_polish_composite_pass3(comp, foreground, foreground, plate, alpha)
    contaminated, _ = verify_and_polish_composite_pass3(
        comp, background_rgb_leak, foreground, plate, alpha
    )

    assert stats["aa_polished_px"] > 0
    np.testing.assert_array_equal(contaminated, clean)
    np.testing.assert_array_equal(clean[alpha == 0], 0)
    np.testing.assert_array_equal(
        clean[alpha == 1], (foreground[alpha == 1] * 255).astype(np.uint8)
    )


def test_binary_diagonal_respects_opaque_subject_and_background() -> None:
    yy, xx = np.mgrid[:24, :24]
    alpha = (xx + yy > 24).astype(np.float32)
    foreground = np.full((*alpha.shape, 3), (32, 64, 112), dtype=np.uint8)
    plate = np.full_like(foreground, 255)
    cfg = RelightConfig(
        ambient_color_strength=0,
        directional_light_strength=0,
        light_wrap_strength=0,
        protect_emissive_screen=False,
        grain_match=0,
    )

    out = composite_with_relighting(foreground, alpha, plate, cfg=cfg)

    np.testing.assert_array_equal(out[alpha == 1], foreground[alpha == 1])
    np.testing.assert_array_equal(out[alpha == 0], plate[alpha == 0])


def test_bright_light_wrap_respects_radius_skin_color_and_highlights() -> None:
    alpha = np.zeros((40, 40), dtype=np.float32)
    alpha[:, 10:] = 1.0
    foreground = np.full((*alpha.shape, 3), (0.40, 0.55, 0.74), dtype=np.float32)
    foreground[12:16] = 1.0  # A white bezel touching the silhouette.
    plate = np.ones_like(foreground)

    narrow = _apply_physical_light_wrap(foreground, plate, alpha, radius=2, strength=1.0)
    wide = _apply_physical_light_wrap(foreground, plate, alpha, radius=8, strength=1.0)

    np.testing.assert_array_equal(narrow[20, 14], foreground[20, 14])
    assert np.all(wide[20, 14] > foreground[20, 14])
    assert np.max(wide - foreground) < 0.05
    np.testing.assert_allclose(
        wide[20, 10] / foreground[20, 10],
        np.full(3, wide[20, 10, 2] / foreground[20, 10, 2]),
        atol=1e-6,
    )
    np.testing.assert_array_equal(wide[12:16], foreground[12:16])
    np.testing.assert_array_equal(wide[:, 19:], foreground[:, 19:])


def test_directional_light_uses_broad_exposure_not_narrow_white_object() -> None:
    h, w = 96, 128
    fg = np.full((h, w, 3), 0.35, dtype=np.float32)
    bg = np.full_like(fg, 0.20)
    bg[:, 62:65] = 1.0  # A small bright object should not become a key light.

    out = _apply_2d_directional_lighting(fg, bg, None, strength=1.0)

    assert float(np.max(out - fg)) < 0.01
    assert float(np.max(np.ptp(out[40:56, 48:80, 0], axis=1))) < 0.01

    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[:, 20:58] = 1.0
    wrapped = _apply_physical_light_wrap(fg, bg, alpha, radius=8, strength=1.0)
    assert float(np.max(wrapped[:, 49:58] - fg[:, 49:58])) < 0.01


def test_directional_light_preserves_broad_left_to_right_gradient() -> None:
    h, w = 96, 128
    fg = np.full((h, w, 3), 0.35, dtype=np.float32)
    gradient = np.linspace(0.16, 0.48, w, dtype=np.float32)[None, :, None]
    bg = np.broadcast_to(gradient, fg.shape).copy()

    out = _apply_2d_directional_lighting(fg, bg, None, strength=1.0)

    assert float(np.mean(out[:, -12:, 0])) > float(np.mean(out[:, :12, 0])) + 0.06


def test_ambient_match_keeps_warm_tone_and_specular_detail() -> None:
    from src.relight_harmonizer import _transfer_ambient_lab

    fg = np.full((48, 48, 3), (112, 150, 190), dtype=np.uint8)
    fg[20:24, 20:24] = 245  # Specular metal highlight.
    plate = np.full_like(fg, (32, 110, 180))
    mask = np.ones((48, 48), dtype=np.uint8)
    mask[8:40, 8:40] = 0

    out = _transfer_ambient_lab(fg, plate, None, mask, strength=0.5)

    assert int(out[22, 22, 2]) >= 240
    # Warm plate should move the skin color slightly toward warm while retaining it.
    assert out[12, 12, 2] > fg[12, 12, 2]
    assert out[12, 12, 0] > 80


def test_drift_copyback_transitions_smoothly_and_limits_large_drift() -> None:
    alpha = np.ones((12, 401), dtype=np.float32)
    original = np.full((12, 401, 3), 0.3, dtype=np.float32)
    plate = np.zeros_like(original)
    for channel, end in ((0, 0.4), (2, 0.15)):
        foreground = original.copy()
        foreground[..., channel] += np.linspace(0, end, 401, dtype=np.float32)
        output, stats = verify_and_polish_composite_pass3(
            foreground, foreground, original, plate, alpha
        )
        values = output[6, :, channel].astype(np.int16)
        assert np.max(np.abs(np.diff(values))) <= 2
        assert stats["drift_reverted_px"] > 0
        # Well above either threshold the original 65% correction still applies.
        assert abs(int(values[-1]) - int((0.3 + end * 0.35) * 255)) <= 1

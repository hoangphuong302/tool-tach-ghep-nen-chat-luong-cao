import numpy as np

from src.alpha_matting import _pass2_recovered_rim_confidence


def test_pass2_rim_confidence_preserves_supported_near_opaque_fg_only():
    img = np.array(
        [[
            [0.25, 0.30, 0.35],  # Pass-2 foreground recovery
            [0.80, 0.80, 0.80],  # Observation agrees with the local background
            [0.25, 0.30, 0.35],  # Supported color, but alpha is below the gate
            [0.25, 0.30, 0.35],  # Already in the coarse mask
        ]],
        dtype=np.float32,
    )
    fg_est = np.array(
        [[[0.26, 0.29, 0.35], [0.30, 0.30, 0.30], [0.26, 0.29, 0.35], [0.26, 0.29, 0.35]]],
        dtype=np.float32,
    )
    bg_est = np.array(
        [[[0.80, 0.80, 0.80], [0.81, 0.80, 0.80], [0.80, 0.80, 0.80], [0.80, 0.80, 0.80]]],
        dtype=np.float32,
    )
    alpha = np.array([[0.95, 0.98, 0.89, 0.99]], dtype=np.float32)
    alpha_before = alpha.copy()
    mask_pass1 = np.array([[0, 0, 0, 1]], dtype=np.uint8)
    mask_pass2 = np.ones((1, 4), dtype=np.uint8)

    ys, xs, confidence = _pass2_recovered_rim_confidence(
        img, fg_est, bg_est, alpha, mask_pass1, mask_pass2
    )

    np.testing.assert_array_equal(np.column_stack((ys, xs)), [[0, 0]])
    np.testing.assert_allclose(confidence, [0.5], atol=1e-6)
    np.testing.assert_array_equal(alpha, alpha_before)

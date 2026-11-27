import cv2
import numpy as np

from src.relight_harmonizer import verify_and_polish_composite_pass3


def _dense_pass3_reference(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    height, width = alpha.shape
    edge_core = (alpha > 0.005) & (alpha < 0.995)
    if not np.any(edge_core):
        composite = bg * (1.0 - alpha[..., None]) + fg * alpha[..., None]
        return np.clip(composite * 255.0, 0, 255).astype(np.uint8)

    gx = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
    gnorm = np.sqrt(gx * gx + gy * gy + 1e-6)
    tx = gy / gnorm
    ty = -gx / gnorm
    lap_alpha = np.abs(cv2.Laplacian(alpha, cv2.CV_32F, ksize=3))
    jagged_w = np.where(edge_core, np.clip(lap_alpha * 3.2, 0.25, 1.0), 0.0)[..., None]

    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    premultiplied = np.dstack((fg * alpha[..., None], alpha))
    accumulated = np.zeros_like(premultiplied)
    taps = ((-2, 0.12), (-1, 0.23), (0, 0.30), (1, 0.23), (2, 0.12))
    for step, weight in taps:
        if step == 0:
            accumulated += premultiplied * weight
        else:
            sample_x = np.clip(grid_x + tx * (step * 0.75), 0.0, float(width - 1))
            sample_y = np.clip(grid_y + ty * (step * 0.75), 0.0, float(height - 1))
            accumulated += cv2.remap(
                premultiplied,
                sample_x,
                sample_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ) * weight

    filtered = premultiplied * (1.0 - jagged_w) + accumulated * jagged_w
    composite = bg * (1.0 - filtered[..., 3:4]) + filtered[..., :3]
    return np.clip(composite * 255.0, 0, 255).astype(np.uint8)


def _make_layers(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = alpha.shape
    yy, xx = np.mgrid[:height, :width]
    foreground = np.stack(
        (
            0.35 + 0.1 * xx / max(width - 1, 1),
            0.48 + 0.08 * yy / max(height - 1, 1),
            np.full((height, width), 0.72, dtype=np.float32),
        ),
        axis=-1,
    ).astype(np.float32)
    background = np.full_like(foreground, (0.12, 0.16, 0.20))
    composite = background * (1.0 - alpha[..., None]) + foreground * alpha[..., None]
    return composite, foreground, foreground.copy(), background


def test_sparse_pass3_matches_dense_reference_for_chunked_and_dense_edges() -> None:
    height, width = 300, 600
    sparse_alpha = np.zeros((height, width), dtype=np.float32)
    rising = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    falling = np.linspace(1.0, 0.0, 20, dtype=np.float32)
    for left in (70, 250, 430):
        sparse_alpha[:, left:left + 20] = rising
        sparse_alpha[:, left + 20:left + 40] = falling

    dense_alpha = np.broadcast_to(
        np.linspace(0.0, 1.0, width, dtype=np.float32), (height, width)
    ).copy()
    sparse_count = int(np.sum((sparse_alpha > 0.005) & (sparse_alpha < 0.995)))
    assert sparse_count > 32_000
    assert sparse_count < height * width * 0.40

    for alpha in (sparse_alpha, dense_alpha):
        comp, fg, original_fg, bg = _make_layers(alpha)
        actual, _ = verify_and_polish_composite_pass3(comp, fg, original_fg, bg, alpha)
        expected = _dense_pass3_reference(fg, bg, alpha)
        np.testing.assert_array_equal(actual, expected)


def test_sparse_pass3_keeps_hard_binary_edges_unchanged() -> None:
    alpha = np.zeros((64, 96), dtype=np.float32)
    alpha[12:52, 16:78] = 1.0
    comp, fg, original_fg, bg = _make_layers(alpha)
    actual, stats = verify_and_polish_composite_pass3(comp, fg, original_fg, bg, alpha)
    expected = _dense_pass3_reference(fg, bg, alpha)
    np.testing.assert_array_equal(actual, expected)
    assert stats["aa_polished_px"] == 0

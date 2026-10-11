"""Unit tests for Laplacian spatial filtering."""

import numpy as np
import pytest

from illumination_lab.laplacian import apply_laplacian, process_laplacian


@pytest.fixture
def step_edge_image() -> np.ndarray:
    """Fixture providing a synthetic step-edge image (left half 40, right half 200)."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, :32] = 40
    img[:, 32:] = 200
    return img


def test_laplacian_preserves_signed_response(step_edge_image: np.ndarray) -> None:
    raw, scaled, abs_resp = apply_laplacian(step_edge_image, ksize=3)

    # raw should be float64
    assert raw.dtype == np.float64
    # The step edge generates both negative and positive second-derivative transitions
    assert np.min(raw) < 0
    assert np.max(raw) > 0

    # scaled should be 8-bit [0, 255]
    assert scaled.dtype == np.uint8
    assert scaled.shape == step_edge_image.shape

    # abs_resp should be 8-bit [0, 255] with edge response centered at column 31/32
    assert abs_resp.dtype == np.uint8
    assert np.max(abs_resp[:, 31:33]) > np.max(abs_resp[:, :20])


def test_process_laplacian_result(step_edge_image: np.ndarray) -> None:
    result = process_laplacian(step_edge_image, ksize=3)

    assert result.original_gray.shape == step_edge_image.shape
    assert result.ksize == 3
    assert result.laplacian_raw.shape == step_edge_image.shape
    assert result.laplacian_scaled.shape == step_edge_image.shape
    assert result.laplacian_abs.shape == step_edge_image.shape


def test_laplacian_invalid_parameters(step_edge_image: np.ndarray) -> None:
    # Even kernel size
    with pytest.raises(ValueError):
        apply_laplacian(step_edge_image, ksize=2)

    # Negative kernel size
    with pytest.raises(ValueError):
        apply_laplacian(step_edge_image, ksize=-1)

    # 3D image
    with pytest.raises(ValueError):
        apply_laplacian(np.zeros((64, 64, 3), dtype=np.uint8))

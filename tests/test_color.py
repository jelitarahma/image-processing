"""Unit tests for color space conversion functions."""

import numpy as np
import pytest

from illumination_lab.color import (
    bgr_to_lab,
    bgr_to_rgb,
    lab_to_bgr,
    lab_to_rgb,
    merge_lab_channels,
    rgb_to_bgr,
    rgb_to_lab,
    split_lab_channels,
)


@pytest.fixture
def sample_bgr() -> np.ndarray:
    """Fixture providing a deterministic 3-channel test image."""
    np.random.seed(42)
    return np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)


def test_bgr_rgb_roundtrip(sample_bgr: np.ndarray) -> None:
    rgb = bgr_to_rgb(sample_bgr)
    assert rgb.shape == sample_bgr.shape
    assert rgb.dtype == np.uint8
    # Channel inversion check: Red and Blue channels should be swapped
    np.testing.assert_array_equal(rgb[:, :, 0], sample_bgr[:, :, 2])
    np.testing.assert_array_equal(rgb[:, :, 1], sample_bgr[:, :, 1])
    np.testing.assert_array_equal(rgb[:, :, 2], sample_bgr[:, :, 0])

    # Invert back to BGR
    recovered_bgr = rgb_to_bgr(rgb)
    np.testing.assert_array_equal(recovered_bgr, sample_bgr)


def test_bgr_to_lab_shape_and_dtype(sample_bgr: np.ndarray) -> None:
    lab = bgr_to_lab(sample_bgr)
    assert lab.shape == sample_bgr.shape
    assert lab.dtype == np.uint8


def test_rgb_to_lab_and_inverse(sample_bgr: np.ndarray) -> None:
    rgb = bgr_to_rgb(sample_bgr)
    lab = rgb_to_lab(rgb)
    recovered_rgb = lab_to_rgb(lab)

    assert lab.shape == sample_bgr.shape
    assert recovered_rgb.shape == rgb.shape
    # Due to integer quantization in color space conversion, allow small tolerance
    assert np.mean(np.abs(rgb.astype(float) - recovered_rgb.astype(float))) < 3.0


def test_split_and_merge_lab(sample_bgr: np.ndarray) -> None:
    lab = bgr_to_lab(sample_bgr)
    l_chan, a_chan, b_chan = split_lab_channels(lab)

    assert l_chan.shape == (64, 64)
    assert a_chan.shape == (64, 64)
    assert b_chan.shape == (64, 64)

    recombined_lab = merge_lab_channels(l_chan, a_chan, b_chan)
    np.testing.assert_array_equal(recombined_lab, lab)


def test_invalid_input_types() -> None:
    with pytest.raises(TypeError):
        bgr_to_rgb([1, 2, 3])  # type: ignore

    with pytest.raises(ValueError):
        bgr_to_rgb(np.zeros((64, 64), dtype=np.uint8))  # 2D instead of 3D

    with pytest.raises(ValueError):
        bgr_to_rgb(np.zeros((64, 64, 3), dtype=np.float32))  # float instead of uint8

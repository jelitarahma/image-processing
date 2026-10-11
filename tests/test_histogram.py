"""Unit tests for histogram computation and L-channel equalization."""

import numpy as np
import pytest

from illumination_lab.histogram import (
    compute_cdf,
    compute_histogram,
    equalize_illumination,
    equalize_l_channel,
    equalize_lab_image,
)


@pytest.fixture
def low_contrast_l_channel() -> np.ndarray:
    """Fixture providing a low-contrast synthetic lightness channel."""
    np.random.seed(42)
    # Intensity concentrated narrowly between 80 and 110
    return np.random.randint(80, 111, size=(64, 64), dtype=np.uint8)


@pytest.fixture
def sample_bgr() -> np.ndarray:
    """Fixture providing a synthetic color image."""
    np.random.seed(42)
    return np.random.randint(30, 180, size=(64, 64, 3), dtype=np.uint8)


def test_compute_histogram(low_contrast_l_channel: np.ndarray) -> None:
    hist = compute_histogram(low_contrast_l_channel)
    assert len(hist) == 256
    assert hist.sum() == low_contrast_l_channel.size
    # Values outside [80, 110] should be 0
    assert hist[:80].sum() == 0
    assert hist[111:].sum() == 0


def test_compute_cdf(low_contrast_l_channel: np.ndarray) -> None:
    hist, cdf = compute_cdf(low_contrast_l_channel)
    assert len(cdf) == 256
    assert cdf[-1] == pytest.approx(1.0)
    # CDF must be monotonically non-decreasing
    assert np.all(np.diff(cdf) >= 0)


def test_equalize_l_channel_expands_dynamic_range(
    low_contrast_l_channel: np.ndarray,
) -> None:
    eq_l = equalize_l_channel(low_contrast_l_channel)
    assert eq_l.shape == low_contrast_l_channel.shape
    assert eq_l.dtype == np.uint8

    orig_range = int(low_contrast_l_channel.max()) - int(low_contrast_l_channel.min())
    eq_range = int(eq_l.max()) - int(eq_l.min())
    assert eq_range > orig_range
    # Dynamic range should be substantially widened
    assert eq_range > 200


def test_equalize_lab_preserves_chroma(sample_bgr: np.ndarray) -> None:
    from illumination_lab.color import bgr_to_lab, split_lab_channels

    lab = bgr_to_lab(sample_bgr)
    orig_l, orig_a, orig_b = split_lab_channels(lab)

    eq_lab, eq_l = equalize_lab_image(lab)
    res_l, res_a, res_b = split_lab_channels(eq_lab)

    # Chromatic channels A and B must remain completely identical
    np.testing.assert_array_equal(res_a, orig_a)
    np.testing.assert_array_equal(res_b, orig_b)

    # L channel should be equalized
    np.testing.assert_array_equal(res_l, eq_l)


def test_equalize_illumination_pipeline(sample_bgr: np.ndarray) -> None:
    result = equalize_illumination(sample_bgr)

    assert result.original_rgb.shape == sample_bgr.shape
    assert result.equalized_rgb.shape == sample_bgr.shape
    assert result.equalized_bgr.shape == sample_bgr.shape
    assert result.equalized_l.shape == (64, 64)

    metrics = result.metrics
    assert "l_mean_before" in metrics
    assert "l_mean_after" in metrics
    assert "l_std_before" in metrics
    assert "l_std_after" in metrics

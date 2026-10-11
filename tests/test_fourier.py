"""Unit tests for 2D Discrete Fourier Transform analysis."""

import numpy as np
import pytest

from illumination_lab.fourier import compute_dft, process_fourier


@pytest.fixture
def sample_gray() -> np.ndarray:
    """Fixture providing a synthetic grayscale image with distinct frequency content."""
    # Create an image with a horizontal sinusoidal grating
    x = np.linspace(0, 4 * np.pi, 64)
    y = np.linspace(0, 4 * np.pi, 64)
    xx, yy = np.meshgrid(x, y)
    pattern = 127 + 100 * np.sin(xx)
    return np.uint8(np.clip(pattern, 0, 255))


def test_compute_dft_shapes_and_types(sample_gray: np.ndarray) -> None:
    dft_shift, mag_spec = compute_dft(sample_gray)

    assert dft_shift.shape == (64, 64, 2)
    assert mag_spec.shape == (64, 64)
    assert mag_spec.dtype == np.float32 or mag_spec.dtype == np.float64
    assert np.all(mag_spec >= 0)


def test_dft_dc_component_at_center(sample_gray: np.ndarray) -> None:
    dft_shift, mag_spec = compute_dft(sample_gray)
    crow, ccol = sample_gray.shape[0] // 2, sample_gray.shape[1] // 2

    # DC component (mean brightness) is at the center after fftshift
    # The center magnitude should be among the highest values in the spectrum
    center_val = mag_spec[crow, ccol]
    assert center_val > np.percentile(mag_spec, 90)


def test_process_fourier_normalized_spectrum(sample_gray: np.ndarray) -> None:
    result = process_fourier(sample_gray)

    assert result.original_gray.shape == sample_gray.shape
    assert result.normalized_spectrum.shape == sample_gray.shape
    assert result.normalized_spectrum.dtype == np.uint8
    assert result.normalized_spectrum.min() >= 0
    assert result.normalized_spectrum.max() <= 255


def test_fourier_invalid_inputs() -> None:
    # 3D image
    with pytest.raises(ValueError):
        compute_dft(np.zeros((64, 64, 3), dtype=np.uint8))

    # Float array
    with pytest.raises(ValueError):
        compute_dft(np.zeros((64, 64), dtype=np.float32))

    # Non-numpy
    with pytest.raises(TypeError):
        compute_dft([[1, 2], [3, 4]])  # type: ignore

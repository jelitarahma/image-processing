"""2D Discrete Fourier Transform (DFT) and frequency spectrum analysis."""

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class FourierResult:
    """Container holding Fourier frequency representations."""

    original_gray: np.ndarray
    dft_shift: np.ndarray
    magnitude_spectrum: np.ndarray
    normalized_spectrum: np.ndarray


def compute_dft(gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D Discrete Fourier Transform and shifted magnitude spectrum.

    Parameters
    ----------
    gray_image : np.ndarray
        Single-channel 2D grayscale image array (uint8).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - dft_shift: Shifted complex frequency array of shape (H, W, 2).
        - magnitude_spectrum: Logarithmically scaled magnitude spectrum (float32).

    Raises
    ------
    ValueError
        If input is not a 2D single-channel uint8 array.
    """
    if not isinstance(gray_image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(gray_image)}.")
    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray_image.shape}.")
    if gray_image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image array, got {gray_image.dtype}.")

    # Convert to float32 for OpenCV DFT
    float_img = np.float32(gray_image)
    dft = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Shift zero-frequency DC component from top-left to center
    dft_shift = np.fft.fftshift(dft)

    # Compute Euclidean magnitude of real and imaginary parts
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    # Logarithmic scaling to compress high dynamic range: 20 * log(1 + magnitude)
    magnitude_spectrum = 20.0 * np.log(magnitude + 1.0)

    return dft_shift, magnitude_spectrum


def process_fourier(gray_image: np.ndarray) -> FourierResult:
    """Execute complete 2D Fourier analysis on a grayscale image.

    Parameters
    ----------
    gray_image : np.ndarray
        Single-channel 2D grayscale image array (uint8).

    Returns
    -------
    FourierResult
        Object containing grayscale input, centered DFT, and magnitude spectrum.
    """
    dft_shift, magnitude_spectrum = compute_dft(gray_image)

    # Normalize magnitude spectrum to 8-bit [0, 255] for image saving and visualization
    norm_spectrum = cv2.normalize(
        magnitude_spectrum,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    return FourierResult(
        original_gray=gray_image.copy(),
        dft_shift=dft_shift,
        magnitude_spectrum=magnitude_spectrum,
        normalized_spectrum=norm_spectrum,
    )

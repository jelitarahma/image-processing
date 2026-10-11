"""Laplacian spatial filtering and second-derivative edge response."""

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class LaplacianResult:
    """Container holding Laplacian filter outputs across different scaling representations."""

    original_gray: np.ndarray
    ksize: int
    laplacian_raw: np.ndarray
    laplacian_scaled: np.ndarray
    laplacian_abs: np.ndarray


def apply_laplacian(
    gray_image: np.ndarray,
    ksize: int = 3,
    ddepth: int = cv2.CV_64F,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate 2nd order spatial derivative using the Laplacian operator.

    Parameters
    ----------
    gray_image : np.ndarray
        Single-channel 8-bit grayscale image array (uint8).
    ksize : int, default=3
        Aperture size used to compute second-derivative filters. Must be a positive odd integer.
    ddepth : int, default=cv2.CV_64F
        Destination image depth. Using CV_64F preserves both positive and
        negative gradient slopes without saturation clipping.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        - laplacian_raw: Float64 2nd derivative response retaining negative and positive values.
        - laplacian_scaled: 8-bit normalized representation where zero-derivative is centered at mid-gray (128).
        - laplacian_abs: 8-bit absolute derivative response (cv2.convertScaleAbs) showing edge magnitudes.

    Raises
    ------
    ValueError
        If input is not 2D grayscale or ksize is invalid.
    """
    if not isinstance(gray_image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(gray_image)}.")
    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray_image.shape}.")
    if gray_image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image array, got {gray_image.dtype}.")
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError(f"Kernel size must be a positive odd integer, got {ksize}.")

    # Compute raw Laplacian in 64-bit float to preserve signs
    laplacian_raw = cv2.Laplacian(gray_image, ddepth=ddepth, ksize=ksize)

    # 1. Scaled representation: normalize min..max to 0..255 (zero centered near 128)
    laplacian_scaled = cv2.normalize(
        laplacian_raw,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    # 2. Absolute response: |L|, showing prominent edge transitions
    laplacian_abs = cv2.convertScaleAbs(laplacian_raw)

    return laplacian_raw, laplacian_scaled, laplacian_abs


def process_laplacian(
    gray_image: np.ndarray,
    ksize: int = 3,
) -> LaplacianResult:
    """Execute complete Laplacian filter pipeline on a grayscale image.

    Parameters
    ----------
    gray_image : np.ndarray
        Single-channel 2D grayscale image array (uint8).
    ksize : int, default=3
        Aperture size of Laplacian kernel.

    Returns
    -------
    LaplacianResult
        Object containing raw float, scaled normalized, and absolute 8-bit Laplacian representations.
    """
    raw, scaled, abs_resp = apply_laplacian(gray_image, ksize=ksize)

    return LaplacianResult(
        original_gray=gray_image.copy(),
        ksize=ksize,
        laplacian_raw=raw,
        laplacian_scaled=scaled,
        laplacian_abs=abs_resp,
    )

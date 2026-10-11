"""Illumination processing, histogram calculation, and L-channel equalization."""

from dataclasses import dataclass
from typing import Dict, Tuple
import cv2
import numpy as np

from illumination_lab.color import (
    bgr_to_lab,
    bgr_to_rgb,
    lab_to_bgr,
    lab_to_rgb,
    merge_lab_channels,
    split_lab_channels,
)


@dataclass(frozen=True)
class IlluminationResult:
    """Container holding all stages of color and illumination processing."""

    original_bgr: np.ndarray
    original_rgb: np.ndarray
    original_lab: np.ndarray
    l_channel: np.ndarray
    a_channel: np.ndarray
    b_channel: np.ndarray
    equalized_l: np.ndarray
    equalized_lab: np.ndarray
    equalized_bgr: np.ndarray
    equalized_rgb: np.ndarray

    @property
    def metrics(self) -> Dict[str, float]:
        """Compute key illumination statistics before and after equalization."""
        return {
            "l_mean_before": float(np.mean(self.l_channel)),
            "l_std_before": float(np.std(self.l_channel)),
            "l_min_before": float(np.min(self.l_channel)),
            "l_max_before": float(np.max(self.l_channel)),
            "l_mean_after": float(np.mean(self.equalized_l)),
            "l_std_after": float(np.std(self.equalized_l)),
            "l_min_after": float(np.min(self.equalized_l)),
            "l_max_after": float(np.max(self.equalized_l)),
        }


def compute_histogram(
    channel: np.ndarray,
    bins: int = 256,
    val_range: Tuple[int, int] = (0, 256),
) -> np.ndarray:
    """Compute the 1D intensity histogram for an image channel.

    Parameters
    ----------
    channel : np.ndarray
        2D single-channel image array (uint8).
    bins : int, default=256
        Number of histogram bins.
    val_range : tuple of (int, int), default=(0, 256)
        Lower and upper range of pixel intensity values.

    Returns
    -------
    np.ndarray
        1D histogram array of shape (bins,).
    """
    if channel.ndim != 2:
        raise ValueError(f"Expected 2D single channel array, got shape {channel.shape}.")

    hist = cv2.calcHist([channel], [0], None, [bins], list(val_range))
    return hist.flatten()


def compute_cdf(channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized cumulative distribution function (CDF) and histogram.

    Parameters
    ----------
    channel : np.ndarray
        2D single-channel image array (uint8).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (hist, cdf_normalized) where cdf_normalized is scaled between 0.0 and 1.0.
    """
    hist = compute_histogram(channel)
    cdf = hist.cumsum()
    cdf_normalized = cdf / (cdf[-1] if cdf[-1] > 0 else 1.0)
    return hist, cdf_normalized


def equalize_l_channel(l_channel: np.ndarray) -> np.ndarray:
    """Apply histogram equalization exclusively to the L (Lightness) channel.

    Equalizing only the lightness component enhances contrast and normalizes
    illumination without altering chromaticity (A and B color channels), thus
    preventing severe color distortion.

    Parameters
    ----------
    l_channel : np.ndarray
        2D array of the L channel (uint8).

    Returns
    -------
    np.ndarray
        Equalized L channel (uint8) of identical shape.
    """
    if l_channel.ndim != 2:
        raise ValueError(f"Expected 2D L-channel array, got shape {l_channel.shape}.")
    if l_channel.dtype != np.uint8:
        raise ValueError(f"Expected uint8 dtype, got {l_channel.dtype}.")

    return cv2.equalizeHist(l_channel)


def equalize_lab_image(lab_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Equalize illumination in LAB color space by processing the L channel only.

    Parameters
    ----------
    lab_image : np.ndarray
        3-channel LAB image array (uint8).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (equalized_lab, equalized_l) where equalized_lab has the modified L channel
        and original A, B channels.
    """
    l_channel, a_channel, b_channel = split_lab_channels(lab_image)
    equalized_l = equalize_l_channel(l_channel)
    equalized_lab = merge_lab_channels(equalized_l, a_channel, b_channel)
    return equalized_lab, equalized_l


def equalize_illumination(bgr_image: np.ndarray) -> IlluminationResult:
    """Run full color-space and illumination equalization pipeline.

    Processes input BGR image through:
    1. RGB conversion (for display/visualization).
    2. LAB conversion and channel splitting (L, A, B).
    3. Histogram equalization strictly on the L channel.
    4. Merging equalized L with untouched chromatic channels A and B.
    5. Inverse color conversion back to BGR and RGB.

    Parameters
    ----------
    bgr_image : np.ndarray
        Original input image in OpenCV BGR format.

    Returns
    -------
    IlluminationResult
        Complete result object containing all pipeline stages.
    """
    rgb_image = bgr_to_rgb(bgr_image)
    lab_image = bgr_to_lab(bgr_image)
    l_chan, a_chan, b_chan = split_lab_channels(lab_image)

    equalized_l = equalize_l_channel(l_chan)
    equalized_lab = merge_lab_channels(equalized_l, a_chan, b_chan)

    equalized_bgr = lab_to_bgr(equalized_lab)
    equalized_rgb = lab_to_rgb(equalized_lab)

    return IlluminationResult(
        original_bgr=bgr_image.copy(),
        original_rgb=rgb_image,
        original_lab=lab_image,
        l_channel=l_chan,
        a_channel=a_chan,
        b_channel=b_chan,
        equalized_l=equalized_l,
        equalized_lab=equalized_lab,
        equalized_bgr=equalized_bgr,
        equalized_rgb=equalized_rgb,
    )

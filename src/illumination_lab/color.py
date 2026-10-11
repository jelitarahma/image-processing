"""Color space conversion operations and channel manipulation."""

from typing import Tuple
import cv2
import numpy as np


def _validate_image_array(image: np.ndarray, expected_channels: int = 3) -> None:
    """Validate that the input is a valid uint8 numpy image array."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}.")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 array, got dtype {image.dtype}.")
    if expected_channels == 3:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image, got shape {image.shape}.")
    elif expected_channels == 1:
        if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] == 1):
            raise ValueError(f"Expected single-channel image, got shape {image.shape}.")


def bgr_to_rgb(bgr_image: np.ndarray) -> np.ndarray:
    """Convert an image from OpenCV BGR ordering to RGB ordering.

    Parameters
    ----------
    bgr_image : np.ndarray
        3-channel BGR image array (uint8).

    Returns
    -------
    np.ndarray
        Converted RGB image array (uint8).
    """
    _validate_image_array(bgr_image, expected_channels=3)
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(rgb_image: np.ndarray) -> np.ndarray:
    """Convert an image from RGB ordering to OpenCV BGR ordering.

    Parameters
    ----------
    rgb_image : np.ndarray
        3-channel RGB image array (uint8).

    Returns
    -------
    np.ndarray
        Converted BGR image array (uint8).
    """
    _validate_image_array(rgb_image, expected_channels=3)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def bgr_to_lab(bgr_image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to the CIE LAB color space.

    Parameters
    ----------
    bgr_image : np.ndarray
        3-channel BGR image array (uint8).

    Returns
    -------
    np.ndarray
        Converted LAB image array (uint8). In OpenCV:
        L ranges [0, 255] (scaled from 0..100)
        A ranges [0, 255] (shifted by 128)
        B ranges [0, 255] (shifted by 128)
    """
    _validate_image_array(bgr_image, expected_channels=3)
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)


def rgb_to_lab(rgb_image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to the CIE LAB color space.

    Parameters
    ----------
    rgb_image : np.ndarray
        3-channel RGB image array (uint8).

    Returns
    -------
    np.ndarray
        Converted LAB image array (uint8).
    """
    _validate_image_array(rgb_image, expected_channels=3)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)


def lab_to_bgr(lab_image: np.ndarray) -> np.ndarray:
    """Convert a CIE LAB image to OpenCV BGR color space.

    Parameters
    ----------
    lab_image : np.ndarray
        3-channel LAB image array (uint8).

    Returns
    -------
    np.ndarray
        Converted BGR image array (uint8).
    """
    _validate_image_array(lab_image, expected_channels=3)
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


def lab_to_rgb(lab_image: np.ndarray) -> np.ndarray:
    """Convert a CIE LAB image to standard RGB color space.

    Parameters
    ----------
    lab_image : np.ndarray
        3-channel LAB image array (uint8).

    Returns
    -------
    np.ndarray
        Converted RGB image array (uint8).
    """
    _validate_image_array(lab_image, expected_channels=3)
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)


def split_lab_channels(lab_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a 3-channel LAB image into individual L, A, and B components.

    Parameters
    ----------
    lab_image : np.ndarray
        3-channel LAB image array (uint8).

    Returns
    -------
    tuple of np.ndarray
        (l_channel, a_channel, b_channel), each of shape (H, W) and dtype uint8.
    """
    _validate_image_array(lab_image, expected_channels=3)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    return l_channel, a_channel, b_channel


def merge_lab_channels(
    l_channel: np.ndarray,
    a_channel: np.ndarray,
    b_channel: np.ndarray,
) -> np.ndarray:
    """Merge separate L, A, and B channels into a single 3-channel LAB image.

    Parameters
    ----------
    l_channel : np.ndarray
        L channel array (H, W), dtype uint8.
    a_channel : np.ndarray
        A channel array (H, W), dtype uint8.
    b_channel : np.ndarray
        B channel array (H, W), dtype uint8.

    Returns
    -------
    np.ndarray
        Merged 3-channel LAB image array (H, W, 3), dtype uint8.
    """
    _validate_image_array(l_channel, expected_channels=1)
    _validate_image_array(a_channel, expected_channels=1)
    _validate_image_array(b_channel, expected_channels=1)

    if l_channel.shape[:2] != a_channel.shape[:2] or l_channel.shape[:2] != b_channel.shape[:2]:
        raise ValueError("Channel dimensions must match for merging.")

    return cv2.merge([l_channel, a_channel, b_channel])


def bgr_to_gray(bgr_image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale.

    Parameters
    ----------
    bgr_image : np.ndarray
        3-channel BGR image array (uint8).

    Returns
    -------
    np.ndarray
        Single-channel grayscale image array (H, W), dtype uint8.
    """
    _validate_image_array(bgr_image, expected_channels=3)
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)


def rgb_to_gray(rgb_image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to single-channel grayscale.

    Parameters
    ----------
    rgb_image : np.ndarray
        3-channel RGB image array (uint8).

    Returns
    -------
    np.ndarray
        Single-channel grayscale image array (H, W), dtype uint8.
    """
    _validate_image_array(rgb_image, expected_channels=3)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


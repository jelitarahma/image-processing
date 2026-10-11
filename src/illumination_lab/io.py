"""Image input and output utilities with robust validation."""

from pathlib import Path
from typing import Union
import cv2
import numpy as np


def load_image(file_path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk using OpenCV.

    Parameters
    ----------
    file_path : str | Path
        Path to the target image file.

    Returns
    -------
    np.ndarray
        Loaded image array in OpenCV's default BGR format (dtype uint8).

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the file cannot be decoded as an image or has invalid dimensions.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path.resolve()}")

    # OpenCV imread does not raise error on failure; returns None instead
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"Failed to read image at {path.resolve()}. The file may be corrupt or unsupported."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected 3-channel color image, got shape {image.shape}."
        )

    return image


def save_image(
    output_path: Union[str, Path],
    image: np.ndarray,
    is_rgb: bool = False,
) -> Path:
    """Save an image array to disk, ensuring directory structure exists.

    Parameters
    ----------
    output_path : str | Path
        Destination file path.
    image : np.ndarray
        Image array to save (2D grayscale or 3D color).
    is_rgb : bool, default=False
        If True and image is 3D (3-channel), converts RGB to BGR before writing
        so that OpenCV saves correct colors.

    Returns
    -------
    Path
        Resolved path of the saved file.

    Raises
    ------
    ValueError
        If the image array is empty or writing fails.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot save empty image array.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img_to_write = image
    if is_rgb and image.ndim == 3 and image.shape[2] == 3:
        img_to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(path), img_to_write)
    if not success:
        raise IOError(f"Failed to write image to {path.resolve()}.")

    return path.resolve()

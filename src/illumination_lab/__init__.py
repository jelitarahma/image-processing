"""Illumination and Color Space Processing Package."""

from illumination_lab.color import (
    bgr_to_gray,
    bgr_to_lab,
    bgr_to_rgb,
    lab_to_bgr,
    lab_to_rgb,
    merge_lab_channels,
    rgb_to_bgr,
    rgb_to_gray,
    rgb_to_lab,
    split_lab_channels,
)
from illumination_lab.fourier import (
    FourierResult,
    compute_dft,
    process_fourier,
)
from illumination_lab.histogram import (
    IlluminationResult,
    compute_cdf,
    compute_histogram,
    equalize_illumination,
    equalize_l_channel,
    equalize_lab_image,
)
from illumination_lab.io import load_image, save_image
from illumination_lab.laplacian import (
    LaplacianResult,
    apply_laplacian,
    process_laplacian,
)
from illumination_lab.pipeline import (
    run_experiment,
    run_fourier_experiment,
    run_laplacian_experiment,
    run_question2_experiments,
)

__all__ = [
    # I/O
    "load_image",
    "save_image",
    # Color conversions
    "bgr_to_rgb",
    "rgb_to_bgr",
    "bgr_to_gray",
    "rgb_to_gray",
    "bgr_to_lab",
    "rgb_to_lab",
    "lab_to_rgb",
    "lab_to_bgr",
    "split_lab_channels",
    "merge_lab_channels",
    # Histogram & Illumination
    "compute_histogram",
    "compute_cdf",
    "equalize_l_channel",
    "equalize_lab_image",
    "equalize_illumination",
    "IlluminationResult",
    # Fourier Transform
    "FourierResult",
    "compute_dft",
    "process_fourier",
    # Laplacian Filtering
    "LaplacianResult",
    "apply_laplacian",
    "process_laplacian",
    # Pipelines
    "run_experiment",
    "run_fourier_experiment",
    "run_laplacian_experiment",
    "run_question2_experiments",
]

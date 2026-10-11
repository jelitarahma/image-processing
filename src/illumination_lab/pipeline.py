"""Pipeline orchestrating experiment execution, figure generation, and output storage."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from illumination_lab.color import bgr_to_gray, bgr_to_rgb
from illumination_lab.fourier import FourierResult, process_fourier
from illumination_lab.histogram import IlluminationResult, equalize_illumination
from illumination_lab.io import load_image, save_image
from illumination_lab.laplacian import LaplacianResult, process_laplacian
from illumination_lab.visualizer import (
    plot_fourier_spectrum,
    plot_histogram_equalization,
    plot_illumination_histograms_comparison,
    plot_indoor_vs_outdoor_comparison,
    plot_lab_channels,
    plot_laplacian_filter,
    plot_rgb_overview,
)

logger = logging.getLogger(__name__)


def run_experiment(
    indoor_path: Path,
    outdoor_path: Path,
    output_dir: Path,
    dpi: int = 300,
    save_raw_images: bool = True,
) -> Dict[str, Any]:
    """Execute complete illumination experiment for indoor and outdoor images (Question 1).

    Parameters
    ----------
    indoor_path : Path
        Path to the indoor image file.
    outdoor_path : Path
        Path to the outdoor image file.
    output_dir : Path
        Base output directory.
    dpi : int, default=300
        Figure resolution.
    save_raw_images : bool, default=True
        Whether to save individual processed image files (RGB & channels).

    Returns
    -------
    dict
        Execution summary with paths, metrics, and generated figures.
    """
    logger.info("Starting illumination experiment (Question 1)...")
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    processed_dir = output_dir / "processed_images"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if save_raw_images:
        processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load images
    logger.info(f"Loading indoor image: {indoor_path}")
    indoor_bgr = load_image(indoor_path)
    logger.info(f"Loading outdoor image: {outdoor_path}")
    outdoor_bgr = load_image(outdoor_path)

    # 2. Process illumination and color spaces
    logger.info("Executing color space conversions and L-channel equalization...")
    indoor_res = equalize_illumination(indoor_bgr)
    outdoor_res = equalize_illumination(outdoor_bgr)

    # 3. Save raw processed images if requested
    saved_images: Dict[str, Path] = {}
    if save_raw_images:
        logger.info("Saving processed images...")
        saved_images["indoor_rgb"] = save_image(
            processed_dir / "indoor_rgb.png", indoor_res.original_rgb, is_rgb=True
        )
        saved_images["indoor_equalized_rgb"] = save_image(
            processed_dir / "indoor_equalized_rgb.png", indoor_res.equalized_rgb, is_rgb=True
        )
        saved_images["indoor_l_channel"] = save_image(
            processed_dir / "indoor_l_channel.png", indoor_res.l_channel
        )
        saved_images["indoor_equalized_l"] = save_image(
            processed_dir / "indoor_equalized_l.png", indoor_res.equalized_l
        )

        saved_images["outdoor_rgb"] = save_image(
            processed_dir / "outdoor_rgb.png", outdoor_res.original_rgb, is_rgb=True
        )
        saved_images["outdoor_equalized_rgb"] = save_image(
            processed_dir / "outdoor_equalized_rgb.png", outdoor_res.equalized_rgb, is_rgb=True
        )
        saved_images["outdoor_l_channel"] = save_image(
            processed_dir / "outdoor_l_channel.png", outdoor_res.l_channel
        )
        saved_images["outdoor_equalized_l"] = save_image(
            processed_dir / "outdoor_equalized_l.png", outdoor_res.equalized_l
        )

    # 4. Generate experimental figures
    logger.info("Generating publication-quality figures for Question 1...")
    generated_figures: Dict[str, Path] = {}

    # Figure 1: RGB Overview
    fig1_path = figures_dir / "01_rgb_overview.png"
    plot_rgb_overview(indoor_res.original_rgb, outdoor_res.original_rgb, fig1_path, dpi=dpi)
    generated_figures["rgb_overview"] = fig1_path

    # Figure 2: LAB Decomposition for Indoor & Outdoor
    fig2a_path = figures_dir / "02_indoor_lab_channels.png"
    plot_lab_channels(
        indoor_res.original_rgb,
        indoor_res.l_channel,
        indoor_res.a_channel,
        indoor_res.b_channel,
        "Indoor",
        fig2a_path,
        dpi=dpi,
    )
    generated_figures["indoor_lab_channels"] = fig2a_path

    fig2b_path = figures_dir / "02_outdoor_lab_channels.png"
    plot_lab_channels(
        outdoor_res.original_rgb,
        outdoor_res.l_channel,
        outdoor_res.a_channel,
        outdoor_res.b_channel,
        "Outdoor",
        fig2b_path,
        dpi=dpi,
    )
    generated_figures["outdoor_lab_channels"] = fig2b_path

    # Figure 3: Histogram Equalization for Indoor & Outdoor
    fig3a_path = figures_dir / "03_indoor_histogram_equalization.png"
    plot_histogram_equalization(indoor_res, "Indoor", fig3a_path, dpi=dpi)
    generated_figures["indoor_equalization"] = fig3a_path

    fig3b_path = figures_dir / "03_outdoor_histogram_equalization.png"
    plot_histogram_equalization(outdoor_res, "Outdoor", fig3b_path, dpi=dpi)
    generated_figures["outdoor_equalization"] = fig3b_path

    # Figure 4: Indoor vs Outdoor Matrix Comparison
    fig4_path = figures_dir / "04_indoor_vs_outdoor_comparison.png"
    plot_indoor_vs_outdoor_comparison(indoor_res, outdoor_res, fig4_path, dpi=dpi)
    generated_figures["cross_scene_comparison"] = fig4_path

    # Figure 5: Histogram Distribution Comparison
    fig5_path = figures_dir / "05_illumination_distribution_comparison.png"
    plot_illumination_histograms_comparison(indoor_res, outdoor_res, fig5_path, dpi=dpi)
    generated_figures["histograms_comparison"] = fig5_path

    logger.info("Question 1 illumination experiment completed.")

    return {
        "indoor_metrics": indoor_res.metrics,
        "outdoor_metrics": outdoor_res.metrics,
        "figures": generated_figures,
        "processed_images": saved_images,
    }


def run_fourier_experiment(
    image_path: Path,
    output_dir: Path,
    dpi: int = 300,
    save_raw_images: bool = True,
) -> Dict[str, Any]:
    """Execute 2D Discrete Fourier Transform experiment (Question 2 - Experiment A).

    Parameters
    ----------
    image_path : Path
        Path to target input image.
    output_dir : Path
        Base output directory.
    dpi : int, default=300
        Figure resolution.
    save_raw_images : bool, default=True
        Whether to save raw magnitude spectrum image.

    Returns
    -------
    dict
        Execution summary with paths and figures.
    """
    logger.info(f"Starting Fourier Transform experiment on {image_path}...")
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    processed_dir = output_dir / "processed_images"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if save_raw_images:
        processed_dir.mkdir(parents=True, exist_ok=True)

    bgr_img = load_image(image_path)
    gray_img = bgr_to_gray(bgr_img)
    rgb_img = bgr_to_rgb(bgr_img)

    fourier_res = process_fourier(gray_img)

    saved_images: Dict[str, Path] = {}
    if save_raw_images:
        saved_images["fourier_magnitude_spectrum"] = save_image(
            processed_dir / "fourier_magnitude_spectrum.png",
            fourier_res.normalized_spectrum,
        )

    fig_path = figures_dir / "06_fourier_transform.png"
    plot_fourier_spectrum(
        image=rgb_img,
        magnitude_spectrum=fourier_res.magnitude_spectrum,
        save_path=fig_path,
        title="Question 2 (Exp A): 2D Discrete Fourier Transform Analysis",
        dpi=dpi,
    )

    logger.info(f"Fourier experiment completed. Figure: {fig_path}")
    return {
        "result": fourier_res,
        "figures": {"fourier_transform": fig_path},
        "processed_images": saved_images,
    }


def run_laplacian_experiment(
    image_path: Path,
    output_dir: Path,
    ksize: int = 3,
    dpi: int = 300,
    save_raw_images: bool = True,
) -> Dict[str, Any]:
    """Execute Laplacian spatial filtering experiment (Question 2 - Experiment B).

    Parameters
    ----------
    image_path : Path
        Path to target input image.
    output_dir : Path
        Base output directory.
    ksize : int, default=3
        Kernel aperture size.
    dpi : int, default=300
        Figure resolution.
    save_raw_images : bool, default=True
        Whether to save raw filtered image files.

    Returns
    -------
    dict
        Execution summary with paths and figures.
    """
    logger.info(f"Starting Laplacian Filter experiment on {image_path} (ksize={ksize})...")
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    processed_dir = output_dir / "processed_images"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if save_raw_images:
        processed_dir.mkdir(parents=True, exist_ok=True)

    bgr_img = load_image(image_path)
    gray_img = bgr_to_gray(bgr_img)
    rgb_img = bgr_to_rgb(bgr_img)

    lap_res = process_laplacian(gray_img, ksize=ksize)

    saved_images: Dict[str, Path] = {}
    if save_raw_images:
        saved_images["laplacian_scaled"] = save_image(
            processed_dir / "laplacian_scaled.png",
            lap_res.laplacian_scaled,
        )
        saved_images["laplacian_abs"] = save_image(
            processed_dir / "laplacian_abs.png",
            lap_res.laplacian_abs,
        )

    fig_path = figures_dir / "07_laplacian_filter.png"
    plot_laplacian_filter(
        original_image=rgb_img,
        laplacian_scaled=lap_res.laplacian_scaled,
        laplacian_abs=lap_res.laplacian_abs,
        save_path=fig_path,
        ksize=ksize,
        title="Question 2 (Exp B): Laplacian Spatial Filter Edge Analysis",
        dpi=dpi,
    )

    logger.info(f"Laplacian experiment completed. Figure: {fig_path}")
    return {
        "result": lap_res,
        "figures": {"laplacian_filter": fig_path},
        "processed_images": saved_images,
    }


def run_question2_experiments(
    image_path: Path,
    output_dir: Path,
    ksize: int = 3,
    dpi: int = 300,
    save_raw_images: bool = True,
) -> Dict[str, Any]:
    """Execute both Question 2 experiments (Fourier and Laplacian) on input image."""
    fourier_out = run_fourier_experiment(
        image_path=image_path,
        output_dir=output_dir,
        dpi=dpi,
        save_raw_images=save_raw_images,
    )
    laplacian_out = run_laplacian_experiment(
        image_path=image_path,
        output_dir=output_dir,
        ksize=ksize,
        dpi=dpi,
        save_raw_images=save_raw_images,
    )

    combined_figures = {**fourier_out["figures"], **laplacian_out["figures"]}
    combined_images = {**fourier_out["processed_images"], **laplacian_out["processed_images"]}

    return {
        "figures": combined_figures,
        "processed_images": combined_images,
    }

"""Visualization module generating professional, publication-quality figures."""

from pathlib import Path
from typing import Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np

from illumination_lab.histogram import IlluminationResult, compute_cdf


def _configure_plot_style() -> None:
    """Set clean, publication-ready styling parameters."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 13,
        "axes.grid": False,
    })


def plot_rgb_overview(
    indoor_rgb: np.ndarray,
    outdoor_rgb: np.ndarray,
    save_path: Union[str, Path],
    indoor_title: str = "Indoor Scene",
    outdoor_title: str = "Outdoor Scene",
    dpi: int = 300,
) -> Path:
    """Generate side-by-side RGB visualization for indoor and outdoor scenes.

    Parameters
    ----------
    indoor_rgb : np.ndarray
        Indoor image in RGB format.
    outdoor_rgb : np.ndarray
        Outdoor image in RGB format.
    save_path : str | Path
        Path to save the generated figure.
    indoor_title : str, default="Indoor Scene"
        Label or title for the indoor image subplot.
    outdoor_title : str, default="Outdoor Scene"
        Label or title for the outdoor image subplot.
    dpi : int, default=300
        Resolution of the saved figure.

    Returns
    -------
    Path
        Resolved path to the saved figure.
    """
    _configure_plot_style()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    axes[0].imshow(indoor_rgb)
    axes[0].set_title(
        f"{indoor_title}\n"
        f"Resolution: {indoor_rgb.shape[1]}x{indoor_rgb.shape[0]} | Mean Intensity: {indoor_rgb.mean():.1f}"
    )
    axes[0].axis("off")

    axes[1].imshow(outdoor_rgb)
    axes[1].set_title(
        f"{outdoor_title}\n"
        f"Resolution: {outdoor_rgb.shape[1]}x{outdoor_rgb.shape[0]} | Mean Intensity: {outdoor_rgb.mean():.1f}"
    )
    axes[1].axis("off")

    fig.suptitle("Input Images: RGB Representation", fontweight="bold")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path.resolve()


def plot_lab_channels(
    rgb_image: np.ndarray,
    l_channel: np.ndarray,
    a_channel: np.ndarray,
    b_channel: np.ndarray,
    scene_name: str,
    save_path: Union[str, Path],
    dpi: int = 300,
) -> Path:
    """Generate 4-panel decomposition showing RGB and separate L, A, B channels.

    Parameters
    ----------
    rgb_image : np.ndarray
        Original image in RGB format.
    l_channel : np.ndarray
        L (Lightness) channel.
    a_channel : np.ndarray
        A (Green-Red chromaticity) channel.
    b_channel : np.ndarray
        B (Blue-Yellow chromaticity) channel.
    scene_name : str
        Descriptive name (e.g. 'Indoor' or 'Outdoor').
    save_path : str | Path
        Path to save the generated figure.
    dpi : int, default=300
        Figure resolution.

    Returns
    -------
    Path
        Resolved path to the saved figure.
    """
    _configure_plot_style()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)

    # RGB Composite
    axes[0].imshow(rgb_image)
    axes[0].set_title("Input RGB Composite")
    axes[0].axis("off")

    # L Channel (Lightness)
    im_l = axes[1].imshow(l_channel, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"L Channel (Lightness)\nMean={l_channel.mean():.1f}, Std={l_channel.std():.1f}")
    axes[1].axis("off")
    fig.colorbar(im_l, ax=axes[1], fraction=0.046, pad=0.04)

    # A Channel (Green - Red)
    im_a = axes[2].imshow(a_channel, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"A Channel (Green - Red)\nMean={a_channel.mean():.1f}, Std={a_channel.std():.1f}")
    axes[2].axis("off")
    fig.colorbar(im_a, ax=axes[2], fraction=0.046, pad=0.04)

    # B Channel (Blue - Yellow)
    im_b = axes[3].imshow(b_channel, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title(f"B Channel (Blue - Yellow)\nMean={b_channel.mean():.1f}, Std={b_channel.std():.1f}")
    axes[3].axis("off")
    fig.colorbar(im_b, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"{scene_name} Scene: CIE LAB Color Space Decomposition", fontweight="bold")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path.resolve()


def plot_histogram_equalization(
    result: IlluminationResult,
    scene_name: str,
    save_path: Union[str, Path],
    dpi: int = 300,
) -> Path:
    """Generate 2x3 comparison grid showing images, L channel, and histograms with CDF.

    Parameters
    ----------
    result : IlluminationResult
        Result container with original and equalized images and channels.
    scene_name : str
        Scene label (e.g. 'Indoor' or 'Outdoor').
    save_path : str | Path
        Path to save figure.
    dpi : int, default=300
        Figure resolution.

    Returns
    -------
    Path
        Resolved path to the saved figure.
    """
    _configure_plot_style()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)

    # Row 1: Original
    axes[0, 0].imshow(result.original_rgb)
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis("off")

    im_orig_l = axes[0, 1].imshow(result.l_channel, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Original L (Lightness)")
    axes[0, 1].axis("off")
    fig.colorbar(im_orig_l, ax=axes[0, 1], fraction=0.046, pad=0.04)

    hist_orig, cdf_orig = compute_cdf(result.l_channel)
    ax_hist_orig = axes[0, 2]
    ax_hist_orig.bar(range(256), hist_orig, width=1.0, color="#4A90E2", alpha=0.7, label="Histogram")
    ax_hist_orig.set_xlim([0, 255])
    ax_hist_orig.set_xlabel("L Intensity")
    ax_hist_orig.set_ylabel("Pixel Count", color="#1E56A0")
    ax_hist_orig.set_title(
        f"Original L Histogram & CDF\n(μ={result.metrics['l_mean_before']:.1f}, σ={result.metrics['l_std_before']:.1f})"
    )
    ax_hist_orig.grid(True, linestyle="--", alpha=0.4)

    ax_cdf_orig = ax_hist_orig.twinx()
    ax_cdf_orig.plot(range(256), cdf_orig, color="#D0021B", linewidth=2.0, label="CDF")
    ax_cdf_orig.set_ylabel("Normalized CDF", color="#D0021B")
    ax_cdf_orig.set_ylim([0, 1.05])

    # Row 2: Equalized
    axes[1, 0].imshow(result.equalized_rgb)
    axes[1, 0].set_title("Equalized RGB (L-Channel Equalized)")
    axes[1, 0].axis("off")

    im_eq_l = axes[1, 1].imshow(result.equalized_l, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Equalized L (Lightness)")
    axes[1, 1].axis("off")
    fig.colorbar(im_eq_l, ax=axes[1, 1], fraction=0.046, pad=0.04)

    hist_eq, cdf_eq = compute_cdf(result.equalized_l)
    ax_hist_eq = axes[1, 2]
    ax_hist_eq.bar(range(256), hist_eq, width=1.0, color="#50E3C2", alpha=0.7, label="Histogram")
    ax_hist_eq.set_xlim([0, 255])
    ax_hist_eq.set_xlabel("L Intensity")
    ax_hist_eq.set_ylabel("Pixel Count", color="#008B74")
    ax_hist_eq.set_title(
        f"Equalized L Histogram & CDF\n(μ={result.metrics['l_mean_after']:.1f}, σ={result.metrics['l_std_after']:.1f})"
    )
    ax_hist_eq.grid(True, linestyle="--", alpha=0.4)

    ax_cdf_eq = ax_hist_eq.twinx()
    ax_cdf_eq.plot(range(256), cdf_eq, color="#D0021B", linewidth=2.0, label="CDF")
    ax_cdf_eq.set_ylabel("Normalized CDF", color="#D0021B")
    ax_cdf_eq.set_ylim([0, 1.05])

    fig.suptitle(f"{scene_name} Scene: Illumination Histogram Equalization", fontweight="bold")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path.resolve()


def plot_indoor_vs_outdoor_comparison(
    indoor_res: IlluminationResult,
    outdoor_res: IlluminationResult,
    save_path: Union[str, Path],
    dpi: int = 300,
) -> Path:
    """Generate cross-scene comparison matrix (Indoor vs Outdoor across stages).

    Parameters
    ----------
    indoor_res : IlluminationResult
        Indoor experiment result.
    outdoor_res : IlluminationResult
        Outdoor experiment result.
    save_path : str | Path
        Path to save figure.
    dpi : int, default=300
        Figure resolution.

    Returns
    -------
    Path
        Resolved path to the saved figure.
    """
    _configure_plot_style()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    # Row 0: Indoor
    axes[0, 0].imshow(indoor_res.original_rgb)
    axes[0, 0].set_title("Indoor: Original RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(indoor_res.l_channel, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Indoor: Original L\nμ={indoor_res.metrics['l_mean_before']:.1f}")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(indoor_res.equalized_rgb)
    axes[0, 2].set_title("Indoor: Equalized RGB")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(indoor_res.equalized_l, cmap="gray", vmin=0, vmax=255)
    axes[0, 3].set_title(f"Indoor: Equalized L\nμ={indoor_res.metrics['l_mean_after']:.1f}")
    axes[0, 3].axis("off")

    # Row 1: Outdoor
    axes[1, 0].imshow(outdoor_res.original_rgb)
    axes[1, 0].set_title("Outdoor: Original RGB")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(outdoor_res.l_channel, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title(f"Outdoor: Original L\nμ={outdoor_res.metrics['l_mean_before']:.1f}")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(outdoor_res.equalized_rgb)
    axes[1, 2].set_title("Outdoor: Equalized RGB")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(outdoor_res.equalized_l, cmap="gray", vmin=0, vmax=255)
    axes[1, 3].set_title(f"Outdoor: Equalized L\nμ={outdoor_res.metrics['l_mean_after']:.1f}")
    axes[1, 3].axis("off")

    fig.suptitle("Comparative Evaluation: Indoor vs Outdoor Illumination Equalization", fontweight="bold")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path.resolve()


def plot_illumination_histograms_comparison(
    indoor_res: IlluminationResult,
    outdoor_res: IlluminationResult,
    save_path: Union[str, Path],
    dpi: int = 300,
) -> Path:
    """Generate comparative histogram distributions for indoor and outdoor illumination.

    Parameters
    ----------
    indoor_res : IlluminationResult
        Indoor experiment result.
    outdoor_res : IlluminationResult
        Outdoor experiment result.
    save_path : str | Path
        Path to save figure.
    dpi : int, default=300
        Figure resolution.

    Returns
    -------
    Path
        Resolved path to the saved figure.
    """
    _configure_plot_style()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    bins = range(256)
    hist_in_orig = compute_cdf(indoor_res.l_channel)[0]
    hist_in_eq = compute_cdf(indoor_res.equalized_l)[0]
    hist_out_orig = compute_cdf(outdoor_res.l_channel)[0]
    hist_out_eq = compute_cdf(outdoor_res.equalized_l)[0]

    # Subplot (0, 0): Indoor Original vs Equalized
    ax = axes[0, 0]
    ax.plot(bins, hist_in_orig, color="#1f77b4", linewidth=2.0, label="Indoor Original L")
    ax.plot(bins, hist_in_eq, color="#2ca02c", linestyle="--", linewidth=1.8, label="Indoor Equalized L")
    ax.set_title("Indoor: Original vs Equalized L Distribution")
    ax.set_xlabel("L Intensity")
    ax.set_ylabel("Pixel Count")
    ax.set_xlim([0, 255])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    # Subplot (0, 1): Outdoor Original vs Equalized
    ax = axes[0, 1]
    ax.plot(bins, hist_out_orig, color="#ff7f0e", linewidth=2.0, label="Outdoor Original L")
    ax.plot(bins, hist_out_eq, color="#d62728", linestyle="--", linewidth=1.8, label="Outdoor Equalized L")
    ax.set_title("Outdoor: Original vs Equalized L Distribution")
    ax.set_xlabel("L Intensity")
    ax.set_ylabel("Pixel Count")
    ax.set_xlim([0, 255])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    # Subplot (1, 0): Cross-Scene Original Illumination
    ax = axes[1, 0]
    ax.plot(bins, hist_in_orig, color="#1f77b4", linewidth=2.0, label=f"Indoor (μ={indoor_res.metrics['l_mean_before']:.1f})")
    ax.plot(bins, hist_out_orig, color="#ff7f0e", linewidth=2.0, label=f"Outdoor (μ={outdoor_res.metrics['l_mean_before']:.1f})")
    ax.set_title("Cross-Scene Comparison: Original Illumination Profiles")
    ax.set_xlabel("L Intensity")
    ax.set_ylabel("Pixel Count")
    ax.set_xlim([0, 255])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    # Subplot (1, 1): Cross-Scene Equalized Illumination
    ax = axes[1, 1]
    ax.plot(bins, hist_in_eq, color="#2ca02c", linewidth=2.0, label=f"Indoor Equalized (μ={indoor_res.metrics['l_mean_after']:.1f})")
    ax.plot(bins, hist_out_eq, color="#d62728", linewidth=2.0, label=f"Outdoor Equalized (μ={outdoor_res.metrics['l_mean_after']:.1f})")
    ax.set_title("Cross-Scene Comparison: Equalized Illumination Profiles")
    ax.set_xlabel("L Intensity")
    ax.set_ylabel("Pixel Count")
    ax.set_xlim([0, 255])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    fig.suptitle("Illumination Distribution Comparative Profiles", fontweight="bold")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path.resolve()


def plot_fourier_spectrum(
    image: np.ndarray,
    magnitude_spectrum: np.ndarray,
    save_path: Union[str, Path],
    title: str = "2D Discrete Fourier Transform (DFT)",
    dpi: int = 300,
) -> Path:
    """Generate 2-panel figure comparing input image and centered magnitude spectrum.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or RGB).
    magnitude_spectrum : np.ndarray
        Log-scaled magnitude spectrum.
    save_path : str | Path
        Path to save figure.
    title : str, default="2D Discrete Fourier Transform (DFT)"
        Figure title.
    dpi : int, default=300
        Figure resolution.

    Returns
    -------
    Path
        Resolved path to the saved figure.
    """
    _configure_plot_style()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    if image.ndim == 2:
        axes[0].imshow(image, cmap="gray")
    else:
        axes[0].imshow(image)
    axes[0].set_title(f"Original Image\nResolution: {image.shape[1]}x{image.shape[0]}")
    axes[0].axis("off")

    im_spec = axes[1].imshow(magnitude_spectrum, cmap="gray")
    axes[1].set_title("Magnitude Spectrum (20 * log(1 + |F|))\nCentered Zero-Frequency")
    axes[1].axis("off")
    fig.colorbar(im_spec, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontweight="bold")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path.resolve()


def plot_laplacian_filter(
    original_image: np.ndarray,
    laplacian_scaled: np.ndarray,
    laplacian_abs: np.ndarray,
    save_path: Union[str, Path],
    ksize: int = 3,
    title: str = "Laplacian Spatial Filter Response",
    dpi: int = 300,
) -> Path:
    """Generate 3-panel figure showing original image and Laplacian responses.

    Parameters
    ----------
    original_image : np.ndarray
        Original image (grayscale or RGB).
    laplacian_scaled : np.ndarray
        Scaled normalized Laplacian response (zero centered near mid-gray 128).
    laplacian_abs : np.ndarray
        Absolute Laplacian response (|Laplacian|) showing edge magnitudes.
    save_path : str | Path
        Path to save figure.
    ksize : int, default=3
        Kernel aperture size.
    title : str, default="Laplacian Spatial Filter Response"
        Figure title.
    dpi : int, default=300
        Figure resolution.

    Returns
    -------
    Path
        Resolved path to the saved figure.
    """
    _configure_plot_style()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    if original_image.ndim == 2:
        axes[0].imshow(original_image, cmap="gray")
    else:
        axes[0].imshow(original_image)
    axes[0].set_title(f"Original Image\nResolution: {original_image.shape[1]}x{original_image.shape[0]}")
    axes[0].axis("off")

    im_scaled = axes[1].imshow(laplacian_scaled, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Scaled Laplacian (ksize={ksize})\nNormalized [0, 255] (Zero at ~128)")
    axes[1].axis("off")
    fig.colorbar(im_scaled, ax=axes[1], fraction=0.046, pad=0.04)

    im_abs = axes[2].imshow(laplacian_abs, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"Absolute Laplacian Response\n|∇²I| Edge Magnitude")
    axes[2].axis("off")
    fig.colorbar(im_abs, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontweight="bold")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path.resolve()


"""Command-line interface for Computer Vision experiments (Question 1 & Question 2)."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure src is in sys.path when executed directly as `python main.py`
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from illumination_lab.pipeline import (
    run_experiment,
    run_fourier_experiment,
    run_laplacian_experiment,
    run_question2_experiments,
)


def _resolve_default_path(base_dir: Path, filename_stem: str) -> Optional[Path]:
    """Find image file by stem, checking multiple extensions and names."""
    candidates = [
        f"{filename_stem}.png",
        f"{filename_stem}.jpg",
        f"{filename_stem}.jpeg",
        f"{filename_stem} 1.jpg",
        f"{filename_stem}_1.jpg",
    ]
    for candidate in candidates:
        path = base_dir / candidate
        if path.is_file():
            return path
    return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Computer Vision Experiments: Illumination (Q1), Fourier Transform & Laplacian (Q2)."
    )
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        choices=["all", "q1", "q2", "illumination", "fourier", "laplacian"],
        default="all",
        help="Experiment to execute: 'all', 'q1' (illumination), 'q2' (Fourier & Laplacian), 'fourier', 'laplacian' (default: all).",
    )
    parser.add_argument(
        "--indoor",
        type=Path,
        default=None,
        help="Path to indoor input image for Q1. Defaults to data/indoor.png or data/indoor.jpg.",
    )
    parser.add_argument(
        "--outdoor",
        type=Path,
        default=None,
        help="Path to outdoor input image for Q1. Defaults to data/outdoor.png or data/outdoor.jpg.",
    )
    parser.add_argument(
        "--q2-image", "--practicum",
        type=Path,
        dest="q2_image",
        default=None,
        help="Path to Question 2 input image. Defaults to data/practicum 1.jpg or data/practicum_1.jpg.",
    )
    parser.add_argument(
        "--ksize",
        type=int,
        default=3,
        help="Aperture size for Laplacian operator (positive odd integer, default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory to save experimental figures and processed images (default: outputs).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution (DPI) for saved matplotlib figures (default: 300).",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip saving individual raw processed images (figures will still be saved).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("main")

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    logger.info("=" * 60)
    logger.info(f"Image Processing Experiments (Mode: {args.experiment})")
    logger.info("=" * 60)

    execute_q1 = args.experiment in ("all", "q1", "illumination")
    execute_fourier = args.experiment in ("all", "q2", "fourier")
    execute_laplacian = args.experiment in ("all", "q2", "laplacian")

    # Run Question 1 (Illumination)
    if execute_q1:
        indoor_path = args.indoor or _resolve_default_path(data_dir, "indoor") or _resolve_default_path(base_dir.parent, "indoor")
        outdoor_path = args.outdoor or _resolve_default_path(data_dir, "outdoor") or _resolve_default_path(base_dir.parent, "outdoor")

        if indoor_path is None or not indoor_path.is_file():
            logger.error(f"Indoor image not found in {data_dir} or parent directory.")
            return 1
        if outdoor_path is None or not outdoor_path.is_file():
            logger.error(f"Outdoor image not found in {data_dir} or parent directory.")
            return 1

        logger.info("[Question 1: Indoor/Outdoor Illumination Experiment]")
        logger.info(f"  Indoor:  {indoor_path}")
        logger.info(f"  Outdoor: {outdoor_path}")

        try:
            results_q1 = run_experiment(
                indoor_path=indoor_path,
                outdoor_path=outdoor_path,
                output_dir=args.output,
                dpi=args.dpi,
                save_raw_images=not args.no_raw,
            )
            in_m = results_q1["indoor_metrics"]
            out_m = results_q1["outdoor_metrics"]
            logger.info(
                f"  Indoor L-mean:  {in_m['l_mean_before']:.1f} -> {in_m['l_mean_after']:.1f}"
            )
            logger.info(
                f"  Outdoor L-mean: {out_m['l_mean_before']:.1f} -> {out_m['l_mean_after']:.1f}"
            )
        except Exception as exc:
            logger.exception(f"Question 1 failed: {exc}")
            return 1

    # Resolve Question 2 image if needed
    if execute_fourier or execute_laplacian:
        q2_img_path = (
            args.q2_image
            or _resolve_default_path(data_dir, "practicum")
            or _resolve_default_path(data_dir, "practicum_1")
            or _resolve_default_path(base_dir.parent, "practicum 1")
            or _resolve_default_path(base_dir.parent, "practicum_1")
        )
        if q2_img_path is None or not q2_img_path.is_file():
            logger.error(f"Question 2 image not found. Expected 'practicum 1.jpg' in {data_dir}.")
            return 1

        logger.info("-" * 60)
        logger.info(f"[Question 2: Frequency & Spatial Edge Experiments on {q2_img_path.name}]")

        if execute_fourier and execute_laplacian:
            try:
                run_question2_experiments(
                    image_path=q2_img_path,
                    output_dir=args.output,
                    ksize=args.ksize,
                    dpi=args.dpi,
                    save_raw_images=not args.no_raw,
                )
            except Exception as exc:
                logger.exception(f"Question 2 experiments failed: {exc}")
                return 1
        elif execute_fourier:
            try:
                run_fourier_experiment(
                    image_path=q2_img_path,
                    output_dir=args.output,
                    dpi=args.dpi,
                    save_raw_images=not args.no_raw,
                )
            except Exception as exc:
                logger.exception(f"Fourier experiment failed: {exc}")
                return 1
        elif execute_laplacian:
            try:
                run_laplacian_experiment(
                    image_path=q2_img_path,
                    output_dir=args.output,
                    ksize=args.ksize,
                    dpi=args.dpi,
                    save_raw_images=not args.no_raw,
                )
            except Exception as exc:
                logger.exception(f"Laplacian experiment failed: {exc}")
                return 1

    logger.info("=" * 60)
    logger.info(f"All requested experiments completed successfully. Outputs in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

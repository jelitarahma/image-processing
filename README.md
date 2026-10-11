# Image Processing Experiments

A modular Python project implementing computer vision experiments for color spaces, illumination processing, 2D Discrete Fourier Transform frequency analysis, and Laplacian spatial edge filtering using OpenCV.

## 1. Requirements and Dependencies

- Python >= 3.9
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `pytest>=7.0.0` (for automated unit tests)

## 2. Environment Setup and Installation

### Clone / Navigate to Project
```bash
cd "image-processing"
```

### Create and Activate Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional) Install project in editable development mode:*
```bash
pip install -e .
```

## 3. Running the Experiments

### Run All Experiments (Question 1 & Question 2)
Executes illumination equalization on indoor/outdoor images alongside Fourier transform and Laplacian filtering on `data/practicum 1.jpg`:
```bash
python main.py
```

### Run Question 1 Only (Illumination & Color Spaces)
```bash
python main.py -e q1
```

### Run Question 2 Only (Fourier Transform & Laplacian Filter)
```bash
python main.py -e q2
```

### Run Individual Experiments
```bash
# Fourier transform only
python main.py -e fourier

# Laplacian filter only
python main.py -e laplacian --ksize 3
```

### Custom Arguments and File Paths
```bash
python main.py --indoor data/indoor.png --outdoor data/outdoor.png --q2-image "data/practicum 1.jpg" --ksize 3 --output outputs
```

### Available CLI Options
- `-e, --experiment {all,q1,q2,illumination,fourier,laplacian}`: Specify experiment suite to run (default: `all`).
- `--indoor PATH`: Path to indoor image file for Q1 (default: auto-detected `data/indoor.png` or `data/indoor.jpg`).
- `--outdoor PATH`: Path to outdoor image file for Q1 (default: auto-detected `data/outdoor.png` or `data/outdoor.jpg`).
- `--q2-image PATH`: Path to input image for Q2 (default: auto-detected `data/practicum 1.jpg` or `data/practicum_1.jpg`).
- `--ksize INT`: Aperture size for Laplacian operator (positive odd integer, default: `3`).
- `--output PATH`: Destination directory for generated outputs (default: `outputs`).
- `--dpi INT`: Resolution (DPI) for exported figures (default: `300`).
- `--no-raw`: Save composite comparison figures only, skipping individual raw image files.
- `-v, --verbose`: Enable detailed debug logging during execution.

### Running Automated Tests
Run unit tests across color conversion, histogram calculations, Fourier transforms, Laplacian filtering, and I/O validation:
```bash
python -m pytest tests -v
```

## 4. Experiment Outputs

### Question 1: Illumination & Color Space Figures
- `outputs/figures/01_rgb_overview.png`: Visualizes indoor and outdoor inputs in RGB.
- `outputs/figures/02_indoor_lab_channels.png`: RGB composite alongside isolated L, A, and B channels for indoor scene.
- `outputs/figures/02_outdoor_lab_channels.png`: RGB composite alongside isolated L, A, and B channels for outdoor scene.
- `outputs/figures/03_indoor_histogram_equalization.png`: Indoor before-and-after comparison of RGB, L channel, and L histogram with CDF.
- `outputs/figures/03_outdoor_histogram_equalization.png`: Outdoor before-and-after comparison of RGB, L channel, and L histogram with CDF.
- `outputs/figures/04_indoor_vs_outdoor_comparison.png`: Cross-scene comparison matrix showing original and equalized images side by side.
- `outputs/figures/05_illumination_distribution_comparison.png`: Comparative histogram curves analyzing illumination changes across both scenes.

### Question 2: Frequency & Spatial Filtering Figures
- `outputs/figures/06_fourier_transform.png`: Original image alongside centered 2D Fourier magnitude spectrum (logarithmic scaling).
- `outputs/figures/07_laplacian_filter.png`: Original image alongside scaled Laplacian response (zero centered at mid-gray) and absolute edge response (|∇²I|).

### Processed Images
- `outputs/processed_images/`: Individual equalized, channel, magnitude spectrum, and Laplacian edge image files.

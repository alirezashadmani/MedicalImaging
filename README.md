# Medical Image Segmentation Toolkit

A modular Python toolkit for medical image segmentation, focusing on bone segmentation from CT scans. Implements and compares multiple classical segmentation algorithms including active contours, level sets, watershed, and edge detection methods.

## Project Structure

```
.
├── config.py                  # Centralized configuration and default paths
├── utils.py                   # Shared utilities (image loading, conversion)
├── requirements.txt           # Python dependencies
│
├── edge_detection/            # Edge detection algorithms
│   ├── canny_skimage.py       # Canny edge detection (scikit-image)
│   ├── canny_manual.py        # Canny from scratch (Gaussian, gradient, NMS, thresholding)
│   └── sobel.py               # Sobel and Laplacian edge detection (OpenCV)
│
├── active_contour/            # Active contour and level-set methods
│   ├── parametric_snake.py    # Parametric active contour (snake) via scikit-image
│   ├── morph_acwe.py          # Morphological Chan-Vese (region-based level set)
│   ├── morph_gac.py           # Morphological Geodesic Active Contour (edge-based)
│   ├── morph_acwe_3d.py       # 3D Chan-Vese with marching cubes visualization
│   └── level_set_evolution.py # Manual level-set PDE solver with curvature/balloon forces
│
├── watershed/                 # Watershed segmentation
│   ├── watershed_opencv.py    # OpenCV pipeline (Otsu + distance transform + watershed)
│   └── watershed_skimage.py   # scikit-image pipeline (mean-shift + peak local max + watershed)
│
├── itk_segmentation/          # ITK-based segmentation
│   ├── geodesic_active_contour.py  # Full ITK GAC pipeline (smoothing → gradient → sigmoid → fast marching → GAC)
│   └── watershed_itk.py       # ITK watershed with gradient magnitude preprocessing
│
├── medical_pipeline/          # End-to-end medical imaging pipelines
│   └── nifti_segmentation.py  # NIfTI volume segmentation with connected components + active contour
│
└── visualization_3d/          # 3D visualization
    └── bone_viewer.py         # PyVista-based volume rendering and multi-slice orthogonal views
```

## Installation

```bash
git clone https://github.com/alirezashadmani/MedicalImaging.git
cd MedicalImaging
pip install -r requirements.txt
```

**Optional dependencies** (install only if needed):

```bash
pip install itk              # For itk_segmentation module
pip install nibabel cc3d     # For medical_pipeline module
pip install pyvista SimpleITK  # For visualization_3d module
pip install PyMCubes         # For 3D level-set visualization
```

## Usage

Every script supports `--help` for full argument documentation. All image paths are passed via CLI arguments (no hardcoded paths).

### Edge Detection

```bash
# Canny with multiple sigma comparison
python -m edge_detection.canny_skimage --image data/scan.png --sigma 1.0 3.0 5.0

# Manual Canny implementation (shows all intermediate stages)
python -m edge_detection.canny_manual --image data/scan.png --sigma 2.5 --low 40 --high 100

# Sobel + Laplacian (optionally with Canny overlay)
python -m edge_detection.sobel --image data/scan.png --with-canny
```

### Active Contour / Level Set

```bash
# Parametric snake with circular initialization
python -m active_contour.parametric_snake --image data/scan.png --center 500 950 --radius 50

# Morphological Chan-Vese (region-based, no edge function needed)
python -m active_contour.morph_acwe --image data/scan.png --smoothing 5 --lambda1 3 --lambda2 1

# Morphological Geodesic Active Contour (edge-based)
python -m active_contour.morph_gac --image data/scan.png --alpha 100 --sigma 4.5

# 3D level set on volumetric data
python -m active_contour.morph_acwe_3d --volume data/confocal.npy --iterations 150

# Manual level-set evolution with curvature and balloon forces
python -m active_contour.level_set_evolution --image data/scan.png --iterations 50 --balloon 1.0
```

### Watershed Segmentation

```bash
# OpenCV-based (with full pipeline visualization)
python -m watershed.watershed_opencv --image data/scan.png --show-steps

# scikit-image-based (mean-shift + distance transform)
python -m watershed.watershed_skimage --image data/scan.png --min-distance 20
```

### ITK Segmentation

```bash
# Geodesic Active Contour (with optional intermediate outputs)
python -m itk_segmentation.geodesic_active_contour \
    --image data/scan.png --output result.png \
    --seed 56 92 --propagation 7.0 --save-intermediates

# ITK Watershed
python -m itk_segmentation.watershed_itk \
    --image data/scan.png --output result.png \
    --threshold 0.005 --level 0.5
```

### Medical Pipeline (NIfTI)

```bash
# Full pipeline: NIfTI volume → connected components → convex hull → active contour
python -m medical_pipeline.nifti_segmentation \
    --img-path data/case.nii.gz \
    --seg-path data/seg.nii.gz \
    --full-path data/full_view.nii.gz \
    --obj-id 4 --alpha 18 --beta 0.01
```

### 3D Visualization

```bash
# Volume rendering (bone + ground truth overlay)
python -m visualization_3d.bone_viewer volume \
    --bone-image data/bone.nii.gz --gt-image data/gt.nii.gz

# Orthogonal slice viewer
python -m visualization_3d.bone_viewer slices --nrrd-file data/volume.nrrd

# Combined 4-panel view (XY, XZ, ZY slices + 3D volume)
python -m visualization_3d.bone_viewer combined \
    --nrrd-file data/slices.nrrd --volume-file data/volume.nrrd
```

### Saving Output

All scripts support `--output` to save figures to disk instead of displaying:

```bash
python -m watershed.watershed_opencv --image data/scan.png --show-steps --output results/watershed.png
```

## Algorithms Overview

| Algorithm | Type | Module | Best For |
|-----------|------|--------|----------|
| Canny | Edge detection | `edge_detection` | Boundary detection, preprocessing |
| Sobel / Laplacian | Edge detection | `edge_detection` | Gradient-based edge maps |
| Parametric Snake | Active contour | `active_contour` | Smooth boundary segmentation |
| Chan-Vese (ACWE) | Level set | `active_contour` | Region-based segmentation without strong edges |
| Geodesic AC (GAC) | Level set | `active_contour` | Edge-based segmentation with balloon force |
| Watershed (OpenCV) | Region growing | `watershed` | Over-segmentation, marker-based |
| Watershed (skimage) | Region growing | `watershed` | Distance-transform-based seed detection |
| ITK GAC | Level set | `itk_segmentation` | Clinical-grade geodesic active contour |
| ITK Watershed | Region growing | `itk_segmentation` | ITK-native watershed with colormap output |

## Tech Stack

- **Core**: NumPy, SciPy, OpenCV, scikit-image, Matplotlib
- **Level sets**: morphsnakes
- **Medical imaging**: ITK, nibabel, SimpleITK, cc3d
- **3D visualization**: PyVista, VTK

## License

This project is for research and educational purposes.

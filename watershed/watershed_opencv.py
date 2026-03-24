"""
Watershed segmentation using OpenCV.

Full pipeline:
  1. Otsu thresholding
  2. Morphological noise removal
  3. Background/foreground separation via distance transform
  4. Marker-based watershed

Consolidated from:
  - Amir/watersheld1.py
  - Amir/watershed2.py
  - Amir/test/watershed_opencv.py

Usage:
    python -m watershed.watershed_opencv --image path/to/image.png
    python -m watershed.watershed_opencv --image path/to/image.png --dilate-iter 3 --fg-threshold 0.7 --show-steps
"""

import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEFAULT_IMAGE, ensure_output_dir
from utils import bgr_to_rgb


def watershed_opencv(img, kernel_size=3, morph_iterations=2,
                     dilate_iterations=3, fg_threshold=0.7):
    """
    Run the full OpenCV watershed pipeline.

    Parameters
    ----------
    img : ndarray
        Input BGR image.
    kernel_size : int
        Morphological kernel size.
    morph_iterations : int
        Opening iterations for noise removal.
    dilate_iterations : int
        Dilation iterations for background estimation.
    fg_threshold : float
        Fraction of max distance for foreground thresholding (0-1).

    Returns
    -------
    result : dict
        Dictionary with intermediate results:
        'gray', 'thresh', 'opening', 'sure_bg', 'dist_transform',
        'sure_fg', 'unknown', 'markers', 'segmented'.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)

    sure_bg = cv2.dilate(opening, kernel, iterations=dilate_iterations)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, fg_threshold * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    segmented = img.copy()
    markers = cv2.watershed(segmented, markers)
    segmented[markers == -1] = [0, 255, 0]

    return {
        "gray": gray,
        "thresh": thresh,
        "opening": opening,
        "sure_bg": sure_bg,
        "dist_transform": dist_transform,
        "sure_fg": sure_fg,
        "unknown": unknown,
        "markers": markers,
        "segmented": segmented,
    }


def plot_pipeline(img_rgb, result):
    """Display the full watershed processing pipeline."""
    titles = [
        "Input Image", "Otsu Threshold", "Morphological Opening",
        "Dilation (Background)", "Distance Transform", "Foreground Threshold",
        "Unknown Region", "Watershed Result",
    ]
    images = [
        img_rgb, result["thresh"], result["opening"],
        result["sure_bg"], result["dist_transform"], result["sure_fg"],
        result["unknown"], bgr_to_rgb(result["segmented"]),
    ]
    cmaps = [None, "gray", "gray", "gray", "gray", "gray", "gray", None]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, title, image, cmap in zip(axes.flat, titles, images, cmaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


def plot_result_only(img_rgb, result):
    """Display just input and segmented output side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image")
    axes[1].imshow(bgr_to_rgb(result["segmented"]))
    axes[1].set_title("Watershed Segmentation")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Watershed segmentation (OpenCV)")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--kernel-size", type=int, default=3, help="Morphological kernel size")
    parser.add_argument("--morph-iter", type=int, default=2, help="Opening iterations")
    parser.add_argument("--dilate-iter", type=int, default=3, help="Dilation iterations")
    parser.add_argument("--fg-threshold", type=float, default=0.7,
                        help="Foreground threshold fraction (0-1)")
    parser.add_argument("--show-steps", action="store_true",
                        help="Show all intermediate processing steps")
    parser.add_argument("--output", default=None, help="Save figure to file")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")

    result = watershed_opencv(
        img,
        kernel_size=args.kernel_size,
        morph_iterations=args.morph_iter,
        dilate_iterations=args.dilate_iter,
        fg_threshold=args.fg_threshold,
    )

    img_rgb = bgr_to_rgb(img)
    if args.show_steps:
        fig = plot_pipeline(img_rgb, result)
    else:
        fig = plot_result_only(img_rgb, result)

    if args.output:
        ensure_output_dir(os.path.dirname(args.output) or None)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

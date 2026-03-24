"""
Sobel and Laplacian edge detection using OpenCV.

Consolidated from:
  - Amir/edge detection/sobel_edge_detection.py
  - Amir/edge detection/edge_finder.py

Usage:
    python -m edge_detection.sobel --image path/to/image.png
    python -m edge_detection.sobel --image path/to/image.png --with-canny --canny-low 50 --canny-high 170
"""

import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEFAULT_IMAGE


def sobel_laplacian(gray_img):
    """Compute Sobel (X, Y) and Laplacian edge maps."""
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    return laplacian, sobel_x, sobel_y


def plot_edges(gray_img, laplacian, sobel_x, sobel_y, canny_edges=None):
    """Display edge detection results."""
    n_plots = 5 if canny_edges is not None else 4
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))

    axes[0].imshow(gray_img, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(laplacian, cmap="gray")
    axes[1].set_title("Laplacian")
    axes[2].imshow(sobel_x, cmap="gray")
    axes[2].set_title("Sobel X")
    axes[3].imshow(sobel_y, cmap="gray")
    axes[3].set_title("Sobel Y")

    if canny_edges is not None:
        axes[4].imshow(canny_edges, cmap="gray")
        axes[4].set_title("Canny")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Sobel & Laplacian edge detection")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--with-canny", action="store_true", help="Also show Canny edges")
    parser.add_argument("--canny-low", type=int, default=50, help="Canny low threshold")
    parser.add_argument("--canny-high", type=int, default=170, help="Canny high threshold")
    parser.add_argument("--output", default=None, help="Save figure to file")
    args = parser.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")

    laplacian, sobel_x, sobel_y = sobel_laplacian(gray)

    canny = None
    if args.with_canny:
        canny = cv2.Canny(gray, args.canny_low, args.canny_high)

    fig = plot_edges(gray, laplacian, sobel_x, sobel_y, canny)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

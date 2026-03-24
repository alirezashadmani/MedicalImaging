"""
Canny edge detection using scikit-image.

Consolidated from:
  - Amir/edge detection/canny.py
  - Amir/edge detection/feature_canny.py

Usage:
    python -m edge_detection.canny_skimage --image path/to/image.png
    python -m edge_detection.canny_skimage --image path/to/image.png --sigma 1.0 3.0 5.0
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.feature

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEFAULT_IMAGE


def canny_edge_detect(image_gray, sigma=2.4, low_threshold=0.05, high_threshold=0.1):
    """Apply Canny edge detection on a grayscale image."""
    return skimage.feature.canny(
        image=image_gray,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )


def compare_sigma_values(image_gray, sigma_values):
    """Compare Canny results across multiple sigma values."""
    n = len(sigma_values)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

    axes[0].imshow(image_gray, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, sigma in enumerate(sigma_values):
        edges = canny_edge_detect(image_gray, sigma=sigma)
        axes[i + 1].imshow(edges, cmap="gray")
        axes[i + 1].set_title(f"Canny, σ={sigma}")
        axes[i + 1].axis("off")

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Canny edge detection (scikit-image)")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--sigma", nargs="+", type=float, default=[1.0, 3.0, 5.0],
                        help="Sigma values to compare (default: 1.0 3.0 5.0)")
    parser.add_argument("--low", type=float, default=0.05, help="Low threshold")
    parser.add_argument("--high", type=float, default=0.1, help="High threshold")
    parser.add_argument("--output", default=None, help="Save figure to file")
    args = parser.parse_args()

    image = skimage.io.imread(fname=args.image, as_gray=True)

    if len(args.sigma) == 1:
        edges = canny_edge_detect(image, sigma=args.sigma[0],
                                  low_threshold=args.low, high_threshold=args.high)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(edges, cmap="gray")
        axes[1].set_title(f"Canny (σ={args.sigma[0]})")
        axes[1].axis("off")
        fig.tight_layout()
    else:
        fig = compare_sigma_values(image, args.sigma)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

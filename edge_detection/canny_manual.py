"""
Manual Canny edge detection implementation from scratch.

Implements the full Canny pipeline:
  1. Gaussian smoothing
  2. Gradient computation (Prewitt-like kernels)
  3. Non-maximum suppression
  4. Double thresholding (weak/strong edges)

Consolidated from:
  - Amir/edge detection/canny_edge_detector.py

Usage:
    python -m edge_detection.canny_manual --image path/to/image.png
    python -m edge_detection.canny_manual --image path/to/image.png --sigma 2.5 --low 40 --high 100
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEFAULT_IMAGE, ensure_output_dir


def canny_edge_manual(img, sigma, low_thresh, high_thresh):
    """
    Full manual Canny edge detection.

    Parameters
    ----------
    img : ndarray
        Grayscale input image (float).
    sigma : float
        Standard deviation for Gaussian smoothing.
    low_thresh : float
        Threshold for weak edges.
    high_thresh : float
        Threshold for strong edges.

    Returns
    -------
    gauss : ndarray
        Gaussian-smoothed image.
    magnitude : ndarray
        Gradient magnitude.
    weak : ndarray
        Weak edge map.
    strong : ndarray
        Strong (binary) edge map.
    """
    # Step 1: Gaussian smoothing
    size = int(2 * np.ceil(3 * sigma) + 1)
    x, y = np.meshgrid(
        np.arange(-size / 2 + 1, size / 2 + 1),
        np.arange(-size / 2 + 1, size / 2 + 1),
    )
    normal = 1.0 / (2.0 * np.pi * sigma ** 2)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2)) / normal

    kern_size = kernel.shape[0]
    gauss = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0] - (kern_size - 1)):
        for j in range(img.shape[1] - (kern_size - 1)):
            window = img[i : i + kern_size, j : j + kern_size] * kernel
            gauss[i, j] = np.sum(window)

    # Step 2: Gradient computation
    edge_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    kern_size = 3
    gx = np.zeros_like(gauss, dtype=float)
    gy = np.zeros_like(gauss, dtype=float)

    for i in range(gauss.shape[0] - (kern_size - 1)):
        for j in range(gauss.shape[1] - (kern_size - 1)):
            window = gauss[i : i + kern_size, j : j + kern_size]
            gx[i, j] = np.sum(window * edge_kernel.T)
            gy[i, j] = np.sum(window * edge_kernel)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.degrees(np.arctan2(gy, gx))
    theta[theta < 0] += 180

    # Step 3: Non-maximum suppression
    nms = np.copy(magnitude)
    for i in range(1, theta.shape[0] - 1):
        for j in range(1, theta.shape[1] - 1):
            angle = theta[i, j]
            if angle <= 22.5 or angle > 157.5:
                neighbors = (magnitude[i - 1, j], magnitude[i + 1, j])
            elif 22.5 < angle <= 67.5:
                neighbors = (magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])
            elif 67.5 < angle <= 112.5:
                neighbors = (magnitude[i, j - 1], magnitude[i, j + 1])
            else:
                neighbors = (magnitude[i + 1, j - 1], magnitude[i - 1, j + 1])

            if magnitude[i, j] <= neighbors[0] or magnitude[i, j] <= neighbors[1]:
                nms[i, j] = 0

    # Step 4: Double thresholding
    weak = np.copy(nms)
    weak[weak < low_thresh] = 0
    weak[weak > high_thresh] = 0

    strong = np.copy(nms)
    strong[strong < high_thresh] = 0
    strong[strong >= high_thresh] = 1

    return gauss, magnitude, weak, strong


def plot_results(img, gauss, magnitude, weak, strong):
    """Display all intermediate Canny edge detection stages."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(gauss, cmap="gray")
    axes[0, 1].set_title("Gaussian Smoothed")

    axes[0, 2].imshow(magnitude, cmap="gray")
    axes[0, 2].set_title("Gradient Magnitude")

    axes[1, 0].imshow(weak, cmap="gray")
    axes[1, 0].set_title("Weak Edges")

    axes[1, 1].imshow(strong, cmap="gray")
    axes[1, 1].set_title("Strong Edges")

    axes[1, 2].imshow(255 - strong, cmap="gray")
    axes[1, 2].set_title("Strong Edges (inverted)")

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Manual Canny edge detection")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--sigma", type=float, default=2.5, help="Gaussian sigma")
    parser.add_argument("--low", type=float, default=40, help="Low threshold")
    parser.add_argument("--high", type=float, default=100, help="High threshold")
    parser.add_argument("--output", default=None, help="Save figure to file")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")
    img_gray = color.rgb2gray(img)

    gauss, magnitude, weak, strong = canny_edge_manual(
        img_gray, args.sigma, args.low, args.high
    )

    fig = plot_results(img_gray, gauss, magnitude, weak, strong)

    if args.output:
        ensure_output_dir(os.path.dirname(args.output) or None)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

"""
Parametric active contour (snake) segmentation using scikit-image.

Supports circular initialization with configurable alpha/beta/gamma parameters.

Consolidated from:
  - Amir/activecountour.py
  - Amir/watersheld.py (snake portion)
  - Amir/Alireza/LevelSet_Segmentation/ActiveContour.py

Usage:
    python -m active_contour.parametric_snake --image path/to/image.png
    python -m active_contour.parametric_snake --image path/to/image.png --center 500 950 --radius 50 --alpha 0.05
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import active_contour

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEFAULT_IMAGE, ensure_output_dir
from utils import normalize_uint8


def create_circle_init(center_r, center_c, radius, n_points=450):
    """Create a circular initial contour."""
    s = np.linspace(0, 2 * np.pi, n_points)
    r = center_r + radius * np.sin(s)
    c = center_c + radius * np.cos(s)
    return np.array([r, c]).T


def run_active_contour(image, init_contour, alpha=0.05, beta=10, gamma=0.001):
    """
    Run parametric active contour segmentation.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale or color).
    init_contour : ndarray of shape (N, 2)
        Initial contour points [row, col].
    alpha : float
        Snake length shape parameter (higher = smoother).
    beta : float
        Snake smoothness shape parameter (higher = smoother).
    gamma : float
        Step size for optimization.

    Returns
    -------
    snake : ndarray of shape (N, 2)
        Final contour points [row, col].
    """
    return active_contour(image, init_contour, alpha=alpha, beta=beta, gamma=gamma)


def plot_result(image, init_contour, snake, output_path=None):
    """Visualize original contour and final snake on the image."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap=plt.cm.gray)
    ax.plot(init_contour[:, 1], init_contour[:, 0], "--r", lw=2, label="Initial")
    ax.plot(snake[:, 1], snake[:, 0], "-b", lw=1.5, label="Snake")
    ax.legend(loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis([0, image.shape[1], image.shape[0], 0])
    fig.tight_layout()

    if output_path:
        ensure_output_dir(os.path.dirname(output_path) or None)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Parametric active contour (snake)")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--center", nargs=2, type=int, default=[500, 950],
                        help="Center of initial circle [row col]")
    parser.add_argument("--radius", type=int, default=50, help="Radius of initial circle")
    parser.add_argument("--alpha", type=float, default=0.05, help="Length parameter")
    parser.add_argument("--beta", type=float, default=10.0, help="Smoothness parameter")
    parser.add_argument("--gamma", type=float, default=0.001, help="Step size")
    parser.add_argument("--output", default=None, help="Save figure to file")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")
    img = normalize_uint8(img)

    init = create_circle_init(args.center[0], args.center[1], args.radius)
    snake = run_active_contour(img, init, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    plot_result(img, init, snake, output_path=args.output)


if __name__ == "__main__":
    main()

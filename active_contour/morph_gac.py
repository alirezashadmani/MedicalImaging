"""
Morphological Geodesic Active Contour (GAC) segmentation.

Edge-based level-set method using an inverse Gaussian gradient as the
stopping function. Uses the morphsnakes library.

Consolidated from:
  - Amir/ACM_morphsnake.py

Usage:
    python -m active_contour.morph_gac --image path/to/image.png
    python -m active_contour.morph_gac --image path/to/image.png --alpha 100 --sigma 4.5
"""

import argparse
import logging
import os

import numpy as np
from imageio.v2 import imread
import matplotlib
from matplotlib import pyplot as plt

import morphsnakes as ms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEFAULT_IMAGE, ensure_output_dir
from utils import rgb2gray

if os.environ.get("DISPLAY", "") == "":
    logging.warning("No display found. Using non-interactive Agg backend.")
    matplotlib.use("Agg")


def visual_callback_2d(background, params, fig=None, output_dir=None):
    """Create a callback for visualizing GAC level-set evolution."""
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):
        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors="r")
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

        if output_dir:
            fname = (
                f"gac_s{params['smoothing']}"
                f"_th{params['threshold']}"
                f"_b{params['balloon']}.png"
            )
            fig.savefig(os.path.join(output_dir, fname), dpi=100, bbox_inches="tight")

    return callback


def run_morph_gac(image_path, alpha=100, sigma=4.5, smoothing=4,
                  threshold=0.5, balloon=3, iterations=50,
                  init_center=None, init_radius=11, output_dir=None):
    """
    Run Morphological Geodesic Active Contour segmentation.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    alpha : float
        Alpha for inverse Gaussian gradient.
    sigma : float
        Sigma for inverse Gaussian gradient.
    smoothing : int
        Number of smoothing iterations.
    threshold : float
        Level-set threshold.
    balloon : float
        Balloon force magnitude.
    iterations : int
        Number of evolution iterations.
    init_center : tuple of (row, col), optional
        Center of initial circle. Defaults to image center.
    init_radius : int
        Radius of the initial circle level set.
    output_dir : str, optional
        Directory to save output frames.
    """
    img = imread(image_path)[..., 0] / 255.0

    gimg = ms.inverse_gaussian_gradient(img, alpha=alpha, sigma=sigma)

    if init_center is None:
        init_center = (img.shape[0] // 2, img.shape[1] // 2)

    init_ls = ms.circle_level_set(img.shape, init_center, init_radius)

    params = {"smoothing": smoothing, "threshold": threshold, "balloon": balloon}
    callback = visual_callback_2d(img, params, output_dir=output_dir)

    ms.morphological_geodesic_active_contour(
        gimg,
        iterations=iterations,
        init_level_set=init_ls,
        smoothing=smoothing,
        threshold=threshold,
        balloon=balloon,
        iter_callback=callback,
    )


def main():
    parser = argparse.ArgumentParser(description="Morphological GAC segmentation")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--alpha", type=float, default=100, help="Inverse Gaussian alpha")
    parser.add_argument("--sigma", type=float, default=4.5, help="Inverse Gaussian sigma")
    parser.add_argument("--smoothing", type=int, default=4, help="Smoothing iterations")
    parser.add_argument("--threshold", type=float, default=0.5, help="Level-set threshold")
    parser.add_argument("--balloon", type=float, default=3.0, help="Balloon force")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--center", nargs=2, type=int, default=None,
                        help="Init circle center [row col]")
    parser.add_argument("--radius", type=int, default=11, help="Init circle radius")
    parser.add_argument("--output-dir", default=None, help="Output directory for frames")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    center = tuple(args.center) if args.center else None

    if args.output_dir:
        ensure_output_dir(args.output_dir)

    run_morph_gac(
        args.image,
        alpha=args.alpha,
        sigma=args.sigma,
        smoothing=args.smoothing,
        threshold=args.threshold,
        balloon=args.balloon,
        iterations=args.iterations,
        init_center=center,
        init_radius=args.radius,
        output_dir=args.output_dir,
    )

    logging.info("Done.")
    plt.show()


if __name__ == "__main__":
    main()

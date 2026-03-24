"""
Morphological Chan-Vese (ACWE) level-set segmentation.

Uses the morphsnakes library for region-based active contour without edges.
Supports real-time visualization of the evolving level set with configurable
smoothing, lambda1, and lambda2 parameters.

Consolidated from:
  - Amir/levelset_morphsnake.py
  - Amir/Alireza/LevelSet_Segmentation/MorphACWE.py

Usage:
    python -m active_contour.morph_acwe --image path/to/image.png
    python -m active_contour.morph_acwe --image path/to/image.png --smoothing 5 --lambda1 3 --lambda2 1
    python -m active_contour.morph_acwe --image path/to/image.png --crop 0 900 700 1100
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

# Use non-interactive backend if no display is available
if os.environ.get("DISPLAY", "") == "":
    logging.warning("No display found. Using non-interactive Agg backend.")
    matplotlib.use("Agg")


def visual_callback_2d(background, params, fig=None, output_dir=None):
    """
    Create a callback for visualizing level-set evolution.

    Parameters
    ----------
    background : ndarray
        Background image to overlay contours on.
    params : dict
        Parameters dict with keys 'smoothing', 'lambda1', 'lambda2'.
    fig : matplotlib.figure.Figure, optional
        Reusable figure.
    output_dir : str, optional
        Directory to save intermediate frames.

    Returns
    -------
    callback : callable
        Function that receives a level set and updates the plot.
    """
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
                f"levelset_s{params['smoothing']}"
                f"_l1_{params['lambda1']}"
                f"_l2_{params['lambda2']}.png"
            )
            fig.savefig(os.path.join(output_dir, fname), dpi=100, bbox_inches="tight")

    return callback


def run_morph_acwe(image_path, smoothing=3, lambda1=1, lambda2=1,
                   iterations=200, init_center=None, init_radius=4,
                   crop=None, output_dir=None):
    """
    Run Morphological Chan-Vese segmentation.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    smoothing : int
        Number of smoothing iterations per level-set step.
    lambda1 : float
        Weight for the 'inside' region.
    lambda2 : float
        Weight for the 'outside' region.
    iterations : int
        Number of evolution iterations.
    init_center : tuple of (row, col), optional
        Center of the initial circle. Defaults to image center.
    init_radius : int
        Radius of the initial circle level set.
    crop : tuple of (top, bottom, left, right), optional
        Crop region [top:bottom, left:right].
    output_dir : str, optional
        Directory to save output frames.
    """
    imgcolor = imread(image_path) / 255.0

    if crop:
        top, bottom, left, right = crop
        imgcolor = imgcolor[top:bottom, left:right]

    img = rgb2gray(imgcolor)

    if init_center is None:
        init_center = (img.shape[0] // 2, img.shape[1] // 2)

    init_ls = ms.circle_level_set(img.shape, init_center, init_radius)

    params = {"smoothing": smoothing, "lambda1": lambda1, "lambda2": lambda2}
    callback = visual_callback_2d(imgcolor, params, output_dir=output_dir)

    ms.morphological_chan_vese(
        img,
        iterations=iterations,
        init_level_set=init_ls,
        smoothing=smoothing,
        lambda1=lambda1,
        lambda2=lambda2,
        iter_callback=callback,
    )


def main():
    parser = argparse.ArgumentParser(description="Morphological Chan-Vese (ACWE)")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--smoothing", type=int, default=3, help="Smoothing iterations")
    parser.add_argument("--lambda1", type=float, default=1.0, help="Inside weight")
    parser.add_argument("--lambda2", type=float, default=1.0, help="Outside weight")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations")
    parser.add_argument("--center", nargs=2, type=int, default=None,
                        help="Init circle center [row col]")
    parser.add_argument("--radius", type=int, default=4, help="Init circle radius")
    parser.add_argument("--crop", nargs=4, type=int, default=None,
                        help="Crop region [top bottom left right]")
    parser.add_argument("--output-dir", default=None, help="Output directory for frames")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    center = tuple(args.center) if args.center else None
    crop = tuple(args.crop) if args.crop else None

    if args.output_dir:
        ensure_output_dir(args.output_dir)

    run_morph_acwe(
        args.image,
        smoothing=args.smoothing,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        iterations=args.iterations,
        init_center=center,
        init_radius=args.radius,
        crop=crop,
        output_dir=args.output_dir,
    )

    logging.info("Done.")
    plt.show()


if __name__ == "__main__":
    main()

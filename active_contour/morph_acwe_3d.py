"""
3D Morphological Chan-Vese (ACWE) level-set segmentation.

Extends Chan-Vese to volumetric data with 3D isosurface visualization
using marching cubes.

Consolidated from:
  - Amir/morphsnake_3D.py

Requirements:
  - morphsnakes
  - PyMCubes (pip install PyMCubes)

Usage:
    python -m active_contour.morph_acwe_3d --volume path/to/confocal.npy
    python -m active_contour.morph_acwe_3d --volume path/to/confocal.npy --smoothing 1 --lambda1 1 --lambda2 2
"""

import argparse
import logging
import os

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import morphsnakes as ms

if os.environ.get("DISPLAY", "") == "":
    logging.warning("No display found. Using non-interactive Agg backend.")
    matplotlib.use("Agg")


def visual_callback_3d(fig=None, plot_each=1):
    """
    Create a callback for 3D level-set evolution visualization.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Reusable figure.
    plot_each : int
        Update the plot every `plot_each` iterations.

    Returns
    -------
    callback : callable
        Callback that renders the 3D isosurface of the level set.
    """
    try:
        import mcubes
    except ImportError:
        raise ImportError("PyMCubes is required: pip install PyMCubes")

    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111, projection="3d")
    plt.pause(0.001)

    counter = [-1]

    def callback(levelset):
        counter[0] += 1
        if (counter[0] % plot_each) != 0:
            return
        if ax.collections:
            del ax.collections[0]
        coords, triangles = mcubes.marching_cubes(levelset, 0.5)
        ax.plot_trisurf(
            coords[:, 0], coords[:, 1], coords[:, 2], triangles=triangles
        )
        plt.pause(0.1)

    return callback


def run_morph_acwe_3d(volume_path, smoothing=1, lambda1=1, lambda2=2,
                      iterations=150, init_center=None, init_radius=25,
                      plot_each=20):
    """
    Run 3D Morphological Chan-Vese segmentation.

    Parameters
    ----------
    volume_path : str
        Path to a .npy volume file.
    smoothing : int
        Number of smoothing iterations.
    lambda1 : float
        Weight for the 'inside' region.
    lambda2 : float
        Weight for the 'outside' region.
    iterations : int
        Number of evolution iterations.
    init_center : tuple of (z, y, x), optional
        Center of initial sphere. Defaults to volume center.
    init_radius : int
        Radius of the initial sphere level set.
    plot_each : int
        Visualize every N iterations.
    """
    img = np.load(volume_path)

    if init_center is None:
        init_center = tuple(s // 2 for s in img.shape)

    init_ls = ms.circle_level_set(img.shape, init_center, init_radius)
    callback = visual_callback_3d(plot_each=plot_each)

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
    parser = argparse.ArgumentParser(description="3D Morphological Chan-Vese (ACWE)")
    parser.add_argument("--volume", required=True, help="Path to .npy volume file")
    parser.add_argument("--smoothing", type=int, default=1, help="Smoothing iterations")
    parser.add_argument("--lambda1", type=float, default=1.0, help="Inside weight")
    parser.add_argument("--lambda2", type=float, default=2.0, help="Outside weight")
    parser.add_argument("--iterations", type=int, default=150, help="Number of iterations")
    parser.add_argument("--center", nargs=3, type=int, default=None,
                        help="Init sphere center [z y x]")
    parser.add_argument("--radius", type=int, default=25, help="Init sphere radius")
    parser.add_argument("--plot-each", type=int, default=20, help="Plot every N iterations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    center = tuple(args.center) if args.center else None

    run_morph_acwe_3d(
        args.volume,
        smoothing=args.smoothing,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        iterations=args.iterations,
        init_center=center,
        init_radius=args.radius,
        plot_each=args.plot_each,
    )

    logging.info("Done.")
    plt.show()


if __name__ == "__main__":
    main()

"""
Manual level-set evolution with curvature and balloon forces.

Implements a basic level-set PDE solver:
  - Stopping function g(I) = 1 / (1 + |grad(I)|^2)
  - Curvature (mean curvature of the level set)
  - Balloon force for expansion/contraction
  - Attachment term for edge attraction

Consolidated from:
  - Amir/test/test2.py

Usage:
    python -m active_contour.level_set_evolution --image path/to/image.png
    python -m active_contour.level_set_evolution --image path/to/image.png --iterations 50 --balloon 1.0
"""

import argparse
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import io

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEFAULT_IMAGE


def grad(x):
    """Compute gradient of array."""
    return np.array(np.gradient(x))


def norm(x, axis=0):
    """Compute L2 norm along axis."""
    return np.sqrt(np.sum(np.square(x), axis=axis))


def stopping_fun(x):
    """Edge stopping function: g(I) = 1 / (1 + |grad(I)|^2)."""
    return 1.0 / (1.0 + norm(grad(x)) ** 2)


def curvature(f):
    """Compute mean curvature of the level set function."""
    fy, fx = grad(f)
    n = np.sqrt(fx ** 2 + fy ** 2) + 1e-8
    nx = fx / n
    ny = fy / n
    fxy, fxx = grad(nx)
    fyy, fyx = grad(ny)
    return fxx + fyy


def dot(x, y, axis=0):
    """Element-wise dot product along axis."""
    return np.sum(x * y, axis=axis)


def default_phi(shape):
    """Initialize phi: -1 inside (5px from border), +1 outside."""
    phi = np.ones(shape[:2])
    phi[5:-5, 5:-5] = -1.0
    return phi


def evolve_level_set(image_path, iterations=20, balloon_force=1.0, dt=1.0,
                     gaussian_sigma=1.0):
    """
    Evolve a level set on the given image.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    iterations : int
        Number of evolution steps.
    balloon_force : float
        Magnitude of the balloon force (positive = expand).
    dt : float
        Time step.
    gaussian_sigma : float
        Sigma for initial Gaussian smoothing.

    Returns
    -------
    phi : ndarray
        Final level set function.
    img : ndarray
        Input image (preprocessed).
    """
    img = io.imread(image_path, as_gray=True).astype(float)
    img = img - np.mean(img)

    img_smooth = scipy.ndimage.gaussian_filter(img, gaussian_sigma)

    g = stopping_fun(img_smooth)
    dg = grad(g)

    phi = default_phi(img.shape)

    for i in range(iterations):
        dphi = grad(phi)
        dphi_norm = norm(dphi)
        kappa = curvature(phi)

        smoothing = g * kappa * dphi_norm
        balloon = g * dphi_norm * balloon_force
        attachment = dot(dphi, dg)

        dphi_t = smoothing + balloon + attachment
        phi = phi + dt * dphi_t

    return phi, img


def main():
    parser = argparse.ArgumentParser(description="Manual level-set evolution")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--iterations", type=int, default=20, help="Evolution steps")
    parser.add_argument("--balloon", type=float, default=1.0, help="Balloon force")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma")
    parser.add_argument("--output", default=None, help="Save figure to file")
    args = parser.parse_args()

    phi, img = evolve_level_set(
        args.image,
        iterations=args.iterations,
        balloon_force=args.balloon,
        dt=args.dt,
        gaussian_sigma=args.sigma,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input Image")
    axes[1].imshow(phi, cmap="RdBu")
    axes[1].set_title("Level Set (phi)")
    axes[2].imshow(img, cmap="gray")
    axes[2].contour(phi, [0], colors="r", linewidths=2)
    axes[2].set_title("Contour on Image")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

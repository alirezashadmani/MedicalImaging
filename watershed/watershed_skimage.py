"""
Watershed segmentation using scikit-image.

Pipeline:
  1. Mean-shift filtering for noise reduction
  2. Otsu thresholding
  3. Euclidean distance transform
  4. Peak local maxima for seed detection
  5. Marker-based watershed

Consolidated from:
  - Amir/Alireza/Watershed_Segmentation/watershed.py
  - Amir/Alireza/Watershed_Segmentation/watershed3.py
  - Amir/test/test_watershed.py

Usage:
    python -m watershed.watershed_skimage --image path/to/image.png
    python -m watershed.watershed_skimage --image path/to/image.png --min-distance 20 --spatial-radius 25
"""

import argparse
import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from matplotlib import pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEFAULT_IMAGE, ensure_output_dir
from utils import bgr_to_rgb


def watershed_skimage(img, spatial_radius=25, color_radius=55,
                      min_distance=20, draw_contours=True):
    """
    Run scikit-image watershed segmentation.

    Parameters
    ----------
    img : ndarray
        Input BGR image.
    spatial_radius : int
        Spatial radius for mean-shift filtering.
    color_radius : int
        Color radius for mean-shift filtering.
    min_distance : int
        Minimum distance between detected peaks.
    draw_contours : bool
        Whether to draw contours on the output image.

    Returns
    -------
    result : dict
        Dictionary with: 'shifted', 'thresh', 'distance', 'labels',
        'n_segments', 'segmented', 'total_area'.
    """
    shifted = cv2.pyrMeanShiftFiltering(img, spatial_radius, color_radius)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    distance = ndimage.distance_transform_edt(thresh)

    local_max = peak_local_max(distance, min_distance=min_distance, labels=thresh)

    # Create marker mask from peak coordinates
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(local_max.T)] = True
    markers = ndimage.label(mask, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance, markers, mask=thresh)

    n_segments = len(np.unique(labels)) - 1

    segmented = img.copy()
    total_area = 0

    if draw_contours:
        for label_val in np.unique(labels):
            if label_val == 0:
                continue

            region_mask = np.zeros(gray.shape, dtype="uint8")
            region_mask[labels == label_val] = 255

            cnts = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                total_area += area
                cv2.drawContours(segmented, [c], -1, (36, 255, 12), 2)

    return {
        "shifted": shifted,
        "thresh": thresh,
        "distance": distance,
        "labels": labels,
        "n_segments": n_segments,
        "segmented": segmented,
        "total_area": total_area,
    }


def plot_results(img, result):
    """Display the 4-panel watershed result."""
    titles = ["Original Image", "Binary Threshold", "Distance Transform", "Watershed Labels"]
    images = [bgr_to_rgb(img), result["thresh"], result["distance"], result["labels"]]
    cmaps = [None, "gray", "gray", "jet"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, title, image, cmap in zip(axes.flat, titles, images, cmaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Found {result['n_segments']} segments", fontsize=14)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Watershed segmentation (scikit-image)")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--spatial-radius", type=int, default=25,
                        help="Mean-shift spatial radius")
    parser.add_argument("--color-radius", type=int, default=55,
                        help="Mean-shift color radius")
    parser.add_argument("--min-distance", type=int, default=20,
                        help="Min distance between peak markers")
    parser.add_argument("--output", default=None, help="Save figure to file")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")

    result = watershed_skimage(
        img,
        spatial_radius=args.spatial_radius,
        color_radius=args.color_radius,
        min_distance=args.min_distance,
    )

    print(f"[INFO] {result['n_segments']} unique segments found")
    print(f"[INFO] Total contour area: {result['total_area']}")

    fig = plot_results(img, result)

    if args.output:
        ensure_output_dir(os.path.dirname(args.output) or None)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

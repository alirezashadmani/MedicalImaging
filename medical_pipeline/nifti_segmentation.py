"""
NIfTI-based medical image segmentation pipeline.

Workflow:
  1. Load NIfTI image and segmentation volumes
  2. Run connected component analysis on the full segmentation
  3. Extract a bounding box around a target object
  4. Compute a convex hull of the dilated segmentation
  5. Initialize and run an active contour from the hull vertices

Consolidated from:
  - Amir/initilize_with_segmentations.py

Requirements:
  - nibabel, cc3d, scipy, skimage

Usage:
    python -m medical_pipeline.nifti_segmentation \\
        --img-path data/case.nii.gz \\
        --seg-path data/seg_case.nii.gz \\
        --full-path data/complete_full_view-seg_case.nii.gz \\
        --obj-id 4

    python -m medical_pipeline.nifti_segmentation \\
        --img-path data/case.nii.gz \\
        --seg-path data/seg.nii.gz \\
        --full-path data/full.nii.gz \\
        --obj-id 4 --alpha 18 --beta 0.01 --w-line 200
"""

import argparse
import numpy as np
import nibabel as nib
import cc3d
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt


def load_nifti_volumes(img_path, seg_path, full_path):
    """Load NIfTI image, segmentation, and full-view volumes."""
    img = nib.load(img_path)
    seg = nib.load(seg_path)
    full = nib.load(full_path)
    return img, seg, full


def extract_object(full_data, obj_id, connectivity=26):
    """
    Extract a specific object from the full segmentation using connected components.

    Parameters
    ----------
    full_data : ndarray
        Full segmentation volume.
    obj_id : int
        Index of the target object (0-based, after sorting by z-centroid).
    connectivity : int
        Connectivity for connected components (6, 18, or 26).

    Returns
    -------
    obj_mask : ndarray (bool)
        Binary mask of the target object.
    labels_out : ndarray
        Connected component labels.
    centroids : list
        Sorted centroids.
    mask_labels : list
        Sorted mask labels.
    counts : list
        Sorted pixel counts.
    """
    labels_out = cc3d.connected_components(full_data, connectivity=connectivity)
    stats = cc3d.statistics(labels_out)
    centroids = stats["centroids"]
    mask_labels, counts = np.unique(labels_out, return_counts=True)

    # Sort by z-centroid (descending), skip background (index 0)
    sorted_data = sorted(
        zip(centroids[1:], mask_labels[1:], counts[1:]),
        key=lambda x: x[0][2],
        reverse=True,
    )
    centroids, mask_labels, counts = map(list, zip(*sorted_data))

    obj_label = mask_labels[obj_id]
    obj_mask = labels_out == obj_label

    return obj_mask, labels_out, centroids, mask_labels, counts


def compute_bounding_box(labels_out, obj_label, spacing, padding_mm=15):
    """Compute a padded bounding box around the target object."""
    normal_padding = np.ceil(np.divide([padding_mm] * 3, spacing)).astype(int)
    coordinates = np.argwhere(labels_out == obj_label)

    min_coords = coordinates.min(axis=0)
    max_coords = coordinates.max(axis=0)
    min_coords -= normal_padding
    max_coords += normal_padding

    min_coords = np.maximum(min_coords, [0, 0, 0])
    max_coords = np.minimum(max_coords, np.array(labels_out.shape) - 1)

    return min_coords, max_coords


def segment_with_active_contour(cut_img, cut_full, alpha=18, beta=0.01,
                                gamma=1.0, w_line=200, dilation_iterations=10):
    """
    Run active contour initialized from the convex hull of a dilated segmentation.

    Parameters
    ----------
    cut_img : ndarray
        2D image slice to segment.
    cut_full : ndarray
        2D binary mask from the full segmentation.
    alpha, beta, gamma : float
        Active contour parameters.
    w_line : float
        Line attraction weight.
    dilation_iterations : int
        Iterations for binary dilation of the mask.

    Returns
    -------
    snake : ndarray
        Final contour points [row, col].
    init_contour : ndarray
        Initial contour from convex hull.
    """
    dilated = ndimage.binary_dilation(cut_full, iterations=dilation_iterations)
    points = np.argwhere(dilated > 0)
    hull = ConvexHull(points)

    y = points[hull.vertices, 0]
    x = points[hull.vertices, 1]
    init_contour = np.array([y, x]).T

    snake = active_contour(
        image=cut_img,
        snake=init_contour,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        w_line=w_line,
    )
    return snake, init_contour


def run_pipeline(img_path, seg_path, full_path, obj_id=4,
                 connectivity=26, alpha=18, beta=0.01, gamma=1.0,
                 w_line=200, dilation_iterations=10, output=None):
    """Run the full NIfTI segmentation pipeline."""
    img, seg, full = load_nifti_volumes(img_path, seg_path, full_path)

    full_data = full.get_fdata()
    obj_mask, labels_out, centroids, mask_labels, counts = extract_object(
        full_data, obj_id, connectivity=connectivity
    )

    spacing = img.header.get_zooms()
    obj_label = mask_labels[obj_id]
    min_coords, max_coords = compute_bounding_box(labels_out, obj_label, spacing)

    obj_centroid = centroids[obj_id]
    img_data = img.get_fdata()

    # Extract sagittal slice at the object centroid
    sagittal_idx = int(obj_centroid[0])
    cut_img = img_data[
        sagittal_idx,
        min_coords[1] : max_coords[1] + 1,
        min_coords[2] : max_coords[2] + 1,
    ]
    cut_full = obj_mask[
        sagittal_idx,
        min_coords[1] : max_coords[1] + 1,
        min_coords[2] : max_coords[2] + 1,
    ]

    snake, init_contour = segment_with_active_contour(
        cut_img, cut_full,
        alpha=alpha, beta=beta, gamma=gamma,
        w_line=w_line, dilation_iterations=dilation_iterations,
    )

    # Visualize
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(cut_img, cmap=plt.cm.gray)
    ax.plot(init_contour[:, 1], init_contour[:, 0], "--r", lw=1, label="Init (hull)")
    ax.plot(snake[:, 1], snake[:, 0], "-b", lw=1, label="Snake")
    ax.legend(loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis([0, cut_img.shape[1], cut_img.shape[0], 0])
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="NIfTI segmentation with active contour initialization"
    )
    parser.add_argument("--img-path", required=True, help="Path to the image .nii.gz")
    parser.add_argument("--seg-path", required=True, help="Path to the segmentation .nii.gz")
    parser.add_argument("--full-path", required=True,
                        help="Path to the full-view segmentation .nii.gz")
    parser.add_argument("--obj-id", type=int, default=4, help="Object index (0-based)")
    parser.add_argument("--connectivity", type=int, default=26,
                        help="CC3D connectivity (6, 18, 26)")
    parser.add_argument("--alpha", type=float, default=18.0, help="Snake alpha")
    parser.add_argument("--beta", type=float, default=0.01, help="Snake beta")
    parser.add_argument("--gamma", type=float, default=1.0, help="Snake gamma")
    parser.add_argument("--w-line", type=float, default=200.0, help="Line weight")
    parser.add_argument("--dilation", type=int, default=10, help="Dilation iterations")
    parser.add_argument("--output", default=None, help="Save figure to file")
    args = parser.parse_args()

    run_pipeline(
        img_path=args.img_path,
        seg_path=args.seg_path,
        full_path=args.full_path,
        obj_id=args.obj_id,
        connectivity=args.connectivity,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        w_line=args.w_line,
        dilation_iterations=args.dilation,
        output=args.output,
    )


if __name__ == "__main__":
    main()

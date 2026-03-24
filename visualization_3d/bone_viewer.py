"""
3D medical image visualization with PyVista.

Supports:
  - Transparent volume rendering of bone + segmentation overlays
  - Multi-slice orthogonal views (XY, XZ, ZY) with 3D
  - STL mesh overlay

Consolidated from:
  - Amir/Visualization/py/viz.py
  - Amir/Visualization/py/bonevisualization.py
  - Amir/Visualization/py/bonevisualization1.py

Requirements:
  - pyvista, SimpleITK, nibabel, vtk

Usage:
    python -m visualization_3d.bone_viewer volume \\
        --bone-image data/bone.nii.gz \\
        --gt-image data/gt.nii.gz

    python -m visualization_3d.bone_viewer slices \\
        --nrrd-file data/gt_masked.nrrd

    python -m visualization_3d.bone_viewer combined \\
        --nrrd-file data/gt_masked.nrrd \\
        --volume-file data/gt_masked.nrrd
"""

import argparse
import SimpleITK as sitk
import pyvista as pv


def render_volume(bone_image_path, gt_image_path,
                  bone_opacity=None, gt_opacity=None,
                  bone_cmap="CMRmap", gt_cmap="coolwarm"):
    """
    Render bone and ground-truth volumes with transparency.

    Parameters
    ----------
    bone_image_path : str
        Path to the bone image (NIfTI format).
    gt_image_path : str
        Path to the ground-truth segmentation image.
    bone_opacity : list of float, optional
        Opacity transfer function for bone volume.
    gt_opacity : list of float, optional
        Opacity transfer function for GT volume.
    bone_cmap : str
        Colormap for bone volume.
    gt_cmap : str
        Colormap for GT volume.
    """
    if bone_opacity is None:
        bone_opacity = [0, 0.6, 0, 0, 0]
    if gt_opacity is None:
        gt_opacity = [0, 1, 0.6, 0.4, 0, 0, 0]

    bone_img = sitk.ReadImage(bone_image_path)
    bone_array = sitk.GetArrayFromImage(bone_img)
    gt_img = sitk.ReadImage(gt_image_path)
    gt_array = sitk.GetArrayFromImage(gt_img)

    plotter = pv.Plotter()
    plotter.add_volume(bone_array, cmap=bone_cmap, opacity=bone_opacity, shade=False)
    plotter.add_volume(gt_array, cmap=gt_cmap, opacity=gt_opacity, shade=True)
    plotter.show()


def render_slices(nrrd_path, stl_path=None):
    """
    Render orthogonal slices with optional STL mesh overlay.

    Parameters
    ----------
    nrrd_path : str
        Path to the NRRD volume file.
    stl_path : str, optional
        Path to an STL mesh to overlay.
    """
    mesh = pv.read(nrrd_path)
    slices = mesh.slice_orthogonal()

    plotter = pv.Plotter()
    plotter.add_mesh(mesh.outline(), color="k")
    plotter.add_mesh(slices, opacity=0.75, show_scalar_bar=False)

    if stl_path:
        reader = pv.get_reader(stl_path)
        mesh_stl = reader.read()
        mesh_reflected = mesh_stl.reflect((1, 0, 0), point=(0, -150, 0))
        plotter.add_mesh(mesh_reflected, opacity=0.5)

    plotter.show_bounds()
    plotter.show_grid()
    plotter.show()


def render_combined(nrrd_path, volume_path,
                    slice_cmap="gist_ncar_r",
                    volume_cmap="CMRmap_r",
                    volume_opacity=None):
    """
    Render 4-panel view: XY, ZY, XZ slices + 3D volume.

    Parameters
    ----------
    nrrd_path : str
        Path to the NRRD file for slicing.
    volume_path : str
        Path to the NRRD file for volume rendering.
    slice_cmap : str
        Colormap for slice views.
    volume_cmap : str
        Colormap for volume rendering.
    volume_opacity : list of float, optional
        Opacity transfer function for volume.
    """
    if volume_opacity is None:
        volume_opacity = [0, 1, 0.8, 0.6, 0.4, 0, 0, 0]

    mesh = pv.read(nrrd_path)
    slices = mesh.slice_orthogonal()

    reader = pv.get_reader(volume_path)
    volume_data = reader.read()

    dargs = dict(cmap=slice_cmap)
    plotter = pv.Plotter(shape=(2, 2))

    # XY view
    plotter.subplot(0, 0)
    plotter.add_mesh(slices, **dargs)
    plotter.show_grid()
    plotter.camera_position = "xy"
    plotter.enable_parallel_projection()

    # ZY view
    plotter.subplot(0, 1)
    plotter.add_mesh(slices, **dargs)
    plotter.show_grid()
    plotter.camera_position = "zy"
    plotter.enable_parallel_projection()

    # XZ view
    plotter.subplot(1, 0)
    plotter.add_mesh(slices, **dargs)
    plotter.show_grid()
    plotter.camera_position = "xz"
    plotter.enable_parallel_projection()

    # 3D volume + slices
    plotter.subplot(1, 1)
    plotter.add_mesh(slices, **dargs)
    plotter.add_volume(
        volume_data,
        cmap=volume_cmap,
        opacity=volume_opacity,
        show_scalar_bar=False,
    )
    plotter.show_grid()

    plotter.show()


def main():
    parser = argparse.ArgumentParser(description="3D medical image viewer")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Volume mode
    vol_parser = subparsers.add_parser("volume", help="Render bone + GT volumes")
    vol_parser.add_argument("--bone-image", required=True, help="Bone NIfTI path")
    vol_parser.add_argument("--gt-image", required=True, help="GT NIfTI path")
    vol_parser.add_argument("--bone-cmap", default="CMRmap")
    vol_parser.add_argument("--gt-cmap", default="coolwarm")

    # Slices mode
    slc_parser = subparsers.add_parser("slices", help="Render orthogonal slices")
    slc_parser.add_argument("--nrrd-file", required=True, help="NRRD volume path")
    slc_parser.add_argument("--stl-file", default=None, help="Optional STL mesh path")

    # Combined mode
    cmb_parser = subparsers.add_parser("combined", help="4-panel slice + volume view")
    cmb_parser.add_argument("--nrrd-file", required=True, help="NRRD slice path")
    cmb_parser.add_argument("--volume-file", required=True, help="NRRD volume path")

    args = parser.parse_args()

    if args.mode == "volume":
        render_volume(
            args.bone_image, args.gt_image,
            bone_cmap=args.bone_cmap, gt_cmap=args.gt_cmap,
        )
    elif args.mode == "slices":
        render_slices(args.nrrd_file, stl_path=args.stl_file)
    elif args.mode == "combined":
        render_combined(args.nrrd_file, args.volume_file)


if __name__ == "__main__":
    main()

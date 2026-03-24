"""
ITK-based Watershed segmentation.

Pipeline:
  1. Gradient magnitude computation
  2. Watershed flooding with threshold and level parameters

Consolidated from:
  - Amir/SegmentWithWatershedImageFilter/Code.py

Requirements:
  - itk (pip install itk)

Usage:
    python -m itk_segmentation.watershed_itk --image input.png --output output.png --threshold 0.005 --level 0.5

Notes:
  - threshold: Minimum height value; raising it removes shallow regions (fewer initial segments).
  - level: Flooding depth (0-1); controls merging of basic segments. Typically < 0.40.
    A rule of thumb: set threshold to ~1/100 of level.
"""

import argparse
import itk


def run_watershed_itk(input_image, output_image, threshold=0.005, level=0.5):
    """
    Run ITK Watershed segmentation.

    Parameters
    ----------
    input_image : str
        Path to the input image.
    output_image : str
        Path for the output colormap image.
    threshold : float
        Minimum height threshold (higher = fewer initial segments).
    level : float
        Flooding level (0-1). Higher values merge more segments.
    """
    dimension = 2

    float_pixel_type = itk.ctype("float")
    float_image_type = itk.Image[float_pixel_type, dimension]

    reader = itk.ImageFileReader[float_image_type].New()
    reader.SetFileName(input_image)

    gradient_magnitude = itk.GradientMagnitudeImageFilter.New(Input=reader.GetOutput())

    watershed_filter = itk.WatershedImageFilter.New(Input=gradient_magnitude.GetOutput())
    watershed_filter.SetThreshold(threshold)
    watershed_filter.SetLevel(level)

    labeled_image_type = type(watershed_filter.GetOutput())

    pixel_type = itk.ctype("unsigned char")
    rgb_pixel_type = itk.RGBPixel[pixel_type]
    rgb_image_type = itk.Image[rgb_pixel_type, dimension]

    colormap_filter = itk.ScalarToRGBColormapImageFilter[
        labeled_image_type, rgb_image_type
    ].New()
    colormap_filter.SetColormap(
        itk.ScalarToRGBColormapImageFilterEnums.RGBColormapFilter_Jet
    )
    colormap_filter.SetInput(watershed_filter.GetOutput())
    colormap_filter.Update()

    writer = itk.ImageFileWriter[rgb_image_type].New()
    writer.SetFileName(output_image)
    writer.SetInput(colormap_filter.GetOutput())
    writer.Update()

    print(f"Watershed segmentation saved to: {output_image}")
    print(f"  threshold={threshold}, level={level}")


def main():
    parser = argparse.ArgumentParser(description="ITK Watershed segmentation")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="output_watershed.png", help="Output image path")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Minimum height threshold (default: 0.005)")
    parser.add_argument("--level", type=float, default=0.5,
                        help="Flooding level 0-1 (default: 0.5)")
    args = parser.parse_args()

    run_watershed_itk(
        input_image=args.image,
        output_image=args.output,
        threshold=args.threshold,
        level=args.level,
    )


if __name__ == "__main__":
    main()

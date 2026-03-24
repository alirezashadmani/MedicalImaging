"""
ITK-based Geodesic Active Contour Level Set segmentation.

Full ITK pipeline:
  1. Curvature anisotropic diffusion smoothing
  2. Gradient magnitude (recursive Gaussian)
  3. Sigmoid mapping (speed image)
  4. Fast marching initialization
  5. Geodesic active contour evolution
  6. Binary thresholding of the final level set

Consolidated from:
  - Amir/SegmentWithGeodesicActiveContourLevelSet/Code.py
  - Amir/test/test.py

Requirements:
  - itk (pip install itk)

Usage:
    python -m itk_segmentation.geodesic_active_contour --image path/to/image.png --output output.png
    python -m itk_segmentation.geodesic_active_contour --image img.png --output out.png --seed 56 92 --propagation 7
"""

import argparse
import itk


def run_geodesic_active_contour(
    input_image,
    output_image,
    seed_x,
    seed_y,
    initial_distance=11.0,
    sigma=1.0,
    sigmoid_alpha=1.0,
    sigmoid_beta=-0.2,
    propagation_scaling=7.0,
    number_of_iterations=200,
    save_intermediates=False,
):
    """
    Run the full ITK Geodesic Active Contour pipeline.

    Parameters
    ----------
    input_image : str
        Path to the input image.
    output_image : str
        Path for the output binary segmentation.
    seed_x, seed_y : int
        Seed point coordinates for the fast marching initialization.
    initial_distance : float
        Initial distance from the seed (controls initial contour size).
    sigma : float
        Sigma for gradient magnitude filter.
    sigmoid_alpha : float
        Alpha parameter for sigmoid speed function.
    sigmoid_beta : float
        Beta parameter for sigmoid speed function.
    propagation_scaling : float
        Propagation scaling for geodesic active contour.
    number_of_iterations : int
        Maximum number of iterations.
    save_intermediates : bool
        If True, save intermediate pipeline images.
    """
    seed_value = -initial_distance
    dimension = 2

    input_pixel_type = itk.F
    output_pixel_type = itk.UC

    input_image_type = itk.Image[input_pixel_type, dimension]
    output_image_type = itk.Image[output_pixel_type, dimension]

    # Read input
    reader = itk.ImageFileReader[input_image_type].New()
    reader.SetFileName(input_image)

    # Smoothing: curvature anisotropic diffusion
    smoothing_filter = itk.CurvatureAnisotropicDiffusionImageFilter[
        input_image_type, input_image_type
    ].New()
    smoothing_filter.SetTimeStep(0.125)
    smoothing_filter.SetNumberOfIterations(5)
    smoothing_filter.SetConductanceParameter(9.0)
    smoothing_filter.SetInput(reader.GetOutput())

    # Gradient magnitude
    gradient_filter = itk.GradientMagnitudeRecursiveGaussianImageFilter[
        input_image_type, input_image_type
    ].New()
    gradient_filter.SetSigma(sigma)
    gradient_filter.SetInput(smoothing_filter.GetOutput())

    # Sigmoid (speed image)
    sigmoid_filter = itk.SigmoidImageFilter[input_image_type, input_image_type].New()
    sigmoid_filter.SetOutputMinimum(0.0)
    sigmoid_filter.SetOutputMaximum(1.0)
    sigmoid_filter.SetAlpha(sigmoid_alpha)
    sigmoid_filter.SetBeta(sigmoid_beta)
    sigmoid_filter.SetInput(gradient_filter.GetOutput())

    # Fast marching initialization
    fast_marching = itk.FastMarchingImageFilter[input_image_type, input_image_type].New()

    # Geodesic active contour
    gac_filter = itk.GeodesicActiveContourLevelSetImageFilter[
        input_image_type, input_image_type, input_pixel_type
    ].New()
    gac_filter.SetPropagationScaling(propagation_scaling)
    gac_filter.SetCurvatureScaling(1.0)
    gac_filter.SetAdvectionScaling(1.0)
    gac_filter.SetMaximumRMSError(0.02)
    gac_filter.SetNumberOfIterations(number_of_iterations)
    gac_filter.SetInput(fast_marching.GetOutput())
    gac_filter.SetFeatureImage(sigmoid_filter.GetOutput())

    # Binary thresholding of the level set output
    thresholder = itk.BinaryThresholdImageFilter[input_image_type, output_image_type].New()
    thresholder.SetLowerThreshold(-1000.0)
    thresholder.SetUpperThreshold(0.0)
    thresholder.SetOutsideValue(itk.NumericTraits[output_pixel_type].min())
    thresholder.SetInsideValue(itk.NumericTraits[output_pixel_type].max())
    thresholder.SetInput(gac_filter.GetOutput())

    # Set seed point
    seed_position = itk.Index[dimension]()
    seed_position[0] = seed_x
    seed_position[1] = seed_y

    node = itk.LevelSetNode[input_pixel_type, dimension]()
    node.SetValue(seed_value)
    node.SetIndex(seed_position)

    seeds = itk.VectorContainer[
        itk.UI, itk.LevelSetNode[input_pixel_type, dimension]
    ].New()
    seeds.Initialize()
    seeds.InsertElement(0, node)

    fast_marching.SetTrialPoints(seeds)
    fast_marching.SetSpeedConstant(1.0)
    fast_marching.SetOutputSize(reader.GetOutput().GetBufferedRegion().GetSize())

    # Write output
    writer = itk.ImageFileWriter[output_image_type].New()
    writer.SetFileName(output_image)
    writer.SetInput(thresholder.GetOutput())
    writer.Update()

    # Print statistics
    print(f"Max iterations:     {gac_filter.GetNumberOfIterations()}")
    print(f"Max RMS error:      {gac_filter.GetMaximumRMSError()}")
    print(f"Elapsed iterations: {gac_filter.GetElapsedIterations()}")
    print(f"RMS change:         {gac_filter.GetRMSChange()}")

    # Save intermediate images
    if save_intermediates:
        cast_filter_type = itk.RescaleIntensityImageFilter[input_image_type, output_image_type]
        writer_type = itk.ImageFileWriter[output_image_type]
        internal_writer_type = itk.ImageFileWriter[input_image_type]

        intermediates = [
            (smoothing_filter.GetOutput(), "GeodesicActiveContour_smoothed.png"),
            (gradient_filter.GetOutput(), "GeodesicActiveContour_gradient.png"),
            (sigmoid_filter.GetOutput(), "GeodesicActiveContour_sigmoid.png"),
            (fast_marching.GetOutput(), "GeodesicActiveContour_fastmarching.png"),
        ]

        for data, filename in intermediates:
            caster = cast_filter_type.New()
            caster.SetInput(data)
            caster.SetOutputMinimum(itk.NumericTraits[output_pixel_type].min())
            caster.SetOutputMaximum(itk.NumericTraits[output_pixel_type].max())
            w = writer_type.New()
            w.SetInput(caster.GetOutput())
            w.SetFileName(filename)
            w.Update()
            print(f"Saved intermediate: {filename}")


def main():
    parser = argparse.ArgumentParser(description="ITK Geodesic Active Contour Level Set")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="output_gac.png", help="Output image path")
    parser.add_argument("--seed", nargs=2, type=int, required=True,
                        help="Seed point [x y]")
    parser.add_argument("--initial-distance", type=float, default=11.0,
                        help="Initial distance from seed")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gradient sigma")
    parser.add_argument("--sigmoid-alpha", type=float, default=1.0, help="Sigmoid alpha")
    parser.add_argument("--sigmoid-beta", type=float, default=-0.2, help="Sigmoid beta")
    parser.add_argument("--propagation", type=float, default=7.0,
                        help="Propagation scaling")
    parser.add_argument("--iterations", type=int, default=200, help="Max iterations")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Save intermediate pipeline images")
    args = parser.parse_args()

    run_geodesic_active_contour(
        input_image=args.image,
        output_image=args.output,
        seed_x=args.seed[0],
        seed_y=args.seed[1],
        initial_distance=args.initial_distance,
        sigma=args.sigma,
        sigmoid_alpha=args.sigmoid_alpha,
        sigmoid_beta=args.sigmoid_beta,
        propagation_scaling=args.propagation,
        number_of_iterations=args.iterations,
        save_intermediates=args.save_intermediates,
    )


if __name__ == "__main__":
    main()

# Postprocessing and export

`JointReconstructor.run()` returns the native image and motion model produced by the solver. Postprocessing is an optional calling-workflow step that turns that reconstruction into the image required for review, analysis, storage, or exchange.

## Why postprocess

Reconstruction estimates an image on its solver grid. In an acquired image, the matrix used for computation can differ from the encoded matrix and the final reconstruction matrix. Zero-filling adds zeros in k-space to interpolate the image onto the encoded grid; a subsequent center crop selects the final matrix. Zero-filling alone does not restore a field of view: cropping changes the covered extent, and the acquisition header supplies the matrix sizes and physical geometry used for export.

A downstream workflow may also need a different intensity scale, magnitude image for display, or a format-specific pixel array for export. These operations change how the image is interpreted or presented; they are not part of the joint image-and-motion optimization.

Postprocessing may change image values, shape, orientation, or intensity scale. Keep the native image whenever the final image is changed so that the solver result remains available for quality assessment, reproducibility, or a different downstream transformation. The returned motion model remains on the reconstruction grid; postprocessing an image does not transform its motion coordinates.

## Apply a workflow transformation

The Siemens 2D and 3D pipelines use the same sequence. They save the solver image as `native_image`, optionally divide the image by the magnitude of its prepared GRICS reference image, then zero-fill in k-space and center-crop to the reconstruction matrix from the ISMRMRD header. The normalization option comes from the corresponding `config/postprocessing/nonrigid_2d_breast.toml` or `nonrigid_3d_breast.toml` file; it is disabled by default and requires Odille spline coil maps when enabled. The zero-fill/crop step runs for both pipelines and leaves the image unchanged if it already has the target shape.

For a prepared `data` object, the image steps are:

```python
from pipelines.utils.postprocessing import (
    grics_reference_image_for_normalization, grics_zero_fill_shapes,
    grics_zero_fill_volume_shapes, normalize_reconstruction_by_grics_reference,
    zero_fill_loaded_reconstruction,
)
from src.runtime.runtime_config import load_postprocessing_config

config_file = f"config/postprocessing/nonrigid_{data.params.data_dimension.lower()}_breast.toml"
postprocessing = load_postprocessing_config(config_file)
if postprocessing.normalize_image_by_grics_reference and data.params.coil_sensitivity_method != "odille-spline":
    raise ValueError("Reference-image normalization requires Odille spline coil maps.")
image, motion = reconstructor.run()
native_image = image.detach().cpu()
if postprocessing.normalize_image_by_grics_reference:
    reference_image = grics_reference_image_for_normalization(data, image)
    image, _ = normalize_reconstruction_by_grics_reference(image, reference_image)
shape_reader = grics_zero_fill_volume_shapes if data.params.data_dimension == "3D" else grics_zero_fill_shapes
target_shape, encoded_shape = shape_reader(data.input_loader)
image = zero_fill_loaded_reconstruction(image, target_shape, encoded_shape)
```

`grics_zero_fill_shapes(...)` reads the in-plane matrix sizes for 2D; `grics_zero_fill_volume_shapes(...)` adds the partition sizes for 3D. Both validate the loaded grid against the encoded grid. When a converted header omits a reconstruction matrix size, the helpers announce that they derive it from the encoded matrix and field-of-view ratio. `zero_fill_loaded_reconstruction(...)` applies the validated shape to the final two or three spatial axes with [`zero_fill_grics_image_to_shape(...)`](../../src/utils/zero_fill.py). The original motion model stays on the solver grid. In the 2D pipeline, this sequence runs in each slice worker; in the 3D pipeline, it runs once on the full volume.

## Save the native and final results

Call `write_reconstruction(...)` after the final image has been chosen. It saves the configured tensors and records metadata about the image stage. Pass `native_image` and set `image_stage="after_postprocessing"` so the output metadata distinguishes the saved final image from the solver image.

```python
from src.runtime.output_layout import write_reconstruction

write_reconstruction(
    data.params, image, motion, native_image=native_image,
    image_stage="after_postprocessing",
)
```

When no image transformation is applied, save the image returned by `run()` and use the default `image_stage`, `"before_postprocessing"`. The [output guide](outputs.md) describes the output files and saving flags.

## Export

Export follows postprocessing because exported pixels should represent the final chosen image. The Siemens 2D pipeline writes one DICOM per prepared slice with [`write_reconstruction_dicom(...)`](../../src/utils/dicom_export.py); the 3D pipeline uses [`write_volume_dicoms(...)`](../../src/utils/dicom_export.py) to expand the slab center into slice positions. Both export the postprocessed image, converting complex values to magnitude and scaling DICOM pixels. The volume exporter derives z spacing from the encoded z field of view and matrix size, preserving spacing even if z was zero-filled or cropped. The DICOM geometry and pixel matrix must describe the final image grid.

The Siemens pipelines provide concrete normalization, zero-filling, and DICOM-export examples. See the [2D T2 pipeline](../pipelines.md#2d-t2-pipeline), the [3D low-resolution pipeline](../pipelines.md#3d-low-resolution-pipeline), and the [output guide](outputs.md).

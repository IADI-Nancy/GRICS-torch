# Postprocessing and export

`JointReconstructor.run()` returns the native image and motion model produced by the solver. Postprocessing is an optional calling-workflow step that turns that reconstruction into the image required for review, analysis, storage, or exchange.

## Why postprocess

The solver produces an image on the grid that is convenient for reconstruction. Before the image is reviewed or exported, the pipeline prepares it for its intended use:

1. **Restore the requested field of view and matrix.** The solver grid, encoded grid, and final reconstruction grid can differ. The pipeline zero-fills k-space to the encoded grid and center-crops to the reconstruction matrix specified in the acquisition header. This places the result on the requested final grid; it does not recreate anatomy outside the acquired field of view.
2. **Optionally normalize the intensity.** MRI intensity is not an absolute physical unit, and receive-coil sensitivity can make the same tissue appear brighter in one location than another. MRI manufacturers commonly offer intensity-correction options for this reason. GRICS-torch can similarly divide the reconstruction by its prepared GRICS reference image, producing a more uniform presentation scale when this is appropriate for the workflow.
3. **Encode pixels for DICOM.** DICOM output uses an unsigned 16-bit container, with 12 bits stored for each pixel. The exporter therefore converts the final magnitude image to integer values from 0 to 4095. This DICOM scaling is required for the file representation and is separate from the optional GRICS-reference normalization.

Keep the native image whenever the final image is changed so the solver result remains available for quality assessment or another downstream transformation. The returned motion model remains on the solver grid; changing the image grid does not transform its motion coordinates.

## Standard pipeline postprocessing

The Siemens 2D and 3D pipelines apply the following sequence after `JointReconstructor.run()`:

1. Preserve the solver output as `native_image`. This is the complex image on the solver grid and remains available for quality checks or a different downstream workflow.
2. Optionally normalize the image by the prepared GRICS reference image. The pipeline divides by the reference magnitude, replacing reference values below its mean magnitude with that mean to avoid unstable amplification in low-signal regions.
3. Restore the requested image grid. The pipeline zero-fills in k-space from the solver grid to the encoded matrix, then center-crops to the reconstruction matrix in the ISMRMRD header. If the image already has the requested shape, this step leaves it unchanged.
4. Save and export the final image. Tensor output uses the normalized and zero-filled/cropped image. DICOM export then takes its magnitude and scales it to the 12-bit stored pixel range; the motion model remains on the solver grid.

The normalization setting comes from `config/postprocessing/nonrigid_2d_breast.toml` or `config/postprocessing/nonrigid_3d_breast.toml`. The zero-fill helpers obtain their encoded and reconstruction sizes from the acquisition header. If a converted header omits a reconstruction size, they derive it from the encoded matrix and field-of-view ratio.

For a prepared `data` object, the corresponding image steps are:

```python
from pipelines.utils.postprocessing import (
    grics_reference_image_for_normalization, grics_zero_fill_shapes,
    grics_zero_fill_volume_shapes, normalize_reconstruction_by_grics_reference,
    zero_fill_loaded_reconstruction,
)
from src.runtime.runtime_config import load_postprocessing_config

config_file = f"config/postprocessing/nonrigid_{data.params.data_dimension.lower()}_breast.toml"
postprocessing = load_postprocessing_config(config_file)
image, motion = reconstructor.run()
native_image = image.detach().cpu()
if postprocessing.normalize_image_by_grics_reference:
    reference_image = grics_reference_image_for_normalization(data, image)
    image, _ = normalize_reconstruction_by_grics_reference(image, reference_image)
shape_reader = grics_zero_fill_volume_shapes if data.params.data_dimension == "3D" else grics_zero_fill_shapes
target_shape, encoded_shape = shape_reader(data.input_loader)
image = zero_fill_loaded_reconstruction(image, target_shape, encoded_shape)
```

`grics_zero_fill_shapes(...)` reads the in-plane matrix sizes for 2D; `grics_zero_fill_volume_shapes(...)` adds the partition sizes for 3D. Both validate the loaded grid against the encoded grid. `zero_fill_loaded_reconstruction(...)` applies the validated shape to the final two or three spatial axes with [`zero_fill_grics_image_to_shape(...)`](../../src/utils/zero_fill.py). The original motion model stays on the solver grid. In the 2D pipeline, this sequence runs in each slice worker; in the 3D pipeline, it runs once on the full volume.

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

Export follows postprocessing because exported pixels should represent the final chosen image. The Siemens 2D pipeline writes one DICOM per prepared slice with [`write_reconstruction_dicom(...)`](../../src/utils/dicom_export.py); the 3D pipeline uses [`write_volume_dicoms(...)`](../../src/utils/dicom_export.py) to expand the slab center into slice positions. Both convert the postprocessed complex image to magnitude, shift its minimum to zero, and scale its maximum to a value no greater than 4095. They store these values as `uint16` pixels with 12 meaningful bits (`BitsAllocated = 16`, `BitsStored = 12`). The volume exporter derives z spacing from the encoded z field of view and matrix size, preserving spacing even if z was zero-filled or cropped. The DICOM geometry and pixel matrix must describe the final image grid.

The Siemens pipelines provide concrete normalization, zero-filling, and DICOM-export examples. See the [2D T2 pipeline](../architecture/pipelines.md#2d-t2-pipeline), the [3D low-resolution pipeline](../architecture/pipelines.md#3d-low-resolution-pipeline), and the [output guide](outputs.md).

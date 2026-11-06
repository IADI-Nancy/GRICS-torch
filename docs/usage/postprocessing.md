# Postprocessing and export

Postprocessing acts on the image returned by `JointReconstructor.run()`. It is
optional and belongs to the calling notebook or pipeline; the solver returns
its native reconstruction. Keep that native image if you need both versions in
the output. The motion model remains on the reconstruction grid.

## Siemens 2D T2 pipeline

The [2D T2 pipeline](../pipelines.md#2d-t2-pipeline) loads
`config/postprocessing/nonrigid_2d_breast.toml` with
`load_postprocessing_config(...)`. Its `normalize_image_by_grics_reference`
setting controls whether the reconstructed image is divided by the magnitude of
the prepared GRICS reference image. Enabling it requires Odille spline coil
maps. The pipeline then zero-fills and crops the image to the target matrix
size from the acquisition header. These steps affect the final image; the
native image is captured before them.

To use the same steps in a custom workflow, apply
`normalize_reconstruction_by_grics_reference(image, reference_image)` when
enabled, then `zero_fill_loaded_reconstruction(image, target_shape,
encoded_shape)` using the acquisition's matrix sizes. The pipeline provides
`grics_reference_image_for_normalization(data, image)` and
`grics_zero_fill_shapes(data.input_loader)` to validate and obtain those
inputs. The helpers are defined in
[postprocessing.py](../../pipelines/utils/postprocessing.py); see
[siemens_breast_T2.py](../../pipelines/siemens_breast_T2.py) for the call sequence.
The [3D low-resolution
pipeline](../pipelines.md#3d-low-resolution-pipeline) does not apply these
image steps.

With a prepared 2D T2 `data` object and the `image, motion` returned by
`reconstructor.run()`, the image steps are:

```python
from pipelines.utils.postprocessing import (
    grics_reference_image_for_normalization, grics_zero_fill_shapes,
    normalize_reconstruction_by_grics_reference, zero_fill_loaded_reconstruction,
)
from src.runtime.output_layout import write_reconstruction
from src.runtime.runtime_config import load_postprocessing_config

postprocessing = load_postprocessing_config("config/postprocessing/nonrigid_2d_breast.toml")
native_image = image
if postprocessing.normalize_image_by_grics_reference:
    if data.params.coil_sensitivity_method != "odille-spline":
        raise ValueError("Reference-image normalization requires Odille spline coil maps.")
    reference_image = grics_reference_image_for_normalization(data, image)
    image, _ = normalize_reconstruction_by_grics_reference(image, reference_image)
target_shape, encoded_shape = grics_zero_fill_shapes(data.input_loader)
image = zero_fill_loaded_reconstruction(image, target_shape, encoded_shape)
write_reconstruction(data.params, image, motion, native_image=native_image, image_stage="after_postprocessing")
```

## Save the result

After any image postprocessing, call `write_reconstruction(params, image,
motion, native_image=native_image, image_stage="after_postprocessing")` from
`src.runtime.output_layout` to save final tensors when tensor saving is enabled
and record their metadata. If no postprocessing was applied, leave
`image_stage` at its default, `"before_postprocessing"`. The [output guide](outputs.md)
describes the saved files and flags.

The Siemens pipelines always export DICOM using
`write_reconstruction_dicom(...)`. The writer accepts a 2D image, converts
complex values to magnitude, and requires source geometry and a slice index;
see [outputs](outputs.md) and the [2D T2
pipeline](../pipelines.md#2d-t2-pipeline).

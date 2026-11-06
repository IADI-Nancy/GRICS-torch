# Postprocessing and export

`JointReconstructor.run()` returns the native image and motion model produced by the solver. Postprocessing is an optional calling-workflow step that turns that reconstruction into the image required for review, analysis, storage, or exchange. It stays outside `JointReconstructor` because its purpose and choices depend on the acquisition and the intended use of the result.

## Why postprocess

Reconstruction estimates an image on the solver grid and in its reconstructed intensity scale. A downstream workflow may need a different representation: for example, a magnitude image for display, a crop or zero-filled matrix for a requested field of view, intensity normalization for comparison, or a format-specific pixel array for export. These operations change how the image is interpreted or presented; they are not part of the joint image-and-motion optimization.

Postprocessing may change image values, shape, orientation, or intensity scale. Keep the native image whenever the final image is changed so that the solver result remains available for quality assessment, reproducibility, or a different downstream transformation. The returned motion model remains on the reconstruction grid; postprocessing an image does not transform its motion coordinates.

## Apply a workflow transformation

Start from the image returned by `run()`. Preserve it before applying any operation that changes it, then use the transformed image as the final result. The transformation itself is owned by the notebook or pipeline.

```python
image, motion = reconstructor.run()
native_image = image.clone()

# Apply the workflow's image transformation.
image = transform_image(image)
```

`transform_image(...)` can be any explicit operation appropriate to the workflow. It should accept and return an image on a well-defined grid. If it changes the spatial shape, record the resulting stage as postprocessed and ensure any export geometry describes that final grid.

## Save the native and final results

Call `write_reconstruction(...)` after the final image has been chosen. It saves the configured tensors and records metadata about the image stage. Pass `native_image` when postprocessing has changed the image; set `image_stage="after_postprocessing"` so the output metadata identifies the saved final image correctly.

```python
from src.runtime.output_layout import write_reconstruction

write_reconstruction(
    data.params, image, motion, native_image=native_image,
    image_stage="after_postprocessing",
)
```

When no image transformation is applied, save the image returned by `run()` and use the default `image_stage`, `"before_postprocessing"`. The [output guide](outputs.md) describes the output files and saving flags.

## Export

Export follows postprocessing because exported pixels should represent the final chosen image. An exporter may impose further requirements, such as a real-valued magnitude image, source geometry, spacing, orientation, or a slice index. Provide metadata that matches the final image grid rather than assuming that preprocessing geometry remains valid after resizing or cropping.

The Siemens pipelines provide concrete normalization, zero-filling, and DICOM-export examples. See the [2D T2 pipeline](../pipelines.md#2d-t2-pipeline), the [3D low-resolution pipeline](../pipelines.md#3d-low-resolution-pipeline), and the [output guide](outputs.md).

# Using JointReconstructor

`JointReconstructor` takes prepared acquisition data and runs joint image and
motion reconstruction. The usual call sequence is:

```python
from src.reconstruction.JointReconstructor import JointReconstructor

reconstructor = JointReconstructor.from_prepared_data(data)
image, motion = reconstructor.run()
reconstructor.finish_run()
```

Here `data` is a `DataPreparer` after `load_input_data()` and
`prepare_reconstructor_inputs(...)`; see [using
DataPreparer](data_preparer.md). Call `finish_run()` after any chosen output
step so the run log includes it.

## Construct the solver

`JointReconstructor.from_prepared_data(data)` is the constructor for normal
workflows. It passes the prepared k-space, coil sensitivity maps, sampling
indices, motion signal, configuration, and optional calibration information to
the solver. It also uses the preparer's intensity scale so the returned image
has acquisition units. Calling it before preparation raises an error.

The direct constructor, `JointReconstructor(KspaceData, smaps, SamplingIndices,
motion_signal, params, ...)`, is available when those inputs have been
assembled without `DataPreparer`. `params` must be the validated configuration
object. Direct callers supply an intensity scale when they need
acquisition-unit output and encoded voxel spacing when voxel-aware
regularization is enabled. When calibration weighting is enabled and no
calibration image is supplied, the constructor computes one from imaging
k-space. The [constructor
signature](../../src/reconstruction/JointReconstructor.py) defines the optional
arguments.

## Run the reconstruction

`run()` executes the configured reconstruction and returns `(image, motion)` as
PyTorch tensors. `image` is a complex 2D image or 3D volume; `motion` contains
the estimated rigid parameters or nonrigid displacement field. The method
writes configured solver logs, metadata, and diagnostic plots during
reconstruction. It does not apply workflow postprocessing or export a final
postprocessed image.

Use [postprocessing and export](postprocessing.md) if the workflow changes the
returned image, or see [outputs](outputs.md) for saved files and output flags.
Solver internals are described in [reconstruction
architecture](../architecture/reconstruction.md).

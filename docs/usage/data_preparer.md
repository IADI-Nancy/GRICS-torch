# Using DataPreparer

`DataPreparer` loads or generates the input data and prepares one slice or volume for `JointReconstructor`. It validates the selected input, estimates coil sensitivity maps, imports or simulates motion, and builds the sampling indices used by reconstruction.

## Construct the preparer

`DataPreparer(params=params, runtime=runtime, filename=...)` requires  `params` created with [configuration loading](configuration/configuration.md) and `runtime` created with `runtime = initialize_runtime(params)`.

The `filename` value depends on `params.data_type`: use one path for an image or preprocessed HDF5 file; use a tuple or dictionary for raw MRI and physiological inputs; omit `filename` for Shepp–Logan data. The [input formats guide](configuration/input_formats.md) lists the required paths and dictionary keys for each source.

## Load or generate the input data

Call `data.load_input_data()` once. It reads or generates the source and retains the full acquisition, including all slices of a 2D real-data input. Calling it again on the same preparer raises an error.

## Prepare reconstruction inputs

Call `data.prepare_reconstructor_inputs(slice_idx=...)` after loading to select a 2D real-data slice. Indices are zero-based. A source with several slices requires an explicit index; a source with exactly one slice selects index `0` when the argument is omitted. Omit `slice_idx` for synthetic and 3D inputs.

## Prepare multiple 2D slices

Load the acquisition once, then create one shallow copy of the loaded `DataPreparer` and one shallow copy of its `params` for every slice. Each copy calls `prepare_reconstructor_inputs(slice_idx=...)` exactly once. It selects its own slice while sharing the original loaded acquisition, so the source file is not read again.

```python
import copy

data.load_input_data()
selected_slice_indices = list(selected_slice_indices)
data.params._output_unit_count = len(selected_slice_indices)
for slice_idx in selected_slice_indices:
    slice_data = copy.copy(data)
    slice_data.params = copy.copy(data.params)
    slice_data.prepare_reconstructor_inputs(slice_idx=slice_idx)
    # Construct and run a JointReconstructor with slice_data.
```

Set `_output_unit_count` before copying the parameters so preparation gives each slice its own output folder when reconstructing more than one slice. This prevents later slices from overwriting earlier results.

Use a different `DataPreparer` copy for each slice. Do not call `prepare_reconstructor_inputs(...)` more than once on the same copy. The [2D T2 pipeline](../pipelines.md#2d-t2-pipeline) provides a complete multi-slice implementation with CPU parallelization.

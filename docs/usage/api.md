# Integration APIs

This section covers the standard reconstruction entry point and the additional interfaces intended for external reconstruction or training code. Exact input types and tensor dimensions are available directly in each Python signature and docstring.

## Data preparation API

`GRICSPreparerAPI` is for an external application that already owns its image or k-space tensors. It loads GRICS configuration, bins chronological motion measurements, and builds GRICS sampling indices. It does not load k-space, calculate sensitivity maps, simulate motion, or reconstruct an image.

```python
from src.preprocessing.GRICSPreparerAPI import GRICSPreparerAPI

preparer = GRICSPreparerAPI(
    reconstruction_config,
    data_type="preprocessed-real",
    overrides=runtime_overrides,
)
prepared = preparer.prepare_acquisition(
    motion_data, ky_indices, nex_indices,
    Nx=Nx, Ny=Ny, Nz=Nz, kz_indices=kz_indices,
    kspace=kspace, seed=seed,
)
sampling_indices = prepared.sampling_indices["all"]
motion_signal = prepared.motion_signal
params = prepared.params
```

When `sampling_masks` is omitted, `"all"` contains every supplied readout.

### Optional undersampling

The preparer supports externally defined undersampling through `sampling_masks`. It does not generate an acceleration pattern: the caller chooses retained chronological readouts, and the preparer converts those selections into GRICS indices.

```python
phase_encode_mask = make_phase_encode_mask(Ny, acceleration, calibration_lines)
retained_readouts = phase_encode_mask[ky_indices]
prepared = preparer.prepare_acquisition(
    motion_data, ky_indices, nex_indices, Nx=Nx, Ny=Ny, Nz=Nz,
    sampling_masks={
        "retained": retained_readouts,
        "heldout": ~retained_readouts,
    },
    kspace=kspace, seed=seed,
)
retained_sampling_indices = prepared.sampling_indices["retained"]
heldout_sampling_indices = prepared.sampling_indices["heldout"]
```

`kspace` is required when using `motion_binning_mode="kspace_energy"`; otherwise it may be omitted. Every mask contains one Boolean per chronological readout. Keys are caller-defined, and all named layouts share the same motion-state labels.

## Standard reconstruction entry point

`JointReconstructor.run()` executes the complete configured multi-resolution reconstruction. Internal workflows use the named factory for a fully prepared `DataPreparer`:

```python
reconstructor = JointReconstructor.from_prepared_data(data)
image, motion_model = reconstructor.run()
```

Call `data.load_input_data()` and `data.prepare_reconstructor_inputs()` first. The factory maps the prepared tensors, configuration, plotting context, enabled calibration image, and required encoded voxel spacing to the explicit constructor. It restores output magnitude automatically. The returned image, `data.image_no_moco`, and simulated `data.image_ground_truth` all use the same acquisition intensity scale, so plots and similarity metrics need no scale argument. Working k-space and `data.input_ifft_image` remain in preprocessing units; the single-iteration and prediction APIs still operate in those working units. The factory accepts keyword options for output scale, initial image, initial motion, and an external image regularizer.

External callers can supply each acquisition field directly:

```python
from src.reconstruction.JointReconstructor import JointReconstructor

reconstructor = JointReconstructor(
    KspaceData, smaps, SamplingIndices, motion_signal, params,
    kspace_scale=1.0, motion_plot_context=None,
    initial_image=None, initial_motion=None,
    external_image_regularizer=None, calibration_image_prior=None,
)
image, motion_model = reconstructor.run()
```

For `params`, use `prepared.params` from `GRICSPreparerAPI`. Choose the appropriate named sampling layout explicitly, such as `prepared.sampling_indices["retained"]`. Explicit tensor inputs are retained without copying or detaching. Output scaling defaults to 1.0; pass the normalization divisor as `kspace_scale` if output magnitude must be restored. Supply `voxel_spacing_mm` when using voxel-aware regularization.

When `use_calibration_weighted_image` is enabled, the explicit form accepts `calibration_image_prior` or calculates it from the supplied imaging k-space using `calibration_image_smoothing`. For held-out sampling experiments, supply a calibration image prepared from the intended calibration data to avoid including held-out measurements. When weighting is disabled, no calibration image is required or calculated by the constructor. Both coil methods support weighting; combining it with an external image regularizer is rejected.

## Full-resolution iteration and prediction APIs

`full_resolutions_gauss_newton_iteration_api()` performs one full-resolution image update followed by an optional motion update.

```python
image, motion_model = reconstructor.full_resolutions_gauss_newton_iteration_api(
    image, motion_model, image_regularizer=None,
    regularization_weight=None, update_motion=True,
    image_cg_iterations=None, motion_cg_iterations=None,
)
```

`predict_kspace_api()` evaluates an image and motion model on the constructor sampling layout or an explicitly supplied layout.

```python
predicted_kspace = reconstructor.predict_kspace_api(
    image, motion_model, sampling_indices=None,
)
```

# Integration APIs

This section covers the standard reconstruction entry point and the additional interfaces intended for external reconstruction or training code. Exact input types and tensor dimensions are available directly in each Python signature and docstring.

## Data preparation API

`GRICSPreparerAPI` is for an external application that already owns its image or k-space tensors. It loads GRICS configuration, bins chronological motion measurements, and builds GRICS sampling indices. It does not load k-space, calculate sensitivity maps, simulate motion, or reconstruct an image.

```python
from src.preprocessing.GRICSPreparerAPI import GRICSPreparerAPI

preparer = GRICSPreparerAPI(
    reconstruction_config,
    "config/coil_sensitivity/odille_spline.toml",
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
    motion_data, ky_indices, nex_indices, Nx=Nx, Ny=Ny,
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

`JointReconstructor.run()` executes the complete configured multi-resolution reconstruction.

```python
from src.reconstruction.JointReconstructor import JointReconstructor

reconstructor = JointReconstructor(
    KspaceData, smaps, SamplingIndices, motion_signal, params,
    kspace_scale=1.0, motion_plot_context=None,
    initial_image=None, initial_motion=None,
    external_image_regularizer=None,
)
image, motion_model = reconstructor.run()
```

For `params`, use `data.params` from `DataLoader` or `prepared.params` from `GRICSPreparerAPI`. When using normalized k-space from `DataLoader`, pass `kspace_scale=data.kspace_scale` to restore the image scale in `run()` outputs.

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

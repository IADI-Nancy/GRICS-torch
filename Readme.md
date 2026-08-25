# GRICS-torch: GRICS MRI motion-corrected reconstruction in PyTorch

This repository contains a 2D/3D MRI reconstruction pipeline with joint image-motion estimation using the GRICS algorithm [1], implemented in PyTorch with GPU support. GRICS is an algorithm based on modeling of MRI acquisition and motion, and do not use any AI priors. However, it requires a data associated with the displacement (e.g. respiratoiry bellow indications, navigators, PilotTone amplitude variation or other similar data). This implementation aims to improve understanding of the algorithm in the MRI community and support its reuse.

Please contact Karyna Isaieva (karyna [dot] isaieva [at] univ-lorraine [dot] fr) for any bug reports, questions or suggestions.

## License and Citation

This project is distributed under the MIT License. See `LICENSE` for full terms.

Please cite the GRICS paper if you use this code for your research work.

```bibtex
@article{odille2008grics,
  title = {Generalized reconstruction by inversion of coupled systems (GRICS) applied to free-breathing MRI},
  author = {Odille, F. and Vuissoz, P. A. and Marie, P. Y. and Felblinger, J.},
  journal = {Magnetic Resonance in Medicine},
  volume = {60},
  number = {1},
  pages = {146--157},
  year = {2008}
}
```

## Repository layout

- `src/preprocessing/`: data loading, sampling simulation, motion simulation, motion binning
- `src/reconstruction/`: joint reconstructor, encoding/motion operators, CG solver, etc.
- `src/runtime/`: config loading and runtime initialization
- `src/utils/`: plotting, diagnostics, notebook display helpers
- `config/`: TOML configs for reconstruction, sampling, motion simulation, and general runtime

\+ four demos. Attention: random initialization was used, therefore the simulated and reconstruction data may look differently and require an adjustment of the reconstruction parameters.

## Environment Setup

A Dockerfile is provided in the `build/` folder. The built image is available at https://github.com/IADI-Nancy/GRICS-torch/pkgs/container/grics-torch. The `docker.sh` script in the repository root can be used for mounting and runtime setup.

## Config System

The pipeline merges config files at runtime via `src/runtime/runtime_config.py`.

Main config groups:
- `config/general.toml`: paths, debug/runtime flags, k-space normalization and coil sensitivities calculation parameters
- `config/reconstruction/*.toml`: solver/reconstruction settings
- `config/sampling_simulation/*.toml`: synthetic k-space ordering
- `config/motion_simulation/*.toml`: synthetic motion model parameters
- `config/shepp_logan_2d.toml`: Shepp-Logan phantom generation parameters (2D default; use `config/shepp_logan_3d.toml` for 3D)

Important consistency rule:
- `GN_iterations_per_level` must match `ResolutionLevels` length exactly.
- `update_motion_on_final_iteration=false` keeps the conventional image-only final GN iteration.
- `gn_early_stopping=true` stops a level when its image residual increases.
- `save_reconstruction_outputs=true` writes logs, plots, and final images.

## Data Types

The `data_type` selected in `load_config(...)` controls how input data is built or loaded.

### `shepp-logan`

Required config files:
- `config/shepp_logan_2d.toml` (or `config/shepp_logan_3d.toml` for 3D data)
- a sampling simulation config file
- a motion simulation config file (otherwise there is nothing to correct)

### `from_image`

Loaded from a 2D image file and converted to synthetic multi-coil k-space using generated coil maps.
Supported inputs include common image formats (e.g. PNG/JPEG/TIFF) and NumPy arrays (`.npy`, `.npz`).

Required config files:
- `config/from_image.toml`
- a sampling simulation config file
- a motion simulation config file

### `preprocessed-real`

Loaded from a preprocessed HDF5 file with datasets:
- `kspace`: shape `(Ncoils, Nex, Nx, Ny, Nslices)`, complex (`complex64`/`complex128`)
- `motion_data`: shape `(Nslices, Nlines)`, real (`float32`/`float64`) - 1D motion data associated with each k-space line (navigator/respiratory bellow indications, etc.)
- `idx_ky`: shape `(Nslices, Nlines)`, integer (`int32`/`int64`)
- `idx_kz`: shape `(Nslices, Nlines)`, integer (`int32`/`int64`)
- `idx_nex`: shape `(Nslices, Nlines)`, integer (`int32`/`int64`)

For 2D `preprocessed-real`, `ismrmrd-saec`, and `siemens-saec`, the `slice_idx` argument of `DataLoader` selects the slice/partition to load (default: `0` if omitted).
For synthetic data and for all 3D data, do not provide `slice_idx`; the loader raises an error if it is set.

No synthetic sampling is needed in this mode: acquisition order and motion signal come from file. However, additional motion simulation can still be applied.

### `ismrmrd-saec`

Loaded from raw scanner and physiological files using `RawDataReader`:
- the MRI raw data in the ISMRMRD format (`ismrmrd_file`)
- physiological data file in SAEC [2, 3] format (`saec_file`)

The reader converts these files to the arrays used by the `preprocessed-real` mode.
The SAEC sensor channel is configured with `rawdata_sensor_type` in `config/general.toml`.

### `siemens-saec`

Loaded from Siemens raw scanner data and physiological files:
- Siemens raw data file (`siemens_file`, `siemens_raw_file`, or `dat_file`)
- physiological data file in SAEC [2, 3] format (`saec_file`)

The loader first converts the Siemens raw file to ISMRMRD using the `siemens_to_ismrmrd` executable, then reads the result with the same path used by `ismrmrd-saec`.
The SAEC sensor channel is configured with `rawdata_sensor_type` in `config/general.toml`.

When `debug_flag=true`, the loader also writes acquisition-order debug plots into `initial_data_folder` using hardcoded filenames:
- ismrmrd-saec: `ky_order_rawdata_slice{slice_idx}.png`
- siemens-saec: `ky_order_rawdata_slice{slice_idx}.png`
- preprocessed-real: `ky_order_realworld_slice{slice_idx}.png`

## Sampling Modes (synthetic acquisition)

Configured with:
- `kspace_sampling_type`
- `NshotsPerNex`
- `Nex`

Implemented in `src/preprocessing/SamplingSimulator.py`.

When synthetic sampling is generated, per-`nex` debug plots are written to `initial_data_folder` with hardcoded names:
- 2D sampling: `ky_order_nex{nex}.png`
- 3D sampling: `ky_kz_order_nex{nex}.png`

For each `nex`, ky lines are split into `NshotsPerNex` chronological shot blocks:

### `linear`

Shot `s` acquires contiguous band:
- start = `s * Ny / NshotsPerNex`
- end = `(s+1) * Ny / NshotsPerNex`

### `interleaved`

Shot `s` acquires:
- `ky = s, s + NshotsPerNex, s + 2*NshotsPerNex, ...`

### `random`

Independent random permutation per `nex`, then split into `NshotsPerNex` chunks.

## Motion Simulation Modes

Configured with:
- `motion_simulation_model_mode`: `"rigid-realistic"`, `"rigid-per-shot"`, `"non-rigid-realistic"`, `"non-rigid-per-shot"`, or `"as-it-is"`
- `motion_state_mode`: `"realistic"` or `"per-shot"`

`simulated_motion_type` is derived from `motion_simulation_model_mode` for internal/backward-compatible code paths (`"rigid"` or `"non-rigid"`), so experiment scripts should not set both values independently.

Reconstruction uses a separate parameter:
- `reconstruction_motion_type`: `"rigid"` or `"non-rigid"`

in `config/motion_simulation/*.toml` (now one file per `{2D,3D} x {rigid,non-rigid}`).
Implemented in `src/preprocessing/MotionSimulator.py`.

### `as-it-is`

No synthetic corruption added. Only valid for `preprocessed-real`/`ismrmrd-saec`/`siemens-saec` (already motion-corrupted).
`motion_state_mode` must not be set for this mode.

### `rigid` + `motion_state_mode = "per-shot"`

Shot-wise rigid states:
- one rigid transform per shot over all `Nshots = Nex * NshotsPerNex`
- optional global multiplier `rigid_motion_amplitude_scale` scales all configured rigid amplitudes
- random `(tx, ty, phi)` (or `(tx, ty, tz, rx, ry, rz)` for the 3D case) per shot in configured ranges
- piecewise-constant motion in ky-time according to shot order

### `rigid` + `motion_state_mode = "realistic"`

Continuous rigid curve over full acquisition:
- random event times over `Ny * Nex` lines
- smooth raised-cosine transitions (`motion_tau`)
- optional global multiplier `rigid_motion_amplitude_scale` scales all configured rigid amplitudes
- random event amplitudes for `tx`, `ty`, `phi` (or `(tx, ty, tz, rx, ry, rz)` for the 3D case)
- data is then reclustered to `N_motion_states` from the simulated navigator signal (first principal component of the simulated rigid motion parameters)

For corruption, simulation uses one global state per acquired line (`Ny * Nz * Nex` states).

### `non-rigid` + `motion_state_mode = "per-shot"`

Shot-wise non-rigid with fixed spatial basis maps:
- displacement field maps `alpha_x`, `alpha_y` (+ `alpha_z` for 3D) simulate respiration
- a per-shot Gaussian scale is configured with `nonrigid_discrete_s_scale`
- one random scalar per shot (`s`) drives the temporal displacement amplitude (can be interpreted as a navigator or respiratory belt signal)
- displacement at state `m`: `[ux, uy, (uz)] = [alpha_x, alpha_y, (alpha_z)] * s[m]`

### `non-rigid` + `motion_state_mode = "realistic"`

Continuous sinusoidal temporal curve:
- random phase
- random cycles per image in `[nonrigid_resp_cycles_min, nonrigid_resp_cycles_max]`
- normalized to unit amplitude

Spatial maps are the same fixed non-rigid basis (`alpha_x`, `alpha_y` + `alpha_z` for 3D) scaled by `nonrigid_motion_amplitude`.
For corruption, simulation uses one state per acquired line (`Ny * Nz * Nex` states).

## Motion Binning and Reconstruction States

After loading or simulation, the motion curve is clustered with k-means into reconstruction states.

Key points:
- simulation state count and reconstruction state count can differ.
- corruption may be line-wise (`Ny * Nz * Nex` states), but reconstruction uses binned virtual states (`N_motion_states`).
- `N_motion_states` is a manual reconstruction setting from the reconstruction TOML (or from an explicit `load_config(..., N_motion_states=...)` override).

State-count rules are set in `runtime_config.refresh_derived(...)`:
- `motion_state_mode = "per-shot"`: `N_motion_states = Nshots`
- `simulated_motion_type = "rigid"` + `motion_state_mode = "realistic"`: `N_motion_states` stays the manual reconstruction value
- `simulated_motion_type = "non-rigid"` + `motion_state_mode = "realistic"`: `N_motion_states` stays the manual reconstruction value
- `as-it-is`: `N_motion_states` stays the manual reconstruction value

For loaded `preprocessed-real` / `ismrmrd-saec` / `siemens-saec` with an explicit per-shot synthetic simulation mode, `DataLoader` recomputes `Nshots = Nex * NshotsPerNex` from the actual loaded data shape and reapplies the `per-shot` rule after loading. This keeps `N_motion_states` consistent with the file content even if the pre-load config values differ.

Note: you can override the reconstruction state count:

Example:

```python
params = load_config(
    data_type="preprocessed-real",
    reconstruction_config="config/reconstruction/nonrigid_2d.toml",
    N_motion_states=6,
)
```

Note: this manual override is not respected for `per-shot` modes, where `N_motion_states` is forced to `Nshots`

## Python API Reference

The modules expose their APIs directly; import objects from the module paths shown below. Names beginning with `_` are implementation details and are not part of the supported API.

### Configuration and runtime

#### Configuration

```python
from src.runtime.runtime_config import load_config

params = load_config(
    *, data_type, reconstruction_config,
    reconstruction_motion_type=None, simulated_motion_type=None,
    shepp_logan_config=None, from_image_config=None,
    sampling_config=None, motion_simulation_config=None,
    motion_simulation_model_mode=None, motion_simulation_type=None,
    motion_state_mode=None, data_dimension=None,
    kspace_sampling_type=None, NshotsPerNex=None, Nex=None,
    N_motion_states=None, flip_for_display=None, overrides=None,
)
```

`load_config` loads and validates TOML files, applies explicit arguments and `overrides`, creates output folders, and returns a `ConfigBundle`. The bundle groups `paths`, `runtime`, `data`, `sampling`, `motion`, and `reconstruction`, while retaining flat attribute access for compatibility. `PathsConfig`, `RuntimeConfig`, `DataConfig`, `SamplingConfig`, `MotionConfig`, and `ReconstructionConfig` provide `to_flat_dict()`; `ConfigBundle` also provides `from_flat_dict(flat_cfg)`.

#### Runtime

```python
from src.runtime.runtime_setup import cleanup_runtime, initialize_runtime

sp_device, t_device = initialize_runtime(params, print_gpu_info=False)
cleanup_runtime()
```

`initialize_runtime` configures output folders, reproducibility, and CPU/GPU backends and returns the SigPy and PyTorch devices. `cleanup_runtime` performs best-effort memory cleanup without probing CUDA during CPU-only runs.

### Preprocessing

#### Data loading

```python
from src.preprocessing.DataLoader import DataLoader

data = DataLoader(
    params, sp_device=None, t_device=None, filename=None,
    slice_idx=None, run_pipeline=True,
)
```

`DataLoader` supports synthetic, image-based, preprocessed, ISMRMRD/SAEC, and Siemens/SAEC sources. `filename` is a path for image/preprocessed input or an `(MRI_file, saec_file)` sequence/dictionary for raw input. `slice_idx` selects a slice in supported 2D real-data modes. With `run_pipeline=False`, call `load_data()` once and `run_slice_pipeline(slice_idx=None)` for each slice. Primary output attributes are `kspace`, `smaps`, `sampling_idx`, `motion_signal`, `params`, `kspace_scale`, and `motion_plot_context`.

#### Coil sensitivities

```python
from src.preprocessing.CoilSensitivityCalculator import CoilSensitivityCalculator

calculator = CoilSensitivityCalculator(params, sp_device=None)
smaps = calculator.calculate(kspace, reference_kspace=None)
```

`calculate` uses the method selected in `params`; `calculate_espirit(...)` and `calculate_iadi_spline(...)` expose the implementations directly. Inputs and outputs are PyTorch tensors.

#### Sampling and motion simulation

```python
from src.preprocessing.Sampling import Sampling
from src.preprocessing.SamplingSimulator import SamplingSimulator
from src.preprocessing.MotionSimulator import MotionSimulator

sampling_indices = Sampling.build_sampling_per_nex_per_motion(
    binned_ky_indices, device, Nx, Ny,
    Nz=1, binned_kz_indices=None,
)
sampling = SamplingSimulator(Ny, params, t_device="cpu")
simulator = MotionSimulator(
    image, smaps, ky_idx, nex_idx, ky_per_motion_state, params,
    sp_device=None, t_device=None, kz_idx=None, kz_per_motion_state=None,
)
```

`Sampling` converts readout indices grouped by excitation and motion state into the flattened sampling layout consumed by reconstruction operators. `SamplingSimulator` generates configured 2D/3D synthetic acquisition orders and is not required when the acquisition order is already known. `MotionSimulator` applies the configured corruption. Its result accessors are `get_corrupted_kspace()`, `get_corrupted_image()`, `get_rigid_motion_information_2d()`, `get_rigid_motion_information_3d()`, and `get_nonrigid_motion_information()`. Mode-specific entry points are `simulate_realistic_rigid_motion()`, `simulate_discrete_rigid_motion()`, `simulate_realistic_non_rigid_motion()`, and `simulate_discrete_non_rigid_motion()`.

#### Raw data

```python
from src.preprocessing.RawDataReader import RawDataReader

reader = RawDataReader(
    ismrmrd_file, saec_file, sensor_type="BELT", device="cpu", debug=False
)
data = reader.read_data_from_rawdata(h5filename=None, slice_idx=None)
```

`RawDataReader` reads ISMRMRD acquisitions and aligns the selected SAEC physiological channel. Supplying `h5filename` also writes the processed representation.

`GRICSPreparerAPI` is the high-level acquisition-preparation interface for external applications and real-world GRICS workflows. Its constructor calls `load_config(...)` and `initialize_runtime(...)` once. `prepare_acquisition(...)` then applies motion binning and constructs one or more caller-named sampling layouts. The Boolean masks supplied through `sampling_masks` contain one value per chronological readout; their names and split policy are entirely caller-defined. The returned `PreparedGRICSAcquisition` contains the per-acquisition parameters, motion centers and labels, named sampling indices, and chronological `ky`, `kz`, and `nex` indices. These values depend only on acquisition metadata and should be cached rather than recomputed in iterative reconstruction or training loops.

```python
from src.preprocessing.GRICSPreparerAPI import GRICSPreparerAPI

preparer = GRICSPreparerAPI(
    reconstruction_config, overrides=runtime_overrides,
)
prepared = preparer.prepare_acquisition(
    motion_data, ky_indices, nex_indices, Nx=Nx, Ny=Ny,
    sampling_masks={
        "reconstruction": retained_readouts,
        "validation": validation_readouts,
    },
    kspace=kspace, seed=seed,
)
reconstruction_indices = prepared.sampling_indices["reconstruction"]
validation_indices = prepared.sampling_indices["validation"]
motion_signal = prepared.motion_signal
params = prepared.params
```

`MotionBinner.bin_motion(...)` publicly groups chronological readouts by motion state; ordinary reconstruction uses it through `DataLoader`. `RespiratoryDataReader` remains internal. `ConstantPhysiologicalSignalError` reports physiological signals without usable motion variation.

### Reconstruction

#### Joint reconstruction

```python
from src.reconstruction.JointReconstructor import JointReconstructor

reconstructor = JointReconstructor(
    KspaceData, smaps, SamplingIndices, motion_signal, params,
    kspace_scale=1.0, motion_plot_context=None,
    initial_image=None, initial_motion=None,
    external_image_regularizer=None,
)
image, motion_model = reconstructor.run()

image, motion_model = reconstructor.full_resolutions_gauss_newton_iteration_api(
    image, motion_model, image_regularizer=None,
    regularization_weight=None, update_motion=True,
    image_cg_iterations=None, motion_cg_iterations=None,
)

predicted = reconstructor.predict_kspace_api(
    image, motion_model, sampling_indices=None,
)
```

`JointReconstructor.run()` runs multi-resolution joint image/motion reconstruction. `initial_image` and `initial_motion` optionally initialize the first level. `full_resolutions_gauss_newton_iteration_api()` performs one image update followed by an optional motion update at full resolution; its CG limits default to `params.max_iter_recon` and `params.max_iter_motion`. When `image_regularizer` is supplied, it produces a prior `z` and the image step minimizes `||E x - y||_2^2 + lambda ||x - z||_2^2`. `predict_kspace_api()` evaluates an image and motion state on either the reconstruction sampling layout or an explicitly supplied layout.

`external_image_regularizer` is an optional callable `regularizer(image) -> image_prior`. It must accept the current complex image, preserve its shape, and return a tensor compatible with the reconstruction device and dtype. It must preserve its computation graph when training through reconstruction. For each image update it supplies `z = regularizer(x0)` to `(EᴴE + λI)x = Eᴴy + λz`. Passing `None` retains the standard GRICS image solve.

#### Encoding and CG

```python
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver

E = EncodingOperator(smaps, Nsamples, SamplingIndices, Nex, motionOperator)
kspace = E.forward(image)
image_adjoint = E.adjoint(kspace)
normal_image = E.normal(image)

solver = ConjugateGradientSolver(
    E, reg_lambda=0.0, regularizer="Tikhonov",
    regularization_shape=None, regularization_spatial_dims=None,
    verbose=False, early_stopping=True, true_residual_interval=10,
    max_stag_steps=3, max_more_steps=None,
    use_reg_scale_proxy=False, reg_scale_num_probes=8,
)
x = solver.cg(b, x0=None, max_iter=20, tol=1e-3, differentiable=False)
```

`EncodingOperator` implements the motion-aware MRI forward, adjoint, and normal operators. `ConjugateGradientSolver` solves their regularized normal equation with Tikhonov or gradient-Tikhonov regularization. `differentiable=True` retains PyTorch autograd through CG; convergence diagnostics are stored in `solver.last_info`.

#### Motion operators

```python
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator

motion = MotionOperator(
    Nx, Ny, alpha, motion_type, centers=None, motion_signal=None, Nz=1
)
J = MotionPerturbationSimulator(
    smaps, Nsamples, SamplingIndices, Nex, image, motion
)
delta_kspace = J.forward(motion_model_perturbation)
delta_motion = J.adjoint(residual_kspace)
normal_delta = J.normal(motion_model_perturbation)
```

`MotionOperator` constructs rigid or non-rigid 2D/3D warp operators. Non-rigid operation requires `motion_signal`; rigid rotation may use explicit `centers`. `MotionPerturbationSimulator` is the Jacobian-like operator used for Gauss-Newton motion updates.

### DICOM and image transforms

`write_reconstruction_dicom(image, output_path, raw_data=None, *, ismrmrd_file=None, slice_index=None, series_description="GRICS reconstruction", study_instance_uid=None, series_instance_uid=None, frame_of_reference_uid=None, images_in_acquisition=None, series_number=None, reference_dicom_path=None)` writes one MR DICOM. It accepts NumPy or PyTorch images shaped `[Nx, Ny]`, `[Nex, Nx, Ny]`, `[Nx, Ny, 1]`, or `[Nex, Nx, Ny, 1]`; repetitions are averaged and complex input is exported as magnitude. Metadata comes from `raw_data` or `ismrmrd_file`, optionally supplemented by a reference DICOM.

Image-transform APIs are:

- `fftnc(x, dims=(-4, -3, -2))` and `ifftnc(X, dims=(-4, -3, -2))`: centered unitary FFT and inverse FFT.
- `zero_fill_image_to_shape(image, target_shape, *, encoded_shape, spatial_dims=(0, 1, 2))`: Gadgetron-style FFT zero-fill and crop.
- `zero_fill_grics_image_to_shape(image, target_shape, *, encoded_shape, spatial_dims=(-2, -1))`: zero-fill a GRICS image to its final matrix.

### Motion and display utilities

Public motion helpers in `src.utils.motion_simulator_utils` are:

- `require_motion_param(params, name)`, `rigid_motion_amplitude_scale(params)`, and `translation_limits_px(params, Nx, Ny, Nz)`.
- `flatten_index_list(values)` and `num_motion_readouts(ky_idx)`.
- `build_sampling_per_line_global_states(ky_idx, nex_idx, kz_idx, *, device, Nx, Ny, Nz, Nex)`.
- `compress_consecutive_rigid_states(alpha, ky_idx, nex_idx, *, device, Nx, Ny, Nz, Nex, centers=None, kz_idx=None)`.
- `globalize_per_shot_readout_layout(per_shot_readout_layout, *, device)` and `expand_motion_states_to_readouts(readout_layout, state_curves, *, device)`.
- `build_event_transition_curve(event_idx, transition_length, n_samples, *, device)`, `build_navigator_from_motion_matrix(motion_matrix)`, and `build_rigid_rotation_centers(translation_limits, n_states, *, device, Nx, Ny, Nz)`.

Non-rigid display conversion is provided by `to_cartesian_components(...)`, `flip_nonrigid_alpha_for_display(...)`, and `split_nonrigid_alpha_components(...)` in `src.utils.nonrigid_display`.

Plotting APIs are `show_and_save_image`, `save_alpha_component_map`, `save_nonrigid_quiver_with_contours`, `save_alpha_component_map_3d`, `save_nonrigid_quiver_with_contours_3d`, `save_nonrigid_alpha_plots`, `compute_motion_plot_y_limits`, `save_clustered_motion_plots`, `save_motion_debug_plots`, `save_line_plot`, `save_residual_subplots`, `save_final_nonrigid_alpha_maps`, and `save_final_rigid_motion_plots`.

Notebook display APIs are `display_run_panels`, `display_input_sampling_motion_panels`, `display_3d_image_matrix`, and `display_logs_and_motion_same_as_2d`. They honor the display flags in `params` where applicable.

## Outputs

Each run writes into folders from `config/general.toml`:

- `initial_data/`: sampling order, motion curves, corrupted and ground-truth images (if they exist), and simulated motion (if it exists)
- `debug_outputs/`: results per reconstruction level
- `logs/`: residual curves and run log
- `results/`: final reconstructed outputs

By default, these folders are cleaned before each run (`clean_output_folders_before_run = true`).

## Disclosure

Parts of this code and its documentation were developed with the assistance of AI tools (e.g., ChatGPT, Codex, Claude, Copilot). All content has been reviewed and validated by a human.

## References

[1] Odille, F., Vuissoz, P. A., Marie, P. Y., & Felblinger, J. (2008). Generalized reconstruction by inversion of coupled systems (GRICS) applied to free‐breathing MRI. Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine, 60(1), 146-157.
[2] Isaieva, K., Fauvel, M., Weber, N., Vuissoz, P. A., Felblinger, J., Oster, J., & Odille, F. (2022). A hardware and software system for MRI applications requiring external device data. Magnetic Resonance in Medicine, 88(3), 1406-1418.
[3] https://github.com/IADI-Nancy/wrapperHDF5 

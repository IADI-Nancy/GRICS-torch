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
- `config/`: TOML configuration root
  - `config/reconstruction/`: solver and reconstruction pipelines
  - `config/sampling_simulation/`: synthetic k-space acquisition ordering
  - `config/motion_simulation/`: synthetic motion models
  - `config/synthetic_data/`: Shepp-Logan phantom and image-source generation settings
- `pipelines/`: executable end-to-end reconstruction pipelines for real acquisitions; `siemens_breast_T2.py` reproduces the Gadgetron pipeline implemented in [2]

\+ four demos. Attention: random initialization was used, therefore the simulated and reconstruction data may look differently and require an adjustment of the reconstruction parameters.

## Environment Setup

A Dockerfile is provided in the `build/` folder. The built image is available at https://github.com/IADI-Nancy/GRICS-torch/pkgs/container/grics-torch. The `docker.sh` script in the repository root can be used for mounting and runtime setup.

## Configuration

Main config types:

- `config/general.toml`: paths, runtime flags, and k-space normalization; loaded automatically
- `config/coil_sensitivity/*.toml`: one explicitly selected coil-sensitivity method and only that method's settings
- `config/reconstruction/*.toml`: reconstruction model, multiresolution GN iterations, regularization, and CG solver settings; always required
- `config/synthetic_data/*.toml`: Shepp-Logan phantom or image-source generation settings
- `config/real_data/saec.toml`: SAEC physiological sensor selection, loaded only for SAEC inputs
- `config/real_data/ismrmrd_reader.toml`: ISMRMRD-reader diagnostics, loaded only for ISMRMRD or Siemens raw inputs
- `config/sampling_simulation/*.toml`: simulated k-space acquisition ordering
- `config/motion_simulation/*.toml`: selected simulated rigid or non-rigid motion modes
- `config/motion_simulation/common/*.toml`: shared motion parameters, loaded only through a selected motion mode

Motion configurations may use one `[motion] include = "relative/path.toml"` entry.
The included common file is loaded first; a selected file cannot redefine any included
setting, include paths cannot leave their directory tree, and include cycles are rejected.
Common files are not runnable configurations because they do not declare a motion mode.

Use `load_config(...)` to load the config files. Use `overrides={...}` for run-specific changes. See the demos for complete configuration, runtime initialization, data loading, and reconstruction examples.

### Configuration ownership and validation

Each TOML file accepts only its own settings and sections: general runtime/paths,
reconstruction, synthetic source, sampling, motion simulation, or postprocessing.
Unknown keys, misplaced keys, old aliases, invalid types, non-finite numbers, and
incompatible combinations raise errors. `load_config` selects configuration files;
run-specific values are supplied only through `overrides`. Notebook mode disables `verbose`
and `print_to_console` unless explicitly supplied in `overrides`. Each automatic
change is announced with an informational message. Outside notebooks, TOML logging
values are preserved.

All numerical and diagnostic settings are specified in TOML. Only `"2D"` and
`"3D"` are valid dimensions. Real input dimensions follow the selected reconstruction
file. Every `load_config` call also selects exactly one CSM file: use
`config/coil_sensitivity/espirit.toml` for ESPIRiT or
`config/coil_sensitivity/odille_spline.toml` for Odille spline maps. The CSM
method itself cannot be overridden. For real data, omitted sampling loads
`config/sampling_simulation/from_data.toml` and reads repetition counts from the
acquisition. Do not supply simulated shot counts or acceleration settings in this mode. Real data
without a motion simulation file loads `config/motion_simulation/as_is.toml`.

To simulate a new acquisition order over real k-space, select a simulated sampling
file and simulated motion. The original acquisition indices and physiological
trace are ignored; `Nex` must equal the repetitions in the k-space array. A
preprocessed HDF5 input then needs only `kspace` (and optional `reference_kspace`).
Raw inputs in this mode can be provided as a single MRI filename, without physiology.
`from-data` is invalid for synthetic sources. Reordering does not fill missing
k-space samples or remove motion already present in the supplied values.

Per-shot simulation automatically replaces the positive integer `N_motion_states`
with the shot count and announces any change. Set
`N_motion_states_per_level="full"` to use all states at every resolution, or provide
a list of counts. Resolution levels must increase in `(0, 1]` and end at `1.0`.
GN iteration counts must be an explicit list of positive integers, one per level.

ESPIRiT settings are named `espirit_calibration_width` and `espirit_kernel_width`.
Requested calibration widths must fit the data; they are never silently reduced.
Set `seed_enabled=false` to disable seeding. GPU unavailability still triggers a
CPU fallback and prints a visible runtime message.

### Postprocessing

`config/postprocessing/nonrigid_2d_breast.toml` owns
`normalize_image_by_grics_reference`. Load it with
`load_postprocessing_config(path, overrides=...)`. The breast pipeline loads this
separately and applies it after reconstruction; it requires Odille spline coil
maps when enabled. Reconstruction files and reconstruction overrides cannot set
postprocessing options.

### Runtime diagnostics

Runtime diagnostics are configured by scope: `save_debug_plots` and `use_deterministic_algorithms` are global runtime settings; `check_simulated_motion_consistency` belongs to non-rigid motion TOMLs; and `print_raw_calibration_lines` belongs to `config/real_data/ismrmrd_reader.toml`. These replace the former combined `debug_flag`; old overrides are rejected as unknown settings. Direct `RawDataReader` and `RawDataPreparer` callers should use `print_raw_calibration_lines=` instead of `debug=`.

## Data Types

The `data_type` selected in `load_config(...)` controls how input data is built or loaded.

### `shepp-logan`

Required config files:
- `config/synthetic_data/shepp_logan_2d.toml` (or `config/synthetic_data/shepp_logan_3d.toml` for 3D data)
- a sampling simulation config file
- a motion simulation config file (otherwise there is nothing to correct)

### `from_image`

Loaded from a 2D image file and converted to synthetic multi-coil k-space using generated coil maps.
Supported inputs include common image formats (e.g. PNG/JPEG/TIFF) and NumPy arrays (`.npy`, `.npz`).

Required config files:
- `config/synthetic_data/from_image.toml`
- a sampling simulation config file
- a motion simulation config file

### `preprocessed-real`

Loaded from a preprocessed HDF5 file with datasets:
- `kspace`: shape `(Ncoils, Nex, Nx, Ny, Nslices)`, complex (`complex64`/`complex128`)
- `motion_data`: shape `(Nslices, Nlines)`, real (`float32`/`float64`) - 1D motion data associated with each k-space line (navigator/respiratory bellow indications, etc.)
- `idx_ky`: shape `(Nslices, Nlines)`, integer (`int32`/`int64`)
- `idx_kz`: shape `(Nslices, Nlines)`, integer (`int32`/`int64`)
- `idx_nex`: shape `(Nslices, Nlines)`, integer (`int32`/`int64`)

For 2D `preprocessed-real`, `ismrmrd-saec`, and `siemens-saec`, `slice_idx` selects the slice/partition to load. It may be omitted only when the source contains exactly one slice; multi-slice sources require an explicit value.
For synthetic data and for all 3D data, do not provide `slice_idx`; the loader raises an error if it is set.

No synthetic sampling is needed in this mode: acquisition order and motion signal come from file. However, additional motion simulation can still be applied.

### `ismrmrd-saec`

Loaded from raw scanner and physiological files using `RawDataReader`:
- the MRI raw data in the ISMRMRD format (`ismrmrd_file`)
- physiological data file in SAEC [3, 4] format (`saec_file`)
- `config/real_data/saec.toml` and `config/real_data/ismrmrd_reader.toml`

The reader converts these files to the arrays used by the `preprocessed-real` mode.
The SAEC sensor channel is configured with `rawdata_sensor_type` in `config/real_data/saec.toml`.

### `ismrmrd-polaris` and `siemens-polaris`

Select these types with `load_config(data_type=...)`. Pass `DataLoader` a pair
`(mri_file, tracking_tsv)` or a dictionary containing `ismrmrd_file` / `siemens_raw_file`
and `polaris_file`. The Siemens variant converts the MRI file to ISMRMRD first.
Both require `config/real_data/ismrmrd_reader.toml`.
Polaris filtering and normalization are handled by `PolarisInfraredTrackerReader`;
no `rawdata_sensor_type` setting is required. Both types support 2D slice selection
and 3D volume loading, with sampling read from the acquisition data.

### `siemens-saec`

Loaded from Siemens raw scanner data and physiological files:
- Siemens raw data file (`siemens_raw_file`)
- physiological data file in SAEC [2, 3] format (`saec_file`)
- `config/real_data/saec.toml` and `config/real_data/ismrmrd_reader.toml`

The loader first converts the Siemens raw file to ISMRMRD using the `siemens_to_ismrmrd` executable, then reads the result with the same path used by `ismrmrd-saec`.
The SAEC sensor channel is configured with `rawdata_sensor_type` in `config/real_data/saec.toml`.

### Planned: `ismrmrd-text` and `siemens-text`

These data types are planned for the near future. They will accept physiological or motion measurements from a text file instead of requiring the SAEC format, enabling raw-data reconstruction for users without SAEC acquisition files. `ismrmrd-text` will use ISMRMRD MRI data, while `siemens-text` will use Siemens raw MRI data. These modes are not implemented yet.

When `save_debug_plots=true`, every real-world input mode uses the same source-independent acquisition-order filename in `initial_data_folder`: `ky_order_acquisition_slice{slice_idx}.png`. This convention also applies to the planned text-based modes.

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

Configured with one variable:
- `simulated_motion_type`: `"rigid-realistic"`, `"rigid-per-shot"`, `"non-rigid-realistic"`, `"non-rigid-per-shot"`, or `"as-it-is"`.

`simulated_motion_type` is configured in `config/motion_simulation/*.toml`. Reconstruction independently uses `reconstruction_motion_type` (`"rigid"` or `"non-rigid"`) from `config/reconstruction/*.toml`.
Implemented in `src/preprocessing/MotionSimulator.py`.

### `as-it-is`

No synthetic corruption added. Valid only for real-data types with `from-data` sampling, including SAEC and Polaris inputs.

### `rigid-per-shot`

Shot-wise rigid states:
- one rigid transform per shot over all `Nshots = Nex * NshotsPerNex`
- explicit global multiplier `rigid_motion_amplitude_scale` scales all configured rigid amplitudes
- random `(tx, ty, phi)` (or `(tx, ty, tz, rx, ry, rz)` for the 3D case) per shot in configured ranges
- piecewise-constant motion in ky-time according to shot order

### `rigid-realistic`

Continuous rigid curve over full acquisition:
- random event times over `Ny * Nex` lines
- smooth raised-cosine transitions (`motion_tau`)
- explicit global multiplier `rigid_motion_amplitude_scale` scales all configured rigid amplitudes
- random event amplitudes for `tx`, `ty`, `phi` (or `(tx, ty, tz, rx, ry, rz)` for the 3D case)
- data is then reclustered to `N_motion_states` from the simulated navigator signal (first principal component of the simulated rigid motion parameters)

For corruption, simulation uses one global state per acquired line (`Ny * Nz * Nex` states).

### `non-rigid-per-shot`

Shot-wise non-rigid with fixed spatial basis maps:
- displacement field maps `alpha_x`, `alpha_y` (+ `alpha_z` for 3D) simulate respiration
- a per-shot Gaussian scale is configured with `nonrigid_discrete_s_scale`
- one random scalar per shot (`s`) drives the temporal displacement amplitude (can be interpreted as a navigator or respiratory belt signal)
- displacement at state `m`: `[ux, uy, (uz)] = [alpha_x, alpha_y, (alpha_z)] * s[m]`

### `non-rigid-realistic`

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
- `N_motion_states` is a manual reconstruction setting from the reconstruction TOML, or an explicit `overrides={"N_motion_states": ...}` value.

State-count rules:
- `rigid-per-shot` and `non-rigid-per-shot`: `N_motion_states` is automatically replaced by the shot count; changes are announced.
- `rigid-realistic`, `non-rigid-realistic`, and `as-it-is`: `N_motion_states` stays at the manual reconstruction value

For real data with generated sampling, configured shot counts are preserved and `Nex` must match the array. With `from-data` sampling and per-shot simulated motion, each recorded readout is a state; the count is resolved from the actual acquisition indices after loading.


## Outputs

Each run writes into folders from `config/general.toml`:

- `initial_data/`: sampling order, motion curves, corrupted and ground-truth images (if they exist), and simulated motion (if it exists)
- `debug_outputs/`: results per reconstruction level
- `logs/`: residual curves and run log
- `results/`: final reconstructed outputs

By default, these folders are cleaned before each run (`clean_output_folders_before_run = true`).

## External Integration APIs

This section covers the standard reconstruction entry point and the additional interfaces intended for external reconstruction or training code. Exact input types and tensor dimensions are available directly in each Python signature and docstring.

### Data preparation API

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

#### Optional undersampling

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

Every mask contains one Boolean per chronological readout. Keys are caller-defined, and all named layouts share the same motion-state labels.

### Standard reconstruction entry point

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

For `params`, use `data.params` from `DataLoader` or `prepared.params` from `GRICSPreparerAPI`.

### Full-resolution iteration and prediction APIs

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

## Disclosure

Parts of this code and its documentation were developed with the assistance of AI tools (e.g., ChatGPT, Codex, Claude, Copilot). All content has been reviewed and validated by a human.

## References

[1] Odille, F., Vuissoz, P. A., Marie, P. Y., & Felblinger, J. (2008). Generalized reconstruction by inversion of coupled systems (GRICS) applied to free‐breathing MRI. Magnetic Resonance in Medicine, 60(1), 146-157.
[2] Isaieva, K., Meullenet, C., Vuissoz, P. A., Fauvel, M., Nohava, L., Laistler, E., ... & Odille, F. (2023). Feasibility of online non‐rigid motion correction for high‐resolution supine breast MRI. Magnetic Resonance in Medicine, 90(5), 2130-2143.
[3] Isaieva, K., Fauvel, M., Weber, N., Vuissoz, P. A., Felblinger, J., Oster, J., & Odille, F. (2022). A hardware and software system for MRI applications requiring external device data. Magnetic Resonance in Medicine, 88(3), 1406-1418.
[4] https://github.com/IADI-Nancy/wrapperHDF5

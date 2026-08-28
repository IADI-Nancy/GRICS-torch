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

- `config/general.toml`: paths, runtime flags, k-space normalization, and coil-sensitivity settings; loaded automatically
- `config/reconstruction/*.toml`: reconstruction model, multiresolution GN iterations, regularization, and CG solver settings; always required
- `config/synthetic_data/*.toml`: Shepp-Logan phantom or image-source generation settings
- `config/sampling_simulation/*.toml`: simulated k-space acquisition ordering
- `config/motion_simulation/*.toml`: simulated rigid or non-rigid motion settings

Use `load_config(...)` to load the config files. Use `overrides={...}` for run-specific changes. See the demos for complete configuration, runtime initialization, data loading, and reconstruction examples.

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

The reader converts these files to the arrays used by the `preprocessed-real` mode.
The SAEC sensor channel is configured with `rawdata_sensor_type` in `config/general.toml`.

### `siemens-saec`

Loaded from Siemens raw scanner data and physiological files:
- Siemens raw data file (`siemens_file`, `siemens_raw_file`, or `dat_file`)
- physiological data file in SAEC [2, 3] format (`saec_file`)

The loader first converts the Siemens raw file to ISMRMRD using the `siemens_to_ismrmrd` executable, then reads the result with the same path used by `ismrmrd-saec`.
The SAEC sensor channel is configured with `rawdata_sensor_type` in `config/general.toml`.

### Planned: `ismrmrd-text` and `siemens-text`

These data types are planned for the near future. They will accept physiological or motion measurements from a text file instead of requiring the SAEC format, enabling raw-data reconstruction for users without SAEC acquisition files. `ismrmrd-text` will use ISMRMRD MRI data, while `siemens-text` will use Siemens raw MRI data. These modes are not implemented yet.

When `debug_flag=true`, every real-world input mode uses the same source-independent acquisition-order filename in `initial_data_folder`: `ky_order_acquisition_slice{slice_idx}.png`. This convention also applies to the planned text-based modes.

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

No synthetic corruption added. Only valid for `preprocessed-real`/`ismrmrd-saec`/`siemens-saec` (already motion-corrupted).

### `rigid-per-shot`

Shot-wise rigid states:
- one rigid transform per shot over all `Nshots = Nex * NshotsPerNex`
- optional global multiplier `rigid_motion_amplitude_scale` scales all configured rigid amplitudes
- random `(tx, ty, phi)` (or `(tx, ty, tz, rx, ry, rz)` for the 3D case) per shot in configured ranges
- piecewise-constant motion in ky-time according to shot order

### `rigid-realistic`

Continuous rigid curve over full acquisition:
- random event times over `Ny * Nex` lines
- smooth raised-cosine transitions (`motion_tau`)
- optional global multiplier `rigid_motion_amplitude_scale` scales all configured rigid amplitudes
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
- `N_motion_states` is a manual reconstruction setting from the reconstruction TOML (or from an explicit `load_config(..., N_motion_states=...)` override).

State-count rules:
- `rigid-per-shot` and `non-rigid-per-shot`: `N_motion_states = Nshots`
- `rigid-realistic`, `non-rigid-realistic`, and `as-it-is`: `N_motion_states` stays at the manual reconstruction value

For loaded `preprocessed-real` / `ismrmrd-saec` / `siemens-saec` with an explicit per-shot synthetic simulation mode, `DataLoader` recomputes `Nshots = Nex * NshotsPerNex` from the actual loaded data shape and reapplies the `per-shot` rule after loading.


## Outputs

Each run writes into folders from `config/general.toml`:

- `initial_data/`: sampling order, motion curves, corrupted and ground-truth images (if they exist), and simulated motion (if it exists)
- `debug_outputs/`: results per reconstruction level
- `logs/`: residual curves and run log
- `results/`: final reconstructed outputs

By default, these folders are cleaned before each run (`clean_output_folders_before_run = true`).

## External Integration APIs

Most public functions in the repository are ordinary GRICS-torch components and are documented by their source docstrings. This section covers only the additional interfaces intended for external reconstruction or training code.

### Standard reconstruction entry point

`JointReconstructor.run()` is the standard GRICS-torch entry point, rather than an additional integration API. It runs the complete configured multi-resolution reconstruction and returns the reconstructed image and motion model.

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

`initial_image` and `initial_motion` optionally initialize the coarsest resolution level. `external_image_regularizer` optionally supplies an image prior during the standard reconstruction.

### Acquisition preparation API

`GRICSPreparerAPI` is a lightweight adapter for an external application that already owns and loads its image or k-space tensors. Given only the chronological motion measurements and acquisition indices, it loads the GRICS configuration, bins the readouts into motion states, and builds the sampling indices expected by GRICS operators. It does not load k-space, calculate sensitivity maps, simulate motion, or reconstruct an image.

The usual external use has one sampling layout containing all acquired readouts:

```python
from src.preprocessing.GRICSPreparerAPI import GRICSPreparerAPI

preparer = GRICSPreparerAPI(
    reconstruction_config, overrides=runtime_overrides,
)
prepared = preparer.prepare_acquisition(
    motion_data, ky_indices, nex_indices,
    Nx=Nx, Ny=Ny, kz_indices=kz_indices,
    kspace=kspace, seed=seed,
)

sampling_indices = prepared.sampling_indices["all"]
motion_signal = prepared.motion_signal
params = prepared.params
```

Here, `sampling_indices` is the `[Nex][N_motion_states]` sampling layout passed to `JointReconstructor` and the encoding operators. Each entry contains the flattened k-space locations acquired for one excitation and one binned motion state. The name `"all"` is created automatically when `sampling_masks` is omitted.

#### Optional undersampling

`GRICSPreparerAPI` supports externally defined undersampling through `sampling_masks`. It does not generate an acceleration pattern: the caller decides which chronological readouts are retained, and the preparer converts that selection into the `[Nex][N_motion_states]` indices required by GRICS.

For example, a 2D phase-encode mask of length `Ny` can be mapped to chronological readouts using their `ky` indices:

```python
# Created by the external application; True means that ky is retained.
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

Each mask must contain one Boolean value per chronological readout. The dictionary keys are arbitrary caller-defined names. The retained layout can be passed to `JointReconstructor`; a held-out layout can be passed to `predict_kspace_api()` for an external loss or evaluation.

Motion binning is performed once using the complete supplied motion trace, and every named sampling layout uses those same motion-state labels.

### Full-resolution iteration and prediction APIs

The following methods can be used by external iterative or unrolled reconstruction codes.

`full_resolutions_gauss_newton_iteration_api()` performs one full-resolution image update followed by an optional motion update. The CG limits default to `params.max_iter_recon` and `params.max_iter_motion`. If `image_regularizer` returns a prior `z`, the image step uses `lambda * ||x - z||_2^2`; passing `None` retains the standard GRICS image solve.

```python
image, motion_model = reconstructor.full_resolutions_gauss_newton_iteration_api(
    image, motion_model,
    image_regularizer=None, regularization_weight=None,
    update_motion=True,
    image_cg_iterations=None, motion_cg_iterations=None,
)
```

`predict_kspace_api()` evaluates an image and motion model using either the reconstruction sampling layout or an explicitly supplied layout.

```
predicted_kspace = reconstructor.predict_kspace_api(
    image, motion_model, sampling_indices=None,
)
```

## Disclosure

Parts of this code and its documentation were developed with the assistance of AI tools (e.g., ChatGPT, Codex, Claude, Copilot). All content has been reviewed and validated by a human.

## References

[1] Odille, F., Vuissoz, P. A., Marie, P. Y., & Felblinger, J. (2008). Generalized reconstruction by inversion of coupled systems (GRICS) applied to free‐breathing MRI. Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine, 60(1), 146-157.
[2] Isaieva, K., Meullenet, C., Vuissoz, P. A., Fauvel, M., Nohava, L., Laistler, E., ... & Odille, F. (2023). Feasibility of online non‐rigid motion correction for high‐resolution supine breast MRI. Magnetic Resonance in Medicine, 90(5), 2130-2143.
[3] Isaieva, K., Fauvel, M., Weber, N., Vuissoz, P. A., Felblinger, J., Oster, J., & Odille, F. (2022). A hardware and software system for MRI applications requiring external device data. Magnetic Resonance in Medicine, 88(3), 1406-1418.
[4] https://github.com/IADI-Nancy/wrapperHDF5

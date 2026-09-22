# GRICS-torch: GRICS MRI motion-corrected reconstruction in PyTorch

This repository contains a 2D/3D MRI reconstruction pipeline with joint image-motion estimation using the GRICS algorithm [1], implemented in PyTorch with GPU support. GRICS models MRI acquisition and motion and does not require AI priors. It requires motion-related data (e.g. respiratory belt measurements, navigators, PilotTone amplitude variations, or similar signals). This implementation aims to improve understanding of the algorithm in the MRI community and support its reuse.

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
## Environment Setup

A Dockerfile is provided in the `build/` folder. The built image is available at https://github.com/IADI-Nancy/GRICS-torch/pkgs/container/grics-torch. The `docker.sh` script in the repository root can be used for mounting and runtime setup.

## Repository layout

- `src/preprocessing/`: data loading, sampling simulation, motion simulation, motion binning
- `src/reconstruction/`: joint reconstructor, encoding/motion operators, CG solver, etc.
- `src/runtime/`: config loading and runtime initialization
- `src/utils/`: plotting, diagnostics, notebook display helpers
- `config/`: TOML configuration root
- `pipelines/`: executable end-to-end reconstruction pipelines for real acquisitions; `siemens_breast_T2.py` reproduces the Gadgetron pipeline implemented in [2]

Four demo notebooks cover simulated and real-data reconstruction. Simulations and some reconstruction steps use random initialization; `seed_enabled` and `seed` in `config/general.toml` control reproducibility. Results can vary with the seed and compute backend.

## Configuration

- `config/general.toml`: paths, runtime flags, and k-space normalization; loaded automatically
- `config/coil_sensitivity/*.toml`: one explicitly selected coil-sensitivity method and only that method's settings
- `config/reconstruction/*.toml`: reconstruction model, multiresolution GN iterations, regularization, and CG solver settings; always required
- `config/synthetic_data/*.toml`: Shepp-Logan phantom or image-source generation settings (only if synthetic data is used)
- `config/real_data/`: configurations for loading real MRI and physiological data
    - `config/real_data/ismrmrd_reader.toml`: ISMRMRD-reader settings, loaded only for ISMRMRD or Siemens raw inputs
    - `config/real_data/saec.toml`: SAEC physiological sensor selection, loaded only if a SAEC file is used
    - `config/real_data/polaris.toml`: Polaris infrared camera channel selection, required only if this sensor is used
    - `config/real_data/physio_text.toml`: clock-shift settings for physiological text inputs, loaded automatically for that mode
    - `config/real_data/physio_array.toml`: clock-shift settings for physiological array inputs, loaded automatically for that mode
- `config/sampling_simulation/*.toml`: simulated k-space acquisition ordering
- `config/motion_simulation/*.toml`: selected simulated rigid or non-rigid motion modes
- `config/motion_simulation/common/*.toml`: shared motion parameters, loaded only through a selected motion mode
- `config/postprocessing/nonrigid_2d_breast.toml`: reference-image normalization for the Siemens breast pipeline, loaded with `load_postprocessing_config(...)`

Use `load_config(...)` to load the config files. Always supply `reconstruction_config` and `coil_sensitivity_config`. Real data defaults to `from-data` sampling and `as-it-is` motion when those configuration files are omitted. Use `overrides={...}` for run-specific changes. See the demos for complete configuration, runtime initialization, data loading, and reconstruction examples.

## Data Types

The `data_type` selected in `load_config(...)` controls how input data is built or loaded.
For 2D real data, `slice_idx` selects the slice/partition to reconstruct. It may be omitted only when the source contains exactly one slice; multi-slice sources require an explicit value before running the slice pipeline.
For synthetic data and for all 3D data, do not provide `slice_idx`; the loader raises an error if it is set.

### Synthetic data types
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

### Real data types
When `save_debug_plots=true` and sampling is `from-data`, every real-world input mode uses the same source-independent acquisition-order filename in the reconstruction’s `preprocessing/` folder: `ky_order_acquisition_slice{slice_idx}.png`.

### `preprocessed-real`

For 2D multi-slice data, load a preprocessed HDF5 file with datasets:
- `kspace`: shape `(Ncoils, Nex, Nx, Ny, Nslices)`, complex (`complex64`/`complex128`)
- `motion_data`: shape `(Nslices, Nlines)` for one channel or `(Nslices, Nlines, Nchannels)` for multiple channels, real (`float32`/`float64`) - motion data associated with each k-space line (navigator/respiratory bellow indications, etc.)
- `idx_ky`: shape `(Nslices, Nlines)`, integer (`int32`/`int64`)
- `idx_kz`: shape `(Nslices, Nlines)`, integer (`int32`/`int64`)
- `idx_nex`: shape `(Nslices, Nlines)`, integer (`int32`/`int64`)

For 3D data, `kspace` has shape `(Ncoils, Nex, Nx, Ny, Nz)` and `motion_data` has shape `(Nreadout, Nchannels)`. Acquisition indices contain one entry per readout in full acquisition order; the reader-generated layout is `(1, Nreadout)`. Index values are zero-based.

No synthetic sampling is needed with `from-data`: acquisition order and motion signals come from the file. Additional motion simulation can still be applied. If simulated sampling and motion are selected instead, only `kspace` is required; the recorded acquisition indices and physiological signals are not used.

### `ismrmrd-physio_array`
- MRI raw data in ISMRMRD format (`ismrmrd_file`) is loaded using `ISMRMRDReader` with `config/real_data/ismrmrd_reader.toml`. `RawDataPreparer` combines it with the physiological data and converts the inputs to the arrays used by the `preprocessed-real` mode.
- External physiological data is provided as two `.npy` files: `physio_timestamps_file`, of shape `(Nsensors, Nsamples, 1)`, containing timestamps in seconds, and `physio_values_file`, of shape `(Nsensors, Nsamples, Ntracks)`, containing real data values. Tracks of the same sensor share timestamps. Each sensor's last timestamp must correspond to the end of the MRI sequence; timestamps must be strictly increasing.
- If the data is already synchronized with MRI readouts, set every timestamp to `-1`. In this case, each sensor/track must contain exactly one value for every retained MRI imaging readout in full-acquisition order, before slice selection; interpolation is skipped and clock correction must be zero. Do not mix channels whose timestamps are all `-1` with timestamped channels. An isolated `-1` in a strictly increasing timestamp sequence is an ordinary time value.

Pass `filename` to `DataLoader` as a dictionary with the three keys above, or as `(ismrmrd_file, physio_timestamps_file, physio_values_file)`. All sensor/track pairs are kept as separate motion channels, ordered by sensor and then track, without filtering or normalization.

### `siemens-physio_array`
Uses the same physiological data format together with Siemens raw scanner data (`siemens_raw_file`). The loader first converts the Siemens raw file to ISMRMRD using the `siemens_to_ismrmrd` executable, then reads the result with `ISMRMRDReader`. Replace `ismrmrd_file` with `siemens_raw_file` in the dictionary or tuple above.

### `ismrmrd-physio_text` and `siemens-physio_text`

Use the corresponding MRI data format together with a whitespace-separated physiological text file (`physio_file`). Each nonempty, non-comment row is one sample from one sensor:

```text
SENSOR TIMESTAMP VALUE1 [VALUE2 ...]
```

The first line may be a header whose first two fields are `SENSOR TIMESTAMP`; lines may include comments after `#`. `SENSOR` is a nonnegative integer identifier. `TIMESTAMP` is in seconds. `VALUE1`, `VALUE2`, and subsequent value columns are that sensor's tracks. Every data row must have the same positive number of value columns, and timestamped rows for each sensor must be in strictly increasing timestamp order; sensors may be interleaved and may have different sample counts or sampling rates. Each sensor’s final timestamp must correspond to the end of the MRI sequence, with clock correction applied after end alignment.

Use `-1` for every timestamp when values are already synchronized to MRI readouts. Then every track must provide one value for every retained full-acquisition MRI imaging readout, before slice selection; interpolation is skipped and clock correction must be zero. Do not mix channels whose timestamps are all `-1` with timestamped channels. An isolated `-1` in a strictly increasing timestamp sequence is an ordinary time value.

Pass `filename` as `(ismrmrd_file, physio_file)` or `(siemens_raw_file, physio_file)`, or as a dictionary with the corresponding keys. Text channels are ordered by numeric sensor ID and then value-column order. These four generic physiological modes require `ismrmrd_reader_config="config/real_data/ismrmrd_reader.toml"` in `load_config(...)`.

### `ismrmrd-saec` and `siemens-saec`
Uses the corresponding MRI data format together with physiological data file in SAEC [3, 4] format (`saec_file`). Pass `real_data_config="config/real_data/saec.toml"` and `ismrmrd_reader_config="config/real_data/ismrmrd_reader.toml"` to `load_config(...)`. Select the sensor type in `saec.toml`.

SAEC processing depends on `rawdata_sensor_type`:

- `BELT`: selects the belt track with the larger standard deviation, applies a first-order zero-phase Butterworth low-pass filter with a 1 Hz cutoff, and removes quadratic drift. The drift is fitted to a copy clipped to the mean plus or minus two standard deviations, then subtracted from the unclipped filtered signal. The returned single channel is not standardized.
- `1MARMOT`: for every MARMOT sensor, each of the three accelerometer tracks is low-pass filtered at 0.3 Hz and then high-pass filtered at 0.03 Hz, using first-order zero-phase Butterworth filters. Tracks identified as displaced, constant, non-finite, or otherwise invalid are rejected. The valid track with the largest standard deviation is selected across all sensors and normalized by its standard deviation.
- `ALL_MARMOTS` (also accepted as `ALL_MARMOTs`): applies the same 0.3/0.03 Hz filtering and displacement checks, selects the highest-variance valid track from each usable sensor, and returns one standard-deviation-normalized channel per usable sensor.

Timestamped SAEC channels are aligned to the full MRI sequence before slice selection and interpolated onto MRI readout times. SAEC timestamps are referenced to the Siemens stop trigger. The filters are applied before this interpolation; readouts outside the SAEC recording use the nearest endpoint value.

### `ismrmrd-polaris` and `siemens-polaris`
Uses the corresponding MRI data format together with a single-tool NDI ToolBox `.tsv` export from a Polaris Vega infrared camera tracker (`polaris_file`). Pass `polaris_config="config/real_data/polaris.toml"` and `ismrmrd_reader_config="config/real_data/ismrmrd_reader.toml"` to `load_config(...)`. Select the tracks and clock shift in `polaris.toml`. The final timestamp must correspond to the end of the MRI sequence; clock correction is applied after end alignment.

Polaris reads the XYZ tool-position channels and applies a first-order zero-phase Butterworth low-pass filter with a 1 Hz cutoff to each axis. The sampling rate is estimated from the recording duration; at least seven samples are required, and the rate must exceed twice the cutoff. The filtered signals are then linearly interpolated onto full-sequence MRI readout times. With `polaris_channel_mode = "all"`, the Tx, Ty, and Tz channels are retained. With `"largest-amplitude"`, only the axis with the largest peak-to-peak range is retained (ties are resolved in Tx, Ty, Tz order). Each retained channel is centered by subtracting its mean, then all retained channels are divided by their largest standard deviation, if nonzero. Channel selection, centering, and scaling are computed at full-sequence MRI readout times, before slice selection.

### Post-processing and synchronization
Text inputs automatically load `config/real_data/physio_text.toml`; array inputs load `config/real_data/physio_array.toml`. Each file owns its clock-shift value independently. To use another sensor configuration, pass `physio_config="path/to/physio_text.toml"` to `load_config(...)`. The clock setting belongs to the sensor configuration, not `ismrmrd_reader.toml`.

Generic `physio_text` and `physio_array` inputs are not filtered, centered, or normalized. Their channels are preserved in sensor/track order after timestamp alignment and interpolation. If all timestamps are `-1`, values are treated as already synchronized and must contain one sample for every full-acquisition MRI readout; interpolation is skipped and clock correction must be zero.

Physiological clock correction (Polaris, `physio_text`, `physio_array`): set `physio_clock_drift_seconds` in `config/real_data/polaris.toml` for Polaris, `config/real_data/physio_text.toml` for text inputs, or `config/real_data/physio_array.toml` for array inputs (default `0.0`), or pass `overrides={"physio_clock_drift_seconds": -0.25}`. Positive shifts samples later; negative shifts earlier, after sequence-end alignment. Already-synchronized (`-1`) inputs reject any nonzero clock correction. Missing coverage at either edge is extrapolated with $\hat{x}[n] = c + \sum_{k=1}^{p} a_k x[n-k]$, fitted per channel (up to 20 lags over the nearest 10 seconds; reversed history at the start). Any required extension greater than 1 second raises an error. SAEC is unchanged.

## Sampling Modes (synthetic acquisition)

Configured with:
- `kspace_sampling_type`
- `NshotsPerNex`
- `Nex`

Implemented in `src/preprocessing/SamplingSimulator.py`.

When synthetic sampling is generated and `save_debug_plots=true`, per-repetition debug plots are written to the reconstruction’s `preprocessing/` folder with hardcoded names:
- 2D sampling: `ky_order_nex{nex}.png`
- 3D sampling: `ky_kz_order_nex{nex}.png`

Here, `nex` in the filename is one-based.

For each repetition, acquired readouts are divided into `NshotsPerNex` chronological shot blocks. `acceleration_factor` selects regularly spaced ky lines, and `calibration_lines` retains a central calibration band when acceleration is greater than one. The ordering below is applied to the retained readouts.

### `linear`

In 2D, ky increases monotonically and is split into blocks of approximately equal size. In 3D, each shot covers a contiguous ky band and all kz partitions.

### `interleaved`

The ky ordering is built from groups `ky = s, s + NshotsPerNex, s + 2*NshotsPerNex, ...`. In 2D, retained lines are then split into approximately equal shot blocks. In 3D, each shot uses its interleaved ky group, with repetition-dependent ky/kz ordering and alternating ky traversal direction between partition blocks.

### `random`

Each repetition independently shuffles ky lines in 2D, or all retained `(ky, kz)` pairs in 3D, then splits them into approximately equal shot blocks.

## Motion Simulation Modes

Configured with one variable:
- `simulated_motion_type`: `"rigid-realistic"`, `"rigid-per-shot"`, `"non-rigid-realistic"`, `"non-rigid-per-shot"`, or `"as-it-is"`.

`simulated_motion_type` is configured in `config/motion_simulation/*.toml`. Reconstruction independently uses `reconstruction_motion_type` (`"rigid"` or `"non-rigid"`) from `config/reconstruction/*.toml`.
Implemented in `src/preprocessing/MotionSimulator.py`.

### `as-it-is`

No synthetic corruption added. Valid only for real-data types with `from-data` sampling, including SAEC and Polaris inputs.

### `rigid-per-shot`

Shot-wise rigid states:
- one rigid transform per shot (`Nshots = Nex * NshotsPerNex` for simulated sampling; one state per recorded readout with `from-data`)
- explicit global multiplier `rigid_motion_amplitude_scale` scales all configured rigid amplitudes
- random `(tx, ty, phi)` (or `(tx, ty, tz, rx, ry, rz)` for the 3D case) per shot in configured ranges
- piecewise-constant motion in ky-time according to shot order

### `rigid-realistic`

Continuous rigid curve over full acquisition:
- random event times over the acquired readouts across all repetitions
- smooth raised-cosine transitions (`motion_tau`)
- explicit global multiplier `rigid_motion_amplitude_scale` scales all configured rigid amplitudes
- random event amplitudes for `tx`, `ty`, `phi` (or `(tx, ty, tz, rx, ry, rz)` for the 3D case)
- data is then reclustered to `N_motion_states` from the simulated navigator signal (first principal component of the simulated rigid motion parameters)

For corruption, motion is defined per acquired readout (`Ny * Nz * Nex` for fully sampled data). Consecutive identical rigid states may share an operator.

### `non-rigid-per-shot`

Shot-wise non-rigid with fixed spatial basis maps:
- displacement field maps `alpha_x`, `alpha_y` (+ `alpha_z` for 3D) simulate respiration
- a per-shot Gaussian scale is configured with `nonrigid_discrete_s_scale`
- one random scalar per shot (`s`) drives the temporal displacement amplitude (can be interpreted as a navigator or respiratory belt signal)
- displacement at state `m`: `[ux, uy, (uz)] = [alpha_x, alpha_y, (alpha_z)] * s[m]`

### `non-rigid-realistic`

Continuous sinusoidal temporal curve:
- random phase
- random cycles per image/volume repetition in `[nonrigid_resp_cycles_min, nonrigid_resp_cycles_max]`
- normalized to unit amplitude

Spatial maps are the same fixed non-rigid basis (`alpha_x`, `alpha_y` + `alpha_z` for 3D) scaled by `nonrigid_motion_amplitude`.
For corruption, simulation uses one state per acquired readout (`Ny * Nz * Nex` for fully sampled data).

## Motion Binning and Reconstruction States

After loading or simulation, motion signals are grouped into reconstruction states using `motion_binning_mode` from the reconstruction configuration:

- `"kmeans"`: clusters the motion-channel values with k-means.
- `"kspace_energy"`: quantizes motion-channel values using `motion_quantization_bins`, selects the highest-energy virtual states, and assigns the others to their nearest selected state. This mode requires k-space data and is used by `nonrigid_2d_breast.toml`.

Key points:
- simulation state count and reconstruction state count can differ.
- corruption may be readout-wise (`Ny * Nz * Nex` states for fully sampled data), but reconstruction uses binned virtual states (`N_motion_states`).
- `N_motion_states` is a manual reconstruction setting from the reconstruction TOML, or an explicit `overrides={"N_motion_states": ...}` value.

State-count rules:
- `rigid-per-shot` and `non-rigid-per-shot`: `N_motion_states` is automatically replaced by the shot count; changes are announced.
- `rigid-realistic`, `non-rigid-realistic`, and `as-it-is`: `N_motion_states` stays at the manual reconstruction value

For real data with generated sampling, configured shot counts are preserved and `Nex` must match the array. With `from-data` sampling and per-shot simulated motion, each recorded readout is a state; the count is resolved from the actual acquisition indices after loading.


## Outputs

### Run outputs

Every notebook and the Siemens pipeline creates a timestamped directory under `output_root/workflow_label/`, configured in `config/general.toml` (`output_root="runs"` by default).

```text
runs/<workflow_label>/YYYYMMDDTHHMMSS/
├── manifest.json
├── config_resolved.json
├── reconstructions/
│   └── slice_001/                    # volume_001 for a 3D reconstruction
│       ├── preprocessing/           # Sampling, input motion, ground truth,
│       │                            # corrupted image, optional synchronization
│       ├── results/
│       │   ├── image_reconstructed.pt
│       │   ├── image_reconstructed.png
│       │   ├── image_reconstructed_nex_001.png  # Multiple repetitions only
│       │   ├── motion_parameters.pt
│       │   └── ...                  # Final motion curves/maps
│       ├── diagnostics/
│       │   ├── level_01/            # Image and available nonrigid motion plots
│       │   ├── level_02/
│       │   ├── .../
│       │   ├── residuals/           # Curves across resolution levels
│       │   ├── consistency_checks/
│       │   └── postprocessing/     # Reference tensors/figures, normalized image,
│       │                            # image_postprocessed.pt and .png
│       └── reconstruction.log
└── exports/
    └── dicom/                       # If requested in the pipeline
        └── slice_001.dcm
```

`image_reconstructed.pt` is the complex reconstruction before reference-image normalization and zero-filling, preserving the repetition dimension. Its PNG shows the magnitude of the repetition mean; individual repetition previews are also saved when multiple repetitions exist. In the Siemens pipeline, `image_postprocessed.pt` contains the complex image after configured reference-image normalization and zero-filling, still preserving repetitions. It is saved only when both saving flags below are enabled. DICOM export uses the magnitude of the repetition mean with DICOM intensity scaling. Motion overlays use the reconstruction grid.

Saving flags:

- `save_debug_plots=true`: save preprocessing figures. Per-level reconstruction diagnostics, residual plots, and pipeline postprocessing tensors additionally require `save_reconstruction_outputs=true`.
- `save_reconstruction_outputs=true`: save reconstruction tensors, final plots, and `reconstruction.log`. Turning it off does not disable explicitly requested DICOM export or preprocessing diagnostics.
- `check_simulated_motion_consistency`: controls the simulated-motion numerical check; its figure additionally requires `save_debug_plots=true`.

To remove inactive run outputs under the configured `output_root`:

```bash
python -m src.utils.clear_runs
```

This permanently removes each matching inactive run directory, including its tensors, figures, logs, manifest, and DICOM exports. Active runs are skipped. Use `--dry-run` to preview or `--workflow-label LABEL` to restrict cleanup.

### Shared data cache

Generated ISMRMRD (`.mrd`) files and optional preprocessed HDF5 files live outside run outputs. Enable `cache_preprocessed_data=true` to cache full-acquisition preprocessing; its default is `false`.

```text
<data-root>/cache/grics/
├── converted/<source-and-converter-key>.mrd
└── preprocessed/<source-and-preprocessing-key>.h5
```

`cache_root = "auto"` defaults to `<repository-parent>/data/cache/grics`, outside this code repository.

Existing entries are reused. `remove_temporary_data_after_run=true` is the default. At the end of a managed execution (`@managed_execution` or `execution_scope()`), generated
cache entries used by that execution are removed when no process still uses them. Set `remove_temporary_data_after_run=false` to retain files for reuse across later executions. Standalone callers release cache leases at interpreter exit or by calling `src.runtime.data_cache.release_leases()`.

To empty the cache:

```bash
python -m src.utils.clear_cache
# For a custom cache location, pass the same path used by your runs:
python -m src.utils.clear_cache --cache-root /path/to/data/cache/grics
```

The utility removes inactive MRD/HDF5 cache entries and abandoned partial files, reports entries still in use, and leaves active files untouched.


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

`kspace` is required when using `motion_binning_mode="kspace_energy"`; otherwise it may be omitted. Every mask contains one Boolean per chronological readout. Keys are caller-defined, and all named layouts share the same motion-state labels.

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

For `params`, use `data.params` from `DataLoader` or `prepared.params` from `GRICSPreparerAPI`. When using normalized k-space from `DataLoader`, pass `kspace_scale=data.kspace_scale` to restore the image scale in `run()` outputs.

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

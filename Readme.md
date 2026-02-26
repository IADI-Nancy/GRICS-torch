# GRICS-torch: GRICS MRI Motion-Corrected Reconstruction in PyTorch

This repository currently contains a 2D MRI reconstruction pipeline with joint image-motion estimation using the GRICS algorithm [1], implemented in PyTorch with GPU support. This implementation aims to improve understanding of the algorithm in the MRI community and support reuse.

Please contact Karyna Isaieva (karyna.isaieva@univ-lorraine.fr) for any bug reports.

## Repository layout

- `src/preprocessing/`: data loading, sampling simulation, motion simulation, motion binning
- `src/reconstruction/`: encoding/motion operators, Jacobian, CG solver, joint reconstructor
- `src/runtime/`: config loading and runtime initialization
- `src/utils/`: plotting, diagnostics, notebook display helpers
- `config/`: TOML configs for reconstruction, sampling, motion simulation, and general runtime

\+ five demos

## Environment Setup

A Dockerfile is provided in the `build/` folder. The built image is available at https://github.com/IADI-Nancy/GRICS-torch/pkgs/container/grics-torch. The `docker.sh` script in the repository root can be used for mounting and runtime setup.

## Config System

The pipeline merges config files at runtime via `src/runtime/runtime_config.py`.

Main config groups:
- `config/general.toml`: paths, debug/runtime flags, k-space normalization and coil sensitivities calculation parameters
- `config/reconstruction/*.toml`: solver/reconstruction settings
- `config/sampling_simulation/*.toml`: synthetic k-space ordering
- `config/motion_simulation/*.toml`: synthetic motion model parameters
- `config/shepp_logan.toml`: Shepp-Logan phantom generation parameters

Important consistency rule:
- `GN_iterations_per_level` must match `ResolutionLevels` length exactly.

## Data Types

The `data_type` selected in `load_config(...)` controls how input data is built or loaded.

### `shepp-logan`

Synthetic phantom data generated in `DataLoader.generate_shepp_logan(...)`.
Required config files:
- `config/shepp_logan.toml`
- a sampling simulation config file
- a motion simulation config file (otherwise there is nothing to correct)

### `real-world`

Loaded via `DataLoader.load_realworld_data(...)` from HDF5 with datasets:
- `kspace`
- `motion_data` - already reordered 
- sampling order (`idx_ky`, `idx_kz` and `idx_nex`)

No synthetic sampling is needed in this mode: acquisition order and motion signal come from file. However, additional motion simulation can still be applied.

### `raw-data`

Loaded from raw scanner and physiological files using `RawDataReader`:
- the MRI raw data in the ISMRMRD format (`ismrmrd_file`)
- physiological data file in SAEC [3, 4] format (`saec_file`)

## Sampling Modes (synthetic acquisition)

Configured with:
- `kspace_sampling_type`
- `NshotsPerNex`
- `Nex`

Implemented in `src/preprocessing/SamplingSimulator.py`.

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

Configured with `motion_simulation_type` in `config/motion_simulation/*.toml`.
Implemented in `src/preprocessing/MotionSimulator.py`.

### `as-it-is`

No synthetic corruption added. Only valid for `real-world`/`raw-data` (already motion-corrupted).

### `no-motion-data`

Adds no synthetic corruption and replaces the available motion signal with zeros. Use this if you want to reconstruct with the same algorithm but without motion correction.

### `discrete-rigid`

Shot-wise states:
- one rigid transform per shot over all `Nshots = Nex * NshotsPerNex`
- random `(tx, ty, phi)` per shot in configured ranges
- piecewise-constant motion in ky-time according to shot order

### `rigid` (realistic rigid)

Continuous rigid curve over full acquisition:
- random event times over `Ny * Nex` lines
- smooth raised-cosine transitions (`motion_tau`)
- random event amplitudes for `tx`, `ty`, `phi`
- data is then reclustered to `N_motion_states` from the simulated navigator signal (first principal component of the simulated rigid motion parameters)

For corruption, simulation uses one global state per acquired line (`Ny * Nex` states).

### `discrete-non-rigid`

Shot-wise non-rigid with fixed spatial basis maps:
- displacement field maps `alpha_x`, `alpha_y` simulate respiration
- one random scalar per shot (`S`) drives the temporal displacement amplitude (can be interpreted as a navigator or respiratory belt signal)
- displacement at state `m`: `[ux, uy] = [alpha_x, alpha_y] * S[m]`

### `non-rigid` (realistic respiratory-like)

Continuous sinusoidal temporal curve:
- random phase
- random cycles per image in `[nonrigid_resp_cycles_min, nonrigid_resp_cycles_max]`
- normalized to unit amplitude

Spatial maps are the same fixed non-rigid basis (`alpha_x`, `alpha_y`) scaled by `nonrigid_motion_amplitude`.
For corruption, simulation uses one state per acquired line (`Ny * Nex` states).

## Motion Binning and Reconstruction States

After loading or simulation, the motion curve is clustered with k-means (`MotionBinner.bin_motion`) into reconstruction states.

Key points:
- simulation state count and reconstruction state count can differ.
- corruption may be line-wise (`Ny * Nex` states), but reconstruction uses binned virtual states (`N_motion_states`).

State-count rules are set in `runtime_config.refresh_derived(...)`:
- `discrete-rigid` and `discrete-non-rigid`: `N_motion_states = Nshots`
- `rigid`: `N_motion_states = num_motion_events + 1`
- `non-rigid`, `as-it-is` and `no-motion-data`: `N_motion_states` stays the reconstruction config value

## Outputs

Each run writes into folders from `config/general.toml`:

- `input_data/`: sampling order, motion curves, corrupted and ground-truth images (if they exist), and simulated motion (if it exists)
- `debug_outputs/`: results per reconstruction level
- `logs/`: residual curves and per-restart logs
- `results/`: final reconstructed outputs

By default, these folders are cleaned before each run (`clean_output_folders_before_run = true`).

## References

[1] Odille, F., Vuissoz, P. A., Marie, P. Y., & Felblinger, J. (2008). Generalized reconstruction by inversion of coupled systems (GRICS) applied to free‐breathing MRI. Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine, 60(1), 146-157.
[2] https://fastmri.med.nyu.edu/ 
[3] Isaieva, K., Fauvel, M., Weber, N., Vuissoz, P. A., Felblinger, J., Oster, J., & Odille, F. (2022). A hardware and software system for MRI applications requiring external device data. Magnetic Resonance in Medicine, 88(3), 1406-1418.
[4] https://github.com/IADI-Nancy/wrapperHDF5 

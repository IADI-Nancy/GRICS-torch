# GRICS MRI Motion-Corrected Reconstruction

This repository contains a 2D MRI reconstruction pipeline with joint image-motion estimation for:
- rigid motion
- non-rigid motion
- synthetic Shepp-Logan simulations
- real-world/as-is motion data

The workflow is currently notebook-first, with five demos at repository root.

## Repository Layout

- `src/preprocessing/`: data loading, sampling simulation, motion simulation, motion binning
- `src/reconstruction/`: encoding/motion operators, Jacobian, CG solver, joint reconstructor
- `src/runtime/`: config loading and runtime initialization
- `src/utils/`: plotting, diagnostics, notebook display helpers
- `config/`: TOML configs for reconstruction, sampling, motion simulation, and general runtime
- `demo_a_shepp_linear_rigid.ipynb`
- `demo_b_shepp_random_rigid.ipynb`
- `demo_c_shepp_interleaved_nonrigid.ipynb`
- `demo_d_shepp_linear_nonrigid.ipynb`
- `demo_e_realworld_as_is.ipynb`

## Environment Setup

Use the conda file in `build/`:

```bash
conda env create -f build/conda_environment.yml
conda activate environment_grics
```

If you use Jupyter:

```bash
python -m ipykernel install --user --name environment_grics
```

## Running Demos

Open one of the demo notebooks and run all cells:

- `demo_a_shepp_linear_rigid.ipynb`: Shepp-Logan, linear sampling, rigid simulation
- `demo_b_shepp_random_rigid.ipynb`: Shepp-Logan, random sampling, rigid simulation
- `demo_c_shepp_interleaved_nonrigid.ipynb`: Shepp-Logan, interleaved sampling, non-rigid simulation
- `demo_d_shepp_linear_nonrigid.ipynb`: Shepp-Logan, linear sampling, realistic non-rigid simulation
- `demo_e_realworld_as_is.ipynb`: real-world data, as-is motion

## Config System

The pipeline merges config files at runtime via `src/runtime/runtime_config.py`.

Main config groups:
- `config/general.toml`: paths, debug/runtime flags, k-space normalization
- `config/reconstruction/*.toml`: solver/reconstruction settings
- `config/sampling_simulation/*.toml`: synthetic k-space ordering
- `config/motion_simulation/*.toml`: synthetic motion model parameters
- `config/shepp_logan.toml`: phantom generation parameters

Important consistency rule:
- `GN_iterations_per_level` must match `ResolutionLevels` length exactly.

## Data Types

The `data_type` selected in `load_config(...)` controls how input data is built/loaded.

### `shepp-logan`

Synthetic phantom data generated in `DataLoader.generate_shepp_logan(...)`:
- 2D Shepp-Logan phantom resized to `N_SheppLogan x N_SheppLogan`
- synthetic complex coil sensitivity maps (Gaussian profiles + random phase)
- FFT to k-space
- synthetic sampling trajectory from `config/sampling_simulation/*.toml`

Used config:
- `config/shepp_logan.toml`
- one sampling config in `config/sampling_simulation/`
- optional motion simulation config in `config/motion_simulation/`

### `fastMRI`

Loaded via `DataLoader.load_fastMRI_data(...)` from a `.npz` file with key `arr_0`.
Expected k-space shape is interpreted as `(coils, Nx, Ny, slices)`, then replicated across `Nex`.
Sampling order is simulated from `config/sampling_simulation/*.toml`.

### `real-world`

Loaded via `DataLoader.load_realworld_data(...)` from HDF5 with datasets:
- `kspace`
- `motion_data`
- `idx_ky`
- `idx_kz`
- `idx_nex`

No synthetic sampling is needed in this mode, acquisition order and motion signal come from file.

### `raw-data`

Loaded from raw scanner + physiological files using `RawDataReader`:
- ISMRMRD file (`ismrmrd_file`)
- SAEC physiological file (`saec_file`)

Pipeline:
- parse acquisitions and k-space coordinates from ISMRMRD
- read/filter respiratory signal from SAEC (`BELT`)
- interpolate respiratory signal to acquisition timestamps
- optional export to HDF5-like structure in memory

## Sampling Modes (Synthetic Acquisition)

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

### `no-motion`

No synthetic corruption; zero motion signal.

### `discrete-rigid`

Shot-wise rigid states:
- one rigid transform per shot over all `Nshots = Nex * NshotsPerNex`
- random `(tx, ty, phi)` per shot in configured ranges
- piecewise-constant motion in ky-time according to shot order

### `rigid` (realistic rigid)

Continuous rigid curve over full acquisition:
- random event times over `Ny * Nex` lines
- smooth raised-cosine transitions (`motion_tau`)
- random event amplitudes for `tx`, `ty`, `phi`

For corruption, simulation uses one global state per acquired line (`Ny * Nex` states).

### `discrete-non-rigid`

Shot-wise non-rigid with fixed spatial basis maps:
- displacement field maps `alpha_x`, `alpha_y` are centered Gaussian-like patterns
- one random scalar per shot (`S`) drives temporal amplitude
- displacement at state `m`: `[ux, uy] = [alpha_x, alpha_y] * S[m]`

### `non-rigid` (realistic respiratory-like)

Continuous sinusoidal temporal curve:
- random phase
- random cycles per image in `[nonrigid_resp_cycles_min, nonrigid_resp_cycles_max]`
- normalized to unit amplitude

Spatial maps are the same fixed non-rigid basis (`alpha_x`, `alpha_y`) scaled by `nonrigid_motion_amplitude`.
For corruption, simulation also uses one global state per acquired line (`Ny * Nex` states).

## Motion Binning and Reconstruction States

After loading/simulation, the motion curve is clustered with k-means (`MotionBinner.bin_motion`) into reconstruction states.

Key point:
- simulation state count and reconstruction state count can differ.
- corruption may be line-wise (`Ny * Nex` states), but reconstruction uses binned virtual states (`N_motion_states`).

State-count rules set in `runtime_config.refresh_derived(...)`:
- `discrete-rigid` and `discrete-non-rigid`: `N_motion_states = Nshots`
- `rigid`: `N_motion_states = num_motion_events + 1`
- `non-rigid`: `N_motion_states` stays the reconstruction config value
- `as-it-is` and `no-motion`: `N_motion_states` stays the reconstruction config value

## Minimal Config Recipes

### Synthetic rigid (Shepp-Logan)

- `data_type = "shepp-logan"`
- sampling config: one of `linear.toml`, `interleaved.toml`, `random.toml`
- motion sim config: `rigid.toml` or `discrete_rigid.toml`
- reconstruction config: `config/reconstruction/rigid_*.toml`

### Synthetic non-rigid (Shepp-Logan)

- `data_type = "shepp-logan"`
- sampling config: one of `config/sampling_simulation/*.toml`
- motion sim config: `nonrigid.toml` or `discrete_nonrigid.toml`
- reconstruction config: `config/reconstruction/nonrigid_*.toml`

### Real-world as-is

- `data_type = "real-world"` (or `raw-data`)
- no synthetic sampling config needed
- motion sim type should be `as-it-is` (default in data-driven mode)
- reconstruction config can be rigid or non-rigid depending on target model

## Outputs

Each run writes into folders from `config/general.toml`:

- `input_data/`: input artifacts (sampling order, motion curves, simulated references)
- `debug_outputs/`: debug diagnostics
- `logs/`: residual curves and per-restart logs
- `results/`: final reconstructed outputs

By default, these folders are cleaned before each run (`clean_output_folders_before_run = true`).

## Reproducibility Notes

- `debug_flag = true` enables deterministic settings where possible.
- For strict CUDA determinism in PyTorch with CuBLAS, set before launching Python:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

- Keep the same seed (`seed` in config) and same environment/package versions.

## Common Issues

### `GN_iterations_per_level` mismatch error

Set the same number of entries in:
- `ResolutionLevels`
- `GN_iterations_per_level`

### Jupyter kernels remain alive after runs

Notebook completion does not automatically terminate kernels. Shut them down manually from Jupyter/VSCode kernel manager.

### Local files that should not be versioned

Already ignored (or should be ignored) in `.gitignore`:
- `.jupyter_local/`
- `.env`
- local IDE/cache artifacts

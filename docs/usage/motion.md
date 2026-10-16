# Motion

## Motion simulation

Configured with one variable:
- `simulated_motion_type`: `"rigid-per-readout"`, `"rigid-per-shot"`, `"non-rigid-per-readout"`, `"non-rigid-per-shot"`, or `"as-it-is"`.

`simulated_motion_type` is configured in `config/motion_simulation/*.toml`. Reconstruction independently uses `reconstruction_motion_type` (`"rigid"` or `"non-rigid"`) from `config/reconstruction/*.toml`.
Implemented in `src/preprocessing/MotionSimulator.py`.

Motion simulation starts on the configured runtime device, which is the GPU by default. If CUDA memory is insufficient, preparation prints an explicit message, clears unused CUDA memory, and restarts the complete simulation on the CPU. Other runtime errors are not treated as memory fallbacks.

### `as-it-is`

No synthetic corruption added. Valid only for real-data types with `from-data` sampling, including SAEC and Polaris inputs.

### `rigid-per-shot`

Shot-wise rigid states:
- one rigid transform per shot (`Nshots = Nex * NshotsPerNex` for simulated sampling; one state per recorded readout with `from-data`)
- explicit global multiplier `rigid_motion_amplitude_scale` scales all configured rigid amplitudes
- random `(tx, ty, phi)` (or `(tx, ty, tz, rx, ry, rz)` for the 3D case) per shot in configured ranges
- piecewise-constant motion in ky-time according to shot order

### `rigid-per-readout`

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

### `non-rigid-per-readout`

Continuous sinusoidal temporal curve:
- random phase
- random cycles per image/volume repetition in `[nonrigid_resp_cycles_min, nonrigid_resp_cycles_max]`
- normalized to unit amplitude

Spatial maps are the same fixed non-rigid basis (`alpha_x`, `alpha_y` + `alpha_z` for 3D) scaled by `nonrigid_motion_amplitude`.
For corruption, simulation uses one state per acquired readout (`Ny * Nz * Nex` for fully sampled data).

## Motion binning and reconstruction states

Physiological signals from real acquisitions are standardized after reader-specific processing and synchronization, before slice grouping and motion binning.
Each sensor is normalized independently as `(signal - mean) / std`, using the
population standard deviation (`sqrt(mean((signal - mean)**2))`, denominator N).
Statistics cover every MRI readout in the full acquisition, before slice selection and motion binning. This gives each sensor zero mean and unit
variance over the acquisition. Constant sensors and non-finite samples raise
an error. Simulated signals are unaffected.

After loading or simulation, motion signals are grouped into reconstruction states using `motion_binning_mode` from the reconstruction configuration:

- `"kmeans"`: clusters the motion-channel values with k-means.
- `"kspace_energy"`: quantizes motion-channel values using `motion_quantization_bins`, selects the highest-energy virtual states, and assigns the others to their nearest selected state. This mode requires k-space data and is used by `nonrigid_2d_breast.toml`.

Key points:
- simulation state count and reconstruction state count can differ.
- corruption may be readout-wise (`Ny * Nz * Nex` states for fully sampled data), but reconstruction uses binned virtual states (`N_motion_states`).
- `N_motion_states` is a manual reconstruction setting from the reconstruction TOML, or an explicit `overrides={"N_motion_states": ...}` value.

State-count rules:
- `rigid-per-shot` and `non-rigid-per-shot`: `N_motion_states` is automatically replaced by the shot count; changes are announced.
- `rigid-per-readout`, `non-rigid-per-readout`, and `as-it-is`: `N_motion_states` stays at the manual reconstruction value

For real data with generated sampling, configured shot counts are preserved and `Nex` must match the array. With `from-data` sampling and per-shot simulated motion, each recorded readout is a state; the count is resolved from the actual acquisition indices after loading.

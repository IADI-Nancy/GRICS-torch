# Configuration

Use `load_config(...)` with a `data_type`, a path to a `reconstruction_config`, and paths to optional configs; the demos provide examples. Use `overrides={...}` for run-specific settings.

Internal ownership and merge order are documented in the [configuration architecture](../../architecture/configuration.md).

## Required

- `data_type`: selects the input source and data format; see [input data types](input_formats.md)
- `config/reconstruction/*.toml`: reconstruction model, multiresolution GN iterations, regularization, and CG solver settings; pass as [`reconstruction_config`](../../implementation/joint_reconstructor.md)

## Required only under certain conditions

- `config/sampling_simulation/linear.toml`, `interleaved.toml`, or `random.toml`: required for both synthetic sources (`data_type="shepp-logan"` or `"from_image"`), which have no recorded acquisition order. For real data, supply one only when deliberately replacing the recorded sampling with a generated acquisition order. Pass as `sampling_config`; omit it to retain recorded sampling for real data. See details in [sampling](sampling.md).
- A rigid or non-rigid mode in `config/motion_simulation/`: required for both synthetic sources and for real data whenever motion is simulated. Replacing recorded sampling also requires simulated motion, because recorded motion cannot be reused with a generated acquisition order. Pass as [`motion_simulation_config`](../../implementation/motion_simulator.md) and choose a 2D or 3D mode matching the input. For real data, omit it when retaining both recorded sampling and recorded motion; simulating motion while retaining recorded sampling is also supported. See details in [motion](motion.md).

## Loaded automatically

These files are selected internally; their paths do not need to be supplied to `load_config(...)`.

- `config/general.toml`: paths, runtime flags, and k-space normalization; loaded automatically
- `config/coil_sensitivity_and_calibration/espirit.toml` or `odille_spline.toml`: selected by [`coil_sensitivity_config`](../../implementation/coil_sensitivity_calculator.md) in the reconstruction TOML
- `config/coil_sensitivity_and_calibration/calibration_image.toml`: imaging-k-space reference smoothing, independent of the coil method; loaded automatically
- `config/synthetic_data/shepp_logan_2d.toml` or `shepp_logan_3d.toml`: phantom settings for `data_type="shepp-logan"`; selected by `reconstruction_dimension` in the reconstruction TOML
- `config/synthetic_data/from_image.toml`: image-source settings for `data_type="from_image"`; supports 2D reconstruction only
- `config/physiological_data/saec.toml`: sensor selection and filters for `ismrmrd-saec` or `siemens-saec`
- `config/physiological_data/polaris.toml`: track selection, filter, and clock shift for `ismrmrd-polaris` or `siemens-polaris`
- `config/physiological_data/physio_text.toml` or `physio_array.toml`: clock shift for the corresponding ISMRMRD or Siemens physiological input mode
- `config/sampling_simulation/from_data.toml`: recorded sampling for real data when `sampling_config` is omitted and sampling is not overridden to a generated mode
- `config/motion_simulation/as_is.toml`: recorded-motion mode for real data when `motion_simulation_config` is omitted; requires recorded sampling
- `config/motion_simulation/common/*.toml`: shared motion parameters, loaded only through a selected motion mode

Source settings come from the automatically selected synthetic or physiological file; custom source configuration paths are not accepted. Edit that file or use `overrides={...}` for run-specific values. Overrides are applied after file selection, so choose the reconstruction dimension in the reconstruction TOML. Preprocessed real data loads no synthetic or physiological configuration because its acquisition and motion measurements are already prepared.

## Pipeline-specific configuration

- `config/postprocessing/nonrigid_2d_breast.toml`: reference-image normalization for the Siemens 2D breast pipeline; loaded separately with `load_postprocessing_config(...)`

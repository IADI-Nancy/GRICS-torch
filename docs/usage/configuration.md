# Configuration

Configuration is loaded through `load_config(...)`; internal ownership and
merge order are documented in the [configuration architecture](../architecture/configuration.md).

- `config/general.toml`: paths, runtime flags, and k-space normalization; loaded automatically
- `config/coil_sensitivity_and_calibration/*.toml`: settings for the coil-sensitivity method named by `coil_sensitivity_config` in the reconstruction TOML; loaded automatically
- `config/coil_sensitivity_and_calibration/calibration_image.toml`: imaging-k-space reference smoothing, independent of the coil method; loaded automatically
- `config/reconstruction/*.toml`: reconstruction model, multiresolution GN iterations, regularization, and CG solver settings; always required
- `config/synthetic_data/*.toml`: Shepp-Logan phantom or image-source generation settings (only if synthetic data is used)
- `config/physiological_data/`: configurations for physiological input processing
    - `config/physiological_data/saec.toml`: SAEC sensor selection, loaded only if a SAEC file is used
    - `config/physiological_data/polaris.toml`: Polaris infrared camera channel selection, required only if this sensor is used
    - `config/physiological_data/physio_text.toml`: clock-shift settings for physiological text inputs, loaded automatically for that mode
    - `config/physiological_data/physio_array.toml`: clock-shift settings for physiological array inputs, loaded automatically for that mode
- `config/sampling_simulation/*.toml`: simulated k-space acquisition ordering
- `config/motion_simulation/*.toml`: selected simulated rigid or non-rigid motion modes
- `config/motion_simulation/common/*.toml`: shared motion parameters, loaded only through a selected motion mode
- `config/postprocessing/nonrigid_2d_breast.toml`: reference-image normalization for the Siemens breast pipeline, loaded with `load_postprocessing_config(...)`

Use `load_config(...)` with a `data_type` and a path to a
`reconstruction_config`; the demos provide complete examples. Source and mode
requirements are documented under [input formats](input_formats.md),
[sampling](sampling.md), and [motion](motion.md). Use `overrides={...}` for
run-specific settings.

## Calibration-weighted image

Set `use_calibration_weighted_image = true` in the reconstruction TOML with either `espirit` or `odille-spline`. Set `calibration_image_smoothing` in `config/coil_sensitivity_and_calibration/calibration_image.toml` (finite number >= 0; zero disables smoothing). This setting is independent of `spline_magnitude_smoothing`, which controls Odille coil maps. The default 1000 preserves the previous calibration image when Odille magnitude smoothing was 1000. If you previously changed that value, use the same value here to preserve the previous image prior.

`DataPreparer` builds the reference from repetition-averaged imaging k-space before simulated motion, using root-sum-of-squares coil combination, in-plane smoothing, and mean-one normalization. It does not use separate reference k-space or estimated coil maps. Use `JointReconstructor.from_prepared_data(data)` to select `data.grics_reference_image` automatically when weighting is enabled. External callers can supply a calibration image explicitly or let the constructor calculate it from their imaging k-space. This image is also available for reference normalization when calibration weighting is disabled.

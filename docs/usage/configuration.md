# Configuration

Configuration is loaded through `load_config(...)`; internal ownership and
merge order are documented in the [configuration architecture](../architecture/configuration.md).

- `config/general.toml`: paths, runtime flags, and k-space normalization; loaded automatically
- `config/coil_sensitivity/*.toml`: settings for the coil-sensitivity method named by `coil_sensitivity_config` in the reconstruction TOML; loaded automatically
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

Use `load_config(...)` with a `data_type` and a path to a
`reconstruction_config`; the demos provide complete examples. Source and mode
requirements are documented under [input formats](input_formats.md),
[sampling](sampling.md), and [motion](motion.md). Use `overrides={...}` for
run-specific settings.

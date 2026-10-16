# Configuration architecture

`src/runtime/runtime_config.py` is the shared configuration boundary for
notebooks, pipelines, and applications.

## Data flow

```mermaid
flowchart TD
    General[_load_general_config] ---|"In: configuration root<br/>Out: general.toml settings"| GeneralGap[" "]
    GeneralGap --> Load[load_config]
    Reconstruction[_load_reconstruction_config] ---|"In: required reconstruction TOML path<br/>Out: reconstruction settings"| ReconstructionGap[" "]
    ReconstructionGap --> Load
    Coil[_load_coil_sensitivity_config] ---|"In: configuration root and accumulated settings<br/>Out: coil-sensitivity settings selected by coil_sensitivity_config"| CoilGap[" "]
    CoilGap --> Load
    Source[_load_source_config] -->|"In: configuration root, data_type and source paths<br/>Out: synthetic or real-data settings"| Load
    Sampling[_load_sampling_simulation_config] -->|"In: configuration root, data_type, sampling path and overrides<br/>Out: explicit sampling settings or recorded-data defaults"| Load
    Motion[_load_motion_simulation_config] -->|"In: configuration root, data_type and motion path<br/>Out: explicit motion settings or recorded-data defaults"| Load
    Overrides[Run-specific overrides] -->|applied after files, before validation| Load
    Load -->|schema and cross-file validation; notebook logging policy| Params[Validated flat params namespace]

    %% Hidden route nodes keep selected labels on the upper edge segments.
    GeneralGap ~~~ Source
    ReconstructionGap ~~~ Sampling
    CoilGap ~~~ Motion

    classDef pythonClass fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef coordinator fill:#bfdbfe,stroke:#1d4ed8,stroke-width:3px,color:#172554
    classDef layoutNode fill:transparent,stroke:transparent,color:transparent
    class General,Reconstruction,Coil,Source,Sampling,Motion pythonClass
    class Load coordinator
    class GeneralGap,ReconstructionGap,CoilGap layoutNode
```

Blue nodes are Python functions; uncolored nodes are parameters or outputs.
`In` and `Out` describe each helper's inputs and returned settings.
`load_config` merges general, reconstruction, coil-sensitivity, source,
sampling, and motion settings in that order. It passes the accumulated settings
to the coil-sensitivity helper; helpers do not call one another to transfer them.

## Configuration boundaries

Schemas and cross-file checks live in `runtime_config.py`. Preprocessing and
reconstruction receive `params`; they do not read TOML files themselves.
Validation of actual acquisition dimensions happens after data loading.

Only keys in the override allowlist can replace file values. Invalid explicit
files or incompatible combinations raise errors. Motion includes share settings
without overriding them: included and selecting files cannot define the same
key, and includes must remain below the including directory without cycles.

Automatic files resolve relative to the repository's `config/` directory.
Caller-supplied paths resolve relative to the working directory unless absolute.

`load_postprocessing_config(...)` validates a separate namespace owned by the
pipeline. `initialize_runtime(params)` then handles device selection and run
state independently of configuration loading.

File choices and usage examples are in the
[configuration guide](../usage/configuration.md).

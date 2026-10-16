# Configuration architecture

`src/runtime/runtime_config.py` is the shared configuration boundary for
notebooks, pipelines, and applications.

## Data flow

```mermaid
flowchart TD
    General[general.toml] -->|automatic| Load[load_config]
    Reconstruction[Reconstruction TOML] -->|required| Load
    Reconstruction -.->|coil_sensitivity_config selects| Coil[Coil-sensitivity TOML]
    Coil -->|automatic| Load
    Source[Synthetic or real-data TOMLs] -->|selected by data_type and source paths| Load
    Sampling[Sampling TOML] -->|explicit; from_data.toml defaults for real data| Load
    Motion[Motion TOML] -->|explicit; as_is.toml defaults for real data| Load
    Common[Common motion TOML] -->|optional include within selected motion file| Motion
    Overrides[Run-specific overrides] -->|applied after files, before validation| Load
    Load -->|schema and cross-file validation; notebook logging policy| Params[Validated flat params namespace]

    classDef pythonClass fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef inputFile fill:#fef3c7,stroke:#d97706,color:#451a03
    class Load pythonClass
    class General,Reconstruction,Coil,Source,Sampling,Motion,Common inputFile
```

Blue nodes are Python functions, amber nodes are files, and uncolored nodes are
parameters or outputs. The dotted edge denotes file selection.

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

File choices and usage examples are in the [README](../../Readme.md#configuration).

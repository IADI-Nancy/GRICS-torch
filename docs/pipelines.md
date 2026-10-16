# Pipeline architecture

The basic pipeline follows the same reconstruction stages as the demos.
Acquisition-specific 2D and 3D pipelines are preimplemented for workflows that
also need input selection, parallel execution, postprocessing, or DICOM export.

## Basic pipeline

```mermaid
flowchart TD
    Files[Configuration files and run overrides] -->|"In: selected data and reconstruction modes<br/>Out: validated params"| Config[load_config]
    Config -->|"In: params<br/>Out: compute devices and run state"| Runtime[initialize_runtime]
    Input[Image, phantom, preprocessed data or raw acquisition] -->|"In: source and params<br/>Out: loaded acquisition"| Load[DataPreparer.load_input_data]
    Runtime --> Load
    Load -->|"In: acquisition and selected slice or volume<br/>Out: k-space, coil maps, motion signal and sampling layout"| Prepare[DataPreparer.prepare_reconstructor_inputs]
    Prepare -->|"In: prepared tensors and params<br/>Out: reconstructed image and motion model"| Reconstruct[JointReconstructor.run]
    Reconstruct -->|"In: reconstruction results<br/>Out: saved results, metadata and completed logs"| Finalize[write_reconstruction and finish_run]

    classDef pythonClass fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef inputFile fill:#fef3c7,stroke:#d97706,color:#451a03
    class Config,Runtime,Load,Prepare,Reconstruct,Finalize pythonClass
    class Files,Input inputFile
```

Blue nodes are Python functions or class methods. Amber nodes are user-selected
inputs. `In` and `Out` describe each stage's inputs and returned state.

Source-specific behavior is contained in `DataPreparer`. `JointReconstructor`
receives the shared interface and does not load configuration or input data;
the notebook or pipeline owns workflow postprocessing and export.

See [configuration](architecture/configuration.md),
[input loading](architecture/input_data_loading.md),
[data preparation](architecture/data_preparation.md), and
[reconstruction](architecture/reconstruction.md) for component boundaries.

## Preimplemented pipelines

`siemens_breast_T2.py` and `siemens_breast_3d_lowres.py` coordinate the shared
components in `src/`. Helpers used only by these entry points live in
`pipelines/utils/`.

### System interaction

```mermaid
flowchart TD
    Inputs[resolve_input_files] ---|"In: raw MRI, SAEC and preprocessed file paths<br/>Out: selected input paths"| InputsGap[" "]
    InputsGap --> Pipeline[run_pipeline]
    Config[load_config] ---|"In: data type, TOML paths and overrides<br/>Out: validated params"| ConfigGap[" "]
    ConfigGap --> Pipeline
    Runtime[initialize_runtime] -->|"In: params<br/>Out: devices and initialized run state"| Pipeline
    Preparer[DataPreparer] -->|"In: input paths, params and devices<br/>Out: prepared k-space, coil sensitivity maps,<br/>motion signal and sampling indices"| Pipeline
    Solver[JointReconstructor] -.->|"In: prepared tensors, sampling indices and params<br/>Out: reconstructed image and motion model"| Execution[timed_reconstruction]
    Execution -->|"In: prepared DataPreparer<br/>Out: reconstructed image, motion model and elapsed time"| Pipeline
    Export[export_reconstruction] ---|"In: final image, native image, motion model and metadata<br/>Out: saved paths and updated result metadata"| ExportGap[" "]
    ExportGap --> Pipeline
    Finish[finish_run] -->|"In: results and timings<br/>Out: run folder, timings and reconstruction results"| Pipeline

    %% Hidden route nodes keep selected labels on the upper edge segments.
    InputsGap ~~~ Runtime
    ConfigGap ~~~ Preparer
    ExportGap ~~~ Finish

    classDef coordinator fill:#bfdbfe,stroke:#1d4ed8,stroke-width:3px,color:#172554
    classDef pythonClass fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef layoutNode fill:transparent,stroke:transparent,color:transparent
    class Pipeline coordinator
    class Inputs,Config,Runtime,Preparer,Solver,Execution,Export,Finish pythonClass
    class InputsGap,ConfigGap,ExportGap layoutNode
```

Arrow labels describe the source component's inputs and outputs.
`run_pipeline` passes data between components; the dotted edge denotes the
solver call inside `timed_reconstruction`. Workflow-specific postprocessing and
optional DICOM export appear in the flows below.

### Shared boundaries

`pipelines/utils/inputs.py` owns file selection. An existing preprocessed file
takes precedence; when it is absent, the pipeline announces its use of raw MRI
and SAEC inputs. Publishing a preprocessing cache is atomic and does not
overwrite an existing destination.

`pipelines/utils/execution.py` adapts prepared data to the solver and output
system. Pipeline-owned runtime overrides take precedence over caller values.
`@managed_execution` surrounds the run to finalize success or failure and
release registered resources.

### 2D T2 pipeline

```mermaid
sequenceDiagram
    participant P as siemens_breast_T2
    participant D as DataPreparer
    participant W as Slice workers
    participant R as JointReconstructor
    participant O as Output and DICOM utilities

    P->>D: load_input_data for all slices
    P->>P: select and validate slice range
    P->>W: share loaded acquisition using fork
    loop Each selected slice
        W->>D: prepare_reconstructor_inputs(slice_idx)
        D-->>W: k-space, maps, sampling, motion
        W->>R: timed_reconstruction
        R-->>W: native image and motion
        W->>W: normalize and zero-fill image
        W-->>P: NumPy result and configuration
    end
    P->>P: order results by slice index
    P->>O: optional DICOM and tensor export
    O-->>P: paths and metadata
```

Each worker copies `DataPreparer` and `params`, selects its own slice, and binds
separate output paths while retaining the shared loaded acquisition. CPU
workers can use fork; CUDA execution requires one worker.

NumPy result transport avoids PyTorch shared-memory lifetime issues. Final
exports occur only after all selected slices succeed.

### 3D low-resolution pipeline

```mermaid
flowchart LR
    Load[Load complete acquisition] --> Prepare[Prepare complete volume]
    Prepare --> Reconstruct[Joint reconstruction]
    Reconstruct --> CPU[Transfer image and motion to CPU]
    CPU --> DICOM[Optional volume DICOM series]
    CPU --> Tensors[Optional tensor export]
    DICOM --> Finish[Finalize manifest]
    Tensors --> Finish
```

The complete volume stays active in one process. This path has no separate
postprocessing configuration or zero-filling step, so the exported and native
images refer to the same solver result.

Output names, flags, and cache lifetime are documented in the
[output guide](usage/outputs.md).

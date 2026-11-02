# Outputs

## Run outputs

Every notebook and Siemens pipeline creates a run directory under `output_root/workflow_label/`, configured in `config/general.toml` (`output_root="runs"` by default). Its name contains only the local date and time, including fractional seconds so simultaneous runs get distinct names. A run with one slice or one volume writes directly under that directory:

```text
runs/<workflow_label>/<YYYY-MM-DD_HH-MM-SS-ffffff>/
├── manifest.json
├── config_resolved.json
├── preprocessing/             # Sampling, input motion and optional debug outputs
├── results/                   # Final image, motion tensors and previews
├── diagnostics/               # Resolution-level and postprocessing plots
├── reconstruction.log         # When text logging is enabled
└── exports/dicom/             # When DICOM export is requested
```

When a pipeline reconstructs multiple slices in one run, each slice's `preprocessing/`, `results/`, `diagnostics/` and `reconstruction.log` are placed under `slice_001/`, `slice_002/`, and so on at the run root. The manifest, resolved configuration and optional DICOM exports remain at the run root.

In notebooks and direct solver runs, `image_reconstructed.pt` is the complex
reconstruction before reference-image normalization and zero-filling, preserving
the repetition dimension. Its PNG shows the magnitude of the repetition mean;
individual repetition previews are also saved when multiple repetitions exist.
The callable Siemens pipelines save both `image_reconstructed_native.pt`, captured
immediately after GRICS, and the final `image_reconstructed.pt`, plus motion and
GRICS text logs. T2's final image includes configured reference-image normalization
and zero-filling; its native image includes neither operation. DICOM export
uses the magnitude of the repetition mean with DICOM intensity scaling. Motion
maps retain the reconstruction grid.

Common output flags are defined under `[runtime]` in `config/general.toml` and
validated by `load_config` for all entry points:

```toml
save_reconstruction_logs = true
save_reconstruction_tensors = true
save_debug_plots = true
```

- `save_reconstruction_logs`: write `reconstruction.log`, independently of tensors and plots.
- `save_reconstruction_tensors`: save the native and final images and motion `.pt` files, independently of logs and plots.
- `save_debug_plots`: save preprocessing, per-level, residual, and final preview/motion figures, independently of logs and tensors.

All three flags can be set through `load_config(overrides={...})` in notebooks
and scripts. `JointReconstructor.run()` reads these settings directly. The old
combined `save_reconstruction_outputs` setting has been removed; migrate custom
configuration files to the common flags above. The pipeline-only `save_outputs`
argument has likewise been replaced with `save_reconstruction_tensors`.

The Siemens pipelines disable plots and defer tensor exports until computation
finishes by calling `JointReconstructor.run(defer_tensor_export=True)`. This
execution option leaves the common saving flags unchanged, so the text log,
resolved configuration, and metadata agree. The log explicitly identifies
deferred exports; metadata records `tensor_export` as `deferred` until the
pipeline takes responsibility for export (`pipeline`), or `disabled` when tensor
saving is off. Direct solver exports use `solver`. Enabled text logs are still written inside the measured reconstruction
call. Their `return_tensors` option controls returning tensors in memory, separately
from saving them to disk. Run/configuration metadata and explicitly requested
DICOM exports have their own lifecycle. Preprocessing and reconstruction metadata
are recorded regardless of the log/tensor/plot flags, including when all three
are disabled. Metadata writes inside reconstruction are included in its timing.

`check_simulated_motion_consistency` controls the simulated-motion numerical check;
its figure additionally requires `save_debug_plots=true`.

To remove inactive run outputs under the configured `output_root`:

```bash
python -m src.utils.clear_runs
```

This permanently removes each matching inactive run directory, including its tensors, figures, logs, manifest, and DICOM exports. Active runs are skipped. Use `--dry-run` to preview or `--workflow-label LABEL` to restrict cleanup.

## Shared data cache

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

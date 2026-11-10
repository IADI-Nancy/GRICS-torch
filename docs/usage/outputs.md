# Outputs

## Run outputs

Every notebook and Siemens pipeline creates a run directory under `output_root/workflow_label/`, configured in `config/general.toml` (`output_root="runs"` by default). Its name contains the local date and time to the second. A run with one slice or one volume writes directly under that directory:

```text
runs/<workflow_label>/<YYYY-MM-DD_HH-MM-SS>/
├── manifest.json
├── config_resolved.json
├── preprocessing/             # Sampling, input motion and optional debug outputs
├── results/                   # Final image, motion tensors and previews
├── diagnostics/               # Resolution-level and postprocessing plots
├── reconstruction.log         # When text logging is enabled
└── exports/dicom/             # Always written by the Siemens pipelines
```

When a pipeline reconstructs multiple slices in one run, each slice's `preprocessing/`, `results/`, `diagnostics/` and `reconstruction.log` are placed under `slice_001/`, `slice_002/`, and so on at the run root. The manifest, resolved configuration and DICOM exports remain at the run root.

`JointReconstructor.run()` returns the complex shared image on the solver grid and the motion model. It records reconstruction metadata and, when debug plots are enabled, writes image and motion previews; it does not save `.pt` tensors. A notebook or direct solver caller saves tensors by calling `write_reconstruction(...)`, which writes the supplied final image and motion. The callable Siemens pipelines additionally save `image_reconstructed_native.pt`, captured immediately after GRICS, and the postprocessed `image_reconstructed.pt`. Both pipelines' final images include configured reference-image normalization and header-based zero-filling/cropping; their native images include neither operation. DICOM export uses the final image magnitude with DICOM intensity scaling. Motion maps retain the reconstruction grid.

See [postprocessing and export](postprocessing.md) for the calls that produce the final and native images in both pipelines.

Common output flags are defined under `[runtime]` in `config/general.toml` and validated by `load_config` for all entry points:

```toml
save_reconstruction_logs = true
save_reconstruction_tensors = true
save_debug_plots = true
```

- `save_reconstruction_logs`: write `reconstruction.log`, independently of tensors and plots.
- `save_reconstruction_tensors`: save the supplied final image and motion `.pt` files, plus a native image when the caller supplies one, independently of logs and plots.
- `save_debug_plots`: save preprocessing, per-level, residual, and final preview/motion figures, independently of logs and tensors.

All three flags can be set through `load_config(overrides={...})` in notebooks and scripts. They control preparation plots, solver logging and plots, and final tensor saving.

The Siemens pipelines accept only Siemens raw `.dat` and SAEC inputs. They call `JointReconstructor.run()` directly, then save tensors with `write_reconstruction(...)`. The 2D pipeline waits for all slice workers before exporting final results. Plot, log, and tensor flags are read from the configuration; callable pipelines also accept `overrides={...}`. The 3D pipeline returns its final and native CPU images, motion, and output paths; the 2D pipeline returns an ordered `reconstructions` list with those values per slice. Both Siemens pipelines always export DICOMs, independently of tensor saving. Metadata is recorded even when logs, tensors, and plots are disabled.

`check_simulated_motion_consistency` controls the simulated-motion numerical check; its figure additionally requires `save_debug_plots=true`.

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

Existing entries are reused. `remove_temporary_data_after_run=true` is the default. At the end of a managed execution (`@managed_execution` or `execution_scope()`), generated cache entries used by that execution are removed when no process still uses them. Set `remove_temporary_data_after_run=false` to retain files for reuse across later executions. Standalone callers release cache leases at interpreter exit or by calling `src.runtime.data_cache.release_leases()`.

To empty the cache:

```bash
python -m src.utils.clear_cache
# For a custom cache location, pass the same path used by your runs:
python -m src.utils.clear_cache --cache-root /path/to/data/cache/grics
```

The utility removes inactive MRD/HDF5 cache entries and abandoned partial files, reports entries still in use, and leaves active files untouched.

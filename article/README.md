# Breast 3D evaluation

Run from the repository root:

```bash
python article/evaluate_breast_3d.py
```

The script discovers all `*_T1_*` HDF5/MRD acquisitions in
`/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/ISMRMRD` and requires
same-named SAEC files in the adjacent `SAEC` directory. Missing prepared files
are written atomically to `/home/pyuser/wkdir/data/GRICS-torch/article_dataset_3D`.
Existing prepared files are validated and reused. Conversion uses the sensor
selected in `config/real_data/saec.toml`; legacy files without sensor metadata
retain their original preprocessing.

Each subject runs through `siemens_breast_3d_lowres.run_pipeline` first with
`config/reconstruction/nonrigid_3d_breast.toml`, then with one motion state.
The one-state run also uses a full one-state schedule at every resolution.
Other reconstruction settings are preserved. The corrected config must request
more than one motion state. Logs are enabled for both runs; tensor saving follows
the general config. DICOM export is disabled for this evaluation.

Timing comes from the pipeline's synchronized `reconstruction_seconds`, including
GRICS logs and metadata but excluding preprocessing, final exports, and scoring.
These are single runs in corrected-then-uncorrected order, without warm-up or
repetitions; first-run initialization and machine load can affect comparisons.

Sharpness uses the implementation copied unchanged from `MedUniVienna` at
`src/utils/sharpness_index.py`, retaining Nora Vogt/Lionel Moisan attribution.
Complex repetitions are averaged before magnitude conversion. Every native axial
partition (including background slices) contributes equally to the subject mean;
mildly oblique axial acquisitions are not resampled. Non-axial volumes are rejected.
Each boxplot observation is one acquisition's mean, with paired observations joined.

The dataset's `results` directory contains:

- `sharpness_boxplot.png` and `time_boxplot.png`.
- `measurements.csv` with subject means, solver times, and run paths.
- `measurements.json` with per-slice scores and all pipeline timing components.
- `runs/` with unique reconstruction folders and resolved configs.

Rerunning reuses prepared inputs but performs fresh reconstructions and replaces
the summary measurements/plots. Old run folders remain available.
Use `--help` for input directories, output dataset directory, device, and
reconstruction-config overrides. Missing inputs or invalid scores stop the run;
completed measurements remain saved and plots include only complete pairs.

# Breast 2D evaluation

Run from the repository root:

```bash
python article/evaluate_breast_2d.py --max-workers 4
```

This follows the 3D evaluation for `00XX_T2_Y` ISMRMRD acquisitions only, where
`XX` is two digits and `Y` is one letter (for example, `0068_T2_m` or
`0068_T2_s`). IDs such as `0154` are excluded. Each acquisition requires a
same-named SAEC file. It atomically caches prepared inputs in
`/home/pyuser/wkdir/data/GRICS-torch/article_dataset_2D`, then invokes the T2
pipeline once with the configured 2D motion correction and once with a single
motion state. Each result includes every reconstructed slice. `--max-workers`
should be fixed across modes and reruns when comparing time; the default uses
the available CPUs.

Acquisitions without a matching SAEC file are excluded (and listed in
`results/excluded_missing_saec.txt`); all complete acquisition pairs run.
Acquisitions without respiratory data for the configured SAEC sensor are also
skipped and recorded with their reasons in
`results/excluded_missing_respiratory_data.json`. Truncated HDF5 inputs are
skipped and recorded in `results/excluded_truncated_files.json`; the damaged
files are left unchanged. Other preparation errors still stop the run.
Existing prepared acquisitions are reused on rerun.
Reruns resume by default from `results/measurements.json`: completed
subject/mode measurements are retained, and only unfinished modes reconstruct.
Previously recorded exclusions are skipped as well. An interrupted mode starts
again from its first slice; uncheckpointed run folders are not imported.
Use the same inputs, configuration, device, and worker count when resuming;
configuration compatibility is not checked automatically. Use `--no-resume`
to rerun all eligible acquisitions and retry exclusions, or a new
`--dataset-dir` for a separate experiment. Old run folders remain available.
If a slice raises `ConstantPhysiologicalSignalError` during reconstruction, the
whole acquisition is excluded from the paired comparison and processing moves
to the next acquisition. Reasons are saved in
`results/excluded_constant_physiological_signal.json`.

The `results` directory contains paired `sharpness_boxplot.png` and
`time_boxplot.png`, plus CSV/JSON measurements and reconstruction runs. Each
sharpness observation is the mean of the per-slice sharpness indices after the
pipeline's postprocessing. The time plot uses the sum of the pipeline's
per-slice synchronized solver times; JSON also retains compute wall time.

# GRICS++ 2D evaluation and Torch timing comparison

```bash
python article/evaluate_grics_cpp_2d.py
```

Reads existing `00XX_T2_Y` outputs from the database's `GRICS-BELT` directory.
The one-state baseline is automatically read from the sibling
`GRICS-BELT_nomoco` directory when available; override with `--nomoco-root`.
Both roots must contain acquisition directories such as `0069_T2_s`, each
containing `Siemens_SingleImage_slice001_image01`, etc. No reconstructions are
launched. Rerun to refresh results as new GRICS++ outputs become available.

The reader uses each slice's XML dimensions and complex-float
`GricsRecon.dat.0000` export, confirmed against the local GRICS++ writer.
Sharpness is scored on the native reconstruction grid, with equal slice
weighting per acquisition. Calibration reference images are not used as
uncorrected reconstructions. Incomplete acquisitions are excluded with reasons.
Paired corrected/one-state sharpness plots require identical slice numbers and
image shapes. Before baseline outputs exist, plots show corrected results only.

Outputs in `article/results_grics_cpp_2d` include sharpness/time boxplots,
CSV/JSON measurements with per-slice details, exclusion reports, and
`time_cpp_vs_torch_corrected.png` (plus the corresponding `nomoco` plot when
available). `time_comparison.csv` records matched acquisition/mode timings and
the GRICS++/Torch time ratio. Only matching slice counts are compared; Torch
measurements are assumed to cover all slices as produced by the 2D evaluator.
Use `--torch-measurements` and `--output-dir` to change paths.

Timing compares sums of per-slice solver times, not acquisition wall time.
GRICS++ uses the log's `Reconstruction time` (around `recon.run()`), while
Torch includes solver construction and logging. GRICS++ total elapsed time is
retained separately. Existing runs may differ in hardware, thread counts,
precision and solver parameters: these are observed runtimes, not a controlled
implementation benchmark. JSON retains GRICS++ thread counts and source paths.
Native-grid GRICS++ sharpness is not directly compared with postprocessed Torch
sharpness because the output grids/postprocessing can differ.

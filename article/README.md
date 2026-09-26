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


The corrected 3D configuration mirrors the supplied GRICS++ breast profile:
16 k-space-energy motion states with 256 quantization bins, acquisition-wide
z-score normalization of the respiratory signal, resolution levels
`[0.25, 0.5, 1.0]`, state counts `[8, 16, 16]`, and Gauss-Newton limits
`[8, 8, 2]`. It uses GRICS++ regularization scaling, image and motion weights
`1.0` and `0.5`, image and motion PCG limits `10` and `15`, tolerances `1e-3`
and `1e-2`, and the motion preconditioner. The normalized Odille calibration
magnitude is applied as the multiplicative image prior corresponding to the
supplied `RegularizationMatrix.dat`. In the reference XML,
`useCalibAsPriorImage=false` selects that supplied matrix; it does not disable
the prior. These choices are explicit in the reconstruction TOML and covered
by `test_breast_3d_matches_grics_cpp_solver_profile`.

This matches the GRICS++ reconstruction schedule and solver profile, but it is
not a bitwise reproduction. The current Torch operators use complex-double
arithmetic, while the supplied 3D GRICS++ XML requests complex-float, and the
two implementations compute their operators independently. The comparison
script therefore reports image and timing agreement rather than asserting
identical voxels.

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
available). `time_comparison.csv` records each matched acquisition/mode's longest slice
timing and
the GRICS++/Torch time ratio. Only matching slice counts are compared; Torch
measurements are assumed to cover all slices as produced by the 2D evaluator.
Use `--torch-measurements` and `--output-dir` to change paths.

Timing plots compare each subject's longest slice solver time, a proxy for the
parallel reconstruction span when slices start together. Measured acquisition
wall time is retained separately; sums of overlapping slice times are not used
for the timing comparison.
GRICS++ uses the log's `Reconstruction time` (around `recon.run()`), while
Torch includes solver construction and logging. GRICS++ total elapsed time is
retained separately. Existing runs may differ in hardware, thread counts,
precision and solver parameters: these are observed runtimes, not a controlled
implementation benchmark. JSON retains GRICS++ thread counts and source paths.
Native-grid GRICS++ sharpness is not directly compared with postprocessed Torch
sharpness because the output grids/postprocessing can differ.

# GRICS++ 3D evaluation and Torch comparison

```bash
python article/evaluate_grics_cpp_3d.py
```

Reads existing `00XX_T1_Y` volumes from `GRICS-BELT-3D` and, when present,
one-state volumes from its `GRICS-BELT-3D_nomoco` sibling. Override those
locations with `--cpp-root` and `--nomoco-root`. The script reads the XML
dimensions, complex-float `GricsRecon.dat.0000`, and completed reconstruction
log for each acquisition; it does not launch reconstructions. Incomplete or
invalid acquisitions are listed in `excluded.json`.

GRICS++ sharpness is the mean of all native axial partitions. Paired C++
corrected/one-state plots require matching volume dimensions. Torch pairs are
matched by acquisition and mode, with the same number of axial partitions.
`comparison.csv` includes both sharpness values, each reconstruction time, and
the C++/Torch time ratio. Cross-implementation sharpness plots explicitly label
the two stages: GRICS++ native export and Torch pipeline output. Their grids
and postprocessing may differ, so those values are descriptive rather than a
controlled quality comparison. Timing similarly reflects existing runs and
may differ in hardware, precision, or solver settings.

By default, outputs go to `article/results_grics_cpp_3d`: measurement CSV/JSON,
exclusion and unmatched reports, C++ corrected/one-state boxplots, and
cross-implementation time and sharpness plots for each available mode. The
script also exports `<subject>_central_planes.png` for each corrected C++
acquisition and `all_subjects_central_planes.png` as a contact sheet. Each
subject shows axial, sagittal, then coronal views. The axial slice is central,
the sagittal slice is at three quarters of the left-right axis, and the
coronal slice is ten voxels before three quarters of the anterior-posterior
axis. Each plane has
four columns: C++ one-state, C++ corrected, Torch one-state, and Torch corrected.
The axial view is rotated 90 degrees counterclockwise; coronal and
sagittal views are flipped vertically. The full slices fill their
panels without clipping. One intensity scale is shared by both
modes within each implementation and subject. Missing reconstructed images are labeled in the plot and recorded in
`image_plot_missing.json`; the script never substitutes a calibration image
for a missing one-state reconstruction. Use `--torch-measurements` and
`--output-dir` to select other results.

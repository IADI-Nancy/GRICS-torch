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

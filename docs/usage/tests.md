# Tests and simulation validation

The simulation scripts in `tests/` validate the complete Shepp-Logan workflow: synthetic acquisition, motion simulation, reconstruction, image comparison, and output writing. Run them directly from the repository root in the installed project environment. An additional [Siemens phantom check](#recorded-siemens-phantom-check) is described at the end for users working with Siemens/ISMRMRD raw data.

## Available simulation cases

| Script in `tests/` | Simulated motion | Quality metric space |
| --- | --- | --- |
| `validate_shepp_logan_2d_nonrigid.py` | Continuous nonrigid motion per readout | Magnitudes without registration |
| `validate_shepp_logan_2d_rigid.py` | Continuous rigid motion per readout | Globally rigid-registered magnitudes |
| `validate_shepp_logan_3d_nonrigid.py` | Nonrigid motion per shot | Magnitudes without registration |
| `validate_shepp_logan_3d_rigid.py` | Rigid motion per shot | Globally rigid-registered magnitudes |

Each simulation script owns its configuration selection and randomized motion parameters. The modules in `tests/utils/` are shared helpers.

## Running the Shepp-Logan validations

The [README installation instructions](../../Readme.md#environment-setup) provide the required simulation dependencies through native CPU/GPU pip installation or the Docker environment.

Run a complete reference-sized batch with an explicit device:

```bash
python tests/validate_shepp_logan_2d_nonrigid.py --runs 20 --device gpu
python tests/validate_shepp_logan_2d_rigid.py --runs 20 --device gpu
python tests/validate_shepp_logan_3d_nonrigid.py --runs 20 --device gpu
python tests/validate_shepp_logan_3d_rigid.py --runs 20 --device gpu
```

`--device` accepts `cpu` or `gpu` and defaults to `gpu` for every validation script. Runtime device fallback notices are recorded in the simulation console log. `--runs` accepts a positive integer and defaults to 20. `--output-root` defaults to the repository's `runs/` directory; use `--help` to inspect these options.

For a shorter execution check:

```bash
python tests/validate_shepp_logan_2d_nonrigid.py --runs 1 --device cpu --output-root /tmp/grics-validation
```

A batch with any count other than 20 reports quality as `INCONCLUSIVE`, even if its simulations complete successfully. The frozen limits were calibrated for batches of 20.

The scripts load phantom, sampling, coil calibration, and reconstruction settings from the configuration TOML files. They override motion samples, execution controls, and the seed (`20261101 + run_number`). Reconstruction logs and tensors are enabled, while debug plots are disabled. The comparison PNG is always saved separately. See [configuration](configuration/configuration.md) for configuration ownership and overrides.

## Progress and verdicts

Each simulation reports its stages on one terminal line:

```text
Simulation 001: reconstruction starting ... SUCCESS; evaluation and output ... PASSED
```

`SUCCESS` means the solver returned finite image and motion tensors, with an image shape matching the ground truth. The subsequent `PASSED` means metric evaluation, residual checks, output writing, and execution-scope finalization completed. These stage messages do not assert that image quality improved. An execution exception prints `FAILED`, stops the batch, and identifies the simulation log in the raised error.

After all simulations finish, the summary prints medians and quartiles, acceptance conditions, individual verdicts, and an overall quality verdict:

```text
==================== SSIM statistics ====================
SSIM relative improvement: 16.43% [11.30%, 20.92%]; criterion: >= SSIM_imp_ref = 11.55% — PASSED

==================== NRMSE statistics ====================
NRMSE relative improvement: 54.55% [38.76%, 61.20%]; criterion: >= NRMSE_imp_ref = 39.27% — PASSED

==================== Overall quality ====================
Overall quality: PASSED
```

Quality misses and inconclusive results are reports, not raised exceptions; they do not by themselves produce a nonzero process exit status. After the run, check the terminal’s `Overall quality` result. `PASSED` means all image-quality checks passed; `QUALITY CHECK NOT MET` means at least one failed; `INCONCLUSIVE` means the run cannot be compared with the reference. You can also find this result in `similarity_summary.json`, under `quality_assessment.status`. A script finishing without an error does not by itself mean that image quality passed.

## Image metrics

SSIM and NRMSE compare image magnitudes in restored acquisition intensity units. NRMSE is `||candidate - reference|| / ||reference||`; SSIM uses the ground-truth magnitude range as its `data_range`. Higher SSIM and lower NRMSE are better. Each printed interval is `[Q1, Q3]` across simulations, not a confidence interval.

Percentage improvements are calculated for each before/after pair, then summarized across simulations:

```text
SSIM improvement (%) = 100 * (corrected - uncorrected) / abs(uncorrected)
NRMSE reduction (%)  = 100 * (uncorrected - corrected) / abs(uncorrected)
```

Both positive values indicate improvement. The console calls both quantities "relative improvement". The reported median is a median of paired percentage changes, not a percentage change between the two aggregate medians. Zero baselines are excluded from percentage summaries and their count is reported.

For rigid cases, the corrupted and corrected images are independently registered to ground truth using one global rotation and translation. No intensity scaling or nonrigid deformation is fitted. Metrics cover the entire original grid. Raw metrics and registration transforms are retained in `simulation_metrics.json`; the comparison PNG shows registered images, while reconstruction tensors retain the solver's coordinate frame. Nonrigid cases do not register comparison images.

Image residual diagnostics measure k-space fit. Motion residual diagnostics measure update size, not motion error relative to ground truth. These diagnostics are distinct from the image-quality acceptance criteria.

## Reference limits

The validation uses provisional reference-based thresholds: the step 2 bootstrap bounds directly, without additional tolerance margins. Limits apply to batch medians. Both percentage improvements must also be positive, and each metric's paired, two-sided Wilcoxon p-value must be below `0.025`.

| Case | SSIM minimum | NRMSE maximum | SSIM improvement minimum | NRMSE reduction minimum |
| --- | ---: | ---: | ---: | ---: |
| 2D nonrigid | 0.89 | 0.07 | 5.56% | 43.84% |
| 2D rigid | 0.91 | 0.08 | 3.47% | 43.02% |
| 3D nonrigid | 0.97 | 0.14 | 1.11% | 28.05% |
| 3D rigid | 0.94 | 0.21 | 11.55% | 39.27% |

Metric values and thresholds are displayed with two decimal places (`%.2f`). Comparisons and JSON output retain full precision, so values that look equal after display rounding can still fall on opposite sides of a threshold. P-values and their threshold are displayed in scientific notation with two decimal places to preserve small values; `0.025` is shown as `2.50e-02`. Full-precision bounds are stored in [validation_quality.py](../../tests/utils/validation_quality.py).

### How the limits were derived

The limits come from reference batches of 20 simulations. A batch median depends on which simulations it contains. To estimate that variation, bootstrap resampling repeatedly draws a new batch from the reference records, allowing a record to be selected more than once. The resulting medians provide the lower and upper bounds used directly as the acceptance limits above.

The calculation is:

1. Order the reference records by simulation number and draw 100,000 bootstrap resamples of 20 paired records, using NumPy's `default_rng(42)`. Keep each corrupted/corrected pair together.
2. Compute corrected medians and medians of paired percentage improvements within each resample. Take the 1st percentile for SSIM and improvement lower bounds, and the 99th percentile for the NRMSE upper bound. Set the acceptance thresholds to these percentile values.
3. Require each Wilcoxon p-value to be below `0.05 / 2 = 0.025`, applying a Bonferroni correction for the two metric tests. Wilcoxon uses the raw paired metric differences; the positive improvement requirements establish the desired direction.

The reference batches each contain 20 simulations. Their identifiers, bootstrap bounds, checked settings, and calculation comments are stored in [validation_quality.py](../../tests/utils/validation_quality.py). Both rigid references use calibration weighting and registered metrics; both nonrigid references have calibration weighting disabled. The 3D rigid reference is `shepp_logan_3d_rigid/2026-11-04_10-20-02_707423`.

Comparability checks cover phantom dimensions, acquisition counts, coil method, calibration weighting, resolution and iteration schedules, regularization settings, and motion-state count. Device is unrestricted. These checks are not a complete configuration fingerprint: changes to other settings or motion sampling may also require new reference batches. Recalibrate with independent batches and devices when changing the validation protocol.

## Saved results

Each invocation creates a batch directory:

```text
<output-root>/simulation_validation/shepp_logan_<case>/<batch timestamp>/
├── similarity_summary.json
├── simulation_001.log
└── simulation_001/<run timestamp>/
    ├── simulation_metrics.json
    ├── image_comparison.png
    ├── config_resolved.json
    ├── manifest.json
    └── results/
        ├── image_reconstructed.pt
        └── motion_parameters.pt
```

The console log captures library progress, warnings, and device fallback notices. `simulation_metrics.json` records the seed, sampled motion parameters, device, residual diagnostics, image metrics, and checked validation settings. `similarity_summary.json` contains aggregate metrics and `quality_assessment`, including limits, reference settings, method, individual checks, and verdict. It is written only after all requested simulations complete.

The comparison PNG shows corrupted, corrected, and ground-truth images: one row for 2D, and three central orthogonal planes for 3D. See [outputs](outputs.md) for the shared run metadata and reconstruction artifacts.

## Siemens phantom check

This additional check is intended for users who will reconstruct Siemens or ISMRMRD raw data.

[validate_siemens_phantom_2d.py](../../tests/validate_siemens_phantom_2d.py) exercises the complete recorded-data path: download, Siemens-to-ISMRMRD conversion, acquisition loading, preparation, sequential reconstruction, and image export. It uses `config/reconstruction/nonrigid_2d.toml` with the standard solver schedule and `N_motion_states=1`. No motion is simulated.

### Running the check

Use the [Docker setup](../../Readme.md#docker), defined in the [Dockerfile](../../build/Dockerfile), for this check. Run from the repository root with that Python environment and ensure `siemens_to_ismrmrd` is on `PATH`:

```bash
python tests/validate_siemens_phantom_2d.py --device gpu
```

`--device` accepts `cpu` or `gpu` and defaults to `gpu`. `--output-root` defaults to `runs/`. To use another local phantom acquisition, pass `--raw-data-file /path/to/acquisition.dat`; its matching `acquisition.physio.txt` must already exist. The zero-physiology checks also apply to custom inputs.

### Inputs and execution

The default input is `meas_MID00020_FID552369_AX_T2_TSE_HR_1NEX_SAT.dat`, a three-slice 2D TSE Siemens acquisition. The script downloads this file and its `.physio.txt` sidecar from Zenodo [1] into `demos_data/` only when each file is absent. Existing inputs are reused without being overwritten, and interrupted downloads are not retained as completed files. The script downloads the pre-generated physiology; it does not generate physiological samples.

The physiological text contains one `0 -1 0` row per retained imaging readout in full acquisition order, before slice selection. The columns are sensor identifier, timestamp marker, and physiological value. The `-1` marker denotes already-synchronized samples without timestamps. The loader uses `siemens-physio_text` and checks alignment with the MRI readouts.

Every invocation creates a fresh temporary conversion cache and disables preprocessing caching, so conversions from previous runs are not reused. The acquisition is converted and read once. All slices are then prepared, reconstructed, and exported sequentially, in zero-based index order. Temporary converted data are removed when the execution ends.

### Checks and progress

The script requires zero loaded physiology, one zero-valued binned motion signal, and all readouts assigned to motion state zero. For each slice, the reconstructed image must match the prepared spatial dimensions, contain finite values, and be nonempty. Reconstructed motion must be finite and exactly zero. A failed assertion or conversion, loading, reconstruction, or export exception stops the run with a nonzero exit status.

Success banners identify conversion, ISMRMRD reading, and each slice's preparation, reconstruction, DICOM export, and PNG export. Export banners include the saved path. A final banner appears only after all slices and exports complete:

```text
========================================================================
[phantom] SUCCESS: ALL 3 SLICES
========================================================================
```

This is an execution check without a ground-truth image or SSIM/NRMSE acceptance threshold. A success banner confirms the completed stage, not an improvement in image quality. Constant physiological channels are supported because one motion state is explicitly requested; multiple-state reconstruction retains the constant-signal error.

### Saved outputs

Each invocation creates a timestamped run directory:

```text
<output-root>/siemens_phantom_2d/<run timestamp>/
├── config_resolved.json
├── manifest.json
├── exports/dicom/
│   ├── slice_001.dcm
│   ├── slice_002.dcm
│   └── slice_003.dcm
├── slice_001/
│   └── results/
│       ├── image_reconstructed.pt
│       ├── image_reconstructed.png
│       └── motion_parameters.pt
├── slice_002/
└── slice_003/
```

Every slice folder contains its own results and reconstruction logs. Folder and DICOM filenames use one-based slice numbers; progress messages use zero-based indices. DICOM export uses the acquisition header and each selected slice's geometry. All exported slices share study, series, and frame-of-reference identifiers. See [outputs](outputs.md) for shared run metadata and reconstruction artifacts.

## References

[1] [Small phantom MRI data for reconstruction code testing, Zenodo record 24314067](https://zenodo.org/records/24314067).

# Tests and simulation validation

The scripts in `tests/` validate the complete Shepp-Logan workflow: synthetic acquisition, motion simulation, reconstruction, image comparison, and output writing. Run them directly from the repository root in the installed project environment. They are manual integration validations with statistical quality checks, rather than a pytest unit-test suite.

## Available cases

| Script in `tests/` | Simulated motion | Quality metric space |
| --- | --- | --- |
| `validate_shepp_logan_2d_nonrigid.py` | Continuous nonrigid motion per readout | Magnitudes without registration |
| `validate_shepp_logan_2d_rigid.py` | Continuous rigid motion per readout | Globally rigid-registered magnitudes |
| `validate_shepp_logan_3d_nonrigid.py` | Nonrigid motion per shot | Magnitudes without registration |
| `validate_shepp_logan_3d_rigid.py` | Rigid motion per shot | Globally rigid-registered magnitudes |

Each script owns its configuration selection and randomized motion parameters. The modules in `tests/utils/` are shared helpers, not executable tests: `validation_metrics.py` computes and prints metrics, `validation_registration.py` registers rigid-case comparison images, and `validation_quality.py` holds reference limits and evaluates batch quality.

## Running the validations

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

The scripts load phantom, sampling, coil calibration, and reconstruction settings from the normal TOML files; they do not use a reduced solver schedule. They override motion samples, execution controls, and the seed (`20261101 + run_number`). Reconstruction logs and tensors are enabled, while debug plots are disabled. The comparison PNG is always saved separately. See [configuration](configuration/configuration.md) for configuration ownership and overrides.

## Progress and verdicts

Each simulation reports its stages on one terminal line:

```text
Simulation 001: reconstruction starting ... SUCCESS; evaluation and output ... PASSED
```

`SUCCESS` means the solver returned finite image and motion tensors, with an image shape matching the ground truth. The subsequent `PASSED` means metric evaluation, residual checks, output writing, and execution-scope finalization completed. These stage messages do not assert that image quality improved. An execution exception prints `FAILED`, stops the batch, and identifies the simulation log in the raised error.

After all simulations finish, the summary prints medians and quartiles, acceptance conditions, individual verdicts, and an overall quality verdict:

```text
==================== SSIM statistics ====================
SSIM relative improvement: 16.43% [11.30%, 20.92%]; criterion: >= SSIM_imp_ref = 10.39% — PASSED

==================== NRMSE statistics ====================
NRMSE relative improvement: 54.55% [38.76%, 61.20%]; criterion: >= NRMSE_imp_ref = 35.34% — PASSED

==================== Overall quality ====================
Overall quality: PASSED
```

| Quality verdict | Meaning |
| --- | --- |
| `PASSED` | All six checks meet their limits for a comparable batch. |
| `QUALITY CHECK NOT MET` | At least one absolute-quality, improvement, or p-value check misses its limit. Individual checks print `NOT MET`. |
| `INCONCLUSIVE` | The batch size, checked configuration settings, or metric space differs from the reference. The reason is printed. |

Quality misses and inconclusive results are reports, not raised exceptions; they do not by themselves produce a nonzero process exit status. Automation must read `quality_assessment.status` in `similarity_summary.json` to enforce a quality gate.

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

Limits apply to batch medians. Both percentage improvements must also be positive, and each metric's paired, two-sided Wilcoxon p-value must be below `0.025`.

| Case | SSIM minimum | NRMSE maximum | SSIM improvement minimum | NRMSE reduction minimum |
| --- | ---: | ---: | ---: | ---: |
| 2D nonrigid | 0.883 | 0.077 | 5.00% | 39.46% |
| 2D rigid | 0.901 | 0.086 | 3.12% | 38.72% |
| 3D nonrigid | 0.964 | 0.150 | 1.00% | 25.24% |
| 3D rigid | 0.931 | 0.232 | 10.39% | 35.34% |

The reference batches each contain 20 CUDA simulations. Their identifiers, bootstrap bounds, checked settings, and calculation comments are stored in [validation_quality.py](../../tests/utils/validation_quality.py). Both rigid references use calibration weighting and registered metrics; both nonrigid references have calibration weighting disabled. The 3D rigid reference is `shepp_logan_3d_rigid/2026-11-04_10-20-02_707423`.

The limits are derived as follows:

1. Order the reference records by simulation number and draw 100,000 bootstrap resamples of 20 paired records, using NumPy's `default_rng(42)`. Keep each corrupted/corrected pair together.
2. Compute corrected medians and medians of paired percentage improvements within each resample. Take the 1st percentile for SSIM and improvement lower bounds, and the 99th percentile for the NRMSE upper bound.
3. Apply explicit engineering margins: subtract 0.01 from the SSIM bound, multiply the NRMSE bound by 1.10, and multiply each improvement bound by 0.90. Round lower limits downward and upper limits upward, to 0.001 for absolute metrics and 0.01 percentage points for improvements.
4. Require each Wilcoxon p-value to be below `0.05 / 2 = 0.025`, applying a Bonferroni correction for the two metric tests. Wilcoxon uses the raw paired metric differences; the positive improvement requirements establish the desired direction.

The bootstrap bounds describe uncertainty in the reference medians. They are not simultaneous confidence bounds or guaranteed prediction limits for new batches or hardware. The extra margins are engineering choices, not measured CPU/GPU tolerances. Limits remain frozen during validation; a batch never sets its own passing thresholds.

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

Additional simulations have their own log and run directory. The console log captures library progress, warnings, and device fallback notices. `simulation_metrics.json` records the seed, sampled motion parameters, device, residual diagnostics, image metrics, and checked validation settings. `similarity_summary.json` contains aggregate metrics and `quality_assessment`, including limits, reference settings, method, individual checks, and verdict. It is written only after all requested simulations complete.

The comparison PNG shows corrupted, corrected, and ground-truth images: one row for 2D, and three central orthogonal planes for 3D. See [outputs](outputs.md) for the shared run metadata and reconstruction artifacts.

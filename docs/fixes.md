# Corrections relative to GRICS-torch 2.0

This document describes corrections and remaining limitations in the current implementation relative to GRICS-torch 2.0. Focused unit tests cover selected corrections. GPU performance changes are described separately in [performance](performance.md).

## Data reading and preparation

**ESPIRiT workspace limits** (`CoilSensitivityCalculator`). GPU calibration queries the cuSOLVER SVD workspace before factorization. When that query fails, the implementation estimates whether a Gram-matrix eigendecomposition fits in GPU memory and selects double or single precision with a notice. Matrix size depends on the calibration width, kernel width, spatial dimension, and coil count; the cuSOLVER workspace limit also depends on the CUDA version. GPU out-of-memory, cuSOLVER, and runtime errors trigger an announced CPU retry. The CPU device is available to all three exception paths.

**Header-declared k-space centres** (`ISMRMRDReader`). Phase-encoding indices are shifted by `grid_size // 2 - declared_center`, placing the declared centre at the centred FFT origin. Each phase-encoding grid size is the maximum of the encoded matrix size and the size derived from encoding limits and acceleration. This also applies to partition encoding in 3D slabs. Placement is unchanged when the calculated shift is zero.

**Repeated imaging lines** (`ISMRMRDReader`). Readouts mapped to the same repetition, phase-encoding index, and slice or partition are accumulated and averaged, with a notice when repeats occur. Each readout retains its physiological sample. The reader groups repetitions using the repetition counter.

**Respiratory clipping** (`SAECReader`). Samples used to fit respiratory drift are clipped to the interval from `mean - 2 * sigma` to `mean + 2 * sigma` with `np.clip`, so both bounds apply to the same signal.

**Voxel spacing without an ISMRMRD header** (`DataPreparer`). With `regularization_scaling = "voxel_aware"`, explicit `voxel_spacing_mm` takes precedence over encoded spacing from the acquisition header. Preprocessed data without a header therefore require explicit spacing for this mode. See the [configuration guide](usage/configuration/configuration.md).

## Solver

**Rigid Jacobian rotation factor** (`MotionPerturbationSimulator`). The rigid Jacobian rotates the gradients of the warped image into the coordinates used for the motion derivatives. This correction is enabled by `jacobian_rotation_factor`. Interpolation contributes to the approximation error when comparing the analytical derivative with finite differences.

**Real motion increments** (`JointReconstructor._solve_motion`). With `motion_step_real = true`, the data term solves `Re(J^H J) d = Re(J^H r)`, where `J` is the motion Jacobian, `r` is the k-space residual, and `d` is the real motion increment. This generally differs from solving `J^H J d_complex = J^H r` and then setting `d = Re(d_complex)`. Regularization is applied separately by the motion solver.

**Image scaling between resolution levels** (`upsample_data`). With `warm_start_rescale = true`, an interpolated image is multiplied by `sqrt(n_previous / n_next)` to account for the intensity scaling introduced by cropping k-space with a unitary FFT.

**CG candidate selection and breakdown guards** (`ConjugateGradientSolver`). CG verifies candidate residuals directly and, when the initial guess would otherwise be selected, also checks the iterate with the smallest recorded recursive residual. It prints a notice if no candidate improves on the initial guess. For a nonzero right-hand side, the solve is normalized to `||b|| = 1` and rescaled on return; breakdown checks use relative scales.

**Incomplete readout lines** (`EncodingOperator`). The optimized normal operator requires complete readout lines. Sampling is checked at construction; when this condition is not met, `normal(image)` explicitly uses `adjoint(forward(image))` and prints a notice.

**External initial-motion scaling** (`initialization.py`). Initial translations and non-rigid displacements are supplied in full-resolution voxels and rescaled to the coarsest reconstruction grid. Rigid angles retain their angular units.

The three reconstruction settings `jacobian_rotation_factor`, `motion_step_real`, and `warm_start_rescale` are enabled in the shipped configurations. `GRICS_COMPAT=1` in the benchmark scripts disables these three settings only. Other solver, reader, and initialization changes remain active. The validation batches below assess reconstruction quality against synthetic ground truth.

## Recorded validation results

The latest batch for each of the four Shepp-Logan validation scripts was started on 2026-11-24. Each contains 20 completed simulations, records device `cuda:0`, and reports `PASSED` with all six quality checks satisfied and matching reference requirements.

SSIM and NRMSE in the table are medians across the 20 simulations. The improvement columns are medians of the per-simulation paired percentage changes relative to the uncorrected reconstruction. Rigid cases use globally rigid-registered magnitude images; non-rigid cases use magnitude images without registration.

| Case | Median SSIM ≥ minimum | Median NRMSE ≤ maximum | Median SSIM improvement ≥ minimum | Median NRMSE reduction ≥ minimum | Recorded verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 2D rigid | 0.92108 ≥ 0.91120 | 0.06752 ≤ 0.07772 | 9.50% ≥ 3.47% | 52.92% ≥ 43.02% | PASSED |
| 2D non-rigid | 0.90120 ≥ 0.89384 | 0.06514 ≤ 0.06946 | 6.78% ≥ 5.56% | 48.87% ≥ 43.84% | PASSED |
| 3D rigid | 0.95183 ≥ 0.94125 | 0.18703 ≤ 0.21001 | 16.54% ≥ 11.55% | 55.22% ≥ 39.27% | PASSED |
| 3D non-rigid | 0.98373 ≥ 0.97484 | 0.10816 ≤ 0.13601 | 1.49% ≥ 1.11% | 34.78% ≥ 28.05% | PASSED |

Both paired, two-sided Wilcoxon p-values are `1.9073486328125e-06` in every batch, below the required `0.025`. All four median comparisons and both p-value checks pass for each case. Batch size, settings, and metric space also match the reference requirements. Values above are rounded for display; acceptance uses full precision.

These batches assess the combined reconstruction pipeline under the four synthetic validation settings. Focused unit tests check selected corrections, including the rigid Jacobian, real motion operator, resolution scaling, CG behavior, and sampling handling.

## Remaining limitations

**Acquisition averages.** Repeated lines are averaged within a repetition. This grouping combines complementary sampling averages and true repeated averages under the same repetition index.

**Breast configurations without a header.** The shipped breast reconstruction configurations select voxel-aware scaling but leave `voxel_spacing_mm` commented out. Inputs without an ISMRMRD header need an explicit spacing override.

**Rigid simulation amplitudes.** `rigid_2d_common.toml` retains `max_tx = 0.2` and `max_ty = 2.0`, in millimetres. These settings give the simulated translations different maximum amplitudes along x and y.

**DICOM orientation.** Export flips the two in-plane pixel axes while constructing orientation and position from the recorded scanner directions. Comparison with scanner-generated DICOMs is needed to validate this orientation convention. See [postprocessing](usage/postprocessing.md) for the export path.

**SAEC timing.** MRI readout timestamps are referenced to the last retained imaging readout; SAEC timestamps are referenced to the sequence stop trigger. A gap between those events can shift alignment. Nonzero `physio_clock_drift_seconds` is supported for Polaris, text, and array physiology, and is rejected for SAEC.

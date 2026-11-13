# GRICS-torch 3.0: corrections with respect to 2.0

Problems found in GRICS-torch 2.0 and corrected in 3.0. Each one was reproduced on that commit before being changed; the scripts that reproduce them are in `tests/unit/`. Three of the fixes change the numerics of the solver and can be switched off in the reconstruction TOML to get the GRICS-torch 2.0 results back (`jacobian_rotation_factor`, `motion_step_real`, `warm_start_rescale`); `benchmarks/recon_dump.py` does that with `GRICS_COMPAT=1`.

## Data reading and preparation

**ESPIRiT in 3D crashed on the GPU with 7 coils or more** (`CoilSensitivityCalculator`). With the shipped widths (calibration 48, kernel 12) the calibration matrix is 50653 x C·1728 whatever the image size, and cuSOLVER refuses the SVD once its workspace passes 2^31 elements; with the CUDA 12.8 libraries that happens at C = 7 (the exact count depends on the cuSOLVER version). Only out-of-memory errors were caught, so the preparation died. Now: the workspace is queried first; if cuSOLVER cannot do it, the SVD is taken from the eigendecomposition of the Gram matrix A^H A on the GPU when that fits in memory, otherwise ESPIRiT runs on the CPU with a message that says why. cuSOLVER and runtime errors fall back to the CPU too.

**Readout oversampling was assumed to be 2x for every ISMRMRD file** (`ISMRMRDReader`). A file without oversampling lost the outer half of its field of view. The factor now comes from `encodedSpace.x / reconSpace.x` (1 or 2).

**The k-space centre declared in the header was ignored** (`ISMRMRDReader`). Lines were stored at their raw `kspace_encode_step_1` index, so a partial-Fourier range indexed from zero put k = 0 away from the FFT centre; the magnitude image is unchanged but the phase, and with it the motion model, is wrong. Lines are now placed at `index - center + N/2` on a grid of `encodedSpace.matrixSize`; files whose indices already span the full grid are read exactly as before. The same applies to `kspace_encode_step_2` for 3D slabs.

**A line acquired twice overwrote the previous one** (`ISMRMRDReader`). Repeated (ky, kz, repetition) lines are now averaged, with a message. Each readout still carries its own physiological sample into the binning; that part of the data model is unchanged.

**Respiratory clipping before the drift fit** (`SAECReader`). The second `np.where` started from the unclipped array, so only the lower bound was applied. Replaced by `np.clip`.

**Voxel spacing without an ISMRMRD header** (`DataPreparer`). `voxel_aware` scaling read the spacing from the header and failed on preprocessed files without one (the Zenodo demo). `voxel_spacing_mm` in the reconstruction TOML is used when present.

## Solver

**Rigid Jacobian without the rotation factor** (`MotionPerturbationSimulator`). The warp is a pull-back, warped(r) = rho(T(r)) with T(r) = c + R (r - c) + t, so the derivative with respect to the motion parameters needs (grad rho)(T(r)) = R grad(warped)(r). The code used grad(warped) directly, i.e. R^T too much. Against finite differences of the forward model the error was 15 %, 35 % and 67 % at 0.1, 0.3 and 0.6 rad; with the factor it is 11 % at every angle (the interpolation). Key: `jacobian_rotation_factor`.

**Motion step solved in complex arithmetic** (`JointReconstructor._solve_motion`). The motion parameters are real, so the Gauss-Newton step is Re(J^H J) d = Re(J^H r); the code solved J^H J d = J^H r and took Re(d). For non-rigid motion the two steps differ by 16 to 41 % and the real step leaves a smaller residual (0.20 to 0.52 of the initial one instead of 0.37 to 0.86 in the test of `verify4/solver`); for rigid motion the difference is about 1 %. Key: `motion_step_real`.

**Warm start between levels** (`upsample_data`). The coarse-level image is brighter by sqrt(n_full / n_level) because the k-space crop is unitary, and the next level started from it unscaled (the first image CG began at a relative residual of 0.92 to 0.97 instead of 0.24 to 0.37). The upsampled image is now rescaled. Key: `warm_start_rescale`.

**Zero motion update returned silently** (`ConjugateGradientSolver`). The returned iterate was the best of x0, the true-residual refresh points and the last iterate; with `max_iter <= cg_true_residual_interval` that is x0 or the last iterate, and when the last one was worse than x0 the solver returned x0, i.e. no motion update, without a message. The iterate with the smallest recursive residual is now kept as a candidate and verified when x0 would otherwise be returned, and a message is printed if the start is still the best.

**Absolute breakdown thresholds** (`ConjugateGradientSolver`). The 1e-15 guards made the result depend on the k-space scale (5.8 % and 14.6 % image change between normalised and raw Siemens scale on the Zenodo slice). The solver now works on the system scaled to ||b|| = 1 and the guards are relative to the initial residual. On normalised data the iterates are unchanged to machine precision.

**`normal()` assumed complete readout lines** (`EncodingOperator`). The phase-mask shortcut is only valid when each state holds all kx samples of its lines; that was never checked and a partial state gave a 36 % wrong result without any error. The operator now checks the sampling at construction and falls back to adjoint(forward) with a message.

**Units of an external initial motion** (`initialization.py`). `initial_motion` was used unscaled at the coarsest level while `upsample_data` rescales translations between levels, so a motion given in full-resolution pixels ended 4x too large with levels [0.25, 0.5, 1]. It is now rescaled to the level grid; the expected unit is full-resolution voxels.

## Validation with the corrections on

The four validation scripts of 2.0 (`tests/validate_shepp_logan_*.py`, 20 simulations each, frozen limits of GRICS-torch 2.0), run on GRICS-torch 3.0 with the default settings on an RTX PRO 6000, 2026-11-13:

| batch | SSIM (limit) | NRMSE (limit) | verdict |
|---|---|---|---|
| 2D rigid | 0.914 (≥ 0.901) | 0.079 (≤ 0.086) | PASSED |
| 2D non-rigid | 0.906 (≥ 0.883) | 0.062 (≤ 0.077) | PASSED |
| 3D rigid | 0.954 (≥ 0.931) | 0.189 (≤ 0.232) | PASSED |
| 3D non-rigid | 0.984 (≥ 0.964) | 0.108 (≤ 0.150) | PASSED |

With the three switches off (`GRICS_COMPAT=1` in `benchmarks/run_all.sh`) the code reproduces GRICS-torch 2.0 to machine precision: operator outputs within 3.6e-16 and the five reconstructions of the benchmark (four synthetic cases and the Zenodo breast slice) within 6.4e-15 in relative norm. With the corrections on, the images differ from GRICS-torch 2.0 by 0.3 to 3.1 % and the motion by 1 to 17 %, and the raw NRMSE against the simulated truth is lower in three of the four synthetic cases (2D non-rigid 0.0546 to 0.0526, 3D rigid 0.2692 to 0.2637, 3D non-rigid 0.1054 to 0.1040; 2D rigid 0.5393 to 0.5416 raw, where the registered metrics above improve). GRICS-torch 2.0's own 2D rigid batch on the same machine gives 0.909 and 0.0785; the other batches sit within their limits with similar margins. The rotations of these simulations are small (under 1°), so the Jacobian correction barely shows here; it matters for rotations of a few degrees. The unit tests in `tests/unit/` cover each fix in isolation (`python -m pytest tests/unit`).

## Left as GRICS-torch 2.0

`max_tx = 0.2` next to `max_ty = 2.0` in `rigid_2d_common.toml` and the breast configurations requiring an ISMRMRD header feed the frozen validation limits, so they are documented rather than changed. The DICOM export flips the pixel array while the orientation tags are derived from the unflipped directions; whether that compensates a scanner convention could not be checked without scanner DICOMs. The physiological time axis is anchored on the last imaging readout while SAEC uses its stop trigger, so a gap between the two shifts the physiological signal by that gap and `physio_clock_drift_seconds` cannot be used to compensate it for SAEC data.

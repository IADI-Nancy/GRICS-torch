from contextlib import nullcontext

import numpy as np
import torch

from src.utils.fftnc import ifftnc
from src.preprocessing.utils.spline import spline_fft_solver


class CoilSensitivityCalculator:
    """Calculate coil sensitivity maps from calibration k-space."""

    VALID_METHODS = {"espirit", "odille-spline"}

    def __init__(self, params, sp_device=None):
        self.params = params
        self.sp_device = sp_device

    def calculate(self, kspace, reference_kspace=None):
        method = str(self.params.coil_sensitivity_method).strip().lower()
        if method not in self.VALID_METHODS:
            raise ValueError(
                "coil_sensitivity_method must be one of "
                f"{sorted(self.VALID_METHODS)}."
            )

        if method == "espirit":
            return self.calculate_espirit(kspace, reference_kspace=reference_kspace)
        return self.calculate_odille_spline(kspace, reference_kspace=reference_kspace)

    @staticmethod
    def _average_repetitions(kspace):
        if kspace.shape[1] == 1:
            return kspace[:, 0]
        return torch.mean(kspace, dim=1)

    def _calibration_kspace(self, kspace, reference_kspace=None):
        """Return the averaged k-space used by the selected CSM method."""
        if reference_kspace is not None:
            return self._average_repetitions(reference_kspace)
        return self._average_repetitions(kspace)

    @staticmethod
    def _cupy_cleanup(cp):
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()

    @staticmethod
    def _maps_numpy_to_torch(maps_np, device):
        maps_np = maps_np.astype(np.complex128, copy=False)
        maps_t = torch.from_numpy(np.stack([maps_np.real, maps_np.imag], axis=-1))
        return torch.complex(maps_t[..., 0], maps_t[..., 1]).to(device)

    def _resolved_espirit_device(self, kspace):
        if self.sp_device is not None:
            return self.sp_device
        from src.runtime.sigpy_backend import resolve_espirit_device

        return resolve_espirit_device(kspace.device.type == "cuda")

    def _run_espirit_calibration_cpu(self, kspace_block, calib_width, kernel_width, sp_device=None):
        import sigpy.mri as spmri

        if sp_device is None:
            sp_device = self._resolved_espirit_device(kspace_block)
        kspace_np = kspace_block.detach().cpu().numpy().astype(np.complex64, copy=False)
        maps_np = spmri.app.EspiritCalib(
            kspace_np,
            calib_width=calib_width,
            kernel_width=kernel_width,
            max_iter=self.params.espirit_max_iter,
            device=sp_device,
        ).run()
        return self._maps_numpy_to_torch(maps_np, kspace_block.device)

    @staticmethod
    def _calibration_matrix_shape(kspace_block, calib_width, kernel_width):
        ndim = kspace_block.ndim - 1
        return (calib_width - kernel_width + 1) ** ndim, int(kspace_block.shape[0]) * kernel_width ** ndim

    @staticmethod
    def _gesvd_workspace_fits(cp, m, n):
        """cuSOLVER refuses gesvd (INVALID_VALUE) once its workspace exceeds 2^31 elements."""
        from cupy_backends.cuda.libs import cusolver
        try:
            cusolver.cgesvd_bufferSize(cp.cuda.device.get_cusolver_handle(), m, n)
            return True
        except cusolver.CUSOLVERError:
            return False

    @staticmethod
    def _gram_svd(mat, full_matrices=False, compute_uv=True, dtype=None):
        """S and V^H of a tall matrix from the eigendecomposition of A^H A, formed in `dtype` (complex128 by default).

        EspiritCalib only uses S and V^H. Maps agree with the direct SVD to about 1e-4 (complex128) or 1e-3 (complex64).
        """
        import cupy as cp
        g = mat.astype(dtype or cp.complex128, copy=False)
        gram = g.conj().T @ g
        w, v = cp.linalg.eigh(gram)
        order = cp.argsort(w)[::-1]
        s = cp.sqrt(cp.maximum(w[order], 0)).astype(cp.float32)
        vh = v[:, order].conj().T.astype(mat.dtype)
        return (None, s, vh) if compute_uv else s

    def _run_espirit_calibration_gpu(self, kspace_block, calib_width, kernel_width, sp_device):
        import cupy as cp
        import sigpy.mri as spmri
        from contextlib import contextmanager

        m, n = self._calibration_matrix_shape(kspace_block, calib_width, kernel_width)
        use_gram = False
        if not self._gesvd_workspace_fits(cp, m, n):
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            if 3 * n * n * 16 <= free_bytes:
                gram_dtype = cp.complex128
            elif 3 * n * n * 8 <= free_bytes:
                gram_dtype = cp.complex64
            else:
                raise RuntimeError(
                    f"cuSOLVER cannot factor the {m} x {n} ESPIRiT calibration matrix and its {n} x {n} Gram matrix "
                    f"does not fit in GPU memory; reduce espirit_calibration_width / espirit_kernel_width or the coil count.")
            print(f"[coil sensitivity] the {m} x {n} ESPIRiT calibration matrix exceeds the cuSOLVER SVD limit; "
                  f"using the Gram-matrix SVD on the GPU ({'double' if gram_dtype == cp.complex128 else 'single'} precision).", flush=True)
            use_gram = True

        @contextmanager
        def gram_svd_in_cupy():
            original = cp.linalg.svd
            cp.linalg.svd = lambda mat, full_matrices=False, compute_uv=True: self._gram_svd(mat, full_matrices, compute_uv, gram_dtype)
            try:
                yield
            finally:
                cp.linalg.svd = original

        kspace_cp = cp.asarray(kspace_block.contiguous(), dtype=cp.complex64)
        with gram_svd_in_cupy() if use_gram else nullcontext():
            maps_cp = spmri.app.EspiritCalib(
                kspace_cp,
                calib_width=calib_width,
                kernel_width=kernel_width,
                max_iter=self.params.espirit_max_iter,
                device=sp_device,
            ).run()
        maps_cp = maps_cp.astype(cp.complex64, copy=False)
        maps_cp = cp.ascontiguousarray(maps_cp)
        maps_t = torch.view_as_real(torch.utils.dlpack.from_dlpack(maps_cp))
        return torch.complex(maps_t[..., 0], maps_t[..., 1]).to(torch.complex128)

    def _run_espirit_calibration(self, kspace_block, calib_width, kernel_width):
        sp_device = self._resolved_espirit_device(kspace_block)
        use_gpu = sp_device.id >= 0

        if not use_gpu:
            return self._run_espirit_calibration_cpu(
                kspace_block, calib_width, kernel_width, sp_device=sp_device,
            )

        import cupy as cp
        from cupy_backends.cuda.libs import cusolver

        try:
            return self._run_espirit_calibration_gpu(
                kspace_block, calib_width, kernel_width, sp_device,
            )
        except cp.cuda.memory.OutOfMemoryError:
            import sigpy as sp

            print("[coil sensitivity] GPU memory exhausted; running ESPIRiT on CPU.", flush=True)
        except (cusolver.CUSOLVERError, RuntimeError) as error:
            # Upstream only caught out-of-memory, so a cuSOLVER failure (e.g. the 2^31 workspace
            # limit with 7+ coils at the shipped 48/12 widths in 3D) crashed the preparation.
            m, n = self._calibration_matrix_shape(kspace_block, calib_width, kernel_width)
            print(f"[coil sensitivity] GPU ESPIRiT failed on the {m} x {n} calibration matrix ({error}); "
                  "running ESPIRiT on CPU. This can take hours in 3D: consider smaller espirit widths.", flush=True)
        finally:
            self._cupy_cleanup(cp)
        self._cupy_cleanup(cp)
        return self._run_espirit_calibration_cpu(
            kspace_block, calib_width, kernel_width, sp_device=sp.Device(-1),
        )

    def calculate_espirit(self, kspace, reference_kspace=None):
        ncha, _, nx, ny, nz = kspace.shape
        kspace_calib = self._calibration_kspace(kspace, reference_kspace=reference_kspace)

        calib_width_eff = self.params.espirit_calibration_width
        kernel_width_eff = self.params.espirit_kernel_width
        spatial_limit = min(nx, ny, nz) if nz > 1 else min(nx, ny)
        if calib_width_eff > spatial_limit:
            raise ValueError("espirit_calibration_width exceeds the smallest encoded spatial dimension.")
        if kernel_width_eff > calib_width_eff:
            raise ValueError("espirit_kernel_width cannot exceed espirit_calibration_width.")

        if nz > 1:
            return self._run_espirit_calibration(kspace_calib, calib_width_eff, kernel_width_eff)

        espirit_maps = torch.zeros((ncha, nx, ny, nz), dtype=torch.complex128, device=kspace.device)
        for z in range(nz):
            espirit_maps[:, :, :, z] = self._run_espirit_calibration(kspace_calib[:, :, :, z], calib_width_eff, kernel_width_eff)
        return espirit_maps

    def calculate_odille_spline(self, kspace, reference_kspace=None):
        del reference_kspace
        kspace_calib = self._average_repetitions(kspace).to(torch.complex64)
        coil_images = ifftnc(kspace_calib, dims=(-3, -2, -1))

        surface_abs = torch.abs(coil_images)
        if not torch.isfinite(surface_abs).all() or surface_abs.max() <= 0:
            raise ValueError("Odille-spline coil images must be finite with positive magnitude; check imaging k-space.")
        reference_abs = torch.sqrt(torch.sum(surface_abs ** 2, dim=0).clamp_min(0.0))

        # Stabilize phase estimation in low-signal regions by flooring coil
        # magnitudes at 5% of the global surface-coil maximum.
        threshold = 0.05 * torch.max(surface_abs)
        nonzero_surface_abs = torch.clamp(surface_abs, min=threshold.item())
        surface_phase = coil_images / nonzero_surface_abs

        lambda_magnitude = float(self.params.spline_magnitude_smoothing)
        lambda_phase = float(self.params.spline_phase_smoothing)

        if lambda_magnitude > 0:
            surface_abs = spline_fft_solver(
                surface_abs.permute(0, 3, 1, 2).to(torch.complex128),
                lambda_magnitude,
            ).abs().permute(0, 2, 3, 1)
            reference_abs = spline_fft_solver(
                reference_abs.permute(2, 0, 1).to(torch.complex128),
                lambda_magnitude,
            ).abs().permute(1, 2, 0)

        if lambda_phase > 0:
            surface_phase = spline_fft_solver(
                surface_phase.permute(0, 3, 1, 2),
                lambda_phase,
            ).permute(0, 2, 3, 1)

        phase_factor = torch.exp(1j * torch.angle(surface_phase))
        eps = float(self.params.coil_sensitivity_eps)
        # Preserve the smoothed reference magnitude wherever it is positive;
        # replace only exact zeros to avoid introducing an intensity-dependent
        # floor into the sensitivity maps.
        reference_denominator = torch.where(
            reference_abs > 0,
            reference_abs,
            torch.full_like(reference_abs, eps),
        )
        smaps = surface_abs * phase_factor / reference_denominator.unsqueeze(0)
        smaps_max = torch.max(torch.abs(smaps))
        if float(smaps_max.real.item()) <= 0.0:
            raise ValueError("GRICS sensitivity maps have non-positive maximum; cannot normalize.")
        return (smaps / smaps_max).to(torch.complex128)

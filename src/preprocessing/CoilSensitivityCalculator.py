import numpy as np
import sigpy as sp
import sigpy.mri as spmri
import torch

from src.utils.fftnc import fftnc, ifftnc


class CoilSensitivityCalculator:
    """Calculate coil sensitivity maps from calibration k-space."""

    VALID_METHODS = {"espirit", "odille-spline"}

    def __init__(self, params, sp_device=None):
        self.params = params
        self.sp_device = sp_device

    def calculate(self, kspace, reference_kspace=None):
        method = str(getattr(self.params, "coil_sensitivity_method", "espirit")).strip().lower()
        if method not in self.VALID_METHODS:
            raise ValueError(
                "coil_sensitivity_method must be one of "
                f"{sorted(self.VALID_METHODS)}."
            )

        if method == "espirit":
            return self.calculate_espirit(kspace, reference_kspace=reference_kspace)
        return self.calculate_iadi_spline(kspace, reference_kspace=reference_kspace)

    @staticmethod
    def _average_repetitions(kspace):
        if kspace.shape[1] == 1:
            return kspace[:, 0]
        return torch.mean(kspace, dim=1)

    def _central_ky_calibration_kspace(self, kspace):
        kspace_calib = self._average_repetitions(kspace)
        nky = int(kspace_calib.shape[2])
        n_lines = int(
            getattr(
                self.params,
                "coil_sensitivity_calibration_lines",
                getattr(self.params, "acs", nky),
            )
        )
        n_lines = max(1, min(n_lines, nky))
        start = (nky - n_lines) // 2
        stop = start + n_lines

        central = torch.zeros_like(kspace_calib)
        central[:, :, start:stop, :] = kspace_calib[:, :, start:stop, :]
        return central

    def _calibration_kspace(self, kspace, reference_kspace=None, *, use_central_lines=False):
        if reference_kspace is not None:
            return self._average_repetitions(reference_kspace)
        if use_central_lines:
            return self._central_ky_calibration_kspace(kspace)
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
        return sp.Device(0 if kspace.device.type == "cuda" else -1)

    def _run_espirit_calibration_cpu(self, kspace_block, calib_width, kernel_width, sp_device=None):
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

    def _run_espirit_calibration_gpu(self, kspace_block, calib_width, kernel_width, sp_device):
        import cupy as cp

        kspace_cp = cp.asarray(kspace_block.contiguous(), dtype=cp.complex64)
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
        use_gpu = kspace_block.device.type == "cuda"
        sp_device = self._resolved_espirit_device(kspace_block)

        if not use_gpu:
            return self._run_espirit_calibration_cpu(
                kspace_block, calib_width, kernel_width, sp_device=sp_device,
            )

        import cupy as cp

        try:
            return self._run_espirit_calibration_gpu(
                kspace_block, calib_width, kernel_width, sp_device,
            )
        except cp.cuda.memory.OutOfMemoryError:
            self._cupy_cleanup(cp)
            return self._run_espirit_calibration_cpu(
                kspace_block, calib_width, kernel_width, sp_device=sp.Device(-1),
            )
        finally:
            self._cupy_cleanup(cp)

    def calculate_espirit(self, kspace, reference_kspace=None):
        ncha, _, nx, ny, nz = kspace.shape
        kspace_calib = self._calibration_kspace(kspace, reference_kspace=reference_kspace)

        if nz > 1:
            calib_width_eff = max(1, min(int(self.params.acs), nx, ny, nz))
        else:
            calib_width_eff = max(1, min(int(self.params.acs), nx, ny))
        kernel_width_eff = max(1, min(int(self.params.kernel_width), calib_width_eff))

        if nz > 1:
            return self._run_espirit_calibration(
                kspace_calib,
                calib_width_eff,
                kernel_width_eff,
            )

        espirit_maps = torch.zeros((ncha, nx, ny, nz), dtype=torch.complex128, device=kspace.device)
        for z in range(nz):
            espirit_maps[:, :, :, z] = self._run_espirit_calibration(
                kspace_calib[:, :, :, z],
                calib_width_eff,
                kernel_width_eff,
            )
        return espirit_maps

    @staticmethod
    def _spline_weight(nx, ny, device, dtype):
        if nx < 3 or ny < 3:
            return torch.zeros((nx, ny), dtype=dtype, device=device)

        ghg_x = torch.zeros(nx, dtype=dtype, device=device)
        ghg_x[nx // 2 - 1] = -0.25
        ghg_x[nx // 2] = 0.5
        ghg_x[nx // 2 + 1] = -0.25

        ghg_y = torch.zeros(ny, dtype=dtype, device=device)
        mid_y = int(np.floor(ny / 2))
        ghg_y[mid_y - 1] = -0.25
        ghg_y[mid_y] = 0.5
        ghg_y[mid_y + 1] = -0.25

        f_x = fftnc(ghg_x, dims=(-1,)).real[:, None]
        f_y = fftnc(ghg_y, dims=(-1,)).real[None, :]
        return (f_x + f_y) * 16.0

    def _spline_fft_solver(self, y, regularization_weight):
        if regularization_weight <= 0:
            return y

        nx, ny = y.shape[-2:]
        weight = self._spline_weight(nx, ny, y.device, y.real.dtype)
        denominator = 1.0 + float(regularization_weight) * weight
        y_fft = fftnc(y, dims=(-2, -1))
        smoothed = ifftnc(y_fft / denominator, dims=(-2, -1))
        return smoothed

    def calculate_iadi_spline(self, kspace, reference_kspace=None):
        kspace_calib = self._calibration_kspace(
            kspace,
            reference_kspace=reference_kspace,
            use_central_lines=True,
        )
        coil_images = ifftnc(kspace_calib, dims=(-3, -2, -1)).to(torch.complex128)

        surface_abs = torch.abs(coil_images)
        reference_abs = torch.sqrt(torch.sum(surface_abs ** 2, dim=0).clamp_min(0.0))

        threshold = 0.01 * torch.max(surface_abs)
        nonzero_surface_abs = torch.clamp(surface_abs, min=threshold.item())
        surface_phase = coil_images / nonzero_surface_abs

        lambda_magnitude = float(getattr(self.params, "spline_magnitude_smoothing", 1000.0))
        lambda_phase = float(getattr(self.params, "spline_phase_smoothing", 1000.0))

        if lambda_magnitude > 0:
            surface_abs = self._spline_fft_solver(
                surface_abs.permute(0, 3, 1, 2).to(torch.complex128),
                lambda_magnitude,
            ).abs().permute(0, 2, 3, 1)
            reference_abs = self._spline_fft_solver(
                reference_abs.permute(2, 0, 1).to(torch.complex128),
                lambda_magnitude,
            ).abs().permute(1, 2, 0)

        reference_threshold = 0.01 * torch.max(torch.abs(reference_abs))
        reference_abs = torch.clamp(reference_abs, min=reference_threshold.item())

        reference_sum = torch.sum(torch.abs(reference_abs))
        if float(reference_sum.real.item()) <= 0.0:
            raise ValueError("GRICS reference image has non-positive sum; cannot normalize.")
        self.grics_reference_image = reference_abs * (
            float(reference_abs.numel()) / float(reference_sum.real.item())
        )

        if lambda_phase > 0:
            surface_phase = self._spline_fft_solver(
                surface_phase.permute(0, 3, 1, 2),
                lambda_phase,
            ).permute(0, 2, 3, 1)

        phase_factor = torch.exp(1j * torch.angle(surface_phase))
        eps = float(getattr(self.params, "coil_sensitivity_eps", 1.0e-8))
        smaps = surface_abs * phase_factor / reference_abs.clamp_min(eps).unsqueeze(0)
        smaps_max = torch.max(torch.abs(smaps))
        if float(smaps_max.real.item()) <= 0.0:
            raise ValueError("GRICS sensitivity maps have non-positive maximum; cannot normalize.")
        return smaps / smaps_max

import torch

from src.preprocessing.utils.spline import spline_fft_solver
from src.utils.fftnc import ifftnc


class CalibrationImageCalculator:
    """Build a mean-one magnitude image independently of coil sensitivity maps.

    Uses repetition-averaged imaging k-space before simulated motion is applied.
    Smoothing acts within each x/y plane, including for 3D volumes.
    """

    def __init__(self, params):
        self.params = params

    def calculate(self, kspace):
        """Convert [coil, repetition, x, y, z] k-space to a real [x, y, z] image."""
        kspace_calib = torch.mean(kspace, dim=1).to(torch.complex64)
        coil_images = ifftnc(kspace_calib, dims=(-3, -2, -1))
        reference_abs = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=0).clamp_min(0.0))
        reference_abs = spline_fft_solver(
            reference_abs.permute(2, 0, 1).to(torch.complex128), self.params.calibration_image_smoothing,
        ).abs().permute(1, 2, 0)
        reference_sum = reference_abs.sum()
        if not torch.isfinite(reference_abs).all() or reference_sum <= 0:
            raise ValueError("Calibration image must be finite with positive sum; check imaging k-space.")
        return reference_abs * (reference_abs.numel() / reference_sum)

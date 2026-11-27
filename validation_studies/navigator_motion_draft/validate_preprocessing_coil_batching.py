"""Compare grouped preprocessing against full-coil formulas on CPU and CUDA."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.CoilSensitivityCalculator import CoilSensitivityCalculator
from src.preprocessing.CalibrationImageCalculator import CalibrationImageCalculator
from src.preprocessing.DataPreparer import DataPreparer
from src.reconstruction.coil_batching import CoilBatchPlanner
from src.preprocessing.utils.spline import spline_fft_solver
from src.utils.fftnc import ifftnc


def reference_maps(kspace, magnitude, phase):
    images = ifftnc(kspace.mean(1).to(torch.complex64), dims=(-3, -2, -1))
    surface = images.abs()
    reference = surface.square().sum(0).sqrt()
    phases = images / surface.clamp_min(.05 * surface.max())
    if magnitude:
        surface = spline_fft_solver(surface.permute(0, 3, 1, 2).to(torch.complex128), magnitude).abs().permute(0, 2, 3, 1)
        reference = spline_fft_solver(reference.permute(2, 0, 1).to(torch.complex128), magnitude).abs().permute(1, 2, 0)
    if phase:
        phases = spline_fft_solver(phases.permute(0, 3, 1, 2), phase).permute(0, 2, 3, 1)
    denominator = torch.where(reference > 0, reference, torch.full_like(reference, 1e-8))
    maps = surface * torch.exp(1j * torch.angle(phases)) / denominator
    return (maps / maps.abs().max()).to(torch.complex128)


def validate(device):
    for nz in (1, 4):
        for repetitions in (1, 2):
            for magnitude, phase in ((0., 0.), (1000., 0.), (0., 1000.), (1000., 1000.)):
                kspace = torch.randn(5, repetitions, 12, 10, nz, device=device, dtype=torch.complex128)
                params = SimpleNamespace(spline_magnitude_smoothing=magnitude, spline_phase_smoothing=phase, coil_sensitivity_eps=1e-8,
                                         calibration_image_smoothing=1000., data_dimension='2D' if nz == 1 else '3D')
                expected = reference_maps(kspace, magnitude, phase)
                unit = kspace[0, 0].numel() * 16 * CoilBatchPlanner.workspace_factor
                with patch.object(CoilBatchPlanner, '_memory_budget', return_value=2 * unit):
                    actual = CoilSensitivityCalculator(params).calculate_odille_spline(kspace)
                    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
                    actual_calibration = CalibrationImageCalculator(params).calculate(kspace)
                    data = SimpleNamespace(kspace=kspace, smaps=actual, params=params, runtime=SimpleNamespace(t_device=device))
                    DataPreparer._compute_ifft_image_from_input_kspace(data)
                reference = ifftnc(kspace.mean(1).to(torch.complex64), dims=(-3, -2, -1)).abs().square().sum(0).sqrt()
                reference = spline_fft_solver(reference.permute(2, 0, 1).to(torch.complex128), 1000.).abs().permute(1, 2, 0)
                reference *= reference.numel() / reference.sum()
                torch.testing.assert_close(actual_calibration, reference, atol=1e-6, rtol=1e-6)
                combined = (ifftnc(kspace.mean(1), dims=(-3, -2, -1)) * actual.conj()).sum(0)
                if nz == 1:
                    combined = combined.squeeze(-1)
                torch.testing.assert_close(data.input_ifft_image, combined, atol=1e-12, rtol=1e-12)
    print(f'PASS: {device}: 16 cases; Odille maps, calibration image, and coil combination match full-coil formulas.')


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.manual_seed(17)
    validate('cpu')
    if torch.cuda.is_available():
        validate('cuda')
    else:
        print('CUDA validation skipped: no GPU available.')

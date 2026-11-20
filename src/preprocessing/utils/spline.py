"""Periodic in-plane spline smoothing shared by preprocessing calculators."""
import torch

from src.utils.fftnc import fftnc, ifftnc


def _spline_weight(nx, ny, device, dtype):
    if nx < 3 or ny < 3:
        return torch.zeros((nx, ny), dtype=dtype, device=device)

    ghg_x = torch.zeros(nx, dtype=dtype, device=device)
    ghg_x[nx // 2 - 1] = -0.25
    ghg_x[nx // 2] = 0.5
    ghg_x[nx // 2 + 1] = -0.25

    ghg_y = torch.zeros(ny, dtype=dtype, device=device)
    mid_y = ny // 2
    ghg_y[mid_y - 1] = -0.25
    ghg_y[mid_y] = 0.5
    ghg_y[mid_y + 1] = -0.25

    f_x = fftnc(ghg_x, dims=(-1,)).real[:, None]
    f_y = fftnc(ghg_y, dims=(-1,)).real[None, :]
    return (f_x + f_y) * 16.0


def spline_fft_solver(y, regularization_weight):
    """Smooth the last two axes with a periodic Fourier-domain spline filter."""
    if regularization_weight <= 0:
        return y

    nx, ny = y.shape[-2:]
    weight = _spline_weight(nx, ny, y.device, y.real.dtype)
    denominator = 1.0 + float(regularization_weight) * weight
    y_fft = fftnc(y, dims=(-2, -1))
    smoothed = ifftnc(y_fft / denominator, dims=(-2, -1))
    return smoothed


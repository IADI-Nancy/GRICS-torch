import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver


class _DummyOp:
    def __init__(self, device):
        self.device = device

    def normal(self, x):
        return x


def _make_solver(nx, ny, device):
    return ConjugateGradientSolver(
        _DummyOp(device=device),
        reg_lambda=0.0,
        regularizer="Tikhonov_laplacian",
        regularization_shape=(2, nx, ny),
        verbose=False,
    )


def test_laplacian_zero_for_linear_field_including_boundaries():
    device = torch.device("cpu")
    nx, ny = 16, 19
    solver = _make_solver(nx, ny, device)

    x = torch.arange(nx, dtype=torch.float64, device=device).view(nx, 1)
    y = torch.arange(ny, dtype=torch.float64, device=device).view(1, ny)

    # Two channels with affine fields.
    f0 = 1.7 * x + 0.3 * y + 2.0
    f1 = -0.8 * x + 1.4 * y - 5.0
    field = torch.stack([f0, f1], dim=0)

    out = solver.laplacian_op(field.reshape(-1)).reshape(2, nx, ny)

    max_abs = torch.max(torch.abs(out)).item()
    print(f"max |L(affine)| = {max_abs:.6e}")
    assert max_abs < 1e-5

    # Explicit boundary-only check.
    boundary_mask = torch.zeros((nx, ny), dtype=torch.bool, device=device)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True

    boundary_vals = out[:, boundary_mask]
    max_abs_boundary = torch.max(torch.abs(boundary_vals)).item()
    print(f"max boundary |L(affine)| = {max_abs_boundary:.6e}")
    assert max_abs_boundary < 1e-5


if __name__ == "__main__":
    test_laplacian_zero_for_linear_field_including_boundaries()

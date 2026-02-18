import os
import sys
from pathlib import Path

import torch
import pytest

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver


class _DiagNormalOp:
    def __init__(self, diag, device):
        self.diag = diag
        self.device = device

    def normal(self, x):
        return self.diag * x


class _IdentityNormalOp:
    def __init__(self, device):
        self.device = device

    def normal(self, x):
        return x


def _diag_case_metrics():
    torch.manual_seed(0)
    device = torch.device("cpu")

    n = 64
    diag = torch.linspace(0.5, 3.0, n, device=device, dtype=torch.float64)
    b = torch.randn(n, device=device, dtype=torch.float64)
    scale = 19.0
    lam = 0.7

    solver = ConjugateGradientSolver(
        _DiagNormalOp(diag=diag, device=device),
        reg_lambda=lam,
        regularizer="Tikhonov",
        early_stopping=False,
    )

    x1 = solver.solve_cg(b, x0=torch.zeros_like(b), max_iter=200, tol=1e-12)
    x2 = solver.solve_cg(scale * b, x0=torch.zeros_like(b), max_iter=200, tol=1e-12)

    rel_scale_err = torch.linalg.norm(x2 - scale * x1) / (torch.linalg.norm(scale * x1) + 1e-12)
    x_ref = b / (diag + lam)
    rel_ref_err = torch.linalg.norm(x1 - x_ref) / (torch.linalg.norm(x_ref) + 1e-12)
    return rel_scale_err.item(), rel_ref_err.item()


def _spatial_case_metric(regularizer):
    torch.manual_seed(1)
    device = torch.device("cpu")

    shape = (2, 12, 10)
    n = shape[0] * shape[1] * shape[2]
    b = torch.randn(n, device=device, dtype=torch.float64)
    scale = 11.0

    solver = ConjugateGradientSolver(
        _IdentityNormalOp(device=device),
        reg_lambda=0.4,
        regularizer=regularizer,
        regularization_shape=shape,
        early_stopping=False,
    )

    x1 = solver.solve_cg(b, x0=torch.zeros_like(b), max_iter=300, tol=1e-10)
    x2 = solver.solve_cg(scale * b, x0=torch.zeros_like(b), max_iter=300, tol=1e-10)

    rel_scale_err = torch.linalg.norm(x2 - scale * x1) / (torch.linalg.norm(scale * x1) + 1e-12)
    return rel_scale_err.item()


def test_cg_solution_scales_with_rhs_for_fixed_lambda_tikhonov():
    rel_scale_err, rel_ref_err = _diag_case_metrics()
    assert rel_scale_err < 1e-8

    # Also validate against the closed-form diagonal solve.
    assert rel_ref_err < 1e-8


@pytest.mark.parametrize("regularizer", ["Tikhonov_gradient", "Tikhonov_laplacian"])
def test_cg_rhs_scale_invariance_for_spatial_regularizers(regularizer):
    rel_scale_err = _spatial_case_metric(regularizer)
    assert rel_scale_err < 2e-7


def run_direct():
    rel_scale_err, rel_ref_err = _diag_case_metrics()
    print(f"[diag+tikhonov] rel_scale_err={rel_scale_err:.3e}, rel_ref_err={rel_ref_err:.3e}")
    assert rel_scale_err < 1e-8
    assert rel_ref_err < 1e-8

    for regularizer in ("Tikhonov_gradient", "Tikhonov_laplacian"):
        rel_scale_err = _spatial_case_metric(regularizer)
        print(f"[{regularizer}] rel_scale_err={rel_scale_err:.3e}")
        assert rel_scale_err < 2e-7

    print("All direct-run checks passed.")


if __name__ == "__main__":
    run_direct()

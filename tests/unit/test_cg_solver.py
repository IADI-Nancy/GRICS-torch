"""Conjugate-gradient solver: scale invariance of the guards and the zero-update trap."""
import torch

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver


class DenseOp:
    def __init__(self, A):
        self.A = A
        self.device = A.device

    def normal(self, x):
        return self.A @ x


def _solver(op, interval, stagnation=False):
    return ConjugateGradientSolver(op, reg_lambda=0.0, regularizer="Tikhonov", regularization_shape=None,
                                   regularization_spatial_dims=None, verbose=False, stop_on_stagnation=stagnation,
                                   true_residual_interval=interval, stagnation_consecutive_steps=6,
                                   stagnation_countdown_steps=3, use_reg_scale_proxy=False, reg_scale_num_probes=None)


def _spd(seed, n=6, spread=4):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    A = (Q @ torch.diag(torch.logspace(0, spread, n, dtype=torch.float64)) @ Q.T).to(torch.complex128)
    b = torch.randn(n, generator=g, dtype=torch.float64).to(torch.complex128)
    return A, b


def test_solution_does_not_depend_on_the_data_scale():
    # with absolute 1e-15 guards the 1e-9 system stopped at the first iteration with x = 0
    A, b = _spd(0, n=40, spread=2)
    exact = torch.linalg.solve(A, b)
    for scale in (1e-9, 1.0, 1e9):
        s = _solver(DenseOp(A), 10)
        x = s.cg(scale * b, max_iter=200, tol=1e-10) / scale
        assert s.last_info["stop_reason"] == "tolerance", (scale, s.last_info["stop_reason"])
        assert ((x - exact).norm() / exact.norm()).item() < 1e-7, scale


def _trap_system():
    # a system where the third CG iterate has a worse true residual than the start
    for seed in range(2000):
        A, b = _spd(seed)
        relres = []
        for k in (1, 2, 3):
            s = _solver(DenseOp(A), 10)
            s.cg(b, max_iter=k, tol=1e-8)
            relres.append(s.last_info["final_relres"])
        if relres[2] > 1.0 and min(relres[:2]) < 0.5:
            return A, b, relres
    raise RuntimeError("no trap system found")


def test_best_intermediate_iterate_is_returned_instead_of_x0():
    A, b, relres = _trap_system()
    s = _solver(DenseOp(A), 10)
    x = s.cg(b, max_iter=3, tol=1e-8)
    info = s.last_info
    assert info["selected_iteration"] in (1, 2)
    assert x.norm().item() > 0
    assert info["relres"] <= min(relres[:2]) + 1e-12


def test_converged_solution_is_unchanged():
    A, b = _spd(1, n=6)
    s = _solver(DenseOp(A), 10)
    x = s.cg(b, max_iter=50, tol=1e-9)
    assert ((A @ x - b).norm() / b.norm()).item() < 1e-9
    assert s.last_info["flag"] == 0
    assert s.last_info["selected_iteration"] == s.last_info["iterations"]

"""Focused tests for GRICS++-style motion PCG and per-level scheduling."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import torch

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.JointReconstructor import JointReconstructor
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator


class _DiagonalOperator:
    device = torch.device("cpu")

    def __init__(self, values):
        self.values = values

    def normal(self, x):
        return self.values * x


def _solver(operator, preconditioner=None, *, shape=None):
    return ConjugateGradientSolver(
        operator, reg_lambda=0.0, regularizer="Tikhonov_gradient" if shape else "Tikhonov",
        regularization_shape=shape, regularization_spatial_dims=(1, 2) if shape else None,
        verbose=False, stop_on_stagnation=False, true_residual_interval=1,
        stagnation_consecutive_steps=6, stagnation_countdown_steps=0,
        use_reg_scale_proxy=False, reg_scale_num_probes=None, preconditioner=preconditioner)


class MotionPreconditionerTests(unittest.TestCase):
    def test_pcg_uses_preconditioned_residual(self):
        diagonal = torch.logspace(0, 4, 32, dtype=torch.float64)
        rhs = torch.ones_like(diagonal)
        solver = _solver(_DiagonalOperator(diagonal), lambda r: r / diagonal)
        solution = solver.cg(rhs, max_iter=1, tol=1e-12)
        torch.testing.assert_close(solution, rhs / diagonal, rtol=1e-12, atol=1e-12)
        self.assertEqual(solver.last_info["flag"], 0)

    def test_regularization_diagonal_matches_operator(self):
        shape = (2, 3, 4, 1)
        solver = _solver(_DiagonalOperator(torch.ones(24)), shape=shape)
        diagonal = solver._gradient_diagonal(dtype=torch.float64, device=torch.device("cpu"))
        for index in (0, 1, 5, 12, 23):
            basis = torch.zeros(24, dtype=torch.float64)
            basis[index] = 1
            self.assertAlmostEqual(diagonal[index].item(), solver._gradient_op(basis)[index].item())

    def test_local_motion_diagonal_uses_gradients_and_signal(self):
        nx, ny = 4, 5
        x = torch.arange(nx, dtype=torch.float64)[:, None]
        y = torch.arange(ny, dtype=torch.float64)[None, :]
        image = (x + 2 * y).to(torch.complex128).unsqueeze(0)
        motion = SimpleNamespace(
            alpha=torch.zeros(2, nx, ny, 1), motion_type="non-rigid",
            motion_signal=torch.tensor([[1.0], [2.0]]),
            _get_sparse_operator=lambda state: torch.eye(nx * ny, dtype=torch.complex128))
        simulator = MotionPerturbationSimulator(
            torch.ones(1, nx, ny, 1, dtype=torch.complex128), nx * ny,
            [[torch.arange(10), torch.arange(10, 20)]], 1, image, motion)
        diagonal = simulator.approximate_normal_diagonal().reshape(2, nx, ny, 1)
        torch.testing.assert_close(diagonal[0], torch.full((nx, ny, 1), 2.5, dtype=torch.float64))
        torch.testing.assert_close(diagonal[1], torch.full((nx, ny, 1), 10.0, dtype=torch.float64))


class _Logger:
    @contextmanager
    def iterations(self, level):
        yield range(self.count)

    def record_residual(self, value):
        pass

    def iteration_finished(self, *args):
        pass


class MotionScheduleTests(unittest.TestCase):
    def test_last_iteration_of_each_level_is_image_only(self):
        recon = JointReconstructor.__new__(JointReconstructor)
        recon.external_image_regularizer = None
        recon.params = SimpleNamespace(lambda_r=0.0, ResolutionLevels=[0.5, 1.0],
                                       image_only_last_iteration_per_level=True)
        recon._current_level_idx = 0
        recon._last_image_cg_info = None
        recon._last_motion_cg_info = None
        updates = []

        def iteration(data, **kwargs):
            updates.append(kwargs["update_motion"])
            image = torch.ones(1)
            motion = torch.zeros(1)
            return SimpleNamespace(image=image, motion_for_residual=motion,
                                   residual=torch.zeros(1), motion=motion)

        recon.gauss_newton_iteration = iteration
        logger = _Logger()
        logger.count = 4
        data = {"KspaceData": torch.ones(1)}
        recon._run_resolution_level(data, level_index=0, gauss_newton_iterations_at_level=4,
                                    level_count=2, update_final_motion=False,
                                    gn_early_stopping=False, logger=logger)
        self.assertEqual(updates, [True, True, True, False])

        updates.clear()
        logger.count = 1
        recon._run_resolution_level(data, level_index=1, gauss_newton_iterations_at_level=1,
                                    level_count=2, update_final_motion=False,
                                    gn_early_stopping=False, logger=logger)
        self.assertEqual(updates, [False])

        updates.clear()
        recon.params.image_only_last_iteration_per_level = False
        logger.count = 4
        recon._run_resolution_level(data, level_index=0, gauss_newton_iterations_at_level=4,
                                    level_count=2, update_final_motion=False,
                                    gn_early_stopping=False, logger=logger)
        self.assertEqual(updates, [True, True, True, True])


if __name__ == "__main__":
    unittest.main()

"""Focused tests for GRICS++-style motion PCG and per-level scheduling."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import torch

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.JointReconstructor import JointReconstructor
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.reconstruction.grics_cpp_motion_preconditioner import GricsCppPseudoGaussSeidelPreconditioner


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

    def test_cpp_factor_weight_is_square_root_of_normal_equation_weight(self):
        # MotionModelOptimizer uses mu = LambdaScaled**2 in J^H J + mu G^H G,
        # but passes LambdaScaled to createPreconditioner().
        normal_equation_weight = 0.81
        factor_weight = JointReconstructor._grics_cpp_preconditioner_lambda_scaled(
            normal_equation_weight)
        self.assertAlmostEqual(factor_weight, 0.9)
        self.assertAlmostEqual(factor_weight ** 2, normal_equation_weight)
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            JointReconstructor._grics_cpp_preconditioner_lambda_scaled(-1.0)

    def test_cpp_pseudo_gauss_seidel_matches_explicit_triangular_factors(self):
        torch.manual_seed(3)
        nx = ny = nz = 2
        count = 3
        local = torch.randn(count, count, nx, ny, nz, dtype=torch.complex64)
        # The C++ factor needs positive, nonzero local diagonal entries.
        for component in range(count):
            local[component, component] = local[component, component].abs() + 3
        preconditioner = GricsCppPseudoGaussSeidelPreconditioner(
            local, lambda_scaled=0.2, voxel_spacing=(1.0, 1.5, 2.0),
        )
        size = count * preconditioner.nvoxels
        lower = torch.zeros((size, size), dtype=torch.complex64)
        upper = torch.zeros_like(lower)
        nxy = nx * ny
        for component in range(count):
            for index in range(preconditioner.nvoxels):
                row = component * preconditioner.nvoxels + index
                diagonal = preconditioner.diagonal[component, index]
                lower[row, row] = diagonal
                upper[row, row] = diagonal.conj() / diagonal
                for previous in range(component):
                    lower[row, previous * preconditioner.nvoxels + index] = preconditioner.lower[component, previous][index]
                for following in range(component + 1, count):
                    upper[row, following * preconditioner.nvoxels + index] = (
                        preconditioner.lower[following, component][index].conj() / diagonal
                    )
                x, remainder = index % nx, index // nx
                y, z = remainder % ny, remainder // ny
                for offset, coordinate, extent, coefficient in (
                    (1, x, nx, preconditioner.off[0]),
                    (nx, y, ny, preconditioner.off[1]),
                    (nxy, z, nz, preconditioner.off[2]),
                ):
                    if coordinate > 0:
                        lower[row, row - offset] = coefficient
                    if coordinate < extent - 1:
                        upper[row, row + offset] = coefficient.conj() / diagonal
        residual = torch.randn(size, dtype=torch.complex64)
        cpp_residual = preconditioner._to_cpp_layout(residual)
        expected = torch.linalg.solve_triangular(
            upper, torch.linalg.solve_triangular(lower, cpp_residual.reshape(-1, 1), upper=False), upper=True,
        ).reshape(-1)
        torch.testing.assert_close(preconditioner(residual), preconditioner._from_cpp_layout(expected))

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
    @staticmethod
    def _reconstructor():
        recon = JointReconstructor.__new__(JointReconstructor)
        recon.external_image_regularizer = None
        recon.params = SimpleNamespace(
            lambda_r=0.0, ResolutionLevels=[0.5, 1.0],
            image_only_last_iteration_per_level=True)
        recon._current_level_idx = 0
        recon._last_image_cg_info = None
        recon._last_motion_cg_info = None
        return recon

    def test_last_iteration_of_each_level_is_image_only(self):
        recon = self._reconstructor()
        image_calls, motion_calls = [], []

        def iteration(data, **kwargs):
            image_calls.append(kwargs["update_motion"])
            image = torch.ones(1)
            motion = data.get("MotionModel", torch.zeros(1))
            return SimpleNamespace(
                image=image, motion_for_residual=motion, residual=torch.zeros(1),
                motion=motion, motion_update=None, motion_elapsed=None)

        def apply_motion(data, result):
            motion_calls.append(len(image_calls) - 1)
            data["MotionModel"] = result.motion_for_residual + 1
            result.motion = data["MotionModel"]
            return result

        recon.gauss_newton_iteration = iteration
        recon._apply_motion_update = apply_motion
        logger = _Logger()
        logger.count = 4
        data = {"KspaceData": torch.ones(1), "MotionModel": torch.zeros(1)}
        recon._run_resolution_level(
            data, level_index=0, gauss_newton_iterations_at_level=4,
            level_count=2, update_final_motion=False,
            gn_early_stopping=False, logger=logger)
        self.assertEqual(image_calls, [False, False, False, False])
        self.assertEqual(motion_calls, [0, 1, 2])

        image_calls.clear()
        motion_calls.clear()
        logger.count = 1
        recon._run_resolution_level(
            data, level_index=1, gauss_newton_iterations_at_level=1,
            level_count=2, update_final_motion=False,
            gn_early_stopping=False, logger=logger)
        self.assertEqual(image_calls, [False])
        self.assertEqual(motion_calls, [])

    def test_rejected_image_does_not_launch_motion_solve(self):
        recon = self._reconstructor()
        residuals = iter((0.2, 0.3))
        motion_calls = []

        def iteration(data, **kwargs):
            motion = data["MotionModel"]
            return SimpleNamespace(
                image=torch.ones(1), motion_for_residual=motion,
                residual=torch.tensor([next(residuals)]), motion=motion,
                motion_update=None, motion_elapsed=None)

        recon.gauss_newton_iteration = iteration
        recon._apply_motion_update = lambda data, result: (
            motion_calls.append(True) or result)
        logger = _Logger()
        logger.count = 4
        logger.iteration_stopped_early = lambda: None
        data = {"KspaceData": torch.ones(1), "MotionModel": torch.zeros(1)}
        recon._run_resolution_level(
            data, level_index=0, gauss_newton_iterations_at_level=4,
            level_count=2, update_final_motion=False,
            gn_early_stopping=True, logger=logger)
        self.assertEqual(motion_calls, [True])


if __name__ == "__main__":
    unittest.main()

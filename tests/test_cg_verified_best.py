"""Regression checks for uniform verified-residual CG selection."""
from types import SimpleNamespace
import unittest

import torch

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver


class VerifiedBestTests(unittest.TestCase):
    def _solver(self, diagonal, preconditioner=None):
        operator = SimpleNamespace(device=diagonal.device, normal=lambda x: diagonal * x)
        return ConjugateGradientSolver(
            operator, reg_lambda=0., regularizer='Tikhonov', regularization_shape=None,
            regularization_spatial_dims=None, verbose=False, stop_on_stagnation=False,
            true_residual_interval=10, stagnation_consecutive_steps=6,
            stagnation_countdown_steps=6, use_reg_scale_proxy=False,
            reg_scale_num_probes=None, preconditioner=preconditioner)

    def _check_residual(self, solver, diagonal, b, x):
        norm = torch.linalg.vector_norm(b - diagonal * x).item()
        self.assertAlmostEqual(solver.last_info['residual_norm'], norm)
        self.assertAlmostEqual(solver.last_info['relres'], norm / (b.norm().item() + 1e-12))

    def test_returns_initial_guess_when_final_residual_is_worse(self):
        diagonal = torch.tensor([1., 10.], dtype=torch.float64)
        b = torch.tensor([1., .5], dtype=torch.float64)
        for preconditioner in (None, lambda r: r / r.new_tensor([1., 2.])):
            with self.subTest(preconditioned=preconditioner is not None):
                solver = self._solver(diagonal, preconditioner)
                x = solver.cg(b, max_iter=1, tol=0.)
                torch.testing.assert_close(x, torch.zeros_like(b))
                self.assertEqual(solver.last_info['selected_iteration'], 0)
                self.assertEqual(solver.last_info['flag'], 1)
                self.assertGreater(solver.last_info['final_relres'], solver.last_info['relres'])
                self._check_residual(solver, diagonal, b, x)

    def test_verifies_final_candidate_between_refreshes(self):
        diagonal = torch.tensor([1., 2.], dtype=torch.float64)
        b = torch.ones_like(diagonal)
        solver = self._solver(diagonal)
        x = solver.cg(b, max_iter=1, tol=0.)
        torch.testing.assert_close(x, b * (2. / 3.))
        self.assertEqual(solver.last_info['selected_iteration'], 1)
        self._check_residual(solver, diagonal, b, x)

    def test_returns_verified_converged_solution(self):
        diagonal = torch.tensor([1., 10.], dtype=torch.float64)
        b = torch.ones_like(diagonal)
        for preconditioner in (None, lambda r: r / diagonal):
            with self.subTest(preconditioned=preconditioner is not None):
                solver = self._solver(diagonal, preconditioner)
                x = solver.cg(b, max_iter=4, tol=1e-10)
                torch.testing.assert_close(x, b / diagonal)
                self.assertEqual(solver.last_info['flag'], 0)
                self.assertEqual(solver.last_info['selected_iteration'], solver.last_info['iterations'])
                self._check_residual(solver, diagonal, b, x)

    def test_returns_better_verified_intermediate_candidate(self):
        diagonal = torch.tensor([1., 2., 10., 100.], dtype=torch.float64)
        b = torch.tensor([-.15, .11, -.11, -1.4], dtype=torch.float64)
        solver = self._solver(diagonal)
        solver.true_residual_interval = 1
        x = solver.cg(b, max_iter=2, tol=0.)
        expected = b * (torch.dot(b, b) / torch.dot(b, diagonal * b))
        torch.testing.assert_close(x, expected)
        self.assertEqual(solver.last_info['selected_iteration'], 1)
        self.assertEqual(solver.last_info['iterations'], 2)
        self.assertGreater(solver.last_info['final_relres'], solver.last_info['relres'])
        self._check_residual(solver, diagonal, b, x)

    def test_initial_convergence_and_zero_iterations(self):
        diagonal = torch.tensor([1., 2.], dtype=torch.float64)
        for b, x0, max_iter in ((torch.zeros_like(diagonal), None, 5),
                                (diagonal, torch.ones_like(diagonal), 5),
                                (diagonal, None, 0)):
            with self.subTest(max_iter=max_iter, x0=x0):
                solver = self._solver(diagonal)
                x = solver.cg(b, x0=x0, max_iter=max_iter)
                self.assertEqual(solver.last_info['iterations'], 0)
                self.assertEqual(solver.last_info['selected_iteration'], 0)
                self._check_residual(solver, diagonal, b, x)

    def test_selected_solution_preserves_gradients(self):
        diagonal = torch.tensor([1., 2.], dtype=torch.float64)
        b = torch.ones_like(diagonal, requires_grad=True)
        x = self._solver(diagonal).cg(b, max_iter=1, tol=0., differentiable=True)
        x.sum().backward()
        self.assertTrue(torch.isfinite(b.grad).all())
        self.assertGreater(b.grad.norm().item(), 0.)


if __name__ == '__main__':
    unittest.main()

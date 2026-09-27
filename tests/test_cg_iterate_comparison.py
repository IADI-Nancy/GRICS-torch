"""Verify an opt-in audit of two candidates from the same CG solve."""
import unittest
import torch
from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver


class DiagonalEncoding:
    device = torch.device('cpu')

    def __init__(self):
        self.calls = 0

    def normal(self, x):
        self.calls += 1
        return x * x.new_tensor([1., 100.])


def solver():
    return ConjugateGradientSolver(
        DiagonalEncoding(), reg_lambda=0., regularizer='Tikhonov',
        regularization_shape=None, regularization_spatial_dims=None,
        verbose=False, stop_on_stagnation=False, true_residual_interval=10,
        stagnation_consecutive_steps=12, stagnation_countdown_steps=6,
        use_reg_scale_proxy=False, reg_scale_num_probes=None)


class IterateComparisonTests(unittest.TestCase):
    def test_audit_exposes_objective_improvement_despite_larger_residual(self):
        b = torch.tensor([1., .1], dtype=torch.float64)
        normal, audited = solver(), solver()
        expected = normal.cg(b, max_iter=1, tol=1e-12)
        actual = audited.cg(b, max_iter=1, tol=1e-12, compare_iterates=True)
        torch.testing.assert_close(actual, expected)
        info = audited.last_info['iterate_comparison']
        self.assertEqual(info['best_iteration'], 0)
        self.assertEqual(info['last_iteration'], 1)
        self.assertAlmostEqual(info['best']['true_relres'], 1.)
        self.assertAlmostEqual(info['last']['true_relres'], 4.95)
        self.assertAlmostEqual(info['best']['quadratic_objective'], 0.)
        self.assertAlmostEqual(info['last']['quadratic_objective'], -.255025)
        self.assertGreater(info['difference_norm'], 0.)
        self.assertEqual(audited.E.calls, normal.E.calls + 2)
        self.assertNotIn('iterate_comparison', normal.last_info)

    def test_matching_candidates_when_system_converges(self):
        s = solver()
        x = s.cg(torch.tensor([1., .1], dtype=torch.float64),
                 max_iter=2, tol=1e-12, compare_iterates=True)
        torch.testing.assert_close(x, torch.tensor([1., .001], dtype=torch.float64))
        info = s.last_info['iterate_comparison']
        self.assertEqual(info['best_iteration'], 2)
        self.assertEqual(info['difference_norm'], 0.)
        self.assertEqual(info['best'], info['last'])


if __name__ == '__main__':
    unittest.main()

"""Regression checks for the physical spacing of motion-field regularization."""
import unittest

import torch

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver


class _Encoding:
    device = torch.device("cpu")


def _solver(spacing, dimensions=(1, 2, 3)):
    return ConjugateGradientSolver(
        _Encoding(), reg_lambda=1.0, regularizer="Tikhonov_gradient",
        regularization_shape=(1, 4, 5, 3, 1), regularization_spatial_dims=dimensions,
        regularization_spacing=spacing, verbose=False, stop_on_stagnation=False,
        true_residual_interval=1, stagnation_consecutive_steps=1,
        stagnation_countdown_steps=1, use_reg_scale_proxy=False, reg_scale_num_probes=None,
    )


class MotionRegularizationSpacingTests(unittest.TestCase):
    def test_gradient_normal_operator_scales_with_physical_spacing_squared(self):
        field = torch.arange(60, dtype=torch.float64)
        unit = _solver((1.0, 1.0, 1.0))._gradient_op(field)
        four_mm = _solver((4.0, 4.0, 4.0))._gradient_op(field)

        # GRICS++ differentiates once in the gradient and once in its adjoint.
        # Therefore G^H G at 4 mm is 1 / 4^2 of the unit-spacing operator.
        torch.testing.assert_close(four_mm, unit / 16.0)

    def test_anisotropic_spacing_weights_each_direction_independently(self):
        field = torch.arange(60, dtype=torch.float64)
        anisotropic = _solver((2.0, 3.0, 5.0))._gradient_op(field)
        expected = sum(
            _solver((spacing,), (dimension,))._gradient_op(field)
            for spacing, dimension in ((2.0, 1), (3.0, 2), (5.0, 3))
        )
        torch.testing.assert_close(anisotropic, expected)

    def test_rejects_invalid_spacing(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            _solver((1.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()

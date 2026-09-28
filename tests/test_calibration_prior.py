"""Unit tests for the GRICS++ multiplicative calibration prior."""
import unittest

import torch

from src.reconstruction.joint_reconstructor_utils.calibration_prior import CalibrationPriorEncodingOperator


class _ScaledEncoding:
    device = torch.device("cpu")

    def forward(self, value):
        return 2 * value

    def adjoint(self, value):
        return 2 * value

    def normal(self, value):
        return 4 * value


class CalibrationPriorEncodingOperatorTests(unittest.TestCase):
    def test_wraps_forward_adjoint_and_normal_with_calibration(self):
        calibration = torch.tensor([2.0, 3.0])
        operator = CalibrationPriorEncodingOperator(_ScaledEncoding(), calibration)
        latent = torch.tensor([1 + 2j, 2 - 1j])
        data = torch.tensor([3 + 1j, 4 - 2j])
        torch.testing.assert_close(operator.forward(latent), 2 * calibration * latent)
        torch.testing.assert_close(operator.adjoint(data), calibration * 2 * data)
        torch.testing.assert_close(operator.normal(latent), calibration * 4 * calibration * latent)


if __name__ == '__main__':
    unittest.main()

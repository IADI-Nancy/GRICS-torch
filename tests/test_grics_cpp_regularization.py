"""Checks for GRICS++ image scaling and its multiplicative calibration prior."""

import unittest
from types import SimpleNamespace

import torch

from src.reconstruction.JointReconstructor import JointReconstructor


class IdentityEncoding:
    device = torch.device("cpu")

    def forward(self, image):
        return image.flatten()

    def adjoint(self, data):
        return data.flatten()

    def normal(self, image):
        return image.flatten()


def image_reconstructor(kspace_scale):
    reconstructor = JointReconstructor.__new__(JointReconstructor)
    reconstructor.device = torch.device("cpu")
    reconstructor.params = SimpleNamespace(
        Nex=1, verbose=False, cg_stop_on_stagnation=False,
        cg_true_residual_interval=10, cg_stagnation_consecutive_steps=6,
        cg_stagnation_countdown_steps=0, cg_use_reg_scale_proxy=False,
        max_iter_recon=20, tol_recon=1e-12,
    )
    reconstructor._current_level_idx = 0
    reconstructor.regularization_scaling = "grics_cpp"
    reconstructor.kspace_scale = kspace_scale
    reconstructor.Data_full = {"Nx": 2, "Ny": 2, "Nz": 1}
    return reconstructor


class GricsCppRegularizationTests(unittest.TestCase):
    def test_calibration_prior_and_raw_kspace_scale(self):
        calibration = torch.tensor([[1.0, 2.0], [0.0, 4.0]], dtype=torch.float64)
        raw_data = torch.tensor([[2.0, 3.0], [5.0, 7.0]], dtype=torch.complex128)
        reg_weight = 0.5

        def solve(scale):
            reconstructor = image_reconstructor(scale)
            data = {
                "Nx": 2, "Ny": 2, "Nz": 1, "E": IdentityEncoding(),
                "KspaceData": (raw_data / scale).flatten(),
                "ReconstructedImage": torch.zeros((1, 2, 2), dtype=torch.complex128),
                "CalibrationImagePrior": calibration,
            }
            return reconstructor._solve_image(
                data, regularization_weight=reg_weight, max_iterations=20
            )[0] * scale

        rhs = calibration * raw_data
        effective_lambda = reg_weight * torch.linalg.norm(rhs).item()
        expected = calibration * rhs / (calibration.square() + effective_lambda)
        torch.testing.assert_close(solve(1.0), expected, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(solve(10.0), expected, rtol=1e-10, atol=1e-10)

    def test_complex_cropped_calibration_prior_with_warm_start(self):
        reconstructor = image_reconstructor(1.0)
        calibration = torch.tensor(
            [[1.0 + 0.5j, -0.4 + 0.2j], [0.0j, 2.0 - 0.3j]],
            dtype=torch.complex128,
        )
        measured = torch.tensor(
            [[2.0 + 1.0j, 3.0 - 2.0j], [1.0j, 4.0 + 0.5j]],
            dtype=torch.complex128,
        )
        data = {
            "Nx": 2, "Ny": 2, "Nz": 1, "E": IdentityEncoding(),
            "KspaceData": measured.flatten(),
            "ReconstructedImage": torch.ones((1, 2, 2), dtype=torch.complex128),
            "CalibrationImagePrior": calibration,
        }
        image = reconstructor._solve_image(data, regularization_weight=0.5)
        rhs = calibration.conj() * measured
        effective_lambda = 0.5 * torch.linalg.norm(rhs).item()
        expected = calibration * rhs / (calibration.abs().square() + effective_lambda)
        torch.testing.assert_close(image[0], expected, rtol=1e-10, atol=1e-10)

    def test_direct_mode_retains_original_image_penalty(self):
        reconstructor = image_reconstructor(1.0)
        reconstructor.regularization_scaling = "direct"
        data = {
            "Nx": 2, "Ny": 2, "Nz": 1, "E": IdentityEncoding(),
            "KspaceData": torch.ones(4, dtype=torch.complex128),
            "ReconstructedImage": torch.zeros((1, 2, 2), dtype=torch.complex128),
        }
        image = reconstructor._solve_image(data, regularization_weight=1.0)
        torch.testing.assert_close(image, torch.full_like(image, 0.5))


if __name__ == "__main__":
    unittest.main()

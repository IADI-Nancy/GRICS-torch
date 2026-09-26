"""Regression tests for coarse-grid and Fourier-cropped spatial inputs."""

import math
from types import SimpleNamespace
import unittest

import torch

from src.reconstruction.JointReconstructor import JointReconstructor
from src.reconstruction.joint_reconstructor_utils.resampling import (
    downsample_data, fourier_crop_spatial,
)
from src.utils.fftnc import fftnc


class FourierCropTests(unittest.TestCase):
    def test_constant_amplitude_and_full_resolution(self):
        image = torch.full((2, 12, 20), 2 + 3j, dtype=torch.complex128)
        cropped = fourier_crop_spatial(image, (6, 8))
        torch.testing.assert_close(
            cropped, torch.full((2, 6, 8), 2 + 3j, dtype=torch.complex128),
            rtol=1e-12, atol=1e-12,
        )
        self.assertIs(fourier_crop_spatial(image, (12, 20)), image)

    def test_central_spectrum_and_3d(self):
        generator = torch.Generator().manual_seed(12)
        image = torch.complex(
            torch.randn((2, 8, 10, 6), generator=generator, dtype=torch.float64),
            torch.randn((2, 8, 10, 6), generator=generator, dtype=torch.float64),
        )
        cropped = fourier_crop_spatial(image, (4, 6, 4))
        original_spectrum = fftnc(image, dims=(-3, -2, -1))
        coarse_spectrum = fftnc(cropped, dims=(-3, -2, -1))
        expected = original_spectrum[:, 2:6, 2:8, 1:5] * math.sqrt(
            (4 * 6 * 4) / (8 * 10 * 6)
        )
        torch.testing.assert_close(coarse_spectrum, expected, rtol=1e-12, atol=1e-12)

    def test_breast_coarse_grids_maps_and_prior(self):
        nx, ny = 288, 476
        maps = torch.ones((1, nx, ny, 1), dtype=torch.complex128)
        kspace = torch.zeros((1, 1, nx, ny, 1), dtype=torch.complex128)
        full = {
            "Nx": nx, "Ny": ny, "Nz": 1,
            "SensitivityMaps": maps, "KspaceData": kspace,
            "SamplingIndices": [[torch.arange(nx * ny)]],
        }
        params = SimpleNamespace(Nex=1)
        signal = torch.zeros((1, 1), dtype=torch.float64)
        prior = torch.ones((nx, ny), dtype=torch.float64)
        recon = SimpleNamespace(
            Data_full=full, motion_states_per_level=[1], _current_level_idx=0,
            motion_signal=signal, params=params, device=torch.device("cpu"),
            calibration_image_prior=prior, Nz_full=1,
        )

        for factor, shape in ((0.125, (36, 58)), (0.25, (72, 118)),
                              (0.5, (144, 238)), (1.0, (288, 476))):
            level = JointReconstructor._prepare_resolution_level(recon, 1, factor)
            self.assertEqual((level["Nx"], level["Ny"]), shape)
            self.assertEqual(level["SensitivityMaps"].shape, (1, *shape, 1))
            torch.testing.assert_close(
                level["SensitivityMaps"][0, ..., 0],
                torch.ones(shape, dtype=torch.complex128), rtol=1e-11, atol=1e-11,
            )
            torch.testing.assert_close(
                level["CalibrationImagePrior"],
                torch.ones(shape, dtype=level["CalibrationImagePrior"].dtype),
                rtol=1e-11, atol=1e-11,
            )


if __name__ == "__main__":
    unittest.main()

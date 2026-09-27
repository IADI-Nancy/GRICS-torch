"""Small image solves checking independent-NEX GRICS++ lambda scaling."""
import unittest
from types import SimpleNamespace

import torch

from src.reconstruction.JointReconstructor import JointReconstructor


class IdentityEncoding:
    device = torch.device('cpu')

    def adjoint(self, values):
        return values.flatten()

    def normal(self, values):
        return values


def solve_images(images, kspace_scale=1.0, scaling='grics_cpp'):
    recon = object.__new__(JointReconstructor)
    recon.device = torch.device('cpu')
    recon.params = SimpleNamespace(
        Nex=images.shape[0], verbose=False, lambda_r=0.3,
        max_iter_recon=20, tol_recon=1e-12,
        cg_stop_on_stagnation=False, cg_true_residual_interval=10,
        cg_stagnation_consecutive_steps=12, cg_stagnation_countdown_steps=6,
        cg_use_reg_scale_proxy=False,
    )
    shape = images.shape[1:]
    grid = dict(Nx=shape[0], Ny=shape[1], Nz=shape[2] if len(shape) == 3 else 1)
    recon.Data_full = grid.copy()
    recon.regularization_scaling = scaling
    recon.kspace_scale = kspace_scale
    recon._current_level_idx = 0
    return recon._solve_image(dict(
        **grid, ReconstructedImage=torch.zeros_like(images),
        E=IdentityEncoding(), KspaceData=images / kspace_scale,
    )) * kspace_scale


class IndependentImageRegularizationTests(unittest.TestCase):
    def test_single_image_matches_cpp_identity_equation(self):
        image = torch.arange(1, 13, dtype=torch.float64).reshape(1, 3, 4).to(torch.complex128)
        expected = image / (1 + 0.3 * torch.linalg.vector_norm(image))
        torch.testing.assert_close(solve_images(image), expected)

    def test_other_repetitions_do_not_change_an_images_penalty(self):
        for shape in ((3, 4), (3, 4, 2)):
            image = torch.arange(1, 1 + torch.tensor(shape).prod().item(),
                                 dtype=torch.float64).reshape(1, *shape).to(torch.complex128)
            for factor in (1.0, 7.0):
                with self.subTest(shape=shape, other_image_amplitude=factor):
                    pair = torch.cat((image, factor * image), dim=0)
                    expected = torch.cat((solve_images(image), solve_images(factor * image)), dim=0)
                    torch.testing.assert_close(solve_images(pair), expected)

    def test_internal_kspace_normalization_does_not_change_image(self):
        image = torch.arange(1, 25, dtype=torch.float64).reshape(2, 3, 4).to(torch.complex128)
        torch.testing.assert_close(solve_images(image, kspace_scale=0.01), solve_images(image))

    def test_legacy_fixed_lambda_is_preserved(self):
        image = torch.arange(1, 25, dtype=torch.float64).reshape(2, 3, 4).to(torch.complex128)
        torch.testing.assert_close(solve_images(image, scaling='none'), image / 1.3)


if __name__ == '__main__':
    unittest.main()

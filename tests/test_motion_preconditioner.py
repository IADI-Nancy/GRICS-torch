"""CPU regression checks for motion preconditioning and its required config."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tomllib
import unittest

import numpy as np
import torch

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.JointReconstructor import JointReconstructor
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.reconstruction.joint_reconstructor_utils.motion_preconditioners import (
    PseudoGaussSeidelMotionPreconditioner,
)
from src.runtime.runtime_config import _validate_reconstruction


class MotionPreconditionerTests(unittest.TestCase):
    def test_3d_factors_match_dense_triangular_solve(self):
        generator = torch.Generator().manual_seed(8)
        gradients = torch.randn(3, 2, 3, 2, dtype=torch.complex64, generator=generator)
        local = gradients.conj()[:, None] * gradients[None, :]
        factor = PseudoGaussSeidelMotionPreconditioner(
            local, lambda_scaled=0.8, voxel_spacing=(1., 1.5, 2.))
        # Assemble the reference matrix independently, retaining its flat
        # spatial offsets across row, plane and component boundaries.
        lower = np.zeros((36, 36), dtype=np.complex128)
        working = local[[1, 2, 0]][:, [1, 2, 0]].permute(0, 1, 2, 4, 3).reshape(3, 3, 12).numpy()
        for row in range(3):
            for col in range(row + 1):
                for voxel in range(12):
                    lower[row * 12 + voxel, col * 12 + voxel] += working[row, col, voxel]
        lower += np.eye(36) * 2 * 0.8 * (1 + 1.5**2 + 2**2)
        for offset, spacing in zip((1, 3, 6), (1.5, 2., 1.)):
            for col in range(36 - offset):
                lower[col + offset, col] -= 0.8 * spacing**2
        upper = lower.conj().T / np.diag(lower)[:, None]
        rhs = torch.randn(36, dtype=torch.complex128, generator=generator)
        expected = np.linalg.solve(upper, np.linalg.solve(lower, factor._to_factor_layout(rhs).numpy()))
        actual = factor(rhs)
        self.assertEqual(actual.dtype, rhs.dtype)
        torch.testing.assert_close(actual, factor._from_factor_layout(torch.from_numpy(expected)),
                                   rtol=2e-6, atol=2e-7)
        self.assertGreater(torch.vdot(rhs, actual).real.item(), 0)

    def test_pcg_solves_diagonal_system_in_one_iteration(self):
        diagonal = torch.logspace(0, 4, 32, dtype=torch.float64)
        operator = SimpleNamespace(device=torch.device('cpu'), normal=lambda x: diagonal * x)
        solver = ConjugateGradientSolver(
            operator, reg_lambda=0., regularizer='Tikhonov', regularization_shape=None,
            regularization_spatial_dims=None, verbose=False, stop_on_stagnation=False,
            true_residual_interval=1, stagnation_consecutive_steps=6,
            stagnation_countdown_steps=0, use_reg_scale_proxy=False,
            reg_scale_num_probes=None, preconditioner=lambda r: r / diagonal)
        torch.testing.assert_close(solver.cg(torch.ones_like(diagonal), max_iter=1, tol=1e-12),
                                   1 / diagonal)

    def test_zero_motion_rhs_with_zero_regularization(self):
        recon = JointReconstructor.__new__(JointReconstructor)
        recon.params = SimpleNamespace(
            reconstruction_motion_type='non-rigid', motion_preconditioner=True,
            max_iter_motion=5, lambda_m=0., verbose=False, cg_stop_on_stagnation=False,
            cg_true_residual_interval=10, cg_stagnation_consecutive_steps=12,
            cg_stagnation_countdown_steps=6, cg_use_reg_scale_proxy=False, tol_motion=1e-6)
        recon.Nalpha, recon.Nphysio, recon.kspace_scale = 3, 1, 1.
        recon.regularization_scaling = 'direct'
        alpha = torch.zeros(3, 2, 2, 2, 1, dtype=torch.float64)
        motion = MotionOperator(2, 2, alpha, 'non-rigid', Nz=2,
                                motion_signal=torch.ones(1, 1, dtype=torch.float64))
        jacobian = MotionPerturbationSimulator(
            torch.ones(1, 2, 2, 2, dtype=torch.complex128), 8, [[torch.arange(8)]], 1,
            torch.zeros(2, 2, 2, dtype=torch.complex128), motion)
        data = dict(Nx=2, Ny=2, Nz=2, J=jacobian, MotionModel=alpha)
        update = recon._solve_motion(data, torch.zeros(8, dtype=torch.complex128))
        torch.testing.assert_close(update, torch.zeros_like(alpha, dtype=torch.complex128))

    def test_2d_factors_match_dense_triangular_solve(self):
        generator = torch.Generator().manual_seed(9)
        gradients = torch.randn(4, 3, 4, dtype=torch.complex64, generator=generator)
        local = gradients.conj()[:, None] * gradients[None, :]
        factor = PseudoGaussSeidelMotionPreconditioner(
            local, lambda_scaled=0.7, voxel_spacing=(1.5, 2.0))
        lower = np.zeros((48, 48), dtype=np.complex128)
        working = local[[2, 3, 0, 1]][:, [2, 3, 0, 1]].reshape(4, 4, 12).numpy()
        for row in range(4):
            for col in range(row + 1):
                for voxel in range(12):
                    lower[row * 12 + voxel, col * 12 + voxel] += working[row, col, voxel]
        lower += np.eye(48) * 2 * 0.7 * (1.5**2 + 2.0**2)
        for offset, spacing in ((1, 2.0), (4, 1.5)):
            for col in range(48 - offset):
                lower[col + offset, col] -= 0.7 * spacing**2
        upper = lower.conj().T / np.diag(lower)[:, None]
        rhs = torch.randn(48, dtype=torch.complex128, generator=generator)
        expected = np.linalg.solve(upper, np.linalg.solve(lower, factor._to_factor_layout(rhs).numpy()))
        actual = factor(rhs)
        torch.testing.assert_close(actual, factor._from_factor_layout(torch.from_numpy(expected)),
                                   rtol=2e-6, atol=2e-7)
        self.assertGreater(torch.vdot(rhs, actual).real.item(), 0)

    def test_2d_nonrigid_motion_solve_uses_pseudo_gauss_seidel_factors(self):
        nx = ny = 4
        signal = torch.ones(1, 1, dtype=torch.float64)
        alpha = torch.zeros(2, nx, ny, 1, dtype=torch.float64)
        motion = MotionOperator(nx, ny, alpha, 'non-rigid', Nz=1, motion_signal=signal)
        jacobian = MotionPerturbationSimulator(
            torch.ones(1, nx, ny, 1, dtype=torch.complex128), nx * ny,
            [[torch.arange(nx * ny)]], 1, torch.randn(nx, ny, dtype=torch.complex128), motion)
        recon = JointReconstructor.__new__(JointReconstructor)
        recon.params = SimpleNamespace(
            reconstruction_motion_type='non-rigid', motion_preconditioner=True,
            max_iter_motion=2, lambda_m=0.5, verbose=False, cg_stop_on_stagnation=False,
            cg_true_residual_interval=10, cg_stagnation_consecutive_steps=12,
            cg_stagnation_countdown_steps=6, cg_use_reg_scale_proxy=False, tol_motion=1e-3)
        recon.Nalpha, recon.Nphysio, recon.Nz_full = 2, 1, 1
        recon.regularization_scaling = 'voxel_aware'
        recon.voxel_spacing_mm = (1.0, 1.5)
        recon.kspace_scale = 1.0
        recon.Data_full = dict(Nx=nx, Ny=ny, Nz=1)
        residual = jacobian.forward(torch.randn_like(alpha, dtype=torch.complex128))
        update = recon._solve_motion(dict(Nx=nx, Ny=ny, Nz=1, J=jacobian, MotionModel=alpha), residual)
        self.assertEqual(update.shape, alpha.shape)
        self.assertTrue(torch.isfinite(update).all())

    def test_reference_factor_uses_level_spacing_and_scaled_lambda(self):
        recon = JointReconstructor.__new__(JointReconstructor)
        recon.params = SimpleNamespace(reconstruction_motion_type='non-rigid')
        recon.regularization_scaling = 'voxel_aware'
        recon.Nz_full = 8
        recon.Data_full = dict(Nx=8, Ny=12, Nz=8)
        recon.voxel_spacing_mm = (1.0, 1.5, 2.0)
        recon.kspace_scale = 3.0
        data = dict(Nx=4, Ny=6, Nz=4)
        spacing, ratio = recon._level_voxel_spacing_and_sqrt_voxel_count_ratio(data)
        self.assertEqual(spacing, (2.0, 3.0, 4.0))
        self.assertAlmostEqual(ratio, 0.125 ** 0.5)
        solver = SimpleNamespace(_effective_lambda=lambda: 0.25)
        with patch.object(PseudoGaussSeidelMotionPreconditioner, 'from_motion_simulator') as build:
            recon._build_pseudo_gauss_seidel_motion_preconditioner(object(), solver, data)
        self.assertEqual(build.call_args.kwargs['voxel_spacing'], spacing)
        self.assertEqual(build.call_args.kwargs['lambda_scaled'], 1.5)
        self.assertEqual(build.call_args.kwargs['image_scale'], 3.0)

    def test_zero_boundary_gradient_used_by_reference_factors(self):
        nx = ny = nz = 5
        image = torch.arange(nx, dtype=torch.float64)[:, None, None].expand(nx, ny, nz)
        alpha = torch.zeros(3, nx, ny, nz, 1, dtype=torch.float64)
        motion = MotionOperator(nx, ny, alpha, 'non-rigid', Nz=nz,
                                motion_signal=torch.ones(1, 1, dtype=torch.float64))
        jacobian = MotionPerturbationSimulator(
            torch.ones(1, nx, ny, nz, dtype=torch.complex128), nx * ny * nz,
            [[torch.arange(nx * ny * nz)]], 1, image.to(torch.complex128), motion,
            gradient_boundary='zero')
        gx, gy, gz = jacobian._gradient_3d(image)
        torch.testing.assert_close(gx[0], torch.zeros_like(gx[0]))
        torch.testing.assert_close(gx[-1], torch.zeros_like(gx[-1]))
        torch.testing.assert_close(gx[1:-1], torch.ones_like(gx[1:-1]))
        torch.testing.assert_close(gy, torch.zeros_like(gy))
        torch.testing.assert_close(gz, torch.zeros_like(gz))

    def test_config_flag_is_required_boolean_with_expected_defaults(self):
        config_dir = Path(__file__).resolve().parents[1] / 'config' / 'reconstruction'
        for path in config_dir.glob('*.toml'):
            with self.subTest(config=path.name):
                cfg = tomllib.loads(path.read_text())['reconstruction']
                if path.stem.startswith('rigid_'):
                    self.assertNotIn('motion_preconditioner', cfg)
                else:
                    self.assertIs(cfg['motion_preconditioner'], 'breast' in path.stem)
                self.assertEqual(cfg['motion_gradient_boundary'],
                                 'zero' if path.stem.endswith('_3d') or '_3d_' in path.stem else 'one_sided')
                cfg.update(coil_sensitivity_method=cfg['coil_sensitivity_config'],
                           simulated_motion_type='as-is')
                _validate_reconstruction(cfg)
                if path.stem.startswith('rigid_'):
                    for value in (False, True, None):
                        with self.assertRaisesRegex(ValueError, 'motion_preconditioner'):
                            _validate_reconstruction({**cfg, 'motion_preconditioner': value})
                else:
                    _validate_reconstruction({**cfg, 'motion_preconditioner': True})
                    with self.assertRaisesRegex(ValueError, 'motion_preconditioner=true requires lambda_m > 0'):
                        _validate_reconstruction({**cfg, 'motion_preconditioner': True, 'lambda_m': 0})
                    _validate_reconstruction({**cfg, 'motion_preconditioner': False, 'lambda_m': 0})
                    for invalid in (None, 0, 1, 'true', 'voxel_aware'):
                        with self.assertRaisesRegex(ValueError, 'motion_preconditioner'):
                            _validate_reconstruction({**cfg, 'motion_preconditioner': invalid})
                    del cfg['motion_preconditioner']
                    with self.assertRaisesRegex(ValueError, 'motion_preconditioner'):
                        _validate_reconstruction(cfg)


if __name__ == '__main__':
    unittest.main()

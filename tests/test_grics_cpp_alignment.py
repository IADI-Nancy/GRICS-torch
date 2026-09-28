"""Small CPU regressions; no subject data or real reconstruction required."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.reconstruction.JointReconstructor import JointReconstructor
from src.reconstruction.grics_cpp_motion_preconditioner import GricsCppPseudoGaussSeidelPreconditioner
from src.reconstruction.joint_reconstructor_utils.resampling.fourier_crop import fourier_resize_spatial
from src.reconstruction.joint_reconstructor_utils.resampling.upsample import upsample_data
from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_shared_encoding_and_motion_adjoint(self):
        shape = (4, 6, 4)
        n = int(np.prod(shape))
        identity = torch.eye(n, dtype=torch.complex128).to_sparse()
        motion = SimpleNamespace(alpha=torch.zeros(3, *shape, 1), Nz=4,
            motion_type='non-rigid', motion_signal=torch.tensor([[-1.], [1.]]),
            _get_sparse_operator=lambda state: identity)
        maps = torch.randn(2, *shape, dtype=torch.complex128)
        grid = torch.arange(n).reshape(shape)
        first, second = grid[:, ::2].flatten(), grid[:, 1::2].flatten()
        indices = [[first, second], [second, first]]
        image = torch.randn(1, *shape, dtype=torch.complex128)
        encoding = EncodingOperator(maps, n, indices, 2, motion, Nimages=1)
        data = torch.randn(4*n, dtype=torch.complex128)
        torch.testing.assert_close(torch.vdot(encoding.forward(image.flatten()), data),
            torch.vdot(image.flatten(), encoding.adjoint(data)))
        torch.testing.assert_close(encoding.normal(image.flatten()),
            encoding.adjoint(encoding.forward(image.flatten())))
        independent = EncodingOperator(maps, n, indices, 2, motion, Nimages=2)
        torch.testing.assert_close(encoding.forward(image.flatten()),
            independent.forward(image.repeat(2, 1, 1, 1).flatten()))
        torch.testing.assert_close(encoding.adjoint(data), independent.adjoint(data).reshape(2, n).sum(0))
        jacobian = MotionPerturbationSimulator(maps, n, indices, 2, image, motion,
            Nimages=1, gradient_boundary='zero')
        perturbation = torch.randn(3*n, dtype=torch.complex128)
        torch.testing.assert_close(torch.vdot(jacobian.forward(perturbation), data),
            torch.vdot(perturbation, jacobian.adjoint(data)))

    def test_small_shared_two_level_integration(self):
        from src.runtime.runtime_config import load_config
        from tests.test_configuration_validation import SYNTH
        params = load_config(**{**SYNTH,
            'reconstruction_config': 'config/reconstruction/nonrigid_3d_breast.toml',
            'shepp_logan_config': 'config/synthetic_data/shepp_logan_3d.toml',
            'motion_simulation_config': 'config/motion_simulation/nonrigid_3d.toml'},
            overrides={'ResolutionLevels':[0.5,1.0], 'GN_iterations_per_level':[2,2],
                       'N_motion_states':2, 'N_motion_states_per_level':[2,2],
                       'max_iter_motion':2, 'max_iter_recon':2, 'verbose':False})
        params.Nex = 2
        shape = (8, 10, 8)
        grid = torch.arange(640).reshape(shape)
        indices = [[grid[:,::2].flatten(), grid[:,1::2].flatten()]]*2
        recon = JointReconstructor(torch.randn(1,2,*shape,dtype=torch.complex128),
            torch.ones(1,*shape,dtype=torch.complex128), indices,
            torch.tensor([[-1.],[1.]]), params, voxel_spacing_mm=(1.,1.5,2.))
        class Logger:
            @contextmanager
            def iterations(self, level): yield range(2)
            def record_residual(self, value): pass
            def iteration_finished(self, *args): pass
            def iteration_stopped_early(self): pass
        previous = None
        for level, resolution in enumerate(params.ResolutionLevels):
            recon._current_level_idx = level
            data = recon._prepare_resolution_level(level, resolution)
            if previous is not None:
                upsample_data(previous, data, params, 3, torch.device('cpu'))
            image, motion = recon._run_resolution_level(data, level_index=level,
                gauss_newton_iterations_at_level=2, level_count=2, update_final_motion=False,
                gn_early_stopping=True, logger=Logger())
            self.assertEqual(image.shape[0], 1)
            self.assertTrue(torch.isfinite(image).all() and torch.isfinite(motion).all())
            previous = data
        self.assertEqual(tuple(image.shape), (1,*shape))
        self.assertEqual(recon._full_resolution_iteration_data(image,motion)['Nimages'], 1)

    def test_subject_grid_sizes_match_cpp(self):
        from unittest.mock import patch
        from contextlib import ExitStack
        from src.reconstruction.joint_reconstructor_utils.resampling.downsample import downsample_data
        module = 'src.reconstruction.joint_reconstructor_utils.resampling.downsample.'
        full = dict(Nx=160, Ny=218, Nz=56, Nimages=1, SensitivityMaps=None, SamplingIndices=None)
        with ExitStack() as stack:
            stack.enter_context(patch(module+'fourier_resize_spatial', return_value=None))
            stack.enter_context(patch(module+'downsample_sampling_indices', return_value=None))
            stack.enter_context(patch(module+'downsample_kspace', return_value=torch.zeros(1,2,1)))
            stack.enter_context(patch(module+'reduce_motion_states', return_value=(None,None)))
            for scale, expected in ((.25,(40,54,14)), (.5,(80,108,28)), (1.,(160,218,56))):
                data = downsample_data(full, scale, 8, None,
                    SimpleNamespace(resolution_resampling='fourier', Nex=2), torch.device('cpu'))
                self.assertEqual(tuple(data[key] for key in ('Nx','Ny','Nz')), expected)

    def test_zero_boundary_gradients(self):
        simulator = MotionPerturbationSimulator.__new__(MotionPerturbationSimulator)
        simulator.gradient_boundary = 'zero'
        x, y, z = torch.meshgrid(*(torch.arange(5.) for _ in range(3)), indexing='ij')
        for axis, gradient in enumerate(simulator._gradient_3d(x+2*y+3*z)):
            torch.testing.assert_close(gradient.select(axis, 0), torch.zeros(5, 5))
            torch.testing.assert_close(gradient.select(axis, 4), torch.zeros(5, 5))
            torch.testing.assert_close(gradient[1:-1, 1:-1, 1:-1], torch.full((3,3,3), float(axis+1)))

    def test_fourier_transfer_preserves_constants_and_physical_motion(self):
        image = torch.full((1, 4, 6, 4), 2+3j, dtype=torch.complex128)
        resized = fourier_resize_spatial(image, (8, 10, 6))
        torch.testing.assert_close(resized, torch.full_like(resized, 2+3j))
        torch.testing.assert_close(fourier_resize_spatial(resized, (4,6,4)), image)
        previous = dict(Nx=4, Ny=6, Nz=4, ReconstructedImage=image, MotionModel=torch.ones(3,4,6,4,1))
        current = dict(Nx=8, Ny=10, Nz=6)
        upsample_data(previous, current, SimpleNamespace(resolution_resampling='fourier',
            reconstruction_motion_type='non-rigid'), 3, torch.device('cpu'))
        for component, ratio in enumerate((2., 10/6, 1.5)):
            torch.testing.assert_close(current['MotionModel'][component], torch.full((8,10,6,1), ratio))

    def test_preconditioner_matches_dense_cpp_diagonals(self):
        gradients = torch.randn(3, 2, 3, 2, dtype=torch.complex64)
        local = gradients.conj()[:,None] * gradients[None,:]
        factor = GricsCppPseudoGaussSeidelPreconditioner(local,
            lambda_scaled=0.8, voxel_spacing=(1., 1.5, 2.))
        n = 36
        lower = np.zeros((n,n), dtype=np.complex128)
        upper = np.zeros_like(lower)
        # Independently assemble C++ band storage, including boundary crossings.
        working = local[[1,2,0]][:,[1,2,0]].permute(0,1,2,4,3).reshape(3,3,12).numpy()
        for row in range(3):
            for col in range(row+1):
                for v in range(12):
                    lower[row*12+v,col*12+v] += working[row,col,v]
        lower += np.eye(n)*2*0.8*(1+1.5**2+2**2)
        for offset, spacing in zip((1,3,6),(1.5,2.,1.)):
            for col in range(n-offset):
                lower[col+offset,col] -= 0.8*spacing**2
        upper = lower.conj().T / np.diag(lower)[:,None]
        rhs = torch.randn(n, dtype=torch.complex128)
        expected = np.linalg.solve(upper, np.linalg.solve(lower, factor._to_cpp_layout(rhs).numpy()))
        torch.testing.assert_close(factor(rhs), factor._from_cpp_layout(torch.from_numpy(expected)), rtol=2e-6, atol=2e-7)
        self.assertGreater(torch.vdot(rhs, factor(rhs)).real.item(), 0)

    def test_pcg_applies_inverse_preconditioner(self):
        diagonal = torch.logspace(0,4,32,dtype=torch.float64)
        operator = SimpleNamespace(device=torch.device('cpu'), normal=lambda x: diagonal*x)
        solver = ConjugateGradientSolver(operator, reg_lambda=0., regularizer='Tikhonov',
            regularization_shape=None, regularization_spatial_dims=None,
            verbose=False, stop_on_stagnation=False, true_residual_interval=1,
            stagnation_consecutive_steps=6, stagnation_countdown_steps=0,
            use_reg_scale_proxy=False, reg_scale_num_probes=None, preconditioner=lambda r: r/diagonal)
        torch.testing.assert_close(solver.cg(torch.ones_like(diagonal), max_iter=1, tol=1e-12), 1/diagonal)

    def test_level_finishes_with_accepted_pair_and_rejects_before_motion(self):
        class Logger:
            @contextmanager
            def iterations(self, level):
                yield range(3)
            def record_residual(self, value): pass
            def iteration_finished(self, *args): pass
            def iteration_stopped_early(self): pass
        recon = JointReconstructor.__new__(JointReconstructor)
        recon.params = SimpleNamespace(lambda_r=0., gn_level_schedule='grics_cpp')
        recon.external_image_regularizer = None
        recon._current_level_idx = 0
        recon._last_image_cg_info = recon._last_motion_cg_info = None
        for residuals, expected_updates, expected_image in (([.3,.2,.1],2,3), ([.3,.4,.1],1,1)):
            calls, updates = [], []
            data = dict(KspaceData=torch.ones(1), MotionModel=torch.zeros(1))
            def iteration(data, **kwargs):
                self.assertFalse(kwargs['update_motion'])
                calls.append(1)
                return SimpleNamespace(image=torch.tensor([float(len(calls))]),
                    motion_for_residual=data['MotionModel'].clone(), residual=torch.tensor([residuals[len(calls)-1]]))
            def update(data, image, motion, residual):
                updates.append(1)
                return motion+1, torch.ones(1), 0.
            recon.gauss_newton_iteration = iteration
            recon._motion_update = update
            recon._run_resolution_level(data, level_index=0, gauss_newton_iterations_at_level=3,
                level_count=2, update_final_motion=False, gn_early_stopping=True, logger=Logger())
            self.assertEqual(len(updates), expected_updates)
            self.assertEqual(data['ReconstructedImage'].item(), expected_image)
            self.assertEqual(data['MotionModel'].item(), expected_image-1)


if __name__ == '__main__':
    unittest.main()

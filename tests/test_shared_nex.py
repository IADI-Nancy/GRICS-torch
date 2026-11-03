"""Small CPU checks for shared-image reconstruction; no subject data required.

Run with: python tests/test_shared_nex.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.reconstruction.JointReconstructor import JointReconstructor
from src.reconstruction.joint_reconstructor_utils.initialization import (
    initialize_zero_image_and_motion, apply_external_initializer,
)


class SharedNexTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_encoding_and_motion_operators(self):
        for nz in (1, 4):
            for nex in (1, 2, 3):
                with self.subTest(nz=nz, nex=nex):
                    nx, ny, nc = 5, 6, 2
                    spatial = (nx, ny, nz) if nz > 1 else (nx, ny)
                    count = nx * ny * nz
                    image = torch.randn(*spatial, dtype=torch.complex128)
                    smaps = torch.randn(nc, nx, ny, nz, dtype=torch.complex128)
                    alpha = .05 * torch.randn(len(spatial), *spatial, 1, dtype=torch.float64)
                    signal = torch.tensor([[-.8], [.5]], dtype=torch.float64)
                    motion = MotionOperator(nx, ny, alpha, 'non-rigid', motion_signal=signal, Nz=nz)
                    indices = torch.arange(count)
                    phase = indices % (ny * nz)
                    # Complete readout lines, with different state assignments per repetition.
                    sampling = [[indices[(phase + rep) % 2 == state] for state in range(2)]
                                for rep in range(nex)]
                    e = EncodingOperator(smaps, count, sampling, nex, motion)
                    singles = [EncodingOperator(smaps, count, [s], 1, motion) for s in sampling]
                    measured = torch.randn(nc, nex, count, dtype=torch.complex128)
                    expected = torch.cat([s.forward(image).reshape(nc, 1, count)
                                          for s in singles], dim=1)
                    torch.testing.assert_close(e.forward(image), expected.flatten())
                    torch.testing.assert_close(e.adjoint(measured), sum(
                        s.adjoint(measured[:, rep]) for rep, s in enumerate(singles)))
                    torch.testing.assert_close(e.normal(image), e.adjoint(e.forward(image)))
                    torch.testing.assert_close(torch.vdot(e.forward(image), measured.flatten()),
                                               torch.vdot(image.flatten(), e.adjoint(measured)))
                    j = MotionPerturbationSimulator(smaps, count, sampling, nex, image, motion)
                    js = [MotionPerturbationSimulator(smaps, count, [s], 1, image, motion)
                          for s in sampling]
                    delta = torch.randn_like(alpha).to(torch.complex128)
                    expected_j = torch.cat([s.forward(delta).reshape(nc, 1, count)
                                            for s in js], dim=1)
                    torch.testing.assert_close(j.forward(delta), expected_j.flatten())
                    torch.testing.assert_close(j.adjoint(measured), sum(
                        s.adjoint(measured[:, rep]) for rep, s in enumerate(js)))
                    torch.testing.assert_close(torch.vdot(j.forward(delta), measured.flatten()),
                                               torch.vdot(delta.flatten(), j.adjoint(measured)))

    def test_solver_uses_both_repetitions_and_keeps_one_image(self):
        for nz in (1, 4):
            with self.subTest(nz=nz):
                nx, ny = 5, 6
                spatial = (nx, ny, nz) if nz > 1 else (nx, ny)
                count = nx * ny * nz
                params = SimpleNamespace(Nex=2, reconstruction_motion_type='non-rigid',
                    N_motion_states=1, ResolutionLevels=[1.0],
                    regularization_scaling='direct', motion_preconditioner=False,
                    motion_gradient_boundary='one_sided',
                    lambda_r=.3, max_iter_recon=10, tol_recon=1e-10, verbose=False,
                    cg_stop_on_stagnation=False, cg_true_residual_interval=10,
                    cg_stagnation_consecutive_steps=12, cg_stagnation_countdown_steps=6,
                    cg_use_reg_scale_proxy=False)
                smaps = torch.ones(1, nx, ny, nz, dtype=torch.complex128)
                alpha = torch.zeros(len(spatial), *spatial, 1, dtype=torch.float64)
                signal = torch.zeros(1, 1, dtype=torch.float64)
                motion = MotionOperator(nx, ny, alpha, 'non-rigid', motion_signal=signal, Nz=nz)
                sampling = [[torch.arange(count)] for _ in range(2)]
                single = EncodingOperator(smaps, count, sampling[:1], 1, motion)
                references = torch.randn(2, *spatial, dtype=torch.complex128)
                kspace = torch.stack([single.forward(x).reshape(1, nx, ny, nz)
                                      for x in references], dim=1)
                recon = JointReconstructor(kspace, smaps, sampling, signal, params)
                data = {'Nx': nx, 'Ny': ny, 'Nz': nz, 'KspaceData': kspace,
                        'E': EncodingOperator(smaps, count, sampling, 2, motion)}
                initialize_zero_image_and_motion(data, params=params, device='cpu',
                                                 Nalpha=len(spatial), Nphysio=1)
                self.assertEqual(data['ReconstructedImage'].shape, spatial)
                image = recon._solve_image(data)
                # Two fully sampled repetitions give (2 I + lambda I)x = x1 + x2.
                expected = references.sum(dim=0) / (2 + params.lambda_r)
                torch.testing.assert_close(image, expected, rtol=1e-8, atol=1e-10)
                full = recon._full_resolution_iteration_data(image, alpha)
                self.assertEqual(full['KspaceData'].shape, (1, 2, count))
                apply_external_initializer(data, initial_image=image, initial_motion=None,
                    params=params, device='cpu', Nalpha=len(spatial), Nphysio=1)
                self.assertEqual(data['ReconstructedImage'].shape, spatial)
                with self.assertRaises(ValueError):
                    apply_external_initializer(data, initial_image=references, initial_motion=None,
                        params=params, device='cpu', Nalpha=len(spatial), Nphysio=1)
                with self.assertRaises(ValueError):
                    recon._full_resolution_iteration_data(image.unsqueeze(0), alpha)


if __name__ == '__main__':
    unittest.main()

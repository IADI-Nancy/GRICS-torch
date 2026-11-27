"""Validate streamed preconditioner factors against explicit sparse-warp gradients."""
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.reconstruction.joint_reconstructor_utils.motion_preconditioners import PseudoGaussSeidelMotionPreconditioner


class CheckedPreconditioner(PseudoGaussSeidelMotionPreconditioner):
    def __init__(self, local_normal, **kwargs):
        assert local_normal.device.type == 'cpu', 'Normal blocks must not occupy GPU memory'
        super().__init__(local_normal, **kwargs)


def validate(device):
    cases = 0
    for nz in (1, 4):
        shape = (8, 6, nz) if nz > 1 else (8, 6)
        dimensions = len(shape)
        for dtype in (torch.complex64, torch.complex128):
            for boundary in ('zero', 'one_sided'):
                signal = torch.randn(4, 2, device=device, dtype=torch.float64)
                alpha = torch.randn(dimensions, *shape, 2, device=device, dtype=torch.float64) * .1
                smaps = torch.randn(2, 8, 6, nz, device=device, dtype=torch.complex128)
                image = torch.randn(*shape, device=device, dtype=torch.complex128)
                samples = [[torch.arange(image.numel(), device=device) for _ in range(4)]]
                sparse = MotionOperator(8, 6, alpha, 'non-rigid', Nz=nz, motion_signal=signal, compute_dtype=dtype, warp_backend='sparse')
                gather = MotionOperator(8, 6, alpha, 'non-rigid', Nz=nz, motion_signal=signal, compute_dtype=dtype, warp_backend='gather')
                jacobian = MotionPerturbationSimulator(smaps, image.numel(), samples, 1, image, gather, gradient_boundary=boundary)
                gradients = []
                for state in range(4):
                    warped = (sparse.sparseMotionOperator[state] @ (image.flatten() * 2.5).to(dtype)).reshape(shape)
                    gradient = jacobian._gradient_3d(warped) if nz > 1 else jacobian._gradient_2d(warped)
                    gradients.append(torch.stack(gradient))
                expected = CheckedPreconditioner.from_warped_gradients(gradients, signal, lambda_scaled=.3, voxel_spacing=(1.2,) * dimensions)
                with patch('torch.sparse_coo_tensor', side_effect=AssertionError('Gather preconditioner must not materialize sparse warps')):
                    actual = CheckedPreconditioner.from_motion_simulator(jacobian, lambda_scaled=.3, voxel_spacing=(1.2,) * dimensions, image_scale=2.5)
                assert '_sparse_view_cache' not in gather.__dict__
                np.testing.assert_allclose(actual.diagonals, expected.diagonals, rtol=2e-5, atol=2e-5)
                residual = torch.randn(alpha.numel(), device=device, dtype=torch.complex128)
                torch.testing.assert_close(actual(residual), expected(residual), rtol=3e-5, atol=3e-5)
                cases += 1
    print(f'PASS: {cases} {device} cases: 2D/3D, both precisions and gradient boundaries; factors and solves match sparse reference, CPU normal blocks, no sparse views.')


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.manual_seed(42)
    validate('cpu')
    if torch.cuda.is_available():
        validate('cuda')
    else:
        print('CUDA checks skipped explicitly: GPU unavailable.')

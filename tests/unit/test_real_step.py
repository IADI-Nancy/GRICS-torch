"""The motion normal operator restricted to real parameters."""
import torch

from conftest import smooth_phantom
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.reconstruction.joint_reconstructor_utils.operators import RealNormalOperator


def test_real_normal_operator_is_the_real_part(device):
    N = 32
    img = smooth_phantom(N, N, device)
    smaps = torch.ones((1, N, N, 1), device=device, dtype=torch.complex128)
    sampling = [[torch.arange(N * N, device=device)[: N * N // 2], torch.arange(N * N, device=device)[N * N // 2:]]]
    alpha = torch.tensor([[0.3, -0.2], [0.1, 0.4], [0.05, -0.02]], device=device)
    J = MotionPerturbationSimulator(smaps, N * N, sampling, 1, img, MotionOperator(N, N, alpha, "rigid", Nz=1))
    R = RealNormalOperator(J)
    v = torch.randn(6, generator=torch.Generator().manual_seed(1), dtype=torch.float64).to(device)
    full = J.adjoint(J.forward(v.to(torch.complex128)))
    assert torch.allclose(R.normal(v), full.real)
    assert not R.normal(v).is_complex()
    assert full.imag.abs().max() > 0          # the discarded part is not zero in general

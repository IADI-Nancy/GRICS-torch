"""Rigid Jacobian against finite differences of the forward model, with and without the rotation factor."""
import torch

from conftest import smooth_phantom
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator

N = 64


def _jacobian_error(device, phi, direction, rotate_gradients, eps=1e-4):
    img = smooth_phantom(N, N, device)
    smaps = torch.ones((1, N, N, 1), device=device, dtype=torch.complex128)
    sampling = [[torch.arange(N * N, device=device)]]
    alpha = torch.tensor([[0.0], [0.0], [phi]], device=device)

    def forward(a):
        return EncodingOperator(smaps, N * N, sampling, 1, MotionOperator(N, N, a, "rigid", Nz=1)).forward(img.flatten())

    d = torch.zeros((3, 1), device=device)
    d[direction, 0] = 1.0
    fd = (forward(alpha + eps * d) - forward(alpha - eps * d)) / (2 * eps)
    J = MotionPerturbationSimulator(smaps, N * N, sampling, 1, img, MotionOperator(N, N, alpha, "rigid", Nz=1),
                                    rotate_gradients=rotate_gradients)
    Jd = J.forward(d.flatten().to(torch.complex128))
    return ((Jd - fd).norm() / fd.norm()).item()


def test_rotation_factor_removes_angle_dependence(device):
    for direction in range(3):
        fixed = [_jacobian_error(device, phi, direction, True) for phi in (0.1, 0.3, 0.6)]
        upstream = [_jacobian_error(device, phi, direction, False) for phi in (0.1, 0.3, 0.6)]
        assert max(fixed) < 0.15, fixed                     # about 11 % (interpolation), flat in the angle
        assert upstream[2] > 2 * fixed[2], (upstream, fixed)  # grows with the angle without the factor


def test_both_paths_agree_at_zero_rotation(device):
    for direction in range(3):
        assert abs(_jacobian_error(device, 0.0, direction, True) - _jacobian_error(device, 0.0, direction, False)) < 1e-12


def test_adjoint_matches_forward_with_rotation(device):
    img = smooth_phantom(N, N, device)
    smaps = torch.ones((1, N, N, 1), device=device, dtype=torch.complex128)
    sampling = [[torch.arange(N * N, device=device)]]
    alpha = torch.tensor([[0.7], [-0.3], [0.3]], device=device)
    J = MotionPerturbationSimulator(smaps, N * N, sampling, 1, img, MotionOperator(N, N, alpha, "rigid", Nz=1))
    g = torch.Generator().manual_seed(3)
    d = torch.randn(3, generator=g, dtype=torch.float64).to(device).to(torch.complex128)
    y = (torch.randn(N * N, generator=g, dtype=torch.float64) + 1j * torch.randn(N * N, generator=g, dtype=torch.float64)).to(device)
    lhs = torch.vdot(J.forward(d), y)
    rhs = torch.vdot(d, J.adjoint(y))
    assert abs(lhs - rhs) < 1e-10 * abs(lhs)

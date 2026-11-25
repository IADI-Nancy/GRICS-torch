"""Rigid warps by k-space phase ramps and Fourier shears (rigid_interpolation = "fourier")."""
import math

import pytest
import torch

from conftest import smooth_phantom
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.runtime.runtime_config import load_config

CASES = [((128, 112), [1.3, -0.7, 3.0]), ((96, 80), [-2.2, 1.1, -12.0]),
         ((64, 56, 48), [1.2, -0.8, 0.6, 2.0, -3.0, 4.0]), ((64, 56, 48), [-0.4, 1.5, -1.1, -6.0, 5.0, -3.0])]


def _operator(shape, alpha, device, **kwargs):
    """Rigid operator of one state; alpha = translations in voxels, then rotations in degrees."""
    d = len(shape)
    values = torch.tensor([*alpha[:d], *(math.radians(v) for v in alpha[d:])], device=device).reshape(-1, 1)
    return MotionOperator(shape[0], shape[1], values, "rigid", Nz=shape[2] if d == 3 else 1, **kwargs)


def _gaussian(points, shape):
    """Gaussian blob near the centre, below 1e-12 at the borders before and after the motions of CASES."""
    return torch.exp(-sum(((p - 0.53 * n) / (0.05 * n)) ** 2 for p, n in zip(points, shape)) / 2)


def _random(n, device, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(n, generator=g), torch.randn(n, generator=g)).to(device)


@pytest.mark.parametrize("shape, alpha", CASES)
def test_warp_is_the_rigid_transform_of_motion_operator(device, shape, alpha):
    # warped(r) = rho(c + R (r - c) + t) in closed form for a band-limited image.
    fourier = _operator(shape, alpha, device, interpolation="fourier")
    grid = torch.meshgrid(*[torch.arange(n, device=device, dtype=torch.float64) for n in shape], indexing="ij")
    R, c, t = fourier.rigid_rotation_matrix(0), [n / 2 for n in shape], alpha[:len(shape)]
    mapped = [sum(R[i, j] * (grid[j] - c[j]) for j in range(len(shape))) + c[i] + t[i] for i in range(len(shape))]
    image, truth = _gaussian(grid, shape).to(torch.complex128).flatten(), _gaussian(mapped, shape).flatten()
    assert (fourier.warp(image, 0) - truth).abs().max() < 1e-11
    assert (_operator(shape, alpha, device).warp(image, 0) - truth).abs().max() > 1e-3     # linear interpolation error


@pytest.mark.parametrize("shape, alpha", CASES + [((81, 97), [0.6, -1.7, 7.0]), ((40, 36, 30), [0.3, 0.9, -0.5, 3.0, 1.0, -2.0])])
def test_adjoint(device, shape, alpha):
    fourier, n = _operator(shape, alpha, device, interpolation="fourier"), math.prod(shape)
    x, y = _random(n, device, 1), _random(n, device, 2)
    lhs, rhs = torch.vdot(y, fourier.warp(x, 0)), torch.vdot(fourier.warp_T(y, 0), x)
    assert abs(lhs - rhs) < 1e-12 * abs(lhs)


@pytest.mark.parametrize("shape, translation", [((128, 112), [3.0, -2.0]), ((48, 40, 32), [-2.0, 1.0, 3.0])])
def test_integer_translation_equals_linear_interpolation(device, shape, translation):
    grid = torch.meshgrid(*[torch.arange(n, device=device, dtype=torch.float64) for n in shape], indexing="ij")
    image = _gaussian(grid, shape).to(torch.complex128).flatten()
    alpha = translation + [0.0] * (1 if len(shape) == 2 else 3)
    linear, fourier = _operator(shape, alpha, device), _operator(shape, alpha, device, interpolation="fourier")
    assert (linear.warp(image, 0) - fourier.warp(image, 0)).abs().max() < 1e-12


def test_single_precision(device):
    shape, alpha = CASES[2]
    x = _random(math.prod(shape), device, 3)
    double = _operator(shape, alpha, device, interpolation="fourier").warp(x, 0)
    single = _operator(shape, alpha, device, interpolation="fourier", compute_dtype=torch.complex64).warp(x.to(torch.complex64), 0)
    assert single.dtype == torch.complex64 and (single.to(torch.complex128) - double).norm() < 1e-5 * double.norm()


def test_encoding_and_jacobian_with_fourier_warps(device):
    Nx, Ny, C, T = 64, 56, 3, 4
    g = torch.Generator().manual_seed(4)
    smaps = torch.complex(torch.randn(C, Nx, Ny, 1, generator=g), torch.randn(C, Nx, Ny, 1, generator=g)).to(device)
    rows = torch.randperm(Ny, generator=g).to(device)
    sampling = [[(torch.arange(Nx, device=device)[:, None] * Ny + rows[t::T][None, :]).reshape(-1) for t in range(T)]]
    alpha = (torch.randn(3, T, generator=g) * torch.tensor([[1.5], [1.5], [math.radians(2.0)]])).to(device)
    motion = MotionOperator(Nx, Ny, alpha, "rigid", interpolation="fourier")
    E = EncodingOperator(smaps, Nx * Ny, sampling, 1, motion)
    x, y = _random(Nx * Ny, device, 5), _random(C * Nx * Ny, device, 6)
    assert abs(torch.vdot(y, E.forward(x)) - torch.vdot(E.adjoint(y), x)) < 1e-12 * abs(torch.vdot(y, E.forward(x)))
    assert (E.normal(x) - E.adjoint(E.forward(x))).norm() < 1e-12 * E.normal(x).norm()
    J = MotionPerturbationSimulator(smaps, Nx * Ny, sampling, 1, _random(Nx * Ny, device, 7).reshape(Nx, Ny), motion)
    d = _random(3 * T, device, 8)
    assert abs(torch.vdot(y, J.forward(d)) - torch.vdot(J.adjoint(y), d)) < 1e-10 * abs(torch.vdot(y, J.forward(d)))


def test_jacobian_against_finite_differences_of_the_fourier_model(device):
    # Linear interpolation leaves 11 % (test_jacobian_rotation); with Fourier warps only the central differences remain.
    N = 64
    img = smooth_phantom(N, N, device)
    smaps = torch.ones((1, N, N, 1), device=device, dtype=torch.complex128)
    sampling = [[torch.arange(N * N, device=device)]]

    def forward(a):
        return EncodingOperator(smaps, N * N, sampling, 1, MotionOperator(N, N, a, "rigid", interpolation="fourier")).forward(img.flatten())

    for phi in (0.1, 0.6):
        alpha = torch.tensor([[0.0], [0.0], [phi]], device=device)
        J = MotionPerturbationSimulator(smaps, N * N, sampling, 1, img, MotionOperator(N, N, alpha, "rigid", interpolation="fourier"))
        for direction in range(3):
            d = torch.zeros((3, 1), device=device)
            d[direction, 0] = 1.0
            fd = (forward(alpha + 1e-4 * d) - forward(alpha - 1e-4 * d)) / 2e-4
            assert (J.forward(d.flatten().to(torch.complex128)) - fd).norm() < 0.07 * fd.norm()


def _config(reconstruction, motion, **overrides):
    return load_config(data_type="shepp-logan", reconstruction_config=f"config/reconstruction/{reconstruction}.toml",
                       sampling_config="config/sampling_simulation/random.toml",
                       motion_simulation_config=f"config/motion_simulation/{motion}.toml", overrides=overrides)


def test_configuration():
    params = _config("rigid_3d", "rigid_per_shot_3d")
    assert (params.rigid_interpolation, params.rigid_motion_damping) == ("linear", "diagonal")
    assert _config("rigid_2d", "rigid_per_readout_2d", rigid_interpolation="fourier").rigid_interpolation == "fourier"
    with pytest.raises(ValueError, match="gather"):
        _config("rigid_2d", "rigid_per_readout_2d", rigid_interpolation="fourier", warp_backend="gather")
    with pytest.raises(ValueError, match="only valid for rigid"):
        _config("nonrigid_2d", "nonrigid_per_readout_2d", rigid_interpolation="fourier")
    with pytest.raises(ValueError):
        MotionOperator(8, 8, torch.zeros(2, 8, 8, 1), "non-rigid", motion_signal=torch.ones(1, 1), interpolation="fourier")

"""E^H E must equal adjoint(forward) whether or not the sampled lines are complete."""
import torch

from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.preprocessing.MotionSamplingPreparer import MotionSamplingPreparer


def _problem(device, Nx=24, Ny=20, Nc=3, Nm=3):
    g = torch.Generator().manual_seed(0)
    smaps = (torch.randn(Nc, Nx, Ny, 1, generator=g) + 1j * torch.randn(Nc, Nx, Ny, 1, generator=g)).to(device)
    chunks = torch.tensor_split(torch.randperm(Ny, generator=g), Nm)
    sampling = MotionSamplingPreparer.build_sampling_per_nex_per_motion([[c.to(device) for c in chunks]], device, Nx, Ny, Nz=1)
    alpha = (0.5 * torch.randn(3, Nm, generator=g)).to(device)
    alpha[2] *= 0.1
    x = (torch.randn(Nx * Ny, generator=g) + 1j * torch.randn(Nx * Ny, generator=g)).to(device)
    return smaps, sampling, MotionOperator(Nx, Ny, alpha, "rigid", Nz=1), x, Ny


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_complete_lines_use_the_fast_path(device):
    smaps, sampling, mo, x, Ny = _problem(device)
    E = EncodingOperator(smaps, x.numel(), sampling, 1, mo)
    assert E._complete_lines
    assert _rel(E.normal(x), E.adjoint(E.forward(x))) < 1e-12


def test_incomplete_lines_fall_back_to_adjoint_forward(device, capsys):
    smaps, sampling, mo, x, Ny = _problem(device)
    s1 = sampling[0][1]
    broken = [[sampling[0][0], s1[s1 >= Ny], sampling[0][2]]]   # state 1 loses its kx = 0 samples
    E = EncodingOperator(smaps, x.numel(), broken, 1, mo)
    assert not E._complete_lines
    assert "not complete readout lines" in capsys.readouterr().out
    assert _rel(E.normal(x), E.adjoint(E.forward(x))) < 1e-12

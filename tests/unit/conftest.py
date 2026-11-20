import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smooth_phantom(nx, ny, device, seed=11, cutoff=0.15):
    """Band-limited complex image: a few gaussians with a smooth phase, low-passed."""
    g = torch.Generator().manual_seed(seed)
    x, y = torch.meshgrid(torch.arange(nx, device=device, dtype=torch.float64),
                          torch.arange(ny, device=device, dtype=torch.float64), indexing="ij")
    mag = torch.zeros((nx, ny), device=device)
    for _ in range(4):
        cx = (0.2 + 0.6 * torch.rand(1, generator=g).item()) * nx
        cy = (0.2 + 0.6 * torch.rand(1, generator=g).item()) * ny
        mag = mag + (0.5 + torch.rand(1, generator=g).item()) * torch.exp(-((x - cx) / (0.12 * nx)) ** 2 - ((y - cy) / (0.12 * ny)) ** 2)
    noise = torch.randn((nx, ny), generator=g, dtype=torch.float64).to(device)
    fx, fy = torch.meshgrid(torch.fft.fftfreq(nx, device=device), torch.fft.fftfreq(ny, device=device), indexing="ij")
    phase = torch.fft.ifftn(torch.fft.fftn(noise) * torch.exp(-(fx ** 2 + fy ** 2) / (2 * 0.12 ** 2))).real
    img = mag * torch.exp(0.5j * phase / phase.abs().max())
    return torch.fft.ifftn(torch.fft.fftn(img) * ((fx ** 2 + fy ** 2) < cutoff ** 2)).to(torch.complex128)

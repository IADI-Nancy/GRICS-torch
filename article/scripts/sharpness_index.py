"""Sharpness index ported from origin/article:article/sharpness_index.py.

Original Python implementation: Nora VOGT, following Lionel Moisan's
https://helios2.mi.parisdescartes.fr/~moisan/sharpness/sharpness_index.m
Higher values indicate greater sharpness. The default preprocessing removes
boundary discontinuities and applies a half-pixel translation (dequantization).
"""

import math

import torch


def sharpness_index(image, pmode=3):
    """Return a scalar sharpness index for one finite 2D image.

    Complex images are evaluated by magnitude. Modes 0/1/2/3 select no
    preprocessing / periodic decomposition / dequantization / both.
    Constant images return zero. Computation uses float64 on the input device.
    """
    image = torch.as_tensor(image)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError('Sharpness index requires a 2D image with both dimensions >= 2.')
    if pmode not in (0, 1, 2, 3):
        raise ValueError('pmode must be 0, 1, 2 or 3.')
    if not torch.isfinite(image).all():
        raise ValueError('Sharpness index requires finite image values.')
    u = (image.abs() if image.is_complex() else image).double()
    if u.max() == u.min():
        return u.new_zeros(())
    if pmode in (1, 3):
        u, _ = perdecomp(u)
    if pmode in (2, 3):
        u = dequant(u)
    ny, nx = u.shape
    gx = torch.roll(u, -1, dims=1) - u
    gy = torch.roll(u, -1, dims=0) - u
    fgx, fgy = torch.fft.fft2(gx), torch.fft.fft2(gy)
    tv = (gx.abs() + gy.abs()).sum()
    gxx = torch.fft.ifft2(fgx * fgx.conj()).real
    gyy = torch.fft.ifft2(fgy * fgy.conj()).real
    gxy = torch.fft.ifft2(fgx * fgy.conj()).real
    axx, ayy = gxx[0, 0].clamp_min(0), gyy[0, 0].clamp_min(0)

    def omega(t):
        # Roundoff can put normalized correlations just outside [-1, 1].
        t = t.clamp(-1, 1)
        return t * t.asin() + (1 - t.square()).clamp_min(0).sqrt() - 1

    variance = u.new_zeros(())
    if axx > 0:
        variance = variance + axx * omega(gxx / axx).sum()
    if ayy > 0:
        variance = variance + ayy * omega(gyy / ayy).sum()
    axy = (axx * ayy).sqrt()
    if axy > 0:
        variance = variance + 2 * axy * omega(gxy / axy).sum()
    variance = variance * 2 / math.pi
    if variance <= 0:
        return u.new_zeros(())
    t = ((axx.sqrt() + ayy.sqrt()) * math.sqrt(2 * nx * ny / math.pi) - tv) / variance.sqrt()
    # Equivalent to -log(erfc(t / sqrt(2)) / 2) / log(10), stable in the tails.
    return -torch.special.log_ndtr(-t) / math.log(10)


def perdecomp(u):
    """Split an image into periodic and smooth components (article algorithm)."""
    ny, nx = u.shape
    v = torch.zeros_like(u)
    v[0, :] = u[0, :] - u[-1, :]
    v[-1, :] = -v[0, :]
    v[:, 0] += u[:, 0] - u[:, -1]
    v[:, -1] -= u[:, 0] - u[:, -1]
    fx = torch.cos(2 * math.pi * torch.arange(nx, device=u.device, dtype=u.dtype) / nx)
    fy = torch.cos(2 * math.pi * torch.arange(ny, device=u.device, dtype=u.dtype) / ny)
    denominator = 2 - fy[:, None] - fx[None, :]
    denominator[0, 0] = 1
    smooth = torch.fft.ifft2(torch.fft.fft2(v) * 0.5 / denominator).real
    return u - smooth, smooth


def dequant(u):
    """Translate the periodic image by half a pixel along each axis."""
    ny, nx = u.shape
    tx = torch.exp(-1j * math.pi * torch.fft.fftfreq(nx, device=u.device, dtype=u.dtype))
    ty = torch.exp(-1j * math.pi * torch.fft.fftfreq(ny, device=u.device, dtype=u.dtype))
    return torch.fft.ifft2(torch.fft.fft2(u) * ty[:, None] * tx[None, :]).real

# Author : Nora VOGT
import torch
import torch.fft as fft
import numpy as np

# python implementation of: https://helios2.mi.parisdescartes.fr/~moisan/sharpness/sharpness_index.m
def sharpness_index(u, pmode=3):
    u = u.double()
    # print(u[0,0:2].numpy())
    if pmode == 1 or pmode == 3:
        u, _ = perdecomp(u)
        # print(u[0,0:2].numpy())
    if pmode == 2 or pmode == 3:
        u = dequant(u)
        # print(u[0,0:2].numpy())

    ny, nx = u.size()

    gx = torch.cat([u[:, 1:], u[:, 0].unsqueeze(1)], dim=1) - u
    fgx = fft.fft2(gx)
    
    gy = torch.cat([u[1:, :], u[0].unsqueeze(0)], dim=0) - u
    fgy = fft.fft2(gy)

    tv = torch.sum(torch.abs(gx) + torch.abs(gy))
    # print(tv.numpy())
    Gxx = torch.real(fft.ifft2(fgx * torch.conj(fgx)))
    Gyy = torch.real(fft.ifft2(fgy * torch.conj(fgy)))
    Gxy = torch.real(fft.ifft2(fgx * torch.conj(fgy)))

    oomega = lambda t: torch.real(t * torch.asin(t) + torch.sqrt(1 - t**2) - 1)

    var = 0
    axx = Gxx[0, 0]
    if axx > 0:
        var += axx * torch.sum(torch.sum(oomega(Gxx / axx)))

    # print(var.numpy())
    ayy = Gyy[0, 0]
    if ayy > 0:
        var += ayy * torch.sum(torch.sum(oomega(Gyy / ayy)))

    # print(var.numpy())
    axy = torch.sqrt(axx * ayy)
    if axy > 0:
        var += 2 * axy * torch.sum(torch.sum(oomega(Gxy / axy)))

    # print(var.numpy())
    var = var * 2 / torch.pi
    # print(var.numpy())

    if var > 0:
        t = ((torch.sqrt(axx) + torch.sqrt(ayy)) * np.sqrt(2 * nx * ny / np.pi) - tv) / torch.sqrt(var)
        # print("t", t / np.sqrt(2))
        # print("logerfc", logerfc(t / np.sqrt(2)))
        si = -logerfc(t / np.sqrt(2)) / torch.log(torch.tensor(10.0)) + np.log10(2)
    else:
        si = torch.tensor(0.0)

    return si

def perdecomp(u):
    ny, nx = u.size()
    X = torch.arange(1, nx+1)
    Y = torch.arange(1, ny+1)
    v = torch.zeros(ny, nx, dtype=u.dtype)
    v[0, X-1] = u[0, X-1] - u[ny-1, X-1]
    v[ny-1, X-1] = -v[0, X-1]
    v[Y-1, 0] = v[Y-1, 0] + u[Y-1, 0] - u[Y-1, nx-1]
    v[Y-1, nx-1] = v[Y-1, nx-1] - u[Y-1, 0] + u[Y-1, nx-1]
    fx = torch.cos(2. * torch.pi * (X.double() - 1) / nx).repeat(ny, 1)
    # fy = torch.cos(2. * torch.pi * (Y.double().conj() - 1) / ny).unsqueeze(1).repeat(1, nx)
    fy = torch.cos(2. * torch.pi * (Y.double() - 1) / ny).unsqueeze(1).repeat(1, nx)

    fx[0, 0] = 0.  # avoid division by 0 in the line below
    s = torch.fft.ifft2(torch.fft.fft2(v) * 0.5 / (2. - fx - fy)).real
    p = u - s
    return p, s

def dequant(u):
    ny, nx = u.size()
    mx = np.floor(nx / 2)
    my = np.floor(ny / 2)

    Tx = torch.exp(-1j * torch.pi / nx * (torch.fmod(torch.arange(mx, mx + nx), nx) - mx))
    Ty = torch.exp(-1j * torch.pi / ny * (torch.fmod(torch.arange(my, my + ny), ny) - my))

    v = (torch.fft.ifft2(torch.fft.fft2(u) * torch.outer(Ty, Tx))).real

    return v

def logerfc(x):
    y = x.clone()
    T = (x > 20)

    if torch.sum(T) > 0:
        X = x[T]
        z = X.pow(-2)
        s = torch.ones_like(X)
        for k in range(8, 0, -1): #8,..,1
            s = 1 - (k - 0.5) * z * s
        y[T] = -0.5 * torch.log(torch.tensor(torch.pi)) - X.pow(2) + torch.log(s / X)

    # erf_tmp = torch.erfc(x[~T]) 
    # print("log ",  torch.log(erf_tmp))
    y[~T] =  torch.log(torch.erfc(x[~T])) 

    return y

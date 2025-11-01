import numpy as np
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def translate(img_cplx,
             t=(0.0, 0.0, 0.0),   # shift in *pixels*: (tz, ty, tx) or (dz, dy, dx)
             order=1,             # 0=nearest, 1=trilinear; 3 maps to trilinear for 3D
             device=None):
    """ do no use this function the kspace version does work better"""
    # Ensure device consistency
    if device is None:
        device = img_cplx.device

    # Pack real/imag as channels -> (1, 2, D, H, W)
    img_ri = img_cplx.real.unsqueeze(0).unsqueeze(0)
    img_imag = img_cplx.imag.unsqueeze(0).unsqueeze(0)

    D, H, W = img_ri.shape[-3:]

    # Convert pixel shifts to normalized coords (align_corners=True)
    # Note: t = (dz, dy, dx)
    t = torch.tensor(t, dtype=torch.float32, device=device)
    tz_n = (2.0 * t[0]) / (D - 1) if D > 1 else 0.0
    ty_n = (2.0 * t[1]) / (H - 1) if H > 1 else 0.0
    tx_n = (2.0 * t[2]) / (W - 1) if W > 1 else 0.0

    # Build 3x4 affine: [ I | t_norm ]
    theta = torch.eye(3, 4, dtype=torch.float32, device=device).unsqueeze(0)
    theta[0, 0, 3] = tx_n
    theta[0, 1, 3] = ty_n
    theta[0, 2, 3] = tz_n

    # Grid & sample
    grid = F.affine_grid(theta, img_ri.shape, align_corners=False)

    # 3D supports 'nearest' and 'bilinear' (trilinear). Map order accordingly.
    if order == 0:
        mode = "nearest"
    else:
        mode = "bilinear"   # use trilinear for 3D; 'bicubic' is not supported for 5D

    img_warp_ri = F.grid_sample(
        img_ri, grid, mode=mode, padding_mode="reflection", align_corners=False
    ).squeeze(0).squeeze(0)
    img_warp_imag = F.grid_sample(
        img_imag, grid, mode=mode, padding_mode="reflection", align_corners=False
    ).squeeze(0).squeeze(0)
    # Back to complex (D, H, W)
    img_warp = torch.complex(img_warp_ri, img_warp_imag).to(img_cplx.dtype)
    return img_warp  # (D, H, W) complex

################## Matrix helper functions ##################

def translate_cplximg(
        img_cplx: torch.Tensor,
        t):
    """ t : tuple of 3 floats (tz, ty, tx) in pixels"""
    device = img_cplx.device
    *_, D, H, W = img_cplx.shape[-3:]
    tz, ty, tx = torch.as_tensor(t, dtype=torch.complex64, device=device)
    
    # --- calculate the phase shift factor ---
    kz = torch.fft.fftfreq(D, d=1.0, device=device).view(D,1,1)   # cycles/voxel
    ky = torch.fft.fftfreq(H, d=1.0, device=device).view(1,H,1)
    kx = torch.fft.fftfreq(W, d=1.0, device=device).view(1,1,W)

    phase = torch.exp(-2j * torch.pi * (tx*kx + ty*ky + tz*kz))

    shifted_cplximg = img_cplx * phase
    return shifted_cplximg


def rotate_cplximg(img_cplx, p,
               order=1,    # 0=nearest, 1=bilinear, 3=bicubic
               device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Rotate a 3-D (φ, θ, ψ) NumPy array with PyTorch.

    Parameters
    ----------
    img_cplx : torch.Tensor (C, D, H, W) k space or image space
    p        : iterable of 3 floats (φ, θ, ψ)
    order    : int, interpolation order (0=nearest, 1=bilinear, 3=bicubic)
    device   : str, device to use ('cuda' or 'cpu')
    Returns
    -------
    torch.Tensor  –  rotated image, complex Tensor with the same shape as img_cplx 
    """
    img_ri = torch.stack((img_cplx.real, img_cplx.imag), dim=0)
    img_ri = img_ri.unsqueeze(0)  # add batch dimension → (1, 2, D, H, W)

    D, H, W = img_ri.shape[-3:]

    # Convert degrees to radians for the rotation parameters and build the rotation matrix.
    p = torch.tensor(p, dtype=torch.float32, device=device)
    p = torch.deg2rad(p)
    A = _build_rotation_matrix(p)
    
    theta = torch.zeros(3, 4, device=device)
    theta[:, :3] = A
    theta = theta.unsqueeze(0)

    grid = F.affine_grid(theta, img_ri.shape, align_corners=True)
    mode = {0: "nearest", 1: "bilinear", 3: "bicubic"}[order]
    img_warp = F.grid_sample(img_ri, grid, mode=mode, padding_mode="zeros", align_corners=True).squeeze(0)

    # Convert back to complex image
    img_warp = torch.complex(img_warp[0], img_warp[1])  

    return img_warp  # shape: (C, D, H, W)


################## Matrix helper functions ##################


def _build_rotation_matrix(p):
    """ Build a 3 × 3 rotation matrix from the 3 Euler angles (φ, θ, ψ) in radians. """
    # p = (t_x, t_y, t_z,
    (phi, theta, psi) = p
    Rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(phi), -torch.sin(phi)],
                       [0, torch.sin(phi),  torch.cos(phi)]])
    Ry = torch.tensor([[ torch.cos(theta), 0, torch.sin(theta)],
                       [0,                1, 0],
                       [-torch.sin(theta), 0, torch.cos(theta)]])
    Rz = torch.tensor([[torch.cos(psi), -torch.sin(psi), 0],
                       [torch.sin(psi),  torch.cos(psi), 0],
                       [0,               0,              1]])
    return Rx @ Ry @ Rz

# if needed later on for now there is no use
def _build_affine_matrix(p):
    """
    Build a 3 × 3 linear part A = R_x R_y R_z G S from the 12 parameters.
    """
    (t_x, t_y, t_z,
     phi, theta, psi,
     G_xy, G_xz, G_yz,
     S_x, S_y, S_z) = p

    R = _build_rotation_matrix(p[3:6])  # φ, θ, ψ

    # --- shear & scale ------------------------------------------------------
    G  = torch.tensor([[1,  G_xy, G_xz],
                    [0,     1, G_yz],
                    [0,     0,    1]])
    S  = torch.diag(torch.tensor([S_x, S_y, S_z]))

    return R @ G @ S  


def _build_affine_matrix_kspace(p):
    (t_x, t_y, t_z,
     phi, theta, psi,
     G_xy, G_xz, G_yz,
     S_x, S_y, S_z) = p

    R = _build_rotation_matrix(p[3:6])  # φ, θ, ψ

    # --- shear & scale ------------------------------------------------------
    G  = torch.tensor([[1,  0, 0],
                       [-G_xy,     1, 0],
                       [G_xy * G_yz - G_xz,     -G_yz,    1]])

    S  = torch.diag(torch.tensor([S_x, S_y, S_z]))

    return R @ G @ S
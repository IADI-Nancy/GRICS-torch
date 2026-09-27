"""Initialize finer levels from Fourier-resized images and motion."""

import torch

from .fourier_crop import fourier_resize_spatial


def upsample_data(Data_prev, Data_res, params, Nalpha, device):
    img_prev = Data_prev["ReconstructedImage"]
    resize_shape = (
        (Data_res["Nx"], Data_res["Ny"], Data_res["Nz"])
        if int(Data_res.get("Nz", 1)) > 1 else
        (Data_res["Nx"], Data_res["Ny"])
    )
    Data_res["ReconstructedImage"] = fourier_resize_spatial(img_prev, resize_shape)

    mot_prev = Data_prev["MotionModel"]
    if params.reconstruction_motion_type == "rigid":
        Data_res["MotionModel"] = torch.zeros((Nalpha, params.N_motion_states), device=device)
        Data_res["MotionModel"][0,:] = mot_prev[0,:] * Data_res["Nx"] / Data_prev["Nx"]
        Data_res["MotionModel"][1,:] = mot_prev[1,:] * Data_res["Ny"] / Data_prev["Ny"]
        if Nalpha > 3:
            Data_res["MotionModel"][2,:] = mot_prev[2,:] * Data_res.get("Nz", 1) / max(1, Data_prev.get("Nz", 1))
            Data_res["MotionModel"][3:,:] = mot_prev[3:,:]
        else:
            Data_res["MotionModel"][2,:] = mot_prev[2,:]
    else:
        # Motion is [component, x, y, (z), sensor], so the spatial axes do
        # not occupy the final tensor positions.
        motion_spatial_dims = tuple(range(1, 1 + len(resize_shape)))
        mot_res = fourier_resize_spatial(
            mot_prev, resize_shape, spatial_dims=motion_spatial_dims,
        ).real.to(dtype=mot_prev.dtype)
        mot_res[0] = mot_res[0] * Data_res["Nx"] / Data_prev["Nx"]
        mot_res[1] = mot_res[1] * Data_res["Ny"] / Data_prev["Ny"]
        if mot_res.shape[0] > 2 and int(Data_res.get("Nz", 1)) > 1:
            mot_res[2] = mot_res[2] * Data_res["Nz"] / max(1, Data_prev["Nz"])
        Data_res["MotionModel"] = mot_res

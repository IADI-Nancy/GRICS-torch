"""Initialize finer levels from reconstructed images and scaled motion."""

import torch

from .resize import resize_img_xy


def upsample_data(Data_prev, Data_res, params, Nalpha, device):
    img_prev = Data_prev["ReconstructedImage"]
    resize_shape = (
        (Data_res["Nx"], Data_res["Ny"], Data_res["Nz"])
        if int(Data_res.get("Nz", 1)) > 1 else
        (Data_res["Nx"], Data_res["Ny"])
    )
    img_res = resize_img_xy(img_prev, resize_shape)
    Data_res["ReconstructedImage"] = img_res

    mot_prev = Data_prev["MotionModel"]
    if params.reconstruction_motion_type == "rigid":
        Data_res["MotionModel"] = torch.zeros((Nalpha, params.N_motion_states), device=device)
        Data_res["MotionModel"][0,:] = mot_prev[0,:] * Data_res["Nx"] / Data_prev["Nx"]  # scale translations
        Data_res["MotionModel"][1,:] = mot_prev[1,:] * Data_res["Ny"] / Data_prev["Ny"]  # scale translations
        if Nalpha > 3:
            Data_res["MotionModel"][2,:] = mot_prev[2,:] * Data_res.get("Nz", 1) / max(1, Data_prev.get("Nz", 1))
            Data_res["MotionModel"][3:,:] = mot_prev[3:,:]
        else:
            Data_res["MotionModel"][2,:] = mot_prev[2,:]  # rotations remain the same
    else:
        resize_shape = (
            (Data_res["Nx"], Data_res["Ny"], Data_res["Nz"])
            if int(Data_res.get("Nz", 1)) > 1 else
            (Data_res["Nx"], Data_res["Ny"])
        )
        mot_res = resize_img_xy(mot_prev, resize_shape)
        mot_res[0] = mot_res[0] * Data_res["Nx"] / Data_prev["Nx"]
        mot_res[1] = mot_res[1] * Data_res["Ny"] / Data_prev["Ny"]
        if mot_res.shape[0] > 2 and int(Data_res.get("Nz", 1)) > 1:
            mot_res[2] = mot_res[2] * Data_res["Nz"] / max(1, Data_prev["Nz"])
        Data_res["MotionModel"] = mot_res

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
        # The full bank preserves state IDs and initial transforms for newly visible states.
        mot_prev = Data_prev["RigidMotionModelFull"]
        motion = mot_prev.clone().to(device)
        motion[0,:] *= Data_res["Nx"] / Data_prev["Nx"]
        motion[1,:] *= Data_res["Ny"] / Data_prev["Ny"]
        if Nalpha > 3:
            motion[2,:] *= Data_res.get("Nz", 1) / max(1, Data_prev.get("Nz", 1))
        Data_res["RigidMotionModelFull"] = motion
        Data_res["MotionModel"] = motion[:, Data_res["MotionStateIndices"]]
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

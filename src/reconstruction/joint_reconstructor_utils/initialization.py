"""Apply optional starting image and motion estimates at the coarsest level."""

import torch

from .resampling import resize_img_xy


def apply_external_initializer(
    data, *, initial_image, initial_motion, params, device, Nalpha, Nphysio,
):
    """Update data in place from supplied estimates, keeping defaults if absent.

    Images and non-rigid fields are interpolated to the level's spatial grid.
    Motion values are used as supplied, without displacement rescaling.
    """
    spatial = ((data["Nx"], data["Ny"], data["Nz"]) if int(data.get("Nz", 1)) > 1 else (data["Nx"], data["Ny"]))
    if initial_image is not None:
        image = torch.as_tensor(initial_image, device=device)
        # Accept an omitted acquisition axis or share one image across Nex.
        if image.ndim == len(spatial):
            image = image.unsqueeze(0)
        if image.shape[0] == 1 and params.Nex > 1:
            image = image.expand(params.Nex, *image.shape[1:])
        if image.ndim != len(spatial) + 1 or image.shape[0] != params.Nex:
            raise ValueError(f"Invalid initial_image shape {tuple(image.shape)}.")
        data["ReconstructedImage"] = resize_img_xy(image.to(torch.complex128), spatial)
    if initial_motion is not None:
        motion = torch.as_tensor(initial_motion, device=device, dtype=torch.float64)
        if params.reconstruction_motion_type == "non-rigid":
            # A field without a sensor axis represents one physiological signal.
            if motion.ndim == len(spatial) + 1:
                motion = motion.unsqueeze(-1)
            if motion.shape[0] != Nalpha or motion.shape[-1] != Nphysio:
                raise ValueError(f"Invalid initial_motion shape {tuple(motion.shape)}; expected Nalpha={Nalpha}, Nsensor={Nphysio}.")
            data["MotionModel"] = resize_img_xy(motion, spatial)
        else:
            # Rigid parameters are indexed by motion state, not spatial position.
            expected = (Nalpha, params.N_motion_states)
            if tuple(motion.shape) != expected:
                raise ValueError(f"Rigid initial_motion must have shape {expected}.")
            data["MotionModel"] = motion


def initialize_zero_image_and_motion(Data_res, *, params, device, Nalpha, Nphysio):
    """Create zero image and motion estimates in place for the coarsest level."""
    if int(Data_res.get("Nz", 1)) > 1:
        Data_res["ReconstructedImage"] = torch.zeros(
            (params.Nex, Data_res["Nx"], Data_res["Ny"], Data_res["Nz"]),
            dtype=torch.complex128, device=device)
    else:
        Data_res["ReconstructedImage"] = torch.zeros((params.Nex, Data_res["Nx"], Data_res["Ny"]), dtype=torch.complex128, device=device)

    if params.reconstruction_motion_type == "rigid":
        Data_res["MotionModel"] = torch.zeros((Nalpha, params.N_motion_states), device=device)
    elif params.reconstruction_motion_type == "non-rigid":
        if int(Data_res.get("Nz", 1)) > 1:
            Data_res["MotionModel"] = torch.zeros(
                (Nalpha, Data_res["Nx"], Data_res["Ny"], Data_res["Nz"], Nphysio),
                device=device)
        else:
            Data_res["MotionModel"] = torch.zeros((Nalpha, Data_res["Nx"], Data_res["Ny"], Nphysio), device=device)

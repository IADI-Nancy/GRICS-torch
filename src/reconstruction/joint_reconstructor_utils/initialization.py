"""Apply optional starting image and motion estimates at the coarsest level."""

import torch

from .resampling import resize_img_xy


def apply_external_initializer(
    data, *, initial_image, initial_motion, params, device, Nalpha, Nphysio, full_shape=None,
):
    """Update data in place from supplied estimates, keeping defaults if absent.

    Images and non-rigid fields are interpolated to the level's spatial grid.
    Displacements in initial_motion are in full-resolution voxels; when full_shape is
    given they are rescaled to this level's grid the same way upsample_data does
    between levels (upstream applied them unscaled at the coarsest level).
    """
    spatial = ((data["Nx"], data["Ny"], data["Nz"]) if int(data.get("Nz", 1)) > 1 else (data["Nx"], data["Ny"]))
    ratios = [1.0] * len(spatial) if full_shape is None else [s / f for s, f in zip(spatial, full_shape)]
    if initial_image is not None:
        image = torch.as_tensor(initial_image, device=device)
        if image.ndim != len(spatial):
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
            motion = resize_img_xy(motion, spatial)
            for axis, ratio in enumerate(ratios):
                motion[axis] = motion[axis] * ratio
            data["MotionModel"] = motion
        else:
            # Rigid parameters are indexed by motion state, not spatial position.
            expected = (Nalpha, params.N_motion_states)
            if tuple(motion.shape) != expected:
                raise ValueError(f"Rigid initial_motion must have shape {expected}.")
            motion = motion.clone()
            for axis, ratio in enumerate(ratios):
                motion[axis, :] *= ratio
            data["RigidMotionModelFull"] = motion
            data["MotionModel"] = motion[:, data["MotionStateIndices"]]


def initialize_zero_image_and_motion(Data_res, *, params, device, Nalpha, Nphysio):
    """Create zero image and motion estimates in place for the coarsest level."""
    if int(Data_res.get("Nz", 1)) > 1:
        Data_res["ReconstructedImage"] = torch.zeros(
            (Data_res["Nx"], Data_res["Ny"], Data_res["Nz"]),
            dtype=torch.complex128, device=device)
    else:
        Data_res["ReconstructedImage"] = torch.zeros((Data_res["Nx"], Data_res["Ny"]), dtype=torch.complex128, device=device)

    if params.reconstruction_motion_type == "rigid":
        # Retain initial transforms for states that first appear at finer resolutions.
        Data_res["RigidMotionModelFull"] = torch.zeros((Nalpha, params.N_motion_states), device=device)
        Data_res["MotionModel"] = Data_res["RigidMotionModelFull"][:, Data_res["MotionStateIndices"]]
    elif params.reconstruction_motion_type == "non-rigid":
        if int(Data_res.get("Nz", 1)) > 1:
            Data_res["MotionModel"] = torch.zeros(
                (Nalpha, Data_res["Nx"], Data_res["Ny"], Data_res["Nz"], Nphysio),
                device=device)
        else:
            Data_res["MotionModel"] = torch.zeros((Nalpha, Data_res["Nx"], Data_res["Ny"], Nphysio), device=device)

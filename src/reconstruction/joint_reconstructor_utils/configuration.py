"""Validate configuration used by the joint reconstruction loop."""

import math
import torch


def configure_motion_states_per_resolution_level(params, motion_signal):
    """Return validated motion-state counts, one per resolution level."""
    full_states = params.N_motion_states
    schedule = getattr(params, "_motion_states_per_level", None)
    if schedule is None:
        if type(full_states) is int:
            schedule = [full_states] * len(params.ResolutionLevels)
        elif isinstance(full_states, (list, tuple)):
            schedule = full_states
            full_states = max(schedule, default=0)
        else:
            raise ValueError("N_motion_states must be a positive integer or a list of positive integers.")
    if len(schedule) != len(params.ResolutionLevels):
        raise ValueError(
            f"N_motion_states has {len(schedule)} entries; ResolutionLevels has {len(params.ResolutionLevels)}."
        )
    if any(type(value) is not int or value < 1 for value in schedule):
        raise ValueError("N_motion_states entries must be positive integers.")
    if any(value > full_states for value in schedule):
        raise ValueError("N_motion_states entries cannot exceed the full motion-state count.")
    motion_states_per_level = list(schedule)
    if int(motion_signal.shape[0]) != full_states:
        raise ValueError(f"motion_signal has {motion_signal.shape[0]} states; expected {full_states}.")
    if params.reconstruction_motion_type == "rigid" and any(
        value != full_states for value in motion_states_per_level
    ):
        raise ValueError("Explicit per-level motion-state merging is supported only for non-rigid reconstruction; empty states are removed automatically.")
    return motion_states_per_level


def _parse_gn_iterations_per_level(params, res_levels):
    gn_cfg = params.GN_iterations_per_level
    if type(gn_cfg) is int:
        gn_cfg = [gn_cfg] * len(res_levels)
    if not isinstance(gn_cfg, (list, tuple)):
        raise ValueError("GN_iterations_per_level must be a positive integer or a list of positive integers.")
    if len(gn_cfg) != len(res_levels):
        raise ValueError(f"GN_iterations_per_level has {len(gn_cfg)} entries; ResolutionLevels has {len(res_levels)}.")
    if any(type(value) is not int or value < 1 for value in gn_cfg):
        raise ValueError("GN_iterations_per_level entries must be positive integers.")
    return list(gn_cfg)


def image_regularization_weight_for_level(params, level_index):
    """Select the configured image regularization weight for the current level.

    A scalar applies at every resolution; a list/tuple supplies one weight
    per ResolutionLevels entry. Optional CG scaling is applied later.
    """
    lambda_r = params.lambda_r
    if isinstance(lambda_r, (list, tuple)):
        if len(lambda_r) == 0:
            raise ValueError("lambda_r list/tuple cannot be empty.")
        if len(lambda_r) != len(params.ResolutionLevels):
            raise ValueError(
                "Inconsistent config: "
                f"lambda_r has {len(lambda_r)} values, "
                f"but ResolutionLevels has {len(params.ResolutionLevels)} values."
            )
        return float(lambda_r[level_index])
    return float(lambda_r)


def validate_voxel_spacing(regularization_scaling, voxel_spacing_mm, Nz_full):
    """Validate scaling mode and return full-resolution voxel spacing in mm."""
    if regularization_scaling not in ("direct", "voxel_aware"):
        raise ValueError('regularization_scaling must be "direct" or "voxel_aware".')
    spatial_dims = 3 if Nz_full > 1 else 2
    if regularization_scaling == "voxel_aware":
        if (voxel_spacing_mm is None or len(voxel_spacing_mm) != spatial_dims
                or any(not math.isfinite(float(value)) or float(value) <= 0
                       for value in voxel_spacing_mm)):
            raise ValueError("Voxel-aware regularization scaling requires positive voxel_spacing_mm for each spatial dimension.")
    return tuple(float(value) for value in voxel_spacing_mm) if voxel_spacing_mm is not None else None


def validate_calibration_image_prior(calibration_image_prior, use_calibration_weighted_image, external_image_regularizer, spatial_shape, device):
    """Validate the enabled calibration prior and return a real device tensor."""
    if not use_calibration_weighted_image:
        return None
    if external_image_regularizer is not None:
        raise ValueError("Calibration-weighted images cannot be combined with an external image regularizer.")
    if calibration_image_prior is None:
        raise ValueError("Calibration-weighted image is enabled, but no calibration image was supplied.")
    Nx_full, Ny_full, Nz_full = spatial_shape
    spatial_shape = spatial_shape if Nz_full > 1 else (Nx_full, Ny_full)
    calibration = torch.as_tensor(calibration_image_prior, device=device)
    if Nz_full == 1 and tuple(calibration.shape) == (Nx_full, Ny_full, 1):
        calibration = calibration[..., 0]
    if (tuple(calibration.shape) != spatial_shape or calibration.is_complex()
            or not torch.isfinite(calibration).all() or torch.any(calibration < 0)):
        raise ValueError(f"Calibration image must be a finite nonnegative real image of shape {spatial_shape}.")
    return calibration.to(torch.float64)


def validate_motion_signal(motion_signal, device):
    """Validate the motion signal and move it to the reconstruction device."""
    if motion_signal is None:
        raise ValueError("motion_signal must be provided.")
    motion_signal = motion_signal.to(device)
    if motion_signal.ndim != 2:
        raise ValueError(
            "motion_signal must have shape [Nstate, Nsensor]. "
            f"Got {tuple(motion_signal.shape)}."
        )
    return motion_signal

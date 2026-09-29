"""Validate configuration used by the joint reconstruction loop."""


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
        raise ValueError("Per-level motion-state reduction is supported only for non-rigid reconstruction.")
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

"""Normalize physiological signals over the complete acquisition."""

import torch


def normalize_physiological_motion_input(motion, *, allow_constant=False):
    """Return normalized channels, acquisition means and population deviations.

    Compute (signal - mean) / sqrt(mean((signal - mean)**2)) independently
    for each synchronized physiological channel across all MRI readouts, before
    slice selection and motion-state binning. Input must be [readout, sensor].
    Population variance uses N samples as its denominator, without correction.
    Constant sensors are centered to zero only when allow_constant is true
    for single-state reconstruction; otherwise they are rejected. Non-finite
    values are always rejected. The input is not modified.
    """
    signal = torch.as_tensor(motion)
    if not torch.is_floating_point(signal):
        signal = signal.to(torch.float64)
    if signal.ndim != 2:
        raise ValueError("Physiological motion input must be [readout, sensor].")
    axes = (0,)
    if not torch.isfinite(signal).all():
        raise ValueError("Motion input contains non-finite values.")
    mean = signal.mean(dim=axes, keepdim=True)
    std = signal.std(dim=axes, correction=0, keepdim=True)
    if not torch.isfinite(std).all() or (not allow_constant and torch.any(std <= 0)):
        raise ValueError("Cannot normalize a constant or non-finite motion sensor.")
    scale = torch.where(std > 0, std, torch.ones_like(std))
    return (signal - mean) / scale, mean.squeeze(), std.squeeze()

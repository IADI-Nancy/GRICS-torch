"""Normalize physiological signals over the complete acquisition."""

import torch


def normalize_physiological_motion_input(motion):
    """Return normalized channels, acquisition means and population deviations.

    Compute (signal - mean) / sqrt(mean((signal - mean)**2)) independently
    for each synchronized physiological channel across all MRI readouts, before
    slice selection and motion-state binning. Input must be [readout, sensor].
    Population variance uses N samples as its denominator, without correction.
    Constant sensors and non-finite values are rejected because they cannot
    produce finite signals with unit variance. The input is not modified.
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
    if not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("Cannot normalize a constant or non-finite motion sensor.")
    return (signal - mean) / std, mean.squeeze(), std.squeeze()

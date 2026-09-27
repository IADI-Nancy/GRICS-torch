"""Normalize real-data physiological inputs before per-slice motion binning."""

import torch


def acquisition_zscore_motion_input(motion, *, data_dimension):
    """Center and scale each sensor over the complete acquired readout set.

    For 2D, input is [slice, readout, sensor] (or a sensorless 2D array).
    For 3D, input is [readout, sensor]. Population standard deviation matches
    the zero-mean/unit-variance GRICS++ breast input files.
    """
    signal = torch.as_tensor(motion)
    if not torch.is_floating_point(signal):
        signal = signal.to(torch.float64)
    if data_dimension == "2D":
        if signal.ndim == 3:
            axes = (0, 1)
        elif signal.ndim == 2:
            axes = (0, 1)
        else:
            raise ValueError("2D motion input must be [slice, readout, sensor] or [slice, readout].")
    elif data_dimension == "3D":
        if signal.ndim != 2:
            raise ValueError("3D motion input must be [readout, sensor].")
        axes = (0,)
    else:
        raise ValueError("data_dimension must be '2D' or '3D'.")
    if not torch.isfinite(signal).all():
        raise ValueError("Motion input contains non-finite values.")
    mean = signal.mean(dim=axes, keepdim=True)
    std = signal.std(dim=axes, correction=0, keepdim=True)
    if not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("Cannot normalize a constant or non-finite motion sensor.")
    return (signal - mean) / std, mean.squeeze(), std.squeeze()

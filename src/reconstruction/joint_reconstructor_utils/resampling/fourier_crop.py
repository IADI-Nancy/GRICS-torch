"""Centered Fourier cropping for coarse-resolution spatial inputs."""

import math

import torch

from src.utils.fftnc import fftnc, ifftnc


def fourier_crop_spatial(image: torch.Tensor, new_size: tuple[int, ...]) -> torch.Tensor:
    """Crop the last spatial axes with unitary FFTs and preserve constant amplitude.

    The coarse sensitivity maps and calibration prior use the same central
    frequency window as the measured k-space. A unitary inverse FFT of that
    window changes a constant image by sqrt(N/n); the explicit sqrt(n/N)
    factor restores its original amplitude.
    """
    if len(new_size) not in (2, 3) or image.ndim < len(new_size):
        raise ValueError("Fourier cropping requires two or three spatial axes.")
    spatial_dims = tuple(range(image.ndim - len(new_size), image.ndim))
    old_size = tuple(image.shape[axis] for axis in spatial_dims)
    if any(
        target < 1 or target > source for source, target in zip(old_size, new_size)
    ):
        raise ValueError(f"Invalid spatial Fourier crop from {old_size} to {new_size}.")
    if old_size == tuple(new_size):
        return image

    frequency = fftnc(image, dims=spatial_dims)
    crop = tuple(
        slice(source // 2 - target // 2, source // 2 - target // 2 + target)
        for source, target in zip(old_size, new_size)
    )
    frequency = frequency[(..., *crop)]
    result = ifftnc(frequency, dims=spatial_dims)
    return result * math.sqrt(math.prod(new_size) / math.prod(old_size))

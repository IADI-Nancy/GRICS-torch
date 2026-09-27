"""Centered Fourier resizing for spatial inputs."""

import math

import torch

from src.utils.fftnc import fftnc, ifftnc


def fourier_resize_spatial(
    image: torch.Tensor,
    new_size: tuple[int, ...],
    *,
    spatial_dims: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Resize spatial axes by centered Fourier cropping or zero padding.

    ``new_size`` may be smaller or larger than the selected axes. The common
    centered frequency window is copied and all other frequencies are zero.
    With unitary FFTs, the explicit normalization preserves a constant field.
    ``spatial_dims`` defaults to the last axes; motion fields can supply their
    spatial axes explicitly because their sensor axis is last.
    """
    if len(new_size) not in (2, 3) or image.ndim < len(new_size):
        raise ValueError("Fourier resizing requires two or three spatial axes.")
    if spatial_dims is None:
        spatial_dims = tuple(range(image.ndim - len(new_size), image.ndim))
    if len(spatial_dims) != len(new_size):
        raise ValueError("The number of spatial axes must match new_size.")
    spatial_dims = tuple(axis % image.ndim for axis in spatial_dims)
    if len(set(spatial_dims)) != len(spatial_dims):
        raise ValueError("Spatial axes must be distinct.")

    old_size = tuple(image.shape[axis] for axis in spatial_dims)
    if any(target < 1 for target in new_size):
        raise ValueError(f"Invalid spatial Fourier resize from {old_size} to {new_size}.")
    if old_size == tuple(new_size):
        return image

    frequency = fftnc(image, dims=spatial_dims)
    resized_shape = list(frequency.shape)
    for axis, target in zip(spatial_dims, new_size):
        resized_shape[axis] = target
    resized_frequency = torch.zeros(
        resized_shape, dtype=frequency.dtype, device=frequency.device,
    )
    source_index = [slice(None)] * image.ndim
    target_index = [slice(None)] * image.ndim
    for axis, source, target in zip(spatial_dims, old_size, new_size):
        width = min(source, target)
        source_start = source // 2 - width // 2
        target_start = target // 2 - width // 2
        source_index[axis] = slice(source_start, source_start + width)
        target_index[axis] = slice(target_start, target_start + width)
    resized_frequency[tuple(target_index)] = frequency[tuple(source_index)]
    result = ifftnc(resized_frequency, dims=spatial_dims)
    return result * math.sqrt(math.prod(new_size) / math.prod(old_size))


def fourier_crop_spatial(image: torch.Tensor, new_size: tuple[int, ...]) -> torch.Tensor:
    """Crop the last spatial axes with centered Fourier cropping.

    This compatibility wrapper keeps coarse-data resampling crop-only; use
    :func:`fourier_resize_spatial` for both cropping and zero padding.
    """
    if len(new_size) not in (2, 3) or image.ndim < len(new_size):
        raise ValueError("Fourier cropping requires two or three spatial axes.")
    old_size = tuple(image.shape[axis] for axis in range(image.ndim - len(new_size), image.ndim))
    if any(target < 1 or target > source for source, target in zip(old_size, new_size)):
        raise ValueError(f"Invalid spatial Fourier crop from {old_size} to {new_size}.")
    return fourier_resize_spatial(image, new_size)

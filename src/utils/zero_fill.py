from __future__ import annotations

from collections.abc import Sequence

import torch

from src.utils.fftnc import fftnc, ifftnc


def zero_fill_image_to_shape(
    image: torch.Tensor,
    target_shape: Sequence[int],
    *,
    encoded_shape: Sequence[int],
    spatial_dims: Sequence[int] = (0, 1, 2),
) -> torch.Tensor:
    """
    Torch equivalent of Gadgetron GRICS zero_filling_and_cropping():

        fft3c(image)
        center-pad or trim to encoded_shape
        ifft3c(image)
        crop to target_shape

    Parameters
    ----------
    image:
        Input image, usually shape (E0, E1_in, E2).

    target_shape:
        Final cropped reconstruction shape:
        (E0, E1_recon, E2_recon).

    encoded_shape:
        Header encoded matrix size (reader-expanded grids are trimmed in k-space):
        (E0, E1, E2).

    spatial_dims:
        Dimensions corresponding to E0, E1, E2.
        Default matches Gadgetron fft3c.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")

    target_shape = tuple(int(v) for v in target_shape)
    encoded_shape = tuple(int(v) for v in encoded_shape)
    dims = tuple(_normalize_dim(d, image.ndim) for d in spatial_dims)

    if len(dims) != len(target_shape):
        raise ValueError("target_shape and spatial_dims must have the same length.")

    if len(dims) != len(encoded_shape):
        raise ValueError("encoded_shape and spatial_dims must have the same length.")

    if any(v <= 0 for v in target_shape):
        raise ValueError(f"target_shape values must be positive, got {target_shape}")

    if any(v <= 0 for v in encoded_shape):
        raise ValueError(f"encoded_shape values must be positive, got {encoded_shape}")

    work = image

    if not torch.is_complex(work):
        work = work.to(torch.complex64)

    # Gadgetron: fft3c(image)
    kspace = fftnc(work, dims=dims)

    # Gadgetron padding, extended to trim a reader-expanded Fourier grid back
    # to the header matrix without changing the image field of view.
    padded_shape = list(kspace.shape)

    for dim, size in zip(dims, encoded_shape):
        padded_shape[dim] = size

    padded = torch.zeros(
        padded_shape,
        dtype=kspace.dtype,
        device=kspace.device,
    )

    src_slices = [slice(None)] * kspace.ndim
    dst_slices = [slice(None)] * kspace.ndim

    for dim in dims:
        src_size = kspace.shape[dim]
        dst_size = padded.shape[dim]

        size = min(src_size, dst_size)
        src_offset = max(0, (src_size - dst_size) // 2)
        dst_offset = max(0, (dst_size - src_size) // 2)
        src_slices[dim] = slice(src_offset, src_offset + size)
        dst_slices[dim] = slice(dst_offset, dst_offset + size)

    padded[tuple(dst_slices)] = kspace[tuple(src_slices)]

    # Gadgetron: ifft3c(image)
    image_zf = ifftnc(padded, dims=dims)

    # Gadgetron: crop(size_cropped, image)
    crop_slices = [slice(None)] * image_zf.ndim

    for dim, size in zip(dims, target_shape):
        if size > image_zf.shape[dim]:
            raise ValueError(
                f"target_shape cannot be larger than zero-filled shape along dim {dim}: "
                f"{size} > {image_zf.shape[dim]}"
            )

        offset = (image_zf.shape[dim] - size) // 2
        crop_slices[dim] = slice(offset, offset + size)

    return image_zf[tuple(crop_slices)]


def zero_fill_grics_image_to_shape(
    image: torch.Tensor,
    target_shape: Sequence[int],
    *,
    encoded_shape: Sequence[int],
    spatial_dims: Sequence[int] = (-2, -1),
) -> torch.Tensor:
    """Zero-fill a loaded GRICS image to the final reconstruction matrix.

    This keeps zero filling as an explicit post-reconstruction step instead of
    hiding it in DICOM export. ``encoded_shape`` must be provided by the caller
    from the same matrix sizes used by the reconstruction path.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")

    target_shape = tuple(int(v) for v in target_shape)
    dims = tuple(_normalize_dim(d, image.ndim) for d in spatial_dims)
    if len(dims) != len(target_shape):
        raise ValueError("target_shape and spatial_dims must have the same length.")

    encoded_shape = tuple(int(v) for v in encoded_shape)

    if all(int(image.shape[dim]) == int(target) for dim, target in zip(dims, target_shape)):
        return image

    return zero_fill_image_to_shape(
        image,
        target_shape=target_shape,
        encoded_shape=encoded_shape,
        spatial_dims=dims,
    )


def _normalize_dim(dim: int, ndim: int) -> int:
    dim = int(dim)
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(
            f"Dimension {dim} is out of range for tensor with {ndim} dimensions."
        )
    return dim

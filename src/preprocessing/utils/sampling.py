"""Shared readout-index handling and flattened sampling coordinates."""

import torch


def flatten_index_list(values):
    if values is None:
        return None
    if torch.is_tensor(values):
        return values.reshape(-1)
    return torch.cat([value.reshape(-1) for value in values], dim=0)


def flatten_sampling_coordinates(ky, kx, Ny, Nz=1, kz=None, *, expand_kz=False):
    """Expand readouts over kx in [kx, ky, kz] tensor order.

    With expand_kz=True, each readout covers every partition, ordered by
    readout, kx, then kz. Otherwise 3D readouts require paired ky/kz values.
    """
    ky = ky.to(device=kx.device, dtype=torch.int64).reshape(-1)
    indices = ky[:, None] + Ny * kx[None, :]
    if Nz > 1:
        if expand_kz:
            if kz is not None:
                raise ValueError("Full-partition expansion cannot also specify kz.")
            kz = torch.arange(Nz, device=kx.device, dtype=torch.int64)
            indices = indices[:, :, None] * Nz + kz[None, None, :]
        else:
            if kz is None:
                raise ValueError("kz is required for paired 3D readouts.")
            kz = kz.to(device=kx.device, dtype=torch.int64).reshape(-1)
            if ky.numel() != kz.numel():
                raise ValueError("Paired ky and kz bins must contain the same number of readouts.")
            indices = indices * Nz + kz[:, None]
    return indices.reshape(-1)

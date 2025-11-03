import numpy as np
import torch
import os
import numpy as np
import sigpy.mri as spmri
import sigpy as sp
try:
    import cupy as cp
except ImportError:
    pass

def calc_espirit_maps(kspace, acs = 48, kernel_width = 6, sp_device = sp.Device(-1)):
    nCha, nX, nY, nSlices = kspace.shape
    espirit_maps = torch.zeros((nCha, nX, nY, nSlices), dtype=torch.complex64, device = "cuda")
    if sp_device.id == -1:

        for i in range(nSlices):
            kspace_np = kspace[:, :, :, i].detach().cpu().numpy()
            maps_np = spmri.app.EspiritCalib(
                kspace_np, calib_width=acs, kernel_width=kernel_width
            ).run()
            espirit_maps[:, :, :, i] = torch.from_numpy(maps_np)
        return espirit_maps

    with sp.Device(sp_device):
        for i in range(nSlices):
            # move slice to CuPy (from Torch); this will copy from host/device as needed
            kspace_cp = cp.asarray(kspace[:, :, :, i].contiguous())

            maps_cp = spmri.app.EspiritCalib(
                kspace_cp, calib_width=acs, kernel_width=kernel_width, device=sp_device
            ).run()

            maps_cp = maps_cp.astype(cp.complex64, copy=False)
            maps_cp = cp.ascontiguousarray(maps_cp)

            maps_t = sp.to_pytorch(maps_cp)

            espirit_maps[:, :, :, i] = torch.complex(maps_t[..., 0], maps_t[..., 1])

            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()

    return espirit_maps

def to_espirit_dims(data):
    """ width, height, slices, coils  <-  coils, height, width, slices """
    return data.permute(3,0,1,2).contiguous()  # width, height, slices, coils

def from_espirit_dims(data):
    """ coils, height, width, slices  <-  width, height, slices, coils """
    return data.permute(1,2,3,0).contiguous()  # coils, height, width, slices
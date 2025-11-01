import h5py
import numpy as np
import torch
import os
import numpy as np
import cupy as cp
import sigpy.mri as spmri
import sigpy as sp

from utils.show_slice import show_slice, show_kspace_slice, header_info
from utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions

# Get all .h5 files from the specified folder
def get_data(path, num = 0):
    folder = path

    # setting device for cuda and cupy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    h5_files = [f for f in os.listdir(folder) if f.endswith(".h5")]
    h5_files.sort() 

    path = os.path.join(folder, h5_files[num])

    with h5py.File(path, "r") as f:
        numpy_kspace = f["kspace"][:] 
        header_bytes = f["ismrmrd_header"][()]  
        header_str = header_bytes.decode("utf-8")

    header_info(header_str)

    kspace = torch.from_numpy(numpy_kspace) # slices, coils, height, width
    kspace = kspace.permute(1, 3, 2, 0).contiguous() # coils, width, height, slices
    return kspace

def espirit_maps(kspace):
    device = kspace.device
    gpu = cp.cuda.Device(0) 
    nCha, nY, nX, nSlices = kspace.shape # coils, width, height, slices
    espirit_maps = torch.zeros((nCha, nY, nX, nSlices), dtype=torch.complex64, device = device)

    """ for i in range(nSlices):
        kspace_slice = kspace[:,:,:,i]
        kspace_cp = cp.asarray(kspace_slice)
        acs = 48
        ecalib = spmri.app.EspiritCalib(
            kspace_cp,
            calib_width=acs,       
            kernel_width=6,
            device=gpu
        )
         maps = ecalib.run()

        maps = cp.ascontiguousarray(maps) 
        maps = sp.to_pytorch(maps) 
        maps = torch.complex(maps[...,0], maps[...,1])
                
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        cp.cuda.Stream.null.synchronize()

        espirit_maps[:,:,:,i] = maps """
    for i in range(nSlices):
        kspace_slice = kspace[:, :, :, i].numpy()
        espirit_map_slice = spmri.app.EspiritCalib(kspace_slice, calib_width=24, kernel_width=6, thresh=0.02).run()
        espirit_maps[:, :, :, i] = torch.from_numpy(espirit_map_slice)

    
    del espirit_maps#, kspace_cp, kspace_slice, ecalib
    return espirit_maps
import torch
import sigpy as sp
import time

from src.config.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.utils.show_and_save_image import show_and_save_image

# --- Optional CuPy import + capability check ---
try:
    import cupy as cp
    _cupy_ok = True
    try:
        _gpu_count = cp.cuda.runtime.getDeviceCount()
        _cupy_ok = _gpu_count > 0
    except Exception:
        _cupy_ok = False
except Exception:
    cp = None
    _cupy_ok = False

torch.cuda.empty_cache()
total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # in GB
print(f"Total GPU memory: {total_mem:.2f} GB")
# --- End optional CuPy import + capability check ---

params = load_config(
    [
        "config/general.toml",
        "config/sampling_simulation/interleaved.toml",
        "config/motion_simulation/discrete_rigid.toml",
        "config/reconstruction/rigid_fast.toml",
    ],
    overrides={
        "path_to_fastMRI_data": "data/kspace.npz",
        "path_to_realworld_data": "data/breast_motion_data.h5",
        "ismrmrd_file": "data/t2_1724.h5",
        "saec_file": "data/2008-003 01-1724_S11_20210323_151329.h5",
    },
)

# Set device for SigPy and PyTorch 
sp_device = sp.Device(0) if _cupy_ok else sp.Device(-1)
t_device = torch.device("cuda:0" if _cupy_ok and torch.cuda.is_available() else "cpu")

if params.debug_flag:
    # Set random seed for reproducibility
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)

data = DataLoader(params=params, t_device=t_device, sp_device=sp_device)
image_ground_truth = data.image_ground_truth.clone()
show_and_save_image(image_ground_truth[0], 'img_ground_truth', params.debug_folder)
image_corrupted = data.image_no_moco.clone()
kspace_corrupted = data.kspace
show_and_save_image(image_corrupted[0], 'img_corrupted', params.debug_folder)

jointReconstructor = JointReconstructor(
    data.kspace,
    data.smaps,
    data.sampling_idx,
    motion_signal=data.motion_signal,
    params=params,
    kspace_scale=data.kspace_scale,
)
start = time.time()
jointReconstructor.run()
end = time.time()
print(f"Elapsed time joint image/motion reconstruction: {end - start:.2f} s")

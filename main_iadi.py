import torch
import sigpy as sp
import matplotlib.pyplot as plt
import time

from src.preprocessing.DataLoader import DataLoader
from src.Parameters import Parameters
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.JointReconstructor import JointReconstructor
from src.utils.show_slice import show_slice


def show_slice_and_save(image, image_name):
    if params.debug_flag:
        show_slice(image, max_images=1, headline=image_name)
        plt.savefig(params.debug_folder + image_name + '.png')

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

# Set parameters
params = Parameters()
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
show_slice_and_save(image_ground_truth[0], 'img_ground_truth')
image_corrupted = data.image_no_moco.clone()
kspace_corrupted = data.kspace
show_slice_and_save(image_corrupted[0], 'img_corrupted')

jointReconstructor = JointReconstructor(data.kspace, data.smaps, data.sampling_idx, params)
start = time.time()
jointReconstructor.run()
end = time.time()
print(f"Elapsed time joint image/motion reconstruction: {end - start:.2f} s")





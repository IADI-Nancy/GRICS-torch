import torch
import sigpy as sp
from utils.show_slice import show_slice
import matplotlib.pyplot as plt

from iadi.Data import Data
from iadi.Parameters import Parameters
from iadi.EncodingOperator import EncodingOperator

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
# --- End optional CuPy import + capability check ---

# Set parameters
params = Parameters()
# Set device for SigPy and PyTorch 
sp_device = sp.Device(0) if _cupy_ok else sp.Device(-1)
t_device = torch.device("cuda:0" if _cupy_ok and torch.cuda.is_available() else "cpu")

# Load data and simulate motion
data = Data("data/kspace.npz", params=params, t_device=t_device, sp_device=sp_device)
image_ground_truth = data.image_no_moco.clone()
show_slice_and_save(image_ground_truth, 'img_ground_truth')

data.create_motion_corrupted_dataset(params=params)
image_corrupted = data.image_no_moco.clone()
kspace_corrupted = data.kspace
show_slice_and_save(image_corrupted, 'img_corrupted')

E = EncodingOperator(data.smaps, data.TotalKspaceSamples, data.SamplingIndices, data.KspaceOffset, data.t_device)

# Test
EHs = E.normal(image_ground_truth, data.MotionOperator)
show_slice_and_save(EHs.reshape(data.Nx, data.Ny, data.Nsli), 'EHs')

# Ax = b
# EH E x = Eh s

# Prepare for reconstruction
b = E.backward(kspace_corrupted, data.MotionOperator)
lambda_scaled = params.lambda_r * torch.norm(b, p=2)
x0 = image_corrupted.flatten()

def A(x):
    return E.normal(x, data.MotionOperator) +  lambda_scaled * x

import torch
import sigpy as sp
from utils.show_slice import show_slice
import matplotlib.pyplot as plt

from utils.EHE import EH, EHE
from utils.conjugate_gradient import cg

from iadi.Data import Data
from iadi.Parameters import Parameters

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

t_n, masks = data.simulate_rigid_motion(params=params)
image_corrupted = data.image_no_moco.clone()
kspace_corrupted = data.kspace
show_slice_and_save(image_corrupted, 'img_corrupted')

# Prepare for reconstruction
image_shape = data.Nx, data.Ny, data.Nsli, data.Ncha
b = EH(kspace_corrupted, t_n = t_n, iterations = params.iterations, masks=masks, sigmas = data.smaps, image_shape = image_shape, model=0)
# EHEp = EHE(p_true, t_n = t_n, iterations = params.iterations, masks= masks, sigmas = params.smaps, image_shape = image_shape, model=0)

lambda_scaled = params.lambda_r * torch.norm(b, p=2)

x0 = torch.zeros_like(b, device=t_device, dtype=torch.complex64)
eye = torch.ones_like(b, device=t_device, dtype=torch.complex64)

A = EHE(eye, t_n=t_n, iterations=params.iterations, masks=masks, sigmas=data.smaps, image_shape=image_shape, model=0) + eye * lambda_scaled

with torch.no_grad():
    image_rec, info = cg(A, b, x0, max_iter=params.max_iter, tol=params.tol, regularisation=lambda_scaled)
show_slice_and_save(image_rec.view(640,320,1), 'img_reconstructed')
import torch
import sigpy as sp
import matplotlib.pyplot as plt
import time

from src.preprocessing.DataLoader import DataLoader
from src.Parameters import Parameters
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.JointReconstructor import JointReconstructor
from src.preprocessing.RawDataReader import DataReader
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
    # Make cuDNN deterministic (may slow training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)



data = DataLoader(params=params, t_device=t_device, sp_device=sp_device)
image_ground_truth = data.image_ground_truth.clone()
show_slice_and_save(image_ground_truth, 'img_ground_truth')
image_corrupted = data.image_no_moco.clone()
kspace_corrupted = data.kspace
show_slice_and_save(image_corrupted, 'img_corrupted')
params = data.params

jointReconstructor = JointReconstructor(data.kspace, data.smaps, data.TotalKspaceSamples, data.sampling_idx, data.nex_offset, params)
start = time.time()
jointReconstructor.run()
end = time.time()
print(f"Elapsed time joint image/motion reconstruction: {end - start:.2f} s")


# Test DataReader
saec_file = 'data/2008-003 01-1724_S11_20210323_151329.h5'
ismrmrd_file = 'data/t2_1724.h5'
data = DataReader.read_kspace_and_motion_data_from_rawdata(ismrmrd_file, saec_file, \
                                                    sensor_type='BELT', Nbins=8,\
                                                    h5filename='data/breast_motion_data.h5')
# Read the data
data = DataReader.read_kspace_and_motion_data_from_h5('data/breast_motion_data.h5')

# Access individual datasets
motion_data = data['motion_data']
prior_image = data['prior_image']
line_idx = data['line_idx']
kspace = data['kspace']
smap = data['smap']
bin_centers = data['bin_centers']
binned_indices = data['binned_indices']

# Print shapes to verify
print(f"motion_data shape: {motion_data.shape}")
print(f"prior_image shape: {prior_image.shape}")
print(f"line_idx shape: {line_idx.shape}")
print(f"kspace shape: {kspace.shape}")
print(f"smap shape: {smap.shape}")
print(f"bin_centers shape: {bin_centers.shape}")
print(f"binned_indices shape: {binned_indices.shape}")

# Access a specific binned index array
print(f"\nExample - binned_indices[0, 0]: {binned_indices[0, 0]}")
print(f"Type: {type(binned_indices[0, 0])}")





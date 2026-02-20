import time

from src.config.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.utils.show_and_save_image import show_and_save_image
from src.utils.runtime_setup import initialize_runtime

params = load_config(
    data_type="fastMRI",
    reconstruction_config="config/reconstruction/rigid_fast.toml",
    sampling_config="config/sampling_simulation/interleaved.toml",
    motion_simulation_config="config/motion_simulation/discrete_rigid.toml",
)

sp_device, t_device = initialize_runtime(params, print_gpu_info=True)

data = DataLoader(
    params=params,
    t_device=t_device,
    sp_device=sp_device,
    filename="data/kspace.npz",
)
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

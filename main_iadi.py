import time

from src.runtime.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.runtime_setup import cleanup_runtime, initialize_runtime

params = load_config(
    data_type="fastMRI",
    reconstruction_config="config/reconstruction/rigid_high_quality.toml",
    sampling_config="config/sampling_simulation/interleaved.toml",
    motion_simulation_config="config/motion_simulation/discrete_rigid.toml",
)

sp_device, t_device = initialize_runtime(params, print_gpu_info=True)

try:
    data = DataLoader(
        params=params,
        t_device=t_device,
        sp_device=sp_device,
        filename="data/kspace.npz",
    )

    jointReconstructor = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        kspace_scale=data.kspace_scale,
        motion_plot_context=getattr(data, "motion_plot_context", None),
    )
    start = time.time()
    jointReconstructor.run()
    end = time.time()
    print(f"Elapsed time joint image/motion reconstruction: {end - start:.2f} s")
finally:
    cleanup_runtime()

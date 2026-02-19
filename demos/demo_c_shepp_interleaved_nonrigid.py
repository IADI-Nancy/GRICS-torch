import time
import torch
import sigpy as sp

from src.config.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.utils.show_and_save_image import show_and_save_image


try:
    import cupy as cp
    _cupy_ok = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    _cupy_ok = False


def main():
    params = load_config(
        [
            "config/general.toml",
            "config/shepp_logan.toml",
            "config/sampling_simulation/interleaved.toml",
            "config/motion_simulation/discrete_nonrigid.toml",
            "config/reconstruction/nonrigid_fast.toml",
        ],
        overrides={
            "path_to_fastMRI_data": "data/kspace.npz",
            "path_to_realworld_data": "data/breast_motion_data.h5",
        },
    )

    sp_device = sp.Device(0) if _cupy_ok else sp.Device(-1)
    t_device = torch.device("cuda:0" if _cupy_ok and torch.cuda.is_available() else "cpu")

    if params.debug_flag:
        torch.manual_seed(params.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(params.seed)
            torch.cuda.manual_seed_all(params.seed)

    data = DataLoader(params=params, t_device=t_device, sp_device=sp_device)
    show_and_save_image(data.image_ground_truth[0], "img_ground_truth", params.debug_folder)
    show_and_save_image(data.image_no_moco[0], "img_corrupted", params.debug_folder)

    recon = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        kspace_scale=data.kspace_scale,
    )
    t0 = time.time()
    recon.run()
    print(f"Elapsed time: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()

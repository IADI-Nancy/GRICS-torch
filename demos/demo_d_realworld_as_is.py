import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.utils.show_and_save_image import show_and_save_image
from src.runtime.runtime_setup import initialize_runtime


def main():
    params = load_config(
        data_type="real-world",
        reconstruction_config="config/reconstruction/rigid_fast.toml",
    )

    sp_device, t_device = initialize_runtime(params)

    data = DataLoader(
        params=params,
        t_device=t_device,
        sp_device=sp_device,
        filename="data/breast_motion_data.h5",
    )
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

import time
import sys
from pathlib import Path

if "__file__" in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    _REPO_ROOT = Path.cwd()
sys.path.insert(0, str(_REPO_ROOT))

from src.runtime.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.utils.notebook_display import display_run_panels
from src.runtime.runtime_setup import initialize_runtime

jupyter_notebook_flag = False


def main():
    params = load_config(
        data_type="shepp-logan",
        shepp_logan_config="config/shepp_logan.toml",
        reconstruction_config="config/reconstruction/rigid_fast.toml",
        sampling_config="config/sampling_simulation/linear.toml",
        motion_simulation_config="config/motion_simulation/rigid.toml",
        overrides={
            "jupyter_notebook_flag": jupyter_notebook_flag,
            "print_to_console": not jupyter_notebook_flag,
            "verbose": not jupyter_notebook_flag,
        },
    )

    sp_device, t_device = initialize_runtime(params)

    data = DataLoader(params=params, t_device=t_device, sp_device=sp_device)

    recon = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        kspace_scale=data.kspace_scale,
        motion_plot_context=getattr(data, "motion_plot_context", None),
    )
    t0 = time.time()
    recon.run()
    print(f"Elapsed time: {time.time() - t0:.2f} s")
    display_run_panels(
        params,
        motion_type=params.motion_type,
        has_ground_truth=(params.data_type == "shepp-logan"),
        jupyter_notebook_flag=params.jupyter_notebook_flag,
    )


if __name__ == "__main__":
    main()

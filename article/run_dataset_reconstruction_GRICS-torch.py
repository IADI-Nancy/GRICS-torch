import sys
from pathlib import Path
if "__file__" in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    _REPO_ROOT = Path.cwd()
sys.path.insert(0, str(_REPO_ROOT))

import time
import torch
import shutil
from pathlib import Path

from src.runtime.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.utils.notebook_display import display_input_sampling_motion_panels, display_run_panels
from src.runtime.runtime_setup import initialize_runtime

article_dataset_folder = "/home/pyuser/wkdir/data/GRICS-torch/article_dataset/"
jupyter_notebook_flag = False
log_file = "logs/joint_reconstruction.log"

params = load_config(
    data_type="real-world",
    reconstruction_config="config/reconstruction/nonrigid_2d.toml",
    overrides={
        "jupyter_notebook_flag": jupyter_notebook_flag,
    },
)
sp_device, t_device = initialize_runtime(params)


files = list(Path(article_dataset_folder).glob("*.h5"))

for f in files:
    subject = Path(f.name).stem
    print(subject)
    subject_dir = Path(article_dataset_folder) / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    if subject=="0080_T2_s":
        Nsli = 68
    else:
        Nsli = 60
    print("[Demo B] Loading data and building operators...")
    for slice_idx in range(Nsli):
        data = DataLoader(
            params=params,
            t_device=t_device,
            sp_device=sp_device,
            filename=f,
            slice_idx=slice_idx
        )
        print("[Demo B] Starting reconstruction...")
        recon = JointReconstructor(
            data.kspace,
            data.smaps,
            data.sampling_idx,
            motion_signal=data.motion_signal,
            params=params,
            motion_plot_context=data.motion_plot_context,
        )
        t0 = time.time()
        image, alpha = recon.run()
        print(f"Elapsed time: {time.time() - t0:.2f} s")
        # Save the results
        slice_folder = f"Siemens_SingleImage_slice{slice_idx:03d}_image01"
        output_dir = Path(subject_dir) / slice_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save PyTorch tensors
        torch.save(recon, output_dir / "GricsRecon.pt")
        torch.save(alpha, output_dir / "GricsAlphaMaps.pt")

        # Copy log file
        log_file = Path(log_file)
        if log_file.exists():
            shutil.copy(log_file, output_dir / "joint_reconstruction.log")
        else:
            print(f"Warning: log file not found: {log_file}")
        
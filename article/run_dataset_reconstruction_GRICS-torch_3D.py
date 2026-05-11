from curses import raw
import sys
from pathlib import Path
if "__file__" in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    _REPO_ROOT = Path.cwd()
sys.path.insert(0, str(_REPO_ROOT))

import os
import re

import h5py
import numpy as np
import time
import torch
import shutil
from pathlib import Path
from multiprocessing import Manager

from src.runtime.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.runtime_setup import initialize_runtime

article_dataset_folder = "/home/pyuser/wkdir/data/GRICS-torch/article_dataset_3D"
GRICS_plusplus_path = "/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/GRICS-BELT-3D"
jupyter_notebook_flag = False

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


import matplotlib.pyplot as plt


def _read_h5_kspace_shape(h5_file):
    with h5py.File(h5_file, "r") as f:
        return tuple(f["kspace"].shape)


def load_grics_plusplus_sensitivity_maps(grics_plusplus_root, h5_file, device):
    subject = Path(h5_file).stem

    sensitivity_file = (
        Path(grics_plusplus_root)
        / subject
        / "SensitivityMaps.dat"
    )

    if not sensitivity_file.is_file():
        raise FileNotFoundError(f"Missing GRICS++ sensitivity maps: {sensitivity_file}")

    Ncoils, _, Nx, Ny, Nz = _read_h5_kspace_shape(h5_file)

    raw = np.fromfile(sensitivity_file, dtype="<f8")
    expected_size = 2 * Ny * Nx * Nz * Ncoils

    if raw.size != expected_size:
        raise ValueError(
            f"{sensitivity_file} contains {raw.size} floats, expected "
            f"{expected_size} for [2, Ny={Ny}, Nx={Nx}, Nz={Nz}, Ncoils={Ncoils}]."
        )

    raw_complex = raw[0::2] + 1j * raw[1::2]

    # GRICS++ writes: c, e2, e0, e1, with e1 fastest.
    maps_np = raw_complex.reshape((Ncoils, Nz, Nx, Ny), order="C")
    maps_np = np.transpose(maps_np, (0, 2, 3, 1))  # [Ncoils, Nx, Ny, Nz]


    # Save central slice of coil 0 absolute sensitivity map
    central_slice = maps_np[0, :, :, Nz // 2]

    output_dir = Path(subject_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.imshow(np.abs(central_slice), cmap="gray")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(
        output_dir / "SensitivityMap_coil000_central_slice.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    return torch.from_numpy(np.ascontiguousarray(maps_np)).to(
        device=device,
        dtype=torch.complex128,
    )

def _tensor_to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_physiological_data(output_dir, data):
    output_dir = Path(output_dir)

    motion_curve = _tensor_to_numpy(getattr(data, "_motion_curve_for_binning", None))
    motion_signal = _tensor_to_numpy(getattr(data, "motion_signal", None))
    motion_labels = _tensor_to_numpy(getattr(data, "motion_labels", None))
    ky_idx = _tensor_to_numpy(getattr(data, "ky_idx_chronological", None))
    kz_idx = _tensor_to_numpy(getattr(data, "kz_idx_chronological", None))
    nex_idx = _tensor_to_numpy(getattr(data, "nex_idx_chronological", None))

    arrays = {
        "motion_curve_chronological": motion_curve,
        "motion_signal_reconstruction": motion_signal,
        "motion_labels": motion_labels,
        "ky_idx_chronological": ky_idx,
        "nex_idx_chronological": nex_idx,
    }
    if kz_idx is not None:
        arrays["kz_idx_chronological"] = kz_idx
    arrays = {name: value for name, value in arrays.items() if value is not None}

    np.savez(output_dir / "PhysiologicalData_torch.npz", **arrays)
    torch.save(
        {
            "motion_curve_chronological": getattr(data, "_motion_curve_for_binning", None),
            "motion_signal_reconstruction": getattr(data, "motion_signal", None),
            "motion_labels": getattr(data, "motion_labels", None),
            "ky_idx_chronological": getattr(data, "ky_idx_chronological", None),
            "kz_idx_chronological": getattr(data, "kz_idx_chronological", None),
            "nex_idx_chronological": getattr(data, "nex_idx_chronological", None),
        },
        output_dir / "PhysiologicalData_torch.pt",
    )

    if motion_curve is not None:
        model_inputs = np.asarray(motion_curve, dtype="<f4").reshape(1, -1)
        model_inputs.tofile(output_dir / "ModelInputs_torch.dat")




def run_one_subject(
    h5_file,
    subject_dir,
    jupyter_notebook_flag=False,
):

    import time
    import torch

    from src.runtime.runtime_config import load_config
    from src.runtime.runtime_setup import initialize_runtime
    from src.preprocessing.DataLoader import DataLoader
    from src.reconstruction.JointReconstructor import JointReconstructor

    print(
    "PID:", os.getpid(),
    "torch threads:", torch.get_num_threads(),
    "interop:", torch.get_num_interop_threads(),
    "OMP:", os.environ.get("OMP_NUM_THREADS"),
    "MKL:", os.environ.get("MKL_NUM_THREADS"),
)

    output_dir = Path(subject_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(output_dir)

    params = load_config(
        data_type="real-world",
        reconstruction_config="config/reconstruction/nonrigid_3d.toml",
        overrides={
            "jupyter_notebook_flag": jupyter_notebook_flag,

            # Important: avoid different processes deleting each other's folders
            "clean_output_folders_before_run": False,

            # Log directly inside final slice folder
            "logs_folder": str(output_dir) + "/",

            "runtime_device": "gpu",

            "N_motion_states": 16,  # important for fair comparison with GRICS++
            "debug_flag": False,
            "verbose": False,
            "print_to_console": False,
        },
    )

    sp_device, t_device = initialize_runtime(params)
    external_smaps = load_grics_plusplus_sensitivity_maps(
        GRICS_plusplus_path,
        h5_file,
        t_device,
    )

    data = DataLoader(
        params=params,
        t_device=t_device,
        sp_device=sp_device,
        filename=h5_file,
        external_smaps=external_smaps,
    )
    save_physiological_data(output_dir, data)

    reconstructor = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        motion_plot_context=data.motion_plot_context,
    )

    # All slices start reconstruction after this point
    t0 = time.time()
    image, alpha = reconstructor.run()
    elapsed_time = time.time() - t0

    torch.save(image, output_dir / "GricsRecon.pt")
    torch.save(alpha, output_dir / "GricsAlphaMaps.pt")

    return elapsed_time



# Main part

# Main part

T1_H5_PATTERN = "[0-9][0-9][0-9][0-9]_T1_[sm].h5"
T1_SUBJECT_PATTERN = re.compile(r"^\d{4}_T1_[sm]$")

files = sorted(Path(article_dataset_folder).glob(T1_H5_PATTERN))

for f in files:
    subject = f.stem

    if not T1_SUBJECT_PATTERN.match(subject):
        continue

    print(subject)

    subject_dir = Path(article_dataset_folder) / subject # + "_nomoco"
    subject_dir.mkdir(parents=True, exist_ok=True)

    elapsed_time = run_one_subject(
        h5_file=str(f),
        subject_dir=str(subject_dir),
        jupyter_notebook_flag=jupyter_notebook_flag,
    )

    print(f"{subject} : {elapsed_time:.2f} s")



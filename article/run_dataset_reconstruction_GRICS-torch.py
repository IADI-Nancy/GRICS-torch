import sys
from pathlib import Path
if "__file__" in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    _REPO_ROOT = Path.cwd()
sys.path.insert(0, str(_REPO_ROOT))

import os
import re

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["SimpleITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

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

article_dataset_folder = "/home/pyuser/wkdir/data/GRICS-torch/article_dataset"
GRICS_plusplus_path = "/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/GRICS-BELT"
jupyter_notebook_flag = False
Ncores = 128

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def _read_h5_kspace_shape(h5_file):
    with h5py.File(h5_file, "r") as f:
        return tuple(f["kspace"].shape)


def load_grics_plusplus_sensitivity_maps(grics_plusplus_root, h5_file, slice_idx, device):
    subject = Path(h5_file).stem
    slice_folder = f"Siemens_SingleImage_slice{slice_idx + 1:03d}_image01"
    sensitivity_file = (
        Path(grics_plusplus_root)
        / subject
        / slice_folder
        / "SensitivityMaps.dat"
    )

    if not sensitivity_file.is_file():
        raise FileNotFoundError(f"Missing GRICS++ sensitivity maps: {sensitivity_file}")

    Ncoils, _, Nx, Ny, Nslices = _read_h5_kspace_shape(h5_file)
    if slice_idx >= Nslices:
        raise IndexError(
            f"slice_idx {slice_idx} is outside h5 kspace slice count {Nslices}."
        )

    raw = np.fromfile(sensitivity_file, dtype="<f4")
    expected_size = 2 * Ny * Nx * Ncoils
    if raw.size != expected_size:
        raise ValueError(
            f"{sensitivity_file} contains {raw.size} floats, expected "
            f"{expected_size} for [2, Ny={Ny}, Nx={Nx}, Ncoils={Ncoils}]."
        )

    maps_ri = raw.reshape((2, Ny, Nx, Ncoils), order="F")
    maps_np = maps_ri[0] + 1j * maps_ri[1]
    maps_np = np.transpose(maps_np, (2, 1, 0))[:, :, :, None]
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




def run_one_slice(
    h5_file,
    subject_dir,
    slice_idx,
    barrier,
    jupyter_notebook_flag=False,
):

    import time
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

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

    # GRICS++-style slice folder
    slice_folder = f"Siemens_SingleImage_slice{slice_idx + 1:03d}_image01"
    output_dir = Path(subject_dir) / slice_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    print(output_dir)

    params = load_config(
        data_type="real-world",
        reconstruction_config="config/reconstruction/nonrigid_2d.toml",
        overrides={
            "jupyter_notebook_flag": jupyter_notebook_flag,

            # Important: avoid different processes deleting each other's folders
            "clean_output_folders_before_run": False,

            # Log directly inside final slice folder
            "logs_folder": str(output_dir) + "/",

            # Optional but recommended if you really want CPU-only slice parallelism
            "runtime_device": "cpu",

            "N_motion_states": 1,  # important for fair comparison with GRICS++
            "debug_flag": False,
            "verbose": False,
            "print_to_console": False,
        },
    )

    sp_device, t_device = initialize_runtime(params)
    external_smaps = load_grics_plusplus_sensitivity_maps(
        GRICS_plusplus_path,
        h5_file,
        slice_idx,
        t_device,
    )

    data = DataLoader(
        params=params,
        t_device=t_device,
        sp_device=sp_device,
        filename=h5_file,
        slice_idx=slice_idx,
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

    # Synchronize all slice processes here
    print(f"Slice {slice_idx + 1:03d} ready, waiting at barrier...")
    barrier.wait()

    # All slices start reconstruction after this point
    t0 = time.time()
    image, alpha = reconstructor.run()
    elapsed_time = time.time() - t0

    torch.save(image, output_dir / "GricsRecon.pt")
    torch.save(alpha, output_dir / "GricsAlphaMaps.pt")

    return slice_idx, elapsed_time



# Main part

# Main part

T2_H5_PATTERN = "[0-9][0-9][0-9][0-9]_T2_[sm].h5"
T2_SUBJECT_PATTERN = re.compile(r"^\d{4}_T2_[sm]$")

files = sorted(Path(article_dataset_folder).glob(T2_H5_PATTERN))

for f in files:
    subject = f.stem

    if not T2_SUBJECT_PATTERN.match(subject):
        continue

    print(subject)

    subject_dir = Path(article_dataset_folder + "_nomoco") / subject
    subject_dir.mkdir(parents=True, exist_ok=True)

    Nsli = 68 if subject == "0080_T2_s" else 60
    max_workers = Nsli

    with Manager() as manager:
        barrier = manager.Barrier(Nsli)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    run_one_slice,
                    str(f),
                    str(subject_dir),
                    slice_idx,
                    barrier,
                    jupyter_notebook_flag,
                )
                for slice_idx in range(Nsli)
            ]

            for future in as_completed(futures):
                slice_idx, elapsed_time = future.result()
                print(f"{subject} slice {slice_idx + 1:03d}: {elapsed_time:.2f} s")









# params = load_config(
#     data_type="real-world",
#     reconstruction_config="config/reconstruction/nonrigid_2d.toml",
#     overrides={
#         "jupyter_notebook_flag": jupyter_notebook_flag,
#     },
# )
# sp_device, t_device = initialize_runtime(params)


# files = list(Path(article_dataset_folder).glob("*.h5"))

# for f in files:
#     subject = Path(f.name).stem
#     print(subject)
#     subject_dir = Path(article_dataset_folder) / subject
#     subject_dir.mkdir(parents=True, exist_ok=True)
#     if subject=="0080_T2_s":
#         Nsli = 68
#     else:
#         Nsli = 60
#     print("[Demo B] Loading data and building operators...")
#     for slice_idx in range(Nsli):
#         data = DataLoader(
#             params=params,
#             t_device=t_device,
#             sp_device=sp_device,
#             filename=f,
#             slice_idx=slice_idx
#         )
#         print("[Demo B] Starting reconstruction...")
#         recon = JointReconstructor(
#             data.kspace,
#             data.smaps,
#             data.sampling_idx,
#             motion_signal=data.motion_signal,
#             params=params,
#             motion_plot_context=data.motion_plot_context,
#         )
#         t0 = time.time()
#         image, alpha = recon.run()
#         print(f"Elapsed time: {time.time() - t0:.2f} s")
#         # Save the results
#         slice_folder = f"Siemens_SingleImage_slice{slice_idx:03d}_image01"
#         output_dir = Path(subject_dir) / slice_folder
#         output_dir.mkdir(parents=True, exist_ok=True)
#         # Save PyTorch tensors
#         torch.save(recon, output_dir / "GricsRecon.pt")
#         torch.save(alpha, output_dir / "GricsAlphaMaps.pt")

#         # Copy log file
#         log_file = Path(log_file)
#         if log_file.exists():
#             shutil.copy(log_file, output_dir / "joint_reconstruction.log")
#         else:
#             print(f"Warning: log file not found: {log_file}")
        

import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.plotting import save_nonrigid_alpha_plots


def subject_number(subject_name):
    match = re.match(r"\d+", subject_name)
    return match.group(0) if match else subject_name


def append_subject_metric(file_path, subject_name, value):
    value_text = value if isinstance(value, str) else f"{value}"
    with open(file_path, "a") as f:
        f.write(f"{subject_number(subject_name)}\t{value_text}\n")


def _as_real_tensor(data):
    tensor = torch.as_tensor(data)
    if torch.is_complex(tensor):
        tensor = tensor.real
    return tensor.detach().cpu().float()


def load_torch_alpha_maps(file_alpha):
    alpha = torch.load(file_alpha, map_location="cpu")
    if not isinstance(alpha, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(alpha)}")
    alpha = alpha.detach().cpu()
    if torch.is_complex(alpha):
        alpha = alpha.real
    return alpha.float()


def load_grics_2d_alpha_maps(file_alpha, ny, nx):
    alpha = np.fromfile(file_alpha, dtype="<f4").reshape(2, ny, nx)
    return _as_real_tensor(alpha)


def load_grics_3d_alpha_maps(file_alpha, nz, ny, nx):
    alpha = np.fromfile(file_alpha, dtype="<f4").reshape(3, nz, ny, nx)
    alpha = np.transpose(alpha, (0, 3, 2, 1))
    return _as_real_tensor(alpha)


def save_alpha_maps_if_available(file_alpha, image, output_dir, base_name="reconstructed"):
    file_alpha = Path(file_alpha)
    if not file_alpha.exists():
        print(f"Missing alpha maps: {file_alpha}")
        return

    alpha = load_torch_alpha_maps(file_alpha)
    save_nonrigid_alpha_plots(
        alpha,
        _as_real_tensor(np.abs(image)),
        base_name,
        output_dir,
        flip_vertical=True,
    )


def save_grics_2d_alpha_maps_if_available(file_alpha, image, ny, nx, output_dir):
    file_alpha = Path(file_alpha)
    if not file_alpha.exists():
        print(f"Missing alpha maps: {file_alpha}")
        return

    alpha = load_grics_2d_alpha_maps(file_alpha, ny, nx)
    save_nonrigid_alpha_plots(
        alpha,
        _as_real_tensor(np.abs(image)),
        "reconstructed",
        output_dir,
        flip_vertical=True,
    )


def save_grics_3d_alpha_maps_if_available(file_alpha, image, nz, ny, nx, output_dir):
    file_alpha = Path(file_alpha)
    if not file_alpha.exists():
        print(f"Missing alpha maps: {file_alpha}")
        return

    alpha = load_grics_3d_alpha_maps(file_alpha, nz, ny, nx)
    image_xyz = np.transpose(np.abs(image), (2, 1, 0))
    save_nonrigid_alpha_plots(
        alpha,
        _as_real_tensor(image_xyz),
        "reconstructed",
        output_dir,
        flip_vertical=True,
    )

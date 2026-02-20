import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor


def _build_context():
    params = load_config(
        data_type="shepp-logan",
        motion_type="non-rigid",
        reconstruction_config="config/reconstruction/nonrigid_fast.toml",
        shepp_logan_config="config/shepp_logan.toml",
        sampling_config="config/sampling_simulation/interleaved.toml",
        kspace_sampling_type="interleaved",
        motion_simulation_config="config/motion_simulation/discrete_nonrigid.toml",
        motion_simulation_type="discrete-non-rigid",
    )
    device = torch.device("cpu")
    torch.manual_seed(params.seed)
    data = DataLoader(params=params, t_device=device, sp_device=None)
    recon = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        kspace_scale=data.kspace_scale,
    )
    d = recon.downsample_data(params.ResolutionLevels[0])
    nx, ny = d["Nx"], d["Ny"]
    img = recon.resize_img_2D(data.image_no_moco.squeeze(-1), (nx, ny)).to(device)
    d["ReconstructedImage"] = img

    if params.motion_type == "rigid":
        d["MotionModel"] = torch.zeros((recon.Nalpha, params.N_mot_states), device=device)
        nparams = recon.Nalpha * params.N_mot_states
    else:
        d["MotionModel"] = torch.zeros((recon.Nalpha, nx, ny), device=device)
        nparams = recon.Nalpha * nx * ny

    d["MotionOperator"] = recon.build_motion_operator(d)
    d["J"] = recon.build_motion_perturbation_simulator(d)
    return params, recon, d, img, nparams


def _unvec_motion(params, recon, d, v):
    if params.motion_type == "rigid":
        return v.reshape(recon.Nalpha, params.N_mot_states)
    return v.reshape(recon.Nalpha, d["Nx"], d["Ny"])


def _E_of_alpha(recon, d, img, alpha):
    d2 = dict(d)
    d2["MotionModel"] = alpha
    d2["MotionOperator"] = recon.build_motion_operator(d2)
    E2 = recon.build_encoding_operator(d2)
    return E2.forward(img.flatten())


def _derivative_rel_err(recon, d, img, v):
    params = recon.params
    v_map = _unvec_motion(params, recon, d, v)
    Jv = d["J"].forward(v)
    h = 1e-4
    fd = (
        _E_of_alpha(recon, d, img, d["MotionModel"] + h * v_map)
        - _E_of_alpha(recon, d, img, d["MotionModel"] - h * v_map)
    ) / (2 * h)
    return (
        torch.linalg.norm((Jv - fd).reshape(-1))
        / (torch.linalg.norm(fd.reshape(-1)) + 1e-12)
    ).item()


def test_current_gradient_convention_beats_swapped_axes():
    _, recon, d, img, nparams = _build_context()
    v = torch.randn(nparams, device=img.device, dtype=torch.float64)

    err_current = _derivative_rel_err(recon, d, img, v)

    j = d["J"]
    grad_orig = j.gradient_2d

    def swapped_grad(z):
        gx, gy = grad_orig(z)
        return gy, gx

    j.gradient_2d = swapped_grad
    err_swapped = _derivative_rel_err(recon, d, img, v)
    j.gradient_2d = grad_orig

    print(f"Derivative rel error (current gradient): {err_current:.6e}")
    print(f"Derivative rel error (swapped gradient): {err_swapped:.6e}")

    assert err_current <= err_swapped + 1e-8, (
        f"Swapped gradient unexpectedly better ({err_swapped:.6e} < {err_current:.6e})."
    )


if __name__ == "__main__":
    test_current_gradient_convention_beats_swapped_axes()

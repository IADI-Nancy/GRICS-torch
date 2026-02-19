import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor


def _build_lowres_context():
    params = load_config(
        [
            "config/general.toml",
            "config/shepp_logan.toml",
            "config/sampling_simulation/interleaved.toml",
            "config/motion_simulation/discrete_nonrigid.toml",
            "config/reconstruction/nonrigid_fast.toml",
        ]
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

    res_factor = params.ResolutionLevels[0]
    d = recon.downsample_data(res_factor)
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
    d["E"] = recon.build_encoding_operator(d)
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


def test_j_matches_directional_derivative_of_e():
    params, recon, d, img, nparams = _build_lowres_context()
    v = torch.randn(nparams, device=img.device, dtype=torch.float64)
    Jv = d["J"].forward(v)
    v_map = _unvec_motion(params, recon, d, v)

    rel_errs = []
    for h in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
        fd = (
            _E_of_alpha(recon, d, img, d["MotionModel"] + h * v_map)
            - _E_of_alpha(recon, d, img, d["MotionModel"] - h * v_map)
        ) / (2 * h)
        rel = (
            torch.linalg.norm((Jv - fd).reshape(-1))
            / (torch.linalg.norm(fd.reshape(-1)) + 1e-12)
        ).item()
        rel_errs.append((h, rel))

    print("J derivative consistency (h, rel_err):", rel_errs)

    # Tight threshold by design: this test should fail if J is not the actual derivative.
    assert rel_errs[-1][1] < 5e-2, f"Derivative mismatch too large at smallest h: {rel_errs[-1][1]:.6e}"


if __name__ == "__main__":
    test_j_matches_directional_derivative_of_e()

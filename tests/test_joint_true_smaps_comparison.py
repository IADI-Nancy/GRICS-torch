import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.runtime_config import load_config

from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor


def _relative_residual(recon, data, img, alpha, params):
    nx, ny = img.shape[-2], img.shape[-1]
    d = {
        "Nx": nx,
        "Ny": ny,
        "SensitivityMaps": recon.Data_full["SensitivityMaps"],
        "SamplingIndices": recon.Data_full["SamplingIndices"],
        "Nsamples": nx * ny,
        "ReconstructedImage": img,
        "MotionModel": alpha,
    }
    d["MotionOperator"] = recon.build_motion_operator(d)
    d["E"] = recon.build_encoding_operator(d)

    s = data.kspace[..., 0].reshape(data.Ncha, params.Nex, nx * ny).flatten()
    y = d["E"].forward(img.flatten())
    rel = (torch.linalg.norm((s - y).flatten()) / (torch.linalg.norm(s.flatten()) + 1e-12)).item()
    return rel


def test_joint_generated_vs_espirit_smaps():
    params = load_config(
        data_type="shepp-logan",
        motion_type="non-rigid",
        reconstruction_config="config/reconstruction/nonrigid_fast.toml",
        shepp_logan_config="config/shepp_logan.toml",
        sampling_config="config/sampling_simulation/interleaved.toml",
        kspace_sampling_type="interleaved",
        motion_simulation_config="config/motion_simulation/discrete_nonrigid.toml",
        motion_simulation_type="discrete-non-rigid",
        overrides={
            "verbose": False,
            "debug_flag": False,
            "max_restarts": 1,
            "ResolutionLevels": [1.0],
            "GN_iterations_per_level": [3],
            "max_iter_recon": 64,
            "max_iter_motion": 64,
        },
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if params.data_type != "shepp-logan":
        raise RuntimeError("This comparison test is intended for shepp-logan data_type.")
    if params.motion_type != "non-rigid" or params.motion_simulation_type != "discrete-non-rigid":
        raise RuntimeError("Set motion_type='non-rigid' and motion_simulation_type='discrete-non-rigid'.")

    data = DataLoader(params=params, t_device=device, sp_device=None)
    if not hasattr(data, "smaps_generated"):
        raise RuntimeError("Generated sensitivity maps are unavailable in DataLoader.")
    if not hasattr(data, "alpha_maps_true"):
        raise RuntimeError("Ground-truth alpha maps are unavailable in DataLoader.")

    # --- ESPIRiT maps ---
    recon_esp = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        kspace_scale=data.kspace_scale,
    )
    img_esp, alpha_esp = recon_esp.run()
    alpha_esp = alpha_esp.real

    # --- Generated (ground-truth) maps ---
    recon_gen = JointReconstructor(
        data.kspace,
        data.smaps_generated,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        kspace_scale=data.kspace_scale,
    )
    img_gen, alpha_gen = recon_gen.run()
    alpha_gen = alpha_gen.real

    alpha_true = data.alpha_maps_true.to(device)
    alpha_rel_esp = (
        torch.linalg.norm((alpha_esp - alpha_true).flatten()) / (torch.linalg.norm(alpha_true.flatten()) + 1e-12)
    ).item()
    alpha_rel_gen = (
        torch.linalg.norm((alpha_gen - alpha_true).flatten()) / (torch.linalg.norm(alpha_true.flatten()) + 1e-12)
    ).item()

    res_rel_esp = _relative_residual(recon_esp, data, img_esp, alpha_esp, params)
    res_rel_gen = _relative_residual(recon_gen, data, img_gen, alpha_gen, params)

    print(f"[JOINT-ESPIRiT] rel_res={res_rel_esp:.6e}, alpha_rel={alpha_rel_esp:.6e}")
    print(f"[JOINT-Generated] rel_res={res_rel_gen:.6e}, alpha_rel={alpha_rel_gen:.6e}")

    # Keep this test diagnostic and lightweight: ensure finite outputs.
    assert torch.isfinite(torch.tensor(res_rel_esp))
    assert torch.isfinite(torch.tensor(res_rel_gen))
    assert torch.isfinite(torch.tensor(alpha_rel_esp))
    assert torch.isfinite(torch.tensor(alpha_rel_gen))


if __name__ == "__main__":
    test_joint_generated_vs_espirit_smaps()

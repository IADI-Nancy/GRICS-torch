import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Parameters import Parameters

# Keep this test reasonably fast while preserving joint behavior.
Parameters.verbose = False
Parameters.debug_flag = False
Parameters.max_restarts = 1
Parameters.ResolutionLevels = [1.0]
Parameters.GN_iterations_per_level = 3
Parameters.max_iter_recon = 64
Parameters.max_iter_motion = 64

from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor


def _relative_residual(recon, data, img, alpha):
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

    s = data.kspace[..., 0].reshape(data.Ncha, Parameters.Nex, nx * ny).flatten()
    y = d["E"].forward(img.flatten())
    rel = (torch.linalg.norm((s - y).flatten()) / (torch.linalg.norm(s.flatten()) + 1e-12)).item()
    return rel


def test_joint_generated_vs_espirit_smaps():
    params = Parameters()
    device = torch.device("gpu")

    if params.data_type != "shepp-logan":
        raise RuntimeError("This comparison test is intended for shepp-logan data_type.")
    if params.motion_type != "non-rigid" or params.simulation_type != "discrete-non-rigid":
        raise RuntimeError("Set motion_type='non-rigid' and simulation_type='discrete-non-rigid'.")

    data = DataLoader(t_device=device, sp_device=None)
    if not hasattr(data, "smaps_generated"):
        raise RuntimeError("Generated sensitivity maps are unavailable in DataLoader.")
    if not hasattr(data, "alpha_maps_true"):
        raise RuntimeError("Ground-truth alpha maps are unavailable in DataLoader.")

    # --- ESPIRiT maps ---
    recon_esp = JointReconstructor(data.kspace, data.smaps, data.sampling_idx, motion_signal=data.motion_signal)
    img_esp, alpha_esp = recon_esp.run()
    alpha_esp = alpha_esp.real

    # --- Generated (ground-truth) maps ---
    recon_gen = JointReconstructor(
        data.kspace, data.smaps_generated, data.sampling_idx, motion_signal=data.motion_signal
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

    res_rel_esp = _relative_residual(recon_esp, data, img_esp, alpha_esp)
    res_rel_gen = _relative_residual(recon_gen, data, img_gen, alpha_gen)

    print(f"[JOINT-ESPIRiT] rel_res={res_rel_esp:.6e}, alpha_rel={alpha_rel_esp:.6e}")
    print(f"[JOINT-Generated] rel_res={res_rel_gen:.6e}, alpha_rel={alpha_rel_gen:.6e}")

    # Keep this test diagnostic and lightweight: ensure finite outputs.
    assert torch.isfinite(torch.tensor(res_rel_esp))
    assert torch.isfinite(torch.tensor(res_rel_gen))
    assert torch.isfinite(torch.tensor(alpha_rel_esp))
    assert torch.isfinite(torch.tensor(alpha_rel_gen))


if __name__ == "__main__":
    test_joint_generated_vs_espirit_smaps()

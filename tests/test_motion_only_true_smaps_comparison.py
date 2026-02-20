import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.utils.fftnc import ifftnc


def _build_data_res(recon, x_fixed, alpha):
    nx, ny = x_fixed.shape[-2], x_fixed.shape[-1]
    data_res = {
        "Nx": nx,
        "Ny": ny,
        "SensitivityMaps": recon.Data_full["SensitivityMaps"],
        "SamplingIndices": recon.Data_full["SamplingIndices"],
        "Nsamples": nx * ny,
        "ReconstructedImage": x_fixed,
        "MotionModel": alpha,
    }
    data_res["MotionOperator"] = recon.build_motion_operator(data_res)
    data_res["E"] = recon.build_encoding_operator(data_res)
    data_res["J"] = recon.build_motion_perturbation_simulator(data_res)
    return data_res


def _run_motion_only(recon, x_fixed, y_meas, alpha_true, n_iter=4):
    alpha_est = torch.zeros_like(alpha_true)
    res_hist = []
    alpha_hist = []

    for _ in range(n_iter):
        d = _build_data_res(recon, x_fixed, alpha_est)
        y_est = d["E"].forward(x_fixed.flatten())
        residual = y_meas - y_est
        dm = recon.solve_motion(d, residual)
        alpha_est = alpha_est + dm.real

        res_norm = torch.linalg.norm(residual).item()
        alpha_rel = (
            torch.linalg.norm((alpha_est - alpha_true).flatten())
            / (torch.linalg.norm(alpha_true.flatten()) + 1e-12)
        ).item()
        res_hist.append(res_norm)
        alpha_hist.append(alpha_rel)

    return res_hist, alpha_hist


def test_motion_only_generated_vs_espirit_smaps():
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

    # Measured (motion-corrupted) sampled k-space vector.
    y_meas = data.kspace[..., 0].reshape(data.Ncha, params.Nex, data.Nx * data.Ny).flatten()

    # Motion-free reference image under each set of maps.
    img_nomotion = ifftnc(data.kspace_nomotion, dims=(-3, -2, -1))
    x_esp = torch.sum(
        img_nomotion * data.smaps.conj().unsqueeze(1).expand(-1, params.Nex, -1, -1, -1), dim=0
    ).squeeze(-1).to(device)
    x_true = torch.sum(
        img_nomotion * data.smaps_generated.conj().unsqueeze(1).expand(-1, params.Nex, -1, -1, -1), dim=0
    ).squeeze(-1).to(device)

    alpha_true = data.alpha_maps_true.to(device)

    recon_esp = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        kspace_scale=data.kspace_scale,
    )
    recon_true = JointReconstructor(
        data.kspace,
        data.smaps_generated,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        kspace_scale=data.kspace_scale,
    )

    res_esp, alpha_esp = _run_motion_only(recon_esp, x_esp, y_meas, alpha_true, n_iter=4)
    res_true, alpha_true_hist = _run_motion_only(recon_true, x_true, y_meas, alpha_true, n_iter=4)

    print(
        f"[ESPIRiT] residual first={res_esp[0]:.6e}, last={res_esp[-1]:.6e}, "
        f"alpha_rel first={alpha_esp[0]:.6e}, last={alpha_esp[-1]:.6e}"
    )
    print(
        f"[Generated] residual first={res_true[0]:.6e}, last={res_true[-1]:.6e}, "
        f"alpha_rel first={alpha_true_hist[0]:.6e}, last={alpha_true_hist[-1]:.6e}"
    )

    # Sanity: both should reduce residual over a few iterations.
    assert res_esp[-1] < res_esp[0]
    assert res_true[-1] < res_true[0]


if __name__ == "__main__":
    test_motion_only_generated_vs_espirit_smaps()

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.utils.show_and_save_image import show_and_save_image


def _make_smooth_alpha(nx, ny, amp_x=2.0, amp_y=2.0, device="cpu"):
    x = torch.arange(nx, device=device, dtype=torch.float64)
    y = torch.arange(ny, device=device, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    sx = nx / 6.0
    sy = ny / 6.0
    g = torch.exp(-((xx - nx / 2.0) ** 2) / (2 * sx * sx) - ((yy - ny / 2.0) ** 2) / (2 * sy * sy))
    alpha = torch.zeros((2, nx, ny), device=device, dtype=torch.float64)
    alpha[0] = amp_x * g
    alpha[1] = amp_y * g
    return alpha


def _build_data_res(recon, x_true, alpha):
    nx, ny = x_true.shape[-2], x_true.shape[-1]
    data_res = {
        "Nx": nx,
        "Ny": ny,
        "SensitivityMaps": recon.Data_full["SensitivityMaps"],
        "SamplingIndices": recon.Data_full["SamplingIndices"],
        "Nsamples": nx * ny,
        "ReconstructedImage": x_true,
        "MotionModel": alpha,
    }
    data_res["MotionOperator"] = recon.build_motion_operator(data_res)
    data_res["E"] = recon.build_encoding_operator(data_res)
    data_res["J"] = recon.build_motion_perturbation_simulator(data_res)
    return data_res


def run_motion_only_checks():
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
    debug_folder = Path(params.debug_folder) / "test_motion_only_optimization"
    debug_folder.mkdir(parents=True, exist_ok=True)

    if params.motion_type != "non-rigid":
        raise RuntimeError("Set motion_type='non-rigid' for this script.")
    if params.simulation_type != "discrete-non-rigid":
        raise RuntimeError("Set simulation_type='discrete-non-rigid' for this script.")

    data = DataLoader(params=params, t_device=device, sp_device=None)
    x_true = data.image_ground_truth.squeeze(-1).to(device)
    recon = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        kspace_scale=data.kspace_scale,
    )

    nx, ny = x_true.shape[-2], x_true.shape[-1]
    alpha_true = _make_smooth_alpha(nx, ny, amp_x=2.0, amp_y=2.0, device=device)

    # Build "measured" k-space from known image + known true motion model.
    d_true = _build_data_res(recon, x_true, alpha_true)
    y_true = d_true["E"].forward(x_true.flatten())

    # -------------------- 1) Linearized recovery test --------------------
    alpha_ref = torch.zeros_like(alpha_true)
    d_lin = _build_data_res(recon, x_true, alpha_ref)

    delta_true = _make_smooth_alpha(nx, ny, amp_x=0.3, amp_y=-0.25, device=device)
    residual_lin = d_lin["J"].forward(delta_true.flatten())
    delta_est = recon.solve_motion(d_lin, residual_lin)

    lin_rel_err = (
        torch.linalg.norm((delta_est - delta_true).flatten())
        / (torch.linalg.norm(delta_true.flatten()) + 1e-12)
    ).item()

    print(f"[MOTION-ONLY] linearized delta recovery rel_err={lin_rel_err:.6e}")

    show_and_save_image(delta_true[0], "motion_only_linear_delta_true_x", str(debug_folder))
    show_and_save_image(delta_true[1], "motion_only_linear_delta_true_y", str(debug_folder))
    show_and_save_image(delta_est[0], "motion_only_linear_delta_est_x", str(debug_folder))
    show_and_save_image(delta_est[1], "motion_only_linear_delta_est_y", str(debug_folder))
    show_and_save_image(torch.abs(delta_est[0] - delta_true[0]), "motion_only_linear_abs_err_x", str(debug_folder))
    show_and_save_image(torch.abs(delta_est[1] - delta_true[1]), "motion_only_linear_abs_err_y", str(debug_folder))

    # -------------------- 2) Nonlinear GN motion-only loop --------------------
    alpha_est = torch.zeros_like(alpha_true)
    n_iter = 12
    res_hist = []
    alpha_hist = []

    for _ in range(n_iter):
        d = _build_data_res(recon, x_true, alpha_est)
        y_est = d["E"].forward(x_true.flatten())
        residual = y_true - y_est

        dm = recon.solve_motion(d, residual)
        alpha_est = alpha_est + dm.real

        res_norm = torch.linalg.norm(residual).item()
        alpha_rel = (
            torch.linalg.norm((alpha_est - alpha_true).flatten())
            / (torch.linalg.norm(alpha_true.flatten()) + 1e-12)
        ).item()
        res_hist.append(res_norm)
        alpha_hist.append(alpha_rel)

    print(f"[MOTION-ONLY] final alpha rel_err={alpha_hist[-1]:.6e}")
    print(f"[MOTION-ONLY] residual first={res_hist[0]:.6e}, last={res_hist[-1]:.6e}")

    show_and_save_image(alpha_true[0], "motion_only_alpha_true_x", str(debug_folder))
    show_and_save_image(alpha_true[1], "motion_only_alpha_true_y", str(debug_folder))
    show_and_save_image(alpha_est[0], "motion_only_alpha_est_x", str(debug_folder))
    show_and_save_image(alpha_est[1], "motion_only_alpha_est_y", str(debug_folder))
    show_and_save_image(torch.abs(alpha_est[0] - alpha_true[0]), "motion_only_alpha_abs_err_x", str(debug_folder))
    show_and_save_image(torch.abs(alpha_est[1] - alpha_true[1]), "motion_only_alpha_abs_err_y", str(debug_folder))

    plt.figure(figsize=(6, 4))
    plt.plot(res_hist, marker="o")
    plt.xlabel("GN iteration")
    plt.ylabel("||residual||2")
    plt.title("Motion-only GN residual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(debug_folder / "motion_only_residual_curve.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(alpha_hist, marker="o")
    plt.xlabel("GN iteration")
    plt.ylabel("relative alpha error")
    plt.title("Motion-only GN alpha error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(debug_folder / "motion_only_alpha_error_curve.png")
    plt.close()


if __name__ == "__main__":
    run_motion_only_checks()

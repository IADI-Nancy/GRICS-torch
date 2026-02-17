import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Parameters import Parameters
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.utils.show_and_save_image import show_and_save_image


def _make_known_nonrigid_alpha(nx, ny, amp_x=2.0, amp_y=-1.5, device="cpu"):
    x = torch.arange(nx, device=device, dtype=torch.float32)
    y = torch.arange(ny, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    sx = nx / 6.0
    sy = ny / 6.0
    g1 = torch.exp(-((xx - nx * 0.35) ** 2) / (2 * sx * sx) - ((yy - ny * 0.45) ** 2) / (2 * sy * sy))
    g2 = torch.exp(-((xx - nx * 0.65) ** 2) / (2 * sx * sx) - ((yy - ny * 0.60) ** 2) / (2 * sy * sy))

    alpha = torch.zeros((2, nx, ny), device=device, dtype=torch.float32)
    alpha[0] = amp_x * (g1 - 0.6 * g2)
    alpha[1] = amp_y * (0.7 * g1 + g2)
    return alpha


def run_known_motion_image_recon():
    params = Parameters()
    device = torch.device("cpu")

    if params.motion_type != "non-rigid":
        raise RuntimeError("Set motion_type='non-rigid' to use this test.")
    if params.simulation_type != "discrete-non-rigid":
        raise RuntimeError("Set simulation_type='discrete-non-rigid' to use this test.")

    torch.manual_seed(params.seed)
    data = DataLoader(t_device=device, sp_device=None)

    x_true = data.image_ground_truth.squeeze(-1).to(device)
    nx, ny = x_true.shape[-2], x_true.shape[-1]
    ncoils = data.smaps.shape[0]
    nsamples = nx * ny

    # Build a known (synthetic) non-rigid motion model and generate synthetic data.
    alpha_known = _make_known_nonrigid_alpha(nx, ny, amp_x=2.0, amp_y=-1.5, device=device)
    motion_known = MotionOperator(
        nx,
        ny,
        alpha_known,
        params.motion_type,
        motion_signal=data.motion_signal,
    )
    E_known = EncodingOperator(
        data.smaps,
        nsamples,
        data.sampling_idx,
        params.Nex,
        motion_known,
    )
    y_meas = E_known.forward(x_true.flatten())

    # Image-only reconstruction with known motion and Laplacian regularization.
    b = E_known.adjoint(y_meas)
    x0 = torch.zeros_like(b)
    solver_lap = ConjugateGradientSolver(
        E_known,
        reg_lambda=params.lambda_m,
        regularizer="Tikhonov_laplacian",
        regularization_shape=(params.Nex, nx, ny),
        verbose=True,
    )
    x_lap_vec = solver_lap.solve_cg_keep_best(
        b.flatten(),
        x0=x0.flatten(),
        max_iter=params.max_iter_recon,
        tol=params.tol_recon,
    )
    x_lap = x_lap_vec.reshape(params.Nex, nx, ny)

    # Baseline: no regularization.
    solver_none = ConjugateGradientSolver(E_known, reg_lambda=0.0, verbose=True)
    x_none_vec = solver_none.solve_cg_keep_best(
        b.flatten(),
        x0=x0.flatten(),
        max_iter=params.max_iter_recon,
        tol=params.tol_recon,
    )
    x_none = x_none_vec.reshape(params.Nex, nx, ny)

    rel_lap = (
        torch.linalg.norm((x_lap - x_true).flatten())
        / (torch.linalg.norm(x_true.flatten()) + 1e-12)
    ).item()
    rel_none = (
        torch.linalg.norm((x_none - x_true).flatten())
        / (torch.linalg.norm(x_true.flatten()) + 1e-12)
    ).item()

    print(f"[KNOWN-MOTION][IMAGE-ONLY] rel_err_laplacian={rel_lap:.6e}")
    print(f"[KNOWN-MOTION][IMAGE-ONLY] rel_err_no_reg={rel_none:.6e}")
    print(f"[KNOWN-MOTION][IMAGE-ONLY] lambda_for_laplacian={params.lambda_m:.6e}")
    print(f"[KNOWN-MOTION][IMAGE-ONLY] kspace_shape={(ncoils, params.Nex, nsamples)}")

    show_and_save_image(x_true[0], "known_motion_image_true", params.debug_folder)
    show_and_save_image(x_lap[0], "known_motion_image_recon_laplacian", params.debug_folder)
    show_and_save_image(x_none[0], "known_motion_image_recon_noreg", params.debug_folder)
    show_and_save_image(torch.abs(x_lap[0] - x_true[0]), "known_motion_image_abs_err_laplacian", params.debug_folder)
    show_and_save_image(torch.abs(x_none[0] - x_true[0]), "known_motion_image_abs_err_noreg", params.debug_folder)
    show_and_save_image(alpha_known[0], "known_motion_alpha_x", params.debug_folder)
    show_and_save_image(alpha_known[1], "known_motion_alpha_y", params.debug_folder)


if __name__ == "__main__":
    run_known_motion_image_recon()

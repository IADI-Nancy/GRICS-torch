import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from Parameters import Parameters


def _complex_randn(shape, device, dtype=torch.complex128):
    real = torch.randn(*shape, device=device, dtype=torch.float64)
    imag = torch.randn(*shape, device=device, dtype=torch.float64)
    return torch.complex(real, imag).to(dtype)


def _relative_adjoint_error(forward, adjoint, x, y):
    ax = forward(x)
    ahy = adjoint(y)
    lhs = torch.vdot(ax.reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), ahy.reshape(-1))
    rel = torch.abs(lhs - rhs) / (torch.abs(lhs) + torch.abs(rhs) + 1e-12)
    return rel.item(), lhs, rhs


def _build_lowres_operators():
    params = Parameters()
    if params.motion_type != "non-rigid":
        raise RuntimeError("This test is intended for non-rigid motion_type.")

    torch.manual_seed(params.seed)
    device = torch.device("cpu")
    data = DataLoader(t_device=device, sp_device=None)

    recon = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
    )

    res_factor = params.ResolutionLevels[0]
    data_res = recon.downsample_data(res_factor)
    nx, ny = data_res["Nx"], data_res["Ny"]

    image_init = data.image_no_moco.squeeze(-1)
    data_res["ReconstructedImage"] = recon.resize_img_2D(image_init, (nx, ny))
    data_res["MotionModel"] = torch.zeros((recon.Nalpha, nx, ny), device=device)
    data_res["MotionOperator"] = recon.build_motion_operator(data_res)
    data_res["E"] = recon.build_encoding_operator(data_res)
    data_res["J"] = recon.build_motion_perturbation_simulator(data_res)

    return params, recon, data_res


def test_encoding_operator_adjointness():
    _, _, data_res = _build_lowres_operators()
    e = data_res["E"]
    ncoils = data_res["SensitivityMaps"].shape[0]
    nex = Parameters().Nex
    nx, ny = data_res["Nx"], data_res["Ny"]
    nsamples = data_res["Nsamples"]

    x = _complex_randn((nex * nx * ny,), device=e.device)
    y = _complex_randn((ncoils * nex * nsamples,), device=e.device)

    rel_err, lhs, rhs = _relative_adjoint_error(e.forward, e.adjoint, x, y)
    assert rel_err < 1e-4, (
        f"EncodingOperator adjoint mismatch: rel_err={rel_err:.3e}, "
        f"<Ex,y>={lhs}, <x,EHy>={rhs}"
    )


def test_motion_perturbation_operator_adjointness():
    _, recon, data_res = _build_lowres_operators()
    j = data_res["J"]
    ncoils = data_res["SensitivityMaps"].shape[0]
    nex = Parameters().Nex
    nx, ny = data_res["Nx"], data_res["Ny"]
    nsamples = data_res["Nsamples"]

    nparams = recon.Nalpha * nx * ny
    x = _complex_randn((nparams,), device=j.device)
    y = _complex_randn((ncoils * nex * nsamples,), device=j.device)

    rel_err, lhs, rhs = _relative_adjoint_error(j.forward, j.adjoint, x, y)
    assert rel_err < 5e-4, (
        f"MotionPerturbationSimulator adjoint mismatch: rel_err={rel_err:.3e}, "
        f"<Jx,y>={lhs}, <x,JHy>={rhs}"
    )


if __name__ == "__main__":
    failures = 0
    try:
        test_encoding_operator_adjointness()
        print("EncodingOperator adjointness: OK")
    except AssertionError as exc:
        failures += 1
        print(f"EncodingOperator adjointness: FAIL\n{exc}")

    try:
        test_motion_perturbation_operator_adjointness()
        print("MotionPerturbationSimulator adjointness: OK")
    except AssertionError as exc:
        failures += 1
        print(f"MotionPerturbationSimulator adjointness: FAIL\n{exc}")

    if failures:
        raise SystemExit(1)

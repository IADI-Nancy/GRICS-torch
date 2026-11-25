"""Rigid motion step damped by lambda_m diag(Re(J^H J)) (rigid_motion_damping = "diagonal")."""
import math
from types import SimpleNamespace

import torch

from conftest import smooth_phantom
from src.reconstruction.JointReconstructor import JointReconstructor
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.reconstruction.joint_reconstructor_utils.operators import DiagonallyScaledOperator, RealNormalOperator, rigid_normal_diagonal

N, NM, NC = 48, 4, 3


def _problem(device, empty_state=None):
    """Rigid Jacobian of NM states with disjoint complete readout lines, and a random k-space residual."""
    g = torch.Generator().manual_seed(5)
    smaps = torch.complex(torch.randn(NC, N, N, 1, generator=g), torch.randn(NC, N, N, 1, generator=g)).to(device)
    rows = torch.randperm(N, generator=g).to(device)
    sampling = [[(torch.arange(N, device=device)[:, None] * N + rows[s::NM][None, :]).reshape(-1) for s in range(NM)]]
    if empty_state is not None:
        sampling[0][empty_state] = sampling[0][empty_state][:0]
    alpha = torch.tensor([[0.3, -0.2, 0.5, 0.1], [0.1, 0.4, -0.3, 0.2], [0.02, -0.01, 0.03, 0.0]], device=device)
    J = MotionPerturbationSimulator(smaps, N * N, sampling, 1, smooth_phantom(N, N, device), MotionOperator(N, N, alpha, "rigid"))
    residual = torch.complex(torch.randn(NC * N * N, generator=g), torch.randn(NC * N * N, generator=g)).to(device)
    return J, alpha, residual


def _dense(operator, device):
    return torch.stack([operator.normal(e) for e in torch.eye(3 * NM, device=device)], dim=1)


def _motion_step(J, alpha, residual, damping):
    """JointReconstructor._solve_motion on the rigid branch with the shipped CG settings."""
    reconstructor = JointReconstructor.__new__(JointReconstructor)
    reconstructor.params = SimpleNamespace(
        reconstruction_motion_type="rigid", rigid_motion_damping=damping, motion_step_real=True, lambda_m=0.2,
        max_iter_motion=128, tol_motion=1e-10, verbose=False, cg_stop_on_stagnation=False, cg_true_residual_interval=10,
        cg_stagnation_consecutive_steps=6, cg_stagnation_countdown_steps=3, cg_use_reg_scale_proxy=True, cg_reg_scale_num_probes=8)
    reconstructor.Nalpha, reconstructor.regularization_scaling = 3, "direct"
    torch.manual_seed(0)
    return reconstructor._solve_motion({"J": J, "MotionModel": alpha}, residual).flatten()


def test_probes_give_the_diagonal_of_the_block_diagonal_normal_matrix(device):
    J, _, _ = _problem(device)
    A = _dense(RealNormalOperator(J), device)
    state = torch.arange(3 * NM, device=device) % NM
    assert A[state[:, None] != state[None, :]].abs().max() == 0          # the states do not couple
    diagonal = rigid_normal_diagonal(RealNormalOperator(J), 3, NM, torch.float64).flatten()
    assert torch.allclose(diagonal, torch.diagonal(A), rtol=1e-12, atol=0)


def test_motion_step_solves_the_diagonally_damped_system(device):
    J, alpha, residual = _problem(device)
    A = _dense(RealNormalOperator(J), device)
    expected = torch.linalg.solve(A + 0.2 * torch.diag(torch.diagonal(A)), RealNormalOperator(J).adjoint(residual))
    step = _motion_step(J, alpha, residual, "diagonal")
    assert (step - expected).norm() < 1e-8 * expected.norm()


def test_step_does_not_depend_on_the_rotation_unit(device):
    # The same Jacobian with rotations in degrees: J C, C = diag(1, 1, pi / 180). The physical step C d_deg must equal
    # d_rad with the diagonal damping; the identity damping weighs the rotation 180 / pi times differently.
    J, alpha, residual = _problem(device)
    c = torch.ones(3, NM, device=device)
    c[2] = math.pi / 180
    c = c.flatten()
    J_degrees = DiagonallyScaledOperator(J, c)
    radians, degrees = _motion_step(J, alpha, residual, "diagonal"), _motion_step(J_degrees, alpha, residual, "diagonal")
    assert (c * degrees - radians).norm() < 1e-8 * radians.norm()
    radians, degrees = _motion_step(J, alpha, residual, "identity"), _motion_step(J_degrees, alpha, residual, "identity")
    assert (c * degrees - radians).norm() > 0.1 * radians.norm()


def test_parameters_of_a_state_without_data_are_not_updated(device, capsys):
    J, alpha, residual = _problem(device, empty_state=2)
    step = _motion_step(J, alpha, residual, "diagonal").reshape(3, NM)
    assert torch.isfinite(step).all() and step[:, 2].abs().max() == 0
    assert "not updated" in capsys.readouterr().out
    A = _dense(RealNormalOperator(J), device)
    keep = (torch.arange(3 * NM, device=device) % NM) != 2
    A, b = A[keep][:, keep], RealNormalOperator(J).adjoint(residual)[keep]
    expected = torch.linalg.solve(A + 0.2 * torch.diag(torch.diagonal(A)), b)
    assert (step.flatten()[keep] - expected).norm() < 1e-8 * expected.norm()

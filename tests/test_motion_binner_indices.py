import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.MotionBinner import MotionBinner
from Parameters import Parameters


def _build_synthetic_inputs(nex, n_states, ny, device):
    # Initial shot/state-wise ky assignment (ground truth to recover)
    init_bins = [[[] for _ in range(n_states)] for _ in range(nex)]
    chunk = ny // n_states
    rem = ny % n_states

    for n in range(nex):
        start = 0
        for s in range(n_states):
            end = start + chunk + (1 if s < rem else 0)
            init_bins[n][s] = torch.arange(start, end, device=device, dtype=torch.int64)
            start = end

    # Chronological vectors as expected by MotionBinner.bin_motion
    ky_idx = []
    nex_idx = []
    motion_curve = []

    # Distinct per-state signal values
    state_signal = torch.tensor([-1.4, -0.2, 0.9, 2.1], device=device, dtype=torch.float32)[:n_states]

    for n in range(nex):
        ky_idx.append(torch.cat(init_bins[n], dim=0).to(torch.int32))
        nex_idx.append(torch.full((ny,), n, device=device, dtype=torch.int32))
        motion_curve.append(
            torch.cat(
                [
                    torch.full((init_bins[n][s].numel(),), state_signal[s], device=device)
                    for s in range(n_states)
                ],
                dim=0,
            )
        )

    motion_curve = torch.cat(motion_curve, dim=0)
    return init_bins, ky_idx, nex_idx, motion_curve, state_signal


def _best_center_permutation(centers, state_signal):
    n = centers.numel()
    best_perm = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(n)):
        cost = torch.sum(torch.abs(centers[list(perm)] - state_signal)).item()
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return list(best_perm)


def test_motion_binner_recovers_initial_sampling_indices():
    params = Parameters()
    device = torch.device("cpu")

    nex = params.Nex
    n_states = params.N_mot_states
    ny = 32

    init_bins, ky_idx, nex_idx, motion_curve, state_signal = _build_synthetic_inputs(
        nex=nex,
        n_states=n_states,
        ny=ny,
        device=device,
    )

    binned_indices, centers = MotionBinner.bin_motion(motion_curve, ky_idx, nex_idx, device)

    # Cluster labels can be permuted; align by center values first.
    perm = _best_center_permutation(centers, state_signal)

    for n in range(nex):
        for s in range(n_states):
            b = perm[s]
            expected = torch.sort(init_bins[n][s]).values
            got = torch.sort(binned_indices[n][b].to(expected.dtype)).values
            assert torch.equal(expected, got), (
                f"Mismatch for nex={n}, state={s}, mapped_cluster={b}. "
                f"expected={expected.tolist()}, got={got.tolist()}"
            )


if __name__ == "__main__":
    test_motion_binner_recovers_initial_sampling_indices()
    print("MotionBinner sampling-index recovery: OK")

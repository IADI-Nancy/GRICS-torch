import torch
import numpy as np

from src.preprocessing.SamplingSimulator import SamplingSimulator


def require_motion_param(params, name):
    if not hasattr(params, name):
        raise AttributeError(f"MotionSimulator requires config parameter '{name}'.")
    return getattr(params, name)


def rigid_motion_amplitude_scale(params):
    scale = float(getattr(params, "rigid_motion_amplitude_scale", 1.0))
    if scale < 0:
        raise ValueError("rigid_motion_amplitude_scale must be >= 0.")
    return scale


def translation_limits_px(params, Nx, Ny, Nz):
    """
    Convert rigid-motion translations and rotation-center offsets from mm to
    pixels/voxels. x/y use FoVxy_mm, z uses FoVz_mm.
    """
    rigid_amp_scale = rigid_motion_amplitude_scale(params)
    mm_to_px_xy = float(Nx) / float(params.FoVxy_mm)
    mm_to_py_xy = float(Ny) / float(params.FoVxy_mm)

    if Nz == 1:
        return {
            "max_tx_px": float(params.max_tx) * rigid_amp_scale * mm_to_px_xy,
            "max_ty_px": float(params.max_ty) * rigid_amp_scale * mm_to_py_xy,
            "max_center_x_px": float(params.max_center_x) * mm_to_px_xy,
            "max_center_y_px": float(params.max_center_y) * mm_to_py_xy,
        }

    mm_to_pz = float(Nz) / float(params.FoVz_mm)
    return {
        "max_tx_3d_px": float(params.max_tx_3d) * rigid_amp_scale * mm_to_px_xy,
        "max_ty_3d_px": float(params.max_ty_3d) * rigid_amp_scale * mm_to_py_xy,
        "max_tz_3d_px": float(params.max_tz_3d) * rigid_amp_scale * mm_to_pz,
        "max_center_x_3d_px": float(params.max_center_x_3d) * mm_to_px_xy,
        "max_center_y_3d_px": float(params.max_center_y_3d) * mm_to_py_xy,
        "max_center_z_3d_px": float(params.max_center_z_3d) * mm_to_pz,
    }


def flatten_index_list(values):
    if values is None:
        return None
    if torch.is_tensor(values):
        return values.reshape(-1)
    return torch.cat([v.reshape(-1) for v in values], dim=0)


def num_motion_readouts(ky_idx):
    ky_flat = flatten_index_list(ky_idx)
    if ky_flat is None:
        raise ValueError("ky_idx is required to determine the number of motion readouts.")
    return int(ky_flat.numel())


def build_sampling_per_line_global_states(ky_idx, nex_idx, kz_idx, *, device, Nx, Ny, Nz, Nex):
    """
    Build sampling indices with one global motion state per acquired readout.
    Output shape is [Nex][Nreadouts_total].
    """
    ky_flat = flatten_index_list(ky_idx).to(torch.int64)
    nex_flat = flatten_index_list(nex_idx).to(torch.int64)
    kz_flat = flatten_index_list(kz_idx)
    if kz_flat is not None:
        kz_flat = kz_flat.to(torch.int64)

    ny_total = int(ky_flat.numel())
    sampling = [
        [torch.empty(0, dtype=torch.int64, device=device) for _ in range(ny_total)]
        for _ in range(int(Nex))
    ]
    kx = torch.arange(Nx, device=device, dtype=torch.int64)
    kz = torch.arange(Nz, device=device, dtype=torch.int64) if Nz > 1 else None

    for state in range(ny_total):
        nex = int(nex_flat[state].item())
        ky = ky_flat[state].reshape(1)
        samp_xy = ky[:, None] + Ny * kx[None, :]
        if Nz > 1 and kz_flat is not None:
            kz_state = kz_flat[state].reshape(1)
            samp = (samp_xy * Nz + kz_state[:, None]).reshape(-1)
        elif Nz > 1:
            samp = (samp_xy[:, :, None] * Nz + kz[None, None, :]).reshape(-1)
        else:
            samp = samp_xy.reshape(-1)
        sampling[nex][state] = samp

    return sampling


def compress_consecutive_rigid_states(
    alpha,
    ky_idx,
    nex_idx,
    *,
    device,
    Nx,
    Ny,
    Nz,
    Nex,
    centers=None,
    kz_idx=None,
):
    """
    Merge consecutive readouts that share the exact same rigid parameters.
    Plateau regions become one motion state; transition readouts stay separate.
    """
    if alpha.ndim != 2:
        raise ValueError("alpha must have shape [Nparams, Nreadouts].")

    n_readouts = int(alpha.shape[1])
    if n_readouts < 1:
        raise ValueError("alpha must contain at least one readout.")

    new_state = torch.ones(n_readouts, dtype=torch.bool, device=device)
    if n_readouts > 1:
        same_as_prev = torch.all(alpha[:, 1:] == alpha[:, :-1], dim=0)
        if centers is not None:
            same_as_prev = same_as_prev & torch.all(centers[:, 1:] == centers[:, :-1], dim=0)
        new_state[1:] = ~same_as_prev

    state_ids = torch.cumsum(new_state.to(torch.int64), dim=0) - 1
    n_states = int(state_ids[-1].item()) + 1

    ky_flat = flatten_index_list(ky_idx).to(torch.int64)
    nex_flat = flatten_index_list(nex_idx).to(torch.int64)
    kz_flat = flatten_index_list(kz_idx)
    if kz_flat is not None:
        kz_flat = kz_flat.to(torch.int64)

    binned_ky = [
        [torch.empty(0, dtype=torch.int64, device=device) for _ in range(n_states)]
        for _ in range(int(Nex))
    ]
    binned_kz = None
    if kz_flat is not None:
        binned_kz = [
            [torch.empty(0, dtype=torch.int64, device=device) for _ in range(n_states)]
            for _ in range(int(Nex))
        ]

    for nex in range(int(Nex)):
        nex_mask = nex_flat == nex
        for state in range(n_states):
            mask = nex_mask & (state_ids == state)
            binned_ky[nex][state] = ky_flat[mask]
            if binned_kz is not None:
                binned_kz[nex][state] = kz_flat[mask]

    compressed_alpha = alpha[:, new_state]
    compressed_centers = None if centers is None else centers[:, new_state]
    sampling_idx = SamplingSimulator._build_sampling_per_nex_per_motion(
        binned_ky,
        device,
        Nx,
        Ny,
        Nz=Nz,
        binned_kz_indices=binned_kz,
    )
    return sampling_idx, compressed_alpha, compressed_centers


def globalize_per_shot_readout_layout(per_shot_readout_layout, *, device):
    """
    Convert a per-Nex shot layout [Nex][NshotsPerNex] into a global shot-state
    layout [Nex][NshotsTotal], where each total shot has its own unique motion
    state and all other states are empty for that Nex.
    """
    if len(per_shot_readout_layout) == 0:
        raise ValueError("per_shot_readout_layout cannot be empty.")

    first_nonempty = None
    for readout_blocks_per_nex in per_shot_readout_layout:
        if len(readout_blocks_per_nex) > 0:
            first_nonempty = readout_blocks_per_nex[0]
            break
    if first_nonempty is None:
        raise ValueError("per_shot_readout_layout must contain at least one shot.")

    total_shot_states = sum(len(readout_blocks_per_nex) for readout_blocks_per_nex in per_shot_readout_layout)

    def empty():
        return torch.empty(0, dtype=first_nonempty.dtype, device=device)

    global_layout = []
    offset = 0
    for readout_blocks_per_nex in per_shot_readout_layout:
        row = [empty() for _ in range(total_shot_states)]
        for local_shot_idx, readout_block in enumerate(readout_blocks_per_nex):
            row[offset + local_shot_idx] = readout_block
        global_layout.append(row)
        offset += len(readout_blocks_per_nex)

    return global_layout


def expand_motion_states_to_readouts(readout_layout, state_curves, *, device):
    """
    Expand per-state motion traces to chronological per-readout traces.

    `readout_layout` is the global shot-state layout [Nex][Nstates], where each
    entry stores the readout indices acquired under that state for that Nex.
    Only the number of readouts in each entry matters here.
    """
    total_readouts = 0
    for readout_blocks_per_nex in readout_layout:
        for readout_block in readout_blocks_per_nex:
            total_readouts += readout_block.numel()

    expanded = {
        name: torch.empty(total_readouts, dtype=values.dtype, device=device)
        for name, values in state_curves.items()
    }

    write_ptr = 0
    for readout_blocks_per_nex in readout_layout:
        for state_idx, readout_block in enumerate(readout_blocks_per_nex):
            n_readouts = readout_block.numel()
            if n_readouts == 0:
                continue
            for name, values in state_curves.items():
                expanded[name][write_ptr:write_ptr + n_readouts] = values[state_idx]
            write_ptr += n_readouts

    return expanded


def build_event_transition_curve(event_idx, transition_length, n_samples, *, device):
    """
    Build a finite raised-cosine transition curve that starts at `event_idx`,
    reaches 1 after `transition_length` samples, and stays there afterwards.
    """
    transition_end = min(int(event_idx) + int(transition_length), int(n_samples))
    curve = torch.zeros(int(n_samples), device=device)

    if transition_end > int(event_idx):
        alpha = np.linspace(0.0, 1.0, transition_end - int(event_idx))
        ramp = 0.5 * (1.0 - np.cos(np.pi * alpha))
        curve[int(event_idx):transition_end] = torch.from_numpy(ramp).to(device)
    if transition_end < int(n_samples):
        curve[transition_end:] = 1.0

    return curve


def build_navigator_from_motion_matrix(motion_matrix):
    """
    Build a 1D navigator by projecting centered motion parameters onto their
    first principal component.

    Input shape: [Nparams, Nreadouts]
    Output shape: [Nreadouts]
    """
    centered = motion_matrix - motion_matrix.mean(dim=1, keepdim=True)
    U, _, _ = torch.linalg.svd(centered, full_matrices=False)
    pc1 = U[:, 0]
    navigator = torch.matmul(pc1.unsqueeze(0), centered).squeeze(0)
    return navigator / torch.clamp(navigator.abs().max(), min=1e-12)


def build_rigid_rotation_centers(translation_limits, n_states, *, device, Nx, Ny, Nz):
    """
    Build constant rigid rotation centers for all motion states.
    Returns shape [2, Nstates] for 2D and [3, Nstates] for 3D.
    """
    if Nz > 1:
        centers = torch.zeros((3, n_states), device=device)
        centers[0, :] = Nx / 2 + translation_limits["max_center_x_3d_px"] * torch.ones(n_states, device=device)
        centers[1, :] = Ny / 2 + translation_limits["max_center_y_3d_px"] * torch.ones(n_states, device=device)
        centers[2, :] = Nz / 2 + translation_limits["max_center_z_3d_px"] * torch.ones(n_states, device=device)
        return centers

    centers = torch.zeros((2, n_states), device=device)
    centers[0, :] = Nx / 2 + translation_limits["max_center_x_px"] * torch.ones(n_states, device=device)
    centers[1, :] = Ny / 2 + translation_limits["max_center_y_px"] * torch.ones(n_states, device=device)
    return centers

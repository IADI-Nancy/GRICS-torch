"""Prepare coarser data with k-space cropping and motion-state reduction."""

import torch

from .resize import resize_img_xy


def downsample_sampling_indices(Data_full, Sampling_full, Nx_res, Ny_res, Nz_res=1):
    Nx_full, Ny_full = Data_full["Nx"], Data_full["Ny"]
    Nz_full = int(Data_full.get("Nz", 1))

    # central crop coordinates
    x0 = (Nx_full - Nx_res) // 2
    y0 = (Ny_full - Ny_res) // 2
    z0 = (Nz_full - Nz_res) // 2

    Sampling_res = []

    for nex in range(len(Sampling_full)):
        Sampling_res.append([])
        for indices in Sampling_full[nex]:
            if Nz_full > 1:
                # Decode flattened 3D index: idx = ((x * Ny) + y) * Nz + z
                z = indices % Nz_full
                xy = indices // Nz_full
                x = xy // Ny_full
                y = xy % Ny_full
            else:
                # compute x,y coordinates for 2D flattening
                x = indices // Ny_full
                y = indices % Ny_full

            # mask inside central region
            if Nz_full > 1:
                mask = (
                    (x >= x0) & (x < x0 + Nx_res)
                    & (y >= y0) & (y < y0 + Ny_res)
                    & (z >= z0) & (z < z0 + Nz_res)
                )
            else:
                mask = (x >= x0) & (x < x0 + Nx_res) & (y >= y0) & (y < y0 + Ny_res)

            # keep only those indices
            x_crop = x[mask] - x0
            y_crop = y[mask] - y0

            if Nz_full > 1:
                z_keep = z[mask] - z0
                # re-flatten for Nx_res × Ny_res × Nz_full grid
                new_inds = (x_crop * Ny_res + y_crop) * Nz_res + z_keep
            else:
                # re-flatten for Nx_res × Ny_res grid
                new_inds = x_crop * Ny_res + y_crop

            Sampling_res[nex].append(new_inds)

    return Sampling_res


def downsample_kspace(Data_full, Nx_res, Ny_res, Nz_res=1):
    Nx_full, Ny_full = Data_full["Nx"], Data_full["Ny"]
    Nz_full = int(Data_full.get("Nz", 1))
    kspace_full = Data_full["KspaceData"]

    # central crop coordinates
    x0 = (Nx_full - Nx_res) // 2
    y0 = (Ny_full - Ny_res) // 2
    z0 = (Nz_full - Nz_res) // 2

    if Nz_full > 1:
        kspace_res = kspace_full[:, :, x0:x0 + Nx_res, y0:y0 + Ny_res, z0:z0 + Nz_res]
    else:
        kspace_res = kspace_full[:, :, x0:x0 + Nx_res, y0:y0 + Ny_res, :]
    kspace_res = kspace_res.reshape(kspace_full.shape[0], kspace_full.shape[1], -1)

    return kspace_res


def reduce_motion_states(sampling_indices, target_states, motion_signal, params, device, kspace=None):
    full_states = int(motion_signal.shape[0])
    if target_states == full_states:
        return sampling_indices, motion_signal

    weights = torch.tensor(
        [sum(sampling_indices[nex][state].numel() for nex in range(params.Nex))
         for state in range(full_states)],
        dtype=motion_signal.dtype, device=device,
    )

    binning_mode = str(
        params.motion_binning_mode
    ).strip().lower()
    if binning_mode == "kspace_energy":
        if kspace is None:
            raise ValueError(
                "Resolution-specific k-space is required for kspace_energy reduction."
            )
        # Recompute state energy after the resolution crop, exactly where
        # GRICS++ selects its resolution-specific virtual times.
        weights.zero_()
        for nex in range(params.Nex):
            for state in range(full_states):
                indices = sampling_indices[nex][state].long()
                if indices.numel() > 0:
                    weights[state] += kspace[:, nex, indices].abs().square().sum()
        # GRICS++ keeps the highest-energy states at each resolution and
        # attaches every remaining state to its nearest retained state.
        selected = torch.argsort(weights, descending=True, stable=True)[:target_states]
        centers = motion_signal[selected].clone()
        labels = torch.cdist(motion_signal, centers).argmin(dim=1)
    else:
        # Preserve the original deterministic weighted K-means reduction.
        selected = [int(torch.argmax(weights).item())]
        min_distance = torch.cdist(
            motion_signal, motion_signal[selected]
        ).squeeze(1)
        while len(selected) < target_states:
            next_idx = int(torch.argmax(min_distance).item())
            selected.append(next_idx)
            distance = torch.cdist(
                motion_signal, motion_signal[[next_idx]]
            ).squeeze(1)
            min_distance = torch.minimum(min_distance, distance)

        centers = motion_signal[selected].clone()
        for _ in range(20):
            distances = torch.cdist(motion_signal, centers)
            labels = distances.argmin(dim=1)
            updated = []
            for cluster in range(target_states):
                mask = labels == cluster
                if not mask.any():
                    updated.append(centers[cluster])
                    continue
                cluster_weights = weights[mask]
                denominator = torch.clamp(cluster_weights.sum(), min=1.0)
                updated.append(
                    (motion_signal[mask] * cluster_weights[:, None]).sum(dim=0)
                    / denominator
                )
            new_centers = torch.stack(updated)
            if torch.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers

    reduced = []
    for nex in range(params.Nex):
        nex_bins = []
        for cluster in range(target_states):
            members = torch.nonzero(labels == cluster, as_tuple=False).reshape(-1).tolist()
            pieces = [sampling_indices[nex][state] for state in members]
            nex_bins.append(torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long, device=device))
        reduced.append(nex_bins)
    return reduced, centers


def downsample_data(Data_full, res_factor, target_states, motion_signal, params, device):
    Nx = int(round(Data_full["Nx"] * res_factor))
    Ny = int(round(Data_full["Ny"] * res_factor))
    Nz_full = int(Data_full.get("Nz", 1))
    Nz = int(round(Nz_full * res_factor)) if Nz_full > 1 else 1
    Nz = max(Nz, 1)

    Data_res = {}
    Data_res["Nx"] = Nx
    Data_res["Ny"] = Ny
    Data_res["Nz"] = Nz

    resize_shape = (Nx, Ny, Nz) if Nz > 1 else (Nx, Ny)
    Data_res["SensitivityMaps"] = resize_img_xy(Data_full["SensitivityMaps"], resize_shape)
    sampling_indices = downsample_sampling_indices(
        Data_full, Data_full["SamplingIndices"], Nx, Ny, Nz_res=Nz
    )
    Data_res["KspaceData"] = downsample_kspace(Data_full, Nx, Ny, Nz_res=Nz)
    Data_res["SamplingIndices"], Data_res["MotionSignal"] = reduce_motion_states(
        sampling_indices, target_states, motion_signal, params, device, kspace=Data_res["KspaceData"])
    Data_res["Nsamples"] = Data_res["KspaceData"].shape[2]

    return Data_res

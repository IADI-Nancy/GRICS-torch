import torch

from src.utils.plotting import save_clustered_motion_plots

def _kmeans_torch(x, k, n_iter=20):
    N, _ = x.shape

    # Better init: random unique points
    perm = torch.randperm(N, device=x.device)
    centers = x[perm[:k]].clone()

    for _ in range(n_iter):
        dist = torch.cdist(x, centers)
        labels = dist.argmin(dim=1)

        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(dim=0)
            else:
                # Empty cluster → reinitialize
                idx = torch.randint(0, N, (1,), device=x.device)
                centers[j] = x[idx]

    return labels, centers

class MotionBinner:
    @staticmethod
    def _bin_motion(
        motion_curve,
        ky_idx,
        nex_idx,
        t_device,
        params,
        tx=None,
        ty=None,
        phi=None,
        y_limits=None,
        return_debug_data=False,
    ):
        motion_curve = motion_curve.to(t_device)

        Nbins = params.N_motion_states
        Nex = params.Nex

        # ---- K-means clustering (global, across all Nex) ----
        labels, centers = _kmeans_torch(motion_curve.unsqueeze(1), Nbins)

        # ---- Allocate output: [Nex][Nbins] ----
        binned_indices = [
            [torch.empty(0, dtype=ky_idx[0].dtype, device=t_device) for _ in range(Nbins)]
            for _ in range(Nex)
        ]
        ky_idx = torch.cat([k.reshape(-1) for k in ky_idx], dim=0)
        nex_idx = torch.cat([nex.reshape(-1) for nex in nex_idx], dim=0)

        # ---- Fill bins ----
        for nex in range(Nex):
            nex_mask = nex_idx == nex

            for b in range(Nbins):
                mask = nex_mask & (labels == b)
                binned_indices[nex][b] = ky_idx[mask]

        # ---- Input data plots (always saved) ----
        save_clustered_motion_plots(
            motion_curve=motion_curve,
            labels=labels,
            ky_idx=ky_idx,
            nex_idx=nex_idx,
            nbins=Nbins,
            output_folder=params.input_data_folder,
            resolution_levels=params.ResolutionLevels,
            tx=tx,
            ty=ty,
            phi=phi,
            data_type=params.data_type,
            y_limits=y_limits,
        )

        if return_debug_data:
            return binned_indices, centers.squeeze(1), labels, ky_idx, nex_idx
        return binned_indices, centers.squeeze(1)

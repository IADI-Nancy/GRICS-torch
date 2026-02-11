import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch

from Parameters import Parameters
params = Parameters()

def kmeans_torch(x, k, n_iter=20):
    N, D = x.shape

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
    def bin_motion(motion_curve, ky_idx, nex_idx, t_device):
        motion_curve = motion_curve.to(t_device)

        Nbins = params.N_mot_states
        Nex = params.Nex
        norm_color = Normalize(vmin=0, vmax=Nbins - 1)

        # ---- K-means clustering (global, across all Nex) ----
        labels, centers = kmeans_torch(motion_curve.unsqueeze(1), Nbins)

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

        # ---- Debug plots ----
        if params.debug_flag:
            markers = ["o", "x", "s", "^", "v", "D", "+", "*", "<", ">"]
            motion_cpu = motion_curve.detach().cpu()
            labels_cpu = labels.detach().cpu()
            nex_cpu = nex_idx.detach().cpu()
            time_idx = torch.arange(len(motion_cpu))
            ky_idx_cpu = ky_idx.detach().cpu()

            # chronological plot
            fig, ax = plt.subplots(figsize=(10, 4))

            for nex in torch.unique(nex_cpu):
                mask = nex_cpu == nex
                sc = ax.scatter(
                    time_idx[mask],
                    motion_cpu[mask],
                    c=labels_cpu[mask],
                    s=12,
                    cmap="tab10",
                    norm=norm_color,                      # ← IMPORTANT
                    marker=markers[int(nex) % len(markers)],
                    label=f"Nex {int(nex)}",
                )

            ax.set_xlabel("Time / acquisition order")
            ax.set_ylabel("Motion amplitude")
            ax.set_title("Chronological motion curve (color = motion state, marker = Nex)")
            ax.legend(title="Nex", loc="best")

            # Proper colorbar (global)
            sm = plt.cm.ScalarMappable(cmap="tab10", norm=norm_color)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Motion bin")
            cbar.set_ticks(range(Nbins))

            fig.tight_layout()
            fig.savefig("debug_outputs/clustered_curve_chronological.png")
            plt.close(fig)

            # ky-sorted plot        
            fig, ax = plt.subplots(figsize=(10, 4))

            for nex in torch.unique(nex_cpu):
                mask = nex_cpu == nex
                sc = ax.scatter(
                    ky_idx_cpu[mask],
                    motion_cpu[mask],
                    c=labels_cpu[mask],
                    s=12,
                    cmap="tab10",
                    norm=norm_color,                      # ← IMPORTANT
                    marker=markers[int(nex) % len(markers)],
                    label=f"Nex {int(nex)}",
                )

            ax.set_xlabel("Line index (ky)")
            ax.set_ylabel("Motion amplitude")
            ax.set_title("Motion curve vs ky (color = motion state, marker = Nex)")
            ax.legend(title="Nex", loc="best")

            sm = plt.cm.ScalarMappable(cmap="tab10", norm=norm_color)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Motion bin")
            cbar.set_ticks(range(Nbins))

            fig.tight_layout()
            fig.savefig("debug_outputs/clustered_curve_sorted_ky.png")
            plt.close(fig)

        return binned_indices
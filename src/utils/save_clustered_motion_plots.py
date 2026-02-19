import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def save_clustered_motion_plots(motion_curve, labels, ky_idx, nex_idx, nbins, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    markers = ["o", "x", "s", "^", "v", "D", "+", "*", "<", ">"]
    norm_color = Normalize(vmin=0, vmax=nbins - 1)

    motion_cpu = motion_curve.detach().cpu()
    labels_cpu = labels.detach().cpu()
    ky_idx_cpu = ky_idx.detach().cpu()
    nex_cpu = nex_idx.detach().cpu()
    time_idx = torch.arange(len(motion_cpu))

    fig, ax = plt.subplots(figsize=(10, 4))
    for nex in torch.unique(nex_cpu):
        mask = nex_cpu == nex
        ax.scatter(
            time_idx[mask],
            motion_cpu[mask],
            c=labels_cpu[mask],
            s=12,
            cmap="tab10",
            norm=norm_color,
            marker=markers[int(nex) % len(markers)],
            label=f"Nex {int(nex)}",
        )
    ax.set_xlabel("Time / acquisition order")
    ax.set_ylabel("Motion amplitude")
    ax.set_title("Chronological motion curve (color = motion state, marker = Nex)")
    ax.legend(title="Nex", loc="best")
    sm = plt.cm.ScalarMappable(cmap="tab10", norm=norm_color)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Motion bin")
    cbar.set_ticks(range(nbins))
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "clustered_curve_chronological.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for nex in torch.unique(nex_cpu):
        mask = nex_cpu == nex
        ax.scatter(
            ky_idx_cpu[mask],
            motion_cpu[mask],
            c=labels_cpu[mask],
            s=12,
            cmap="tab10",
            norm=norm_color,
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
    cbar.set_ticks(range(nbins))
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "clustered_curve_sorted_ky.png"))
    plt.close(fig)

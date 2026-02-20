import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


def _add_resolution_center_lines(ax, ky_idx_cpu, resolution_levels):
    if resolution_levels is None:
        return

    if len(ky_idx_cpu) == 0:
        return

    ky_min = float(torch.min(ky_idx_cpu).item())
    ky_max = float(torch.max(ky_idx_cpu).item())
    ny = int(round(ky_max - ky_min + 1))
    if ny <= 0:
        return

    center = ky_min + (ny - 1) / 2.0
    colors = [
        "green",
        "red",
        "tab:blue",
        "tab:orange",
        "tab:purple",
        "tab:brown",
    ]

    for i, frac in enumerate(resolution_levels):
        try:
            frac = float(frac)
        except Exception:
            continue
        if frac <= 0:
            continue
        frac = min(frac, 1.0)
        half_width = frac * ny / 2.0
        left = center - half_width
        right = center + half_width
        color = colors[i % len(colors)]

        ax.axvline(left, color=color, linestyle="--", linewidth=1.4, alpha=0.9, label=f"center {frac:g}")
        ax.axvline(right, color=color, linestyle="--", linewidth=1.4, alpha=0.9)


def save_clustered_motion_plots(
    motion_curve,
    labels,
    ky_idx,
    nex_idx,
    nbins,
    output_folder,
    resolution_levels=None,
    tx=None,
    ty=None,
    phi=None,
    data_type=None,
):
    os.makedirs(output_folder, exist_ok=True)

    markers = ["o", "x", "s", "^", "v", "D", "+", "*", "<", ">"]
    # Build a larger discrete palette, then keep only the number of bins in use.
    base_colors = []
    for cmap_name in ["tab20", "tab20b", "tab20c", "Set3", "Paired"]:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            base_colors.extend(list(cmap.colors))
        else:
            base_colors.extend([cmap(i / 20.0) for i in range(20)])
    if nbins > len(base_colors):
        reps = int(np.ceil(nbins / len(base_colors)))
        base_colors = (base_colors * reps)[:nbins]
    used_colors = base_colors[:nbins]
    cluster_cmap = ListedColormap(used_colors)
    norm_color = BoundaryNorm(np.arange(-0.5, nbins + 0.5, 1), cluster_cmap.N)

    motion_cpu = motion_curve.detach().cpu()
    labels_cpu = labels.detach().cpu()
    ky_idx_cpu = ky_idx.detach().cpu()
    nex_cpu = nex_idx.detach().cpu()
    time_idx = torch.arange(len(motion_cpu))

    fig, ax = plt.subplots(figsize=(10, 4))
    # Optional rigid parameters: thin solid lines (chronological domain).
    if tx is not None:
        ax.plot(
            torch.as_tensor(tx).detach().cpu().numpy(),
            linewidth=1.0,
            linestyle="-",
            alpha=0.9,
            label="Rigid tx (line)",
        )
    if ty is not None:
        ax.plot(
            torch.as_tensor(ty).detach().cpu().numpy(),
            linewidth=1.0,
            linestyle="-",
            alpha=0.9,
            label="Rigid ty (line)",
        )
    if phi is not None:
        ax.plot(
            torch.as_tensor(phi).detach().cpu().numpy(),
            linewidth=1.0,
            linestyle="-",
            alpha=0.9,
            label="Rigid phi (line)",
        )

    if data_type in {"real-world", "raw-data"}:
        sample_label_prefix = "Measured motion signal samples"
    elif data_type in {"shepp-logan", "fastMRI"}:
        sample_label_prefix = "PC1 motion samples"
    else:
        sample_label_prefix = "Motion signal samples"

    for nex in torch.unique(nex_cpu):
        mask = nex_cpu == nex
        ax.scatter(
            time_idx[mask],
            motion_cpu[mask],
            c=labels_cpu[mask],
            s=12,
            cmap=cluster_cmap,
            norm=norm_color,
            marker=markers[int(nex) % len(markers)],
            label=f"{sample_label_prefix} (rep {int(nex) + 1})",
        )
    ax.set_xlabel("Time / acquisition order")
    ax.set_ylabel("Motion amplitude")
    ax.set_title("Chronological rigid parameter curves + clustered PC1 samples")
    ax.legend(title="Legend", loc="best")
    sm = plt.cm.ScalarMappable(cmap=cluster_cmap, norm=norm_color)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Motion bin")
    cbar.set_ticks(range(nbins))
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "clustered_motion_curves_chronological.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for nex in torch.unique(nex_cpu):
        mask = nex_cpu == nex
        ax.scatter(
            ky_idx_cpu[mask],
            motion_cpu[mask],
            c=labels_cpu[mask],
            s=12,
            cmap=cluster_cmap,
            norm=norm_color,
            marker=markers[int(nex) % len(markers)],
            label=f"{sample_label_prefix} (rep {int(nex) + 1})",
        )
    ax.set_xlabel("Line index (ky)")
    ax.set_ylabel("Motion amplitude")
    ax.set_title("Clustered PC1 motion samples vs ky")
    _add_resolution_center_lines(ax, ky_idx_cpu, resolution_levels)
    ax.legend(title="Legend", loc="best")
    sm = plt.cm.ScalarMappable(cmap=cluster_cmap, norm=norm_color)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Motion bin")
    cbar.set_ticks(range(nbins))
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "clustered_motion_curve_sorted_ky.png"))
    plt.close(fig)

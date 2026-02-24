import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


def _add_mesh(ax):
    # Add major+minor grid lines for easier reading of rigid motion curves.
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth=0.6, alpha=0.45)
    ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.3)


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


def compute_motion_y_limits(motion_curve, tx=None, ty=None, phi=None, pad_ratio=0.05):
    vals = [torch.as_tensor(motion_curve).detach().flatten().cpu()]
    if tx is not None:
        vals.append(torch.as_tensor(tx).detach().flatten().cpu())
    if ty is not None:
        vals.append(torch.as_tensor(ty).detach().flatten().cpu())
    if phi is not None:
        vals.append(torch.as_tensor(phi).detach().flatten().cpu())

    all_vals = torch.cat(vals) if vals else torch.tensor([0.0])
    y_min = float(torch.min(all_vals).item())
    y_max = float(torch.max(all_vals).item())
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None
    if abs(y_max - y_min) < 1e-12:
        d = max(abs(y_max), 1.0) * 0.1
        return (y_min - d, y_max + d)
    pad = (y_max - y_min) * float(pad_ratio)
    return (y_min - pad, y_max + pad)


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
    y_limits=None,
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
    elif data_type == "shepp-logan":
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
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])
    _add_mesh(ax)
    ax.legend(title="Legend", loc="best")
    sm = plt.cm.ScalarMappable(cmap=cluster_cmap, norm=norm_color)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Motion bin")
    cbar.set_ticks(range(nbins))
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "clustered_motion_curves_chronological.png"))
    plt.close(fig)

    unique_nex = torch.unique(nex_cpu)
    n_nex = int(unique_nex.numel())
    fig, axes = plt.subplots(
        n_nex,
        1,
        figsize=(10, max(3.6, 2.9 * n_nex)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_nex == 1:
        axes = [axes]

    for ax, nex in zip(axes, unique_nex):
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
        ax.set_ylabel("Motion curve value")
        ax.set_title(f"Clustered PC1 motion samples vs ky (rep {int(nex) + 1})")
        if y_limits is not None:
            ax.set_ylim(y_limits[0], y_limits[1])
        _add_resolution_center_lines(ax, ky_idx_cpu[mask], resolution_levels)
        _add_mesh(ax)
        ax.legend(title="Legend", loc="best")

    axes[-1].set_xlabel("Line index (ky)")

    sm = plt.cm.ScalarMappable(cmap=cluster_cmap, norm=norm_color)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes,
        location="right",
        fraction=0.03,
        pad=0.02,
    )
    cbar.set_label("Motion bin")
    cbar.set_ticks(range(nbins))
    fig.savefig(os.path.join(output_folder, "clustered_motion_curve_sorted_ky.png"))
    plt.close(fig)

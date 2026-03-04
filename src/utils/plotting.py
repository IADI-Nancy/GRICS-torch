import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm


def show_and_save_image(
    img: torch.Tensor,
    image_name: str,
    folder: str,
    flip_for_display: bool = False,
    jupyter_notebook_flag: bool = False,
    jupyter_display: bool | None = None,
):
    """
    Display and save:
    - a single 2D image, or
    - for 3D volumes, a 1x3 panel of central XY/XZ/YZ planes.
    """
    if img.ndim == 3 and img.shape[-1] not in (3, 4):
        vol = img.detach().cpu()
        if torch.is_complex(vol):
            vol = torch.abs(vol)

        ix = vol.shape[0] // 2
        iy = vol.shape[1] // 2
        iz = vol.shape[2] // 2
        planes = [vol[:, :, iz], vol[:, iy, :], vol[ix, :, :]]
        titles = ["Axial (XY)", "Coronal (XZ)", "Sagittal (YZ)"]

        if flip_for_display:
            planes = [torch.flipud(p) for p in planes]

        all_vals = torch.cat([p.reshape(-1) for p in planes]).numpy()
        vmin = np.percentile(all_vals, 2)
        vmax = np.percentile(all_vals, 98)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for j in range(3):
            ax = axes[j]
            im = ax.imshow(planes[j].numpy(), vmin=vmin, vmax=vmax, cmap="gray")
            ax.set_title(f"{image_name} | {titles[j]}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        os.makedirs(folder, exist_ok=True)
        fig.savefig(os.path.join(folder, image_name + ".png"), bbox_inches="tight", pad_inches=0)
        should_display = jupyter_notebook_flag if jupyter_display is None else bool(jupyter_display)
        if should_display:
            plt.show()
        plt.close(fig)
        return

    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    np_img = img.detach().cpu().numpy()
    if np.iscomplexobj(np_img):
        np_img = np.abs(np_img)

    if flip_for_display:
        np_img = np.flipud(np_img)

    vmin = np.percentile(np_img, 2)
    vmax = np.percentile(np_img, 98)

    plt.figure(figsize=(5, 5))
    im = plt.imshow(np_img, vmin=vmin, vmax=vmax, cmap="gray")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.title(image_name)

    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, image_name + ".png"), bbox_inches="tight", pad_inches=0)
    should_display = jupyter_notebook_flag if jupyter_display is None else bool(jupyter_display)
    if should_display:
        plt.show()
    plt.close()


def save_alpha_component_map(comp, title, out_path, flip_vertical=True, abs_max=None):
    comp = comp.detach().cpu()
    if flip_vertical:
        comp = torch.flip(comp, dims=[0])

    vmax = float(abs_max) if abs_max is not None else torch.max(torch.abs(comp)).item()
    if vmax <= 0:
        vmax = 1e-12
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(comp.numpy(), cmap="bwr", norm=norm, origin="upper")
    plt.colorbar()
    plt.axis("off")
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_nonrigid_quiver_with_contours(
    alpha_axis0,
    alpha_axis1,
    image,
    title,
    out_path,
    flip_vertical=True,
    amp_vmax=None,
):
    alpha_axis0 = alpha_axis0.detach().cpu()
    alpha_axis1 = alpha_axis1.detach().cpu()
    img = image.detach().cpu()

    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    elif img.ndim != 2:
        img = img[..., 0]
    if img.is_complex():
        img = img.abs()

    if flip_vertical:
        alpha_axis0 = torch.flip(alpha_axis0, dims=[0])
        alpha_axis1 = torch.flip(alpha_axis1, dims=[0])
        img = torch.flip(img, dims=[0])

    nx, ny = alpha_axis0.shape
    step = max(1, min(nx, ny) // 32)
    yy, xx = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing="ij")
    xx = xx[::step, ::step].numpy()
    yy = yy[::step, ::step].numpy()
    ux = (-alpha_axis1[::step, ::step]).numpy()
    uy = (alpha_axis0[::step, ::step]).numpy()
    amp = torch.sqrt(alpha_axis0 * alpha_axis0 + alpha_axis1 * alpha_axis1)[::step, ::step].numpy()
    img_np = img.numpy()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("white")
    norm = None
    if amp_vmax is not None:
        norm = Normalize(vmin=0.0, vmax=float(amp_vmax))
    q = ax.quiver(
        xx, yy, ux, uy, amp, cmap="cividis_r", norm=norm, angles="xy", scale_units="xy", scale=None
    )
    ax.contour(
        torch.arange(ny).cpu().numpy(),
        torch.arange(nx).cpu().numpy(),
        img_np,
        levels=8,
        colors="k",
        linewidths=0.7,
        alpha=0.8,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, ny - 0.5)
    ax.set_ylim(nx - 0.5, -0.5)
    ax.set_title(title)
    fig.colorbar(q, ax=ax, label="|u|")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _add_mesh(ax):
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
    colors = ["green", "red", "tab:blue", "tab:orange", "tab:purple", "tab:brown"]

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
    if tx is not None:
        ax.plot(torch.as_tensor(tx).detach().cpu().numpy(), linewidth=1.0, linestyle="-", alpha=0.9, label="Rigid tx (line)")
    if ty is not None:
        ax.plot(torch.as_tensor(ty).detach().cpu().numpy(), linewidth=1.0, linestyle="-", alpha=0.9, label="Rigid ty (line)")
    if phi is not None:
        ax.plot(torch.as_tensor(phi).detach().cpu().numpy(), linewidth=1.0, linestyle="-", alpha=0.9, label="Rigid phi (line)")

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
    fig, axes = plt.subplots(n_nex, 1, figsize=(10, max(3.6, 2.9 * n_nex)), sharex=True, sharey=True, constrained_layout=True)
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
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.03, pad=0.02)
    cbar.set_label("Motion bin")
    cbar.set_ticks(range(nbins))
    fig.savefig(os.path.join(output_folder, "clustered_motion_curve_sorted_ky.png"))
    plt.close(fig)


def save_motion_debug_plots(motion_curve, tx, ty, phi, output_folder, event_times=None):
    os.makedirs(output_folder, exist_ok=True)

    plt.figure()
    plt.plot(motion_curve.detach().cpu().numpy())
    plt.title("Motion Curve")
    plt.savefig(os.path.join(output_folder, "motion_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(tx.detach().cpu().numpy())
    plt.title("tx curve")
    plt.savefig(os.path.join(output_folder, "tx_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(ty.detach().cpu().numpy())
    plt.title("ty curve")
    plt.savefig(os.path.join(output_folder, "ty_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(phi.detach().cpu().numpy())
    plt.title("phi curve")
    plt.savefig(os.path.join(output_folder, "phi_curve.png"))
    plt.close()


def save_line_plot(x, y, out_path, title=None, xlabel=None, ylabel=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure()
    plt.plot(x, y)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_residual_subplots(values_by_level, title, y_label, out_path):
    n_levels = len(values_by_level)
    if n_levels == 0:
        return

    global_max = None
    for vals in values_by_level:
        if len(vals) == 0:
            continue
        local_max = max(vals)
        global_max = local_max if global_max is None else max(global_max, local_max)

    global_min = 0.0
    if global_max is None:
        global_max = 1.0
    if global_max <= global_min:
        global_max = global_min + 1e-12

    fig, axes = plt.subplots(1, n_levels, figsize=(4.4 * n_levels, 3.8), sharey=True)
    if n_levels == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        vals = values_by_level[idx]
        if len(vals) > 0:
            ax.plot(vals, marker="o")
            ax.set_xlim(0, max(1, len(vals) - 1))
        else:
            ax.text(0.5, 0.5, "No iterations", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(global_min, global_max)
        ax.grid(True)
        ax.set_ylabel(y_label)
        ax.set_title(f"Resolution level {idx + 1}")
        ax.set_xlabel("GN iteration")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_residual_convergence(residual_norms, title, res_level, logs_folder):
    os.makedirs(logs_folder, exist_ok=True)

    plt.figure()
    plt.plot(residual_norms, marker="o")
    plt.xlabel("GN iteration")
    plt.ylabel("||residual||2")
    plt.title(f"Residual convergence ({title}, resolution level {res_level})")
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(logs_folder, f"residual_convergence_{title}_res{res_level}.png"))
    plt.close()

    with open(os.path.join(logs_folder, f"residual_convergence_{title}_res{res_level}.txt"), "w") as f:
        for i, v in enumerate(residual_norms):
            f.write(f"{i+1}\t{v}\n")


def _visualize_ky_order(ky_per_shot, ny, folder, fname="ky_sampling_order.png"):
    img = torch.zeros((ny, ny, 3), dtype=torch.float64)
    all_ky = torch.cat(ky_per_shot)

    order_map = torch.zeros(ny, dtype=torch.float64)
    order_map[all_ky] = torch.linspace(0, 1, len(all_ky))

    cmap = plt.get_cmap("viridis")
    for ky in range(ny):
        img[ky, :, :] = torch.tensor(cmap(order_map[ky].item())[:3], dtype=torch.float64)

    os.makedirs(folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img.numpy())
    ax.axis("off")
    ax.set_title("Ky Acquisition Order")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(all_ky)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Acquisition order (first -> last)")

    plt.tight_layout()
    plt.savefig(os.path.join(folder, fname))
    plt.close(fig)


def _visualize_ky_kz_order(ky_per_block, kz_per_block, ny, nz, folder, fname="ky_kz_sampling_order.png"):
    if nz <= 1:
        return

    os.makedirs(folder, exist_ok=True)

    order_map = torch.full((ny, nz), -1.0, dtype=torch.float64)
    order = 0
    nblocks = min(len(ky_per_block), len(kz_per_block))

    for b in range(nblocks):
        ky_vals = ky_per_block[b].to(torch.int64).reshape(-1)
        kz_vals = kz_per_block[b].to(torch.int64).reshape(-1)
        for ky in ky_vals.tolist():
            if ky < 0 or ky >= ny:
                continue
            for kz in kz_vals.tolist():
                if kz < 0 or kz >= nz:
                    continue
                order_map[ky, kz] = float(order)
                order += 1

    fig, ax = plt.subplots(figsize=(7, 5))
    masked = np.ma.masked_less(order_map.numpy(), 0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="black")
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap)

    ax.set_xlabel("kz")
    ax.set_ylabel("ky")
    ax.set_title("Sampling order in (ky, kz)")

    if order > 0:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Acquisition order (first -> last)")

    plt.tight_layout()
    plt.savefig(os.path.join(folder, fname))
    plt.close(fig)

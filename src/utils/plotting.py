import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle


_ORIENT_LABELS = ["Axial (XY)", "Coronal (XZ)", "Sagittal (YZ)"]


def _ensure_output_dir(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _as_magnitude_cpu(tensor):
    out = tensor.detach().cpu()
    if torch.is_complex(out):
        out = torch.abs(out)
    return out


def _prepare_display_image(img):
    out = _as_magnitude_cpu(img)
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    return out


def _rotation_curve_for_display(curve):
    return torch.rad2deg(torch.as_tensor(curve).detach().cpu())


def _extract_3d_central_slices(vol):
    ix = vol.shape[0] // 2
    iy = vol.shape[1] // 2
    iz = vol.shape[2] // 2
    return [vol[:, :, iz], vol[:, iy, :], vol[ix, :, :]]


def _flip_vertical(items, flip_vertical):
    if not flip_vertical:
        return items
    return [torch.flip(item, dims=[0]) for item in items]


def _robust_vrange(tensors, p_low=2.0, p_high=98.0):
    vals = torch.cat([tensor.reshape(-1) for tensor in tensors]).numpy()
    return np.percentile(vals, p_low), np.percentile(vals, p_high)


def _component_norm(comp, abs_max=None):
    vmax = float(abs_max) if abs_max is not None else torch.max(torch.abs(comp)).item()
    if vmax <= 0:
        vmax = 1e-12
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _quiver_step(nx, ny, divisor=32):
    return max(1, min(nx, ny) // divisor)


def _quiver_fields(alpha_axis0, alpha_axis1, divisor=32):
    nx, ny = alpha_axis0.shape
    step = _quiver_step(nx, ny, divisor=divisor)
    yy, xx = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing="ij")
    return (
        xx[::step, ::step].numpy(),
        yy[::step, ::step].numpy(),
        (-alpha_axis1[::step, ::step]).numpy(),
        (alpha_axis0[::step, ::step]).numpy(),
        torch.sqrt(alpha_axis0 * alpha_axis0 + alpha_axis1 * alpha_axis1)[::step, ::step].numpy(),
    )


def _save_or_show(fig, out_path, should_display=False):
    _ensure_output_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    if should_display:
        plt.show()
    plt.close(fig)


def _plot_component_panels(planes, titles, out_path, abs_max=None, cmap="bwr", figsize=None):
    comp = torch.cat([plane.reshape(-1) for plane in planes])
    norm = _component_norm(comp, abs_max=abs_max)
    figsize = figsize or ((5, 5) if len(planes) == 1 else (12, 4))
    fig, axes = plt.subplots(1, len(planes), figsize=figsize)
    if len(planes) == 1:
        axes = [axes]
    for ax, plane, title in zip(axes, planes, titles):
        im = ax.imshow(plane.numpy(), cmap=cmap, norm=norm, origin="upper")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_or_show(fig, out_path)


def _plot_quiver_panel(ax, alpha_axis0, alpha_axis1, img, title, amp_vmax=None, divisor=32):
    xx, yy, ux, uy, amp = _quiver_fields(alpha_axis0, alpha_axis1, divisor=divisor)
    img_np = img.numpy()

    ax.set_facecolor("white")
    norm = Normalize(vmin=0.0, vmax=float(amp_vmax)) if amp_vmax is not None else None
    q = ax.quiver(xx, yy, ux, uy, amp, cmap="cividis_r", norm=norm, angles="xy", scale_units="xy", scale=None)
    nx, ny = alpha_axis0.shape
    ax.contour(torch.arange(ny).numpy(), torch.arange(nx).numpy(), img_np, levels=8, colors="k", linewidths=0.7, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, ny - 0.5)
    ax.set_ylim(nx - 0.5, -0.5)
    ax.set_title(title)
    return q


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
    should_display = jupyter_notebook_flag if jupyter_display is None else bool(jupyter_display)
    out_path = os.path.join(folder, image_name + ".png")
    img_disp = _prepare_display_image(img)

    if img_disp.ndim == 3 and img_disp.shape[-1] not in (3, 4):
        planes = _flip_vertical(_extract_3d_central_slices(img_disp), flip_for_display)
        vmin, vmax = _robust_vrange(planes)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, plane, title in zip(axes, planes, _ORIENT_LABELS):
            im = ax.imshow(plane.numpy(), vmin=vmin, vmax=vmax, cmap="gray")
            ax.set_title(f"{image_name} | {title}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _save_or_show(fig, out_path, should_display=should_display)
        return

    np_img = img_disp.numpy()
    if flip_for_display:
        np_img = np.flipud(np_img)
    vmin, vmax = np.percentile(np_img, 2), np.percentile(np_img, 98)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(np_img, vmin=vmin, vmax=vmax, cmap="gray")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")
    ax.set_title(image_name)
    _save_or_show(fig, out_path, should_display=should_display)


def save_alpha_component_map(comp, title, out_path, flip_vertical=True, abs_max=None):
    comp = _as_magnitude_cpu(comp) if torch.is_complex(comp) else comp.detach().cpu()
    planes = _flip_vertical([comp], flip_vertical)
    _plot_component_panels(planes, [title], out_path, abs_max=abs_max, figsize=(5, 5))


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
    img = _prepare_display_image(image)
    if img.ndim != 2:
        img = img[..., 0]
    alpha_axis0, alpha_axis1, img = _flip_vertical([alpha_axis0, alpha_axis1, img], flip_vertical)

    fig, ax = plt.subplots(figsize=(6, 6))
    q = _plot_quiver_panel(ax, alpha_axis0, alpha_axis1, img, title, amp_vmax=amp_vmax)
    fig.colorbar(q, ax=ax, label="|u|")
    _save_or_show(fig, out_path)


def save_alpha_component_map_3d(comp_3d, title, out_path, flip_vertical=True, abs_max=None):
    comp_3d = comp_3d.detach().cpu()
    planes = _flip_vertical(_extract_3d_central_slices(comp_3d), flip_vertical)
    titles = [f"{title} | {label}" for label in _ORIENT_LABELS]
    _plot_component_panels(planes, titles, out_path, abs_max=abs_max, figsize=(12, 4))


def save_nonrigid_quiver_with_contours_3d(
    alpha_axis0_3d,
    alpha_axis1_3d,
    image_3d,
    title,
    out_path,
    flip_vertical=True,
    amp_vmax=None,
):
    alpha_axis0_3d = alpha_axis0_3d.detach().cpu()
    alpha_axis1_3d = alpha_axis1_3d.detach().cpu()
    img = _prepare_display_image(image_3d)

    a0_slices = _extract_3d_central_slices(alpha_axis0_3d)
    a1_slices = _extract_3d_central_slices(alpha_axis1_3d)
    img_slices = _extract_3d_central_slices(img)
    a0_slices = _flip_vertical(a0_slices, flip_vertical)
    a1_slices = _flip_vertical(a1_slices, flip_vertical)
    img_slices = _flip_vertical(img_slices, flip_vertical)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, a0, a1, img_slice, label in zip(axes, a0_slices, a1_slices, img_slices, _ORIENT_LABELS):
        q = _plot_quiver_panel(ax, a0, a1, img_slice, f"{title} | {label}", amp_vmax=amp_vmax)
        fig.colorbar(q, ax=ax, label="|u|")
    _save_or_show(fig, out_path)


def save_nonrigid_alpha_plots(
    alpha_maps, image, base_name, folder, flip_vertical=True,
    abs_max_x=None, abs_max_y=None, amp_max=None,
):
    """
    Unified save for non-rigid alpha maps.
    2D (ndim==3): one figure per component + quiver.
    3D (ndim==4): 1×3 panel per component + quiver (axial/coronal/sagittal).
    """
    from src.utils.nonrigid_display import to_cartesian_components, split_nonrigid_alpha_components

    os.makedirs(folder, exist_ok=True)
    alpha_x, alpha_y, alpha_z = split_nonrigid_alpha_components(alpha_maps)

    alpha_axis0 = alpha_x.real if torch.is_complex(alpha_x) else alpha_x
    alpha_axis1 = alpha_y.real if torch.is_complex(alpha_y) else alpha_y
    alpha_x_cart, alpha_y_cart = to_cartesian_components(alpha_axis0, alpha_axis1)

    is_3d = alpha_maps.ndim == 4

    if is_3d:
        save_alpha_component_map_3d(
            alpha_x_cart, f"{base_name}_alpha_x",
            os.path.join(folder, f"{base_name}_alpha_x.png"),
            flip_vertical=flip_vertical, abs_max=abs_max_x,
        )
        save_alpha_component_map_3d(
            alpha_y_cart, f"{base_name}_alpha_y",
            os.path.join(folder, f"{base_name}_alpha_y.png"),
            flip_vertical=flip_vertical, abs_max=abs_max_y,
        )
        if alpha_z is not None:
            save_alpha_component_map_3d(
                alpha_z, f"{base_name}_alpha_z",
                os.path.join(folder, f"{base_name}_alpha_z.png"),
                flip_vertical=flip_vertical,
            )
        save_nonrigid_quiver_with_contours_3d(
            alpha_axis0, alpha_axis1, image,
            f"{base_name}_quiver",
            os.path.join(folder, f"{base_name}_quiver.png"),
            flip_vertical=flip_vertical, amp_vmax=amp_max,
        )
    else:
        save_alpha_component_map(
            alpha_x_cart, f"{base_name}_alpha_x",
            os.path.join(folder, f"{base_name}_alpha_x.png"),
            flip_vertical=flip_vertical, abs_max=abs_max_x,
        )
        save_alpha_component_map(
            alpha_y_cart, f"{base_name}_alpha_y",
            os.path.join(folder, f"{base_name}_alpha_y.png"),
            flip_vertical=flip_vertical, abs_max=abs_max_y,
        )
        save_nonrigid_quiver_with_contours(
            alpha_axis0, alpha_axis1, image,
            f"{base_name}_quiver",
            os.path.join(folder, f"{base_name}_quiver.png"),
            flip_vertical=flip_vertical, amp_vmax=amp_max,
        )


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


def _add_resolution_center_boxes(ax, ny, nz, resolution_levels):
    if resolution_levels is None:
        return
    if ny <= 0 or nz <= 0:
        return

    colors = ["green", "red", "tab:blue", "tab:orange", "tab:purple", "tab:brown"]
    for i, frac in enumerate(resolution_levels):
        try:
            frac = float(frac)
        except Exception:
            continue
        if frac <= 0:
            continue
        frac = min(frac, 1.0)
        height = frac * ny
        width = frac * nz
        x0 = (nz - width) / 2.0 - 0.5
        y0 = (ny - height) / 2.0 - 0.5
        rect = Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor=colors[i % len(colors)],
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
        )
        ax.add_patch(rect)


def _motion_value_norm(motion_cpu, y_limits=None):
    if y_limits is not None:
        vmin, vmax = float(y_limits[0]), float(y_limits[1])
    else:
        vmin = float(torch.min(motion_cpu).item())
        vmax = float(torch.max(motion_cpu).item())

    if vmin < 0 < vmax:
        vmax_abs = max(abs(vmin), abs(vmax), 1e-12)
        return TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs), "coolwarm"
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12
    return Normalize(vmin=vmin, vmax=vmax), "viridis"


def compute_motion_plot_y_limits(
    motion_curve,
    tx=None,
    ty=None,
    phi=None,
    tz=None,
    rx=None,
    ry=None,
    rz=None,
    pad_ratio=0.05,
):
    vals = [torch.as_tensor(motion_curve).detach().flatten().cpu()]
    if tx is not None:
        vals.append(torch.as_tensor(tx).detach().flatten().cpu())
    if ty is not None:
        vals.append(torch.as_tensor(ty).detach().flatten().cpu())
    if phi is not None:
        vals.append(_rotation_curve_for_display(phi).flatten())
    if tz is not None:
        vals.append(torch.as_tensor(tz).detach().flatten().cpu())
    if rx is not None:
        vals.append(_rotation_curve_for_display(rx).flatten())
    if ry is not None:
        vals.append(_rotation_curve_for_display(ry).flatten())
    if rz is not None:
        vals.append(_rotation_curve_for_display(rz).flatten())

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
    kz_idx,
    nbins,
    output_folder,
    resolution_levels=None,
    tx=None,
    ty=None,
    phi=None,
    tz=None,
    rx=None,
    ry=None,
    rz=None,
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

    def _add_rigid_param_legends(ax, handle_by_name):
        translation_names = [name for name in ["tx", "ty", "tz"] if name in handle_by_name]
        rotation_names = [name for name in ["phi", "rx", "ry", "rz"] if name in handle_by_name]
        legend_style = {
            "framealpha": 0.9,
            "borderpad": 0.25,
            "labelspacing": 0.2,
            "handlelength": 1.6,
            "handletextpad": 0.35,
            "borderaxespad": 0.35,
            "fontsize": 9,
        }

        translation_legend = None
        if translation_names:
            translation_handles = [handle_by_name[name] for name in translation_names]
            translation_legend = ax.legend(
                handles=translation_handles,
                loc="lower left",
                bbox_to_anchor=(0.01, 0.01),
                **legend_style,
            )
            ax.add_artist(translation_legend)

        if rotation_names:
            rotation_handles = [handle_by_name[name] for name in rotation_names]
            x_anchor = 0.15 if translation_legend is not None else 0.01
            ax.legend(
                handles=rotation_handles,
                loc="lower left",
                bbox_to_anchor=(x_anchor, 0.01),
                **legend_style,
            )

    fig, ax = plt.subplots(figsize=(10, 4))
    line_handles = {}
    if tx is not None:
        line_handles["tx"] = ax.plot(torch.as_tensor(tx).detach().cpu().numpy(), linewidth=1.0, linestyle="-", alpha=0.9, label="tx")[0]
    if ty is not None:
        line_handles["ty"] = ax.plot(torch.as_tensor(ty).detach().cpu().numpy(), linewidth=1.0, linestyle="-", alpha=0.9, label="ty")[0]
    if phi is not None:
        line_handles["phi"] = ax.plot(_rotation_curve_for_display(phi).numpy(), linewidth=1.0, linestyle="-", alpha=0.9, label="phi [deg]")[0]
    if tz is not None:
        line_handles["tz"] = ax.plot(torch.as_tensor(tz).detach().cpu().numpy(), linewidth=1.0, linestyle="-", alpha=0.9, label="tz")[0]
    if rx is not None:
        line_handles["rx"] = ax.plot(_rotation_curve_for_display(rx).numpy(), linewidth=1.0, linestyle="-", alpha=0.9, label="rx [deg]")[0]
    if ry is not None:
        line_handles["ry"] = ax.plot(_rotation_curve_for_display(ry).numpy(), linewidth=1.0, linestyle="-", alpha=0.9, label="ry [deg]")[0]
    if rz is not None:
        line_handles["rz"] = ax.plot(_rotation_curve_for_display(rz).numpy(), linewidth=1.0, linestyle="-", alpha=0.9, label="rz [deg]")[0]

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
        )
    ax.set_xlabel("Time / acquisition order")
    ax.set_ylabel("Motion amplitude")
    ax.set_title("Chronological rigid parameter curves + clustered motion samples")
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])
    _add_mesh(ax)
    if line_handles:
        _add_rigid_param_legends(ax, line_handles)
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

    if kz_idx is None:
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
            )
            ax.set_ylabel("Motion curve value")
            ax.set_title(f"Clustered motion samples vs ky (rep {int(nex) + 1})")
            if y_limits is not None:
                ax.set_ylim(y_limits[0], y_limits[1])
            _add_resolution_center_lines(ax, ky_idx_cpu[mask], resolution_levels)
            _add_mesh(ax)

        axes[-1].set_xlabel("Line index (ky)")

        sm = plt.cm.ScalarMappable(cmap=cluster_cmap, norm=norm_color)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.03, pad=0.02)
        cbar.set_label("Motion bin")
        cbar.set_ticks(range(nbins))
        fig.savefig(os.path.join(output_folder, "clustered_motion_curve_sorted_ky.png"))
        plt.close(fig)
        return

    if torch.is_tensor(kz_idx):
        kz_idx_cpu = kz_idx.detach().cpu()
    else:
        kz_idx_cpu = torch.cat([k.reshape(-1) for k in kz_idx], dim=0).detach().cpu()
    fig, axes = plt.subplots(
        n_nex,
        1,
        figsize=(7.0, max(5.0, 5.0 * n_nex)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_nex == 1:
        axes = [axes]

    ny_dim = int(torch.max(ky_idx_cpu).item()) + 1 if ky_idx_cpu.numel() > 0 else 0
    nz_dim = int(torch.max(kz_idx_cpu).item()) + 1 if kz_idx_cpu.numel() > 0 else 0

    for ax, nex in zip(axes, unique_nex):
        mask = nex_cpu == nex
        label_map = torch.full((ny_dim, nz_dim), -1, dtype=torch.int64)
        ky_vals = ky_idx_cpu[mask].to(torch.int64)
        kz_vals = kz_idx_cpu[mask].to(torch.int64)
        label_vals = labels_cpu[mask].to(torch.int64)
        label_map[ky_vals, kz_vals] = label_vals

        masked = np.ma.masked_less(label_map.numpy(), 0)
        cmap = cluster_cmap.copy()
        cmap.set_bad(color="black")
        im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, norm=norm_color)
        _add_resolution_center_boxes(ax, ny_dim, nz_dim, resolution_levels)
        ax.set_ylabel("ky")
        ax.set_title(f"Motion bins in k-space (rep {int(nex) + 1})")

    axes[-1].set_xlabel("kz")
    cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.03, pad=0.02)
    cbar.set_label("Motion bin")
    cbar.set_ticks(range(nbins))
    fig.savefig(os.path.join(output_folder, "clustered_motion_curve_sorted_kykz.png"))
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
    plt.plot(_rotation_curve_for_display(phi).numpy())
    plt.title("phi curve [deg]")
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
        if ky_vals.numel() != kz_vals.numel():
            raise ValueError("ky_per_block and kz_per_block must contain paired sequences of equal length.")
        for ky, kz in zip(ky_vals.tolist(), kz_vals.tolist()):
            if ky < 0 or ky >= ny or kz < 0 or kz >= nz:
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

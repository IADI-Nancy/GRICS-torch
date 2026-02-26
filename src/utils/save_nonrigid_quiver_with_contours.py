import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


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
    # Internal inverse-warp axis components -> forward Cartesian quiver:
    # x = -axis1, y = +axis0
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

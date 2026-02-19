import os
import torch
import matplotlib.pyplot as plt


def save_nonrigid_quiver_with_contours(alpha_x, alpha_y, image, title, out_path, flip_vertical=True):
    alpha_x = alpha_x.detach().cpu()
    alpha_y = alpha_y.detach().cpu()
    img = image.detach().cpu()

    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    elif img.ndim != 2:
        img = img[..., 0]
    if img.is_complex():
        img = img.abs()

    if flip_vertical:
        alpha_x = torch.flip(alpha_x, dims=[0])
        alpha_y = torch.flip(alpha_y, dims=[0])
        img = torch.flip(img, dims=[0])

    nx, ny = alpha_x.shape
    step = max(1, min(nx, ny) // 32)
    yy, xx = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing="ij")
    xx = xx[::step, ::step].numpy()
    yy = yy[::step, ::step].numpy()
    ux = (-alpha_y[::step, ::step]).numpy()
    uy = (alpha_x[::step, ::step]).numpy()
    amp = torch.sqrt(alpha_x * alpha_x + alpha_y * alpha_y)[::step, ::step].numpy()
    img_np = img.numpy()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np, cmap="gray", origin="upper", alpha=0.35)
    q = ax.quiver(xx, yy, ux, uy, amp, cmap="bwr", angles="xy", scale_units="xy", scale=None)
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

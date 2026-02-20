import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def save_alpha_component_map(comp, title, out_path, flip_vertical=True, abs_max=None):
    comp = comp.detach().cpu()
    if flip_vertical:
        comp = torch.flip(comp, dims=[0])

    # Always center color scale at zero with symmetric limits.
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

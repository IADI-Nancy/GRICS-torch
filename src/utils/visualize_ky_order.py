import os
import torch
import matplotlib.pyplot as plt


def visualize_ky_order(ky_per_shot, ny, folder, fname="ky_sampling_order.png"):
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

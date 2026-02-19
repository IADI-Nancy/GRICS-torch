import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def show_and_save_image(img: torch.Tensor, image_name: str, folder: str):
    """
    Display and save a single 2D image (real or complex),
    scaled between the 2nd and 98th percentile.
    """

    # Remove singleton channel if present
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    # Convert to numpy on CPU
    np_img = img.detach().cpu().numpy()

    # Magnitude if complex
    if np.iscomplexobj(np_img):
        np_img = np.abs(np_img)

    # Flip vertically to match expected visual orientation.
    np_img = np.flipud(np_img)

    # Compute percentile-based intensity limits
    vmin = np.percentile(np_img, 2)
    vmax = np.percentile(np_img, 98)

    plt.figure(figsize=(5, 5))
    im = plt.imshow(np_img, vmin=vmin, vmax=vmax, cmap="gray")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.title(image_name)

    os.makedirs(folder, exist_ok=True)
    plt.savefig(
        os.path.join(folder, image_name + ".png"),
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()

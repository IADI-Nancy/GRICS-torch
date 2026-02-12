import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def show_and_save_image(img: torch.Tensor, image_name: str, folder: str):
    """
    Display and save a single 2D image (real or complex).

    Parameters
    ----------
    img : (H, W) or (H, W, 1) torch.Tensor
        Real or complex image.
    image_name : str
        Filename without extension.
    folder : str
        Output directory.
    """
    # Remove singleton channel if present
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    # Convert to numpy on CPU
    np_img = img.detach().cpu().numpy()

    # Magnitude if complex
    if np.iscomplexobj(np_img):
        np_img = np.abs(np_img)

    plt.figure(figsize=(5, 5))
    plt.imshow(np_img)
    plt.axis("off")
    plt.title(image_name)

    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, image_name + ".png"),
                bbox_inches="tight",
                pad_inches=0)
    plt.close()

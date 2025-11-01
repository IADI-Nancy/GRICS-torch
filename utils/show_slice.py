# this function shows a slice of the image,
# converting it to a numpy complex value and moving it to CPU

import math
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import torch
import xml.etree.ElementTree as ET

def show_slice(
        img: torch.Tensor,
        slice_idx: int = 0,
        max_images = 2,
        headline: str = 'Slice',
        verbose: bool = False):
    """
    Plot the magnitude of up to *max_images* channels from a 2-D slice.

    Parameters
    ----------
    img : (H, W, C) tensor
        Real or complex image data; complex values are shown as |x|.
    slice_idx : int, optional
        Label added to the figure title (does not affect what is plotted).
    max_images : int, optional
        Number of channels to display (≤ 2 per row). Default 2.
    headline : str, optional
        Figure title. Default “Slice”.
    verbose : bool, optional
        Print max |x| per channel if True.

    Notes
    -----
    • Works in-place: it only shows the figure, returns None.  
    • Unused subplots are hidden.
    """
    # Move data to CPU and convert to NumPy
    np_img = img.detach().cpu().numpy()

    max_cols = 2  # Maximum columns in the subplot grid

    # Determine layout: max_cols per row
    cols = min(max_cols, max_images)
    rows = math.ceil(max_images / cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 4, rows * 4),
                             squeeze=False)

    for cha in range(max_images):
        r, c = divmod(cha, cols)
        magnitude = np.abs(np_img[:, :, cha])

        if verbose:
            print(f"Channel {cha + 1} – Max value: {magnitude.max():.3g}")

        ax = axes[r][c]
        ax.imshow(magnitude)
        ax.set_title(f'Channel {cha + 1}')
        ax.axis('off')

    # Hide any unused subplot axes
    for idx in range(max_images, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis('off')

    fig.suptitle(headline)
    plt.tight_layout()
    plt.show()

    


def show_kspace_slice(
        kspace: torch.Tensor,
        slice_idx: int = 0,
        max_images: int = 2,
        headline: str = "k-space slice",
        eps: float = 1e-12):
    """
    Plot the log-magnitude of up to *max_images* channels from a 2-D k-space slice.

    Parameters
    ----------
    kspace : (H, W) or (H, W, C) tensor
        Complex k-space data; complex values are shown as log10(|x|).
    slice_idx : int, optional
        Label appended to the figure title. Default 0.
    max_images : int, optional
        Number of channels to display (<= 2 per row). Default 2.
    headline : str, optional
        Figure title stem. Default "k-space slice".
    eps : float, optional
        Offset to avoid log(0). Default 1e-6.

    Notes
    -----
    - Operates in-place: shows a figure and returns None.
    - Unused subplots are hidden.
    """
    # Move to CPU and NumPy
    k = kspace.detach().cpu().numpy()
    if k.ndim == 2:                     # single-coil -> (H, W, 1)
        k = k[..., None]

    # fftshift and log-magnitude
    k   = np.fft.fftshift(k, axes=(0, 1))
    log = np.log10(np.abs(k) + eps)
    vmin, vmax = log.min(), log.max()

    # Subplot layout
    max_cols = 2
    cols = min(max_cols, max_images)
    rows = math.ceil(max_images / cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 4, rows * 4),
                             squeeze=False)

    # Plot channels
    for cha in range(max_images):
        r, c = divmod(cha, cols)
        img = log[:, :, cha]

        ax = axes[r][c]
        ax.imshow(img, vmin=vmin, vmax=vmax)
        ax.set_title(f"Channel {cha + 1}")
        ax.axis("off")

    # Hide unused axes
    for idx in range(max_images, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"{headline} (log)")
    plt.tight_layout()
    plt.show()


def header_info(header_str: str):
    '''Extract and print key metadata from an ISMRMRD XML header string.'''
    root = ET.fromstring(header_str)
    ns = {'ismrmrd': 'http://www.ismrm.org/ISMRMRD'}

    protocol = root.findtext(".//ismrmrd:protocolName", namespaces=ns)
    vendor = root.findtext(".//ismrmrd:systemVendor", namespaces=ns)
    model = root.findtext(".//ismrmrd:systemModel", namespaces=ns)
    field_strength = root.findtext(".//ismrmrd:systemFieldStrength_T", namespaces=ns)
    ncoils = root.findtext(".//ismrmrd:receiverChannels", namespaces=ns)
    matrix_x = root.findtext(".//ismrmrd:encoding/ismrmrd:encodedSpace/ismrmrd:matrixSize/ismrmrd:x", namespaces=ns)
    matrix_y = root.findtext(".//ismrmrd:encoding/ismrmrd:encodedSpace/ismrmrd:matrixSize/ismrmrd:y", namespaces=ns)
    fov_x = root.findtext(".//ismrmrd:encoding/ismrmrd:encodedSpace/ismrmrd:fieldOfView_mm/ismrmrd:x", namespaces=ns)
    fov_y = root.findtext(".//ismrmrd:encoding/ismrmrd:encodedSpace/ismrmrd:fieldOfView_mm/ismrmrd:y", namespaces=ns)
    tr = root.findtext(".//ismrmrd:sequenceParameters/ismrmrd:TR", namespaces=ns)
    te = root.findtext(".//ismrmrd:sequenceParameters/ismrmrd:TE", namespaces=ns)

    print("Protocol:", protocol)
    print("Scanner:", vendor, model, f"{field_strength} T")
    print("Coils:", ncoils)
    print("Matrix:", f"{matrix_x} x {matrix_y}")
    print("FOV:", f"{fov_x} mm x {fov_y} mm")
    print("TR:", tr, "ms")
    print("TE:", te, "ms")
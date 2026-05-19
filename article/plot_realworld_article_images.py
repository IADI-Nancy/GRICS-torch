import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_2D_ROOT = Path("/home/pyuser/wkdir/data/GRICS-torch/article_dataset")
DEFAULT_3D_ROOT = Path("/home/pyuser/wkdir/data/GRICS-torch/article_dataset_3D")
DEFAULT_2D_OUTPUT = Path("/home/pyuser/wkdir/data/GRICS-torch/article_dataset_image_plots_2D")
DEFAULT_3D_OUTPUT = Path("/home/pyuser/wkdir/data/GRICS-torch/article_dataset_image_plots_3D")
DEFAULT_SLICE_NUMBER = 15


def ifftnc_numpy(kspace, axes):
    shifted = np.fft.ifftshift(kspace, axes=axes)
    image = np.fft.ifftn(shifted, axes=axes, norm="ortho")
    return np.fft.fftshift(image, axes=axes)


def rss_combine(image, coil_axis=0):
    return np.sqrt(np.sum(np.abs(image) ** 2, axis=coil_axis))


def normalize_image(image):
    image = np.asarray(image, dtype=np.float64)
    image = np.nan_to_num(image, copy=False)
    image -= np.min(image)
    vmax = np.max(image)
    if vmax > 0:
        image = image / vmax
    return image


def load_corrected(path):
    image = torch.load(path, map_location="cpu")
    image = torch.abs(image.detach().cpu()).squeeze()
    if image.ndim == 4:
        image = image.mean(dim=0)
    return image.numpy().astype(np.float64, copy=False)


def load_uncorrected_2d(h5_file, slice_index):
    with h5py.File(h5_file, "r") as f:
        kspace = f["kspace"][:, :, :, :, slice_index]

    coil_images = ifftnc_numpy(kspace, axes=(-2, -1))
    rss_per_nex = rss_combine(coil_images, coil_axis=0)
    return np.mean(rss_per_nex, axis=0)


def load_uncorrected_3d(h5_file):
    with h5py.File(h5_file, "r") as f:
        kspace = f["kspace"][:]

    coil_images = ifftnc_numpy(kspace, axes=(-3, -2, -1))
    rss_per_nex = rss_combine(coil_images, coil_axis=0)
    return np.mean(rss_per_nex, axis=0)


def plot_2d_pair(uncorrected, corrected, output_file, title):
    uncorrected = normalize_image(uncorrected)
    corrected = normalize_image(corrected)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=220)
    vmax = max(float(np.max(uncorrected)), float(np.max(corrected)), 1.0)

    for ax, image, label in zip(
        axes,
        [uncorrected, corrected],
        ["Uncorrected", "Corrected"],
    ):
        ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(pad=0.2)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def volume_views(volume):
    nx, ny, nz = volume.shape
    axial = volume[:, :, nz // 2]
    sagittal = volume[int(round(0.7 * (nx - 1))), :, :]
    coronal = volume[:, int(round(0.3 * (ny - 1))), :]
    return axial, sagittal, coronal


def plot_3d_views(uncorrected, corrected, output_file, title):
    uncorrected = normalize_image(uncorrected)
    corrected = normalize_image(corrected)

    uncorrected_views = volume_views(uncorrected)
    corrected_views = volume_views(corrected)

    labels = [
        "Mid-axial",
        "30% sagittal",
        "30% coronal",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(8, 10), dpi=220)

    for row, label in enumerate(labels):
        row_images = [uncorrected_views[row], corrected_views[row]]
        vmax = max(float(np.max(row_images[0])), float(np.max(row_images[1])), 1.0)

        for col, image in enumerate(row_images):
            ax = axes[row, col]
            ax.imshow(np.rot90(image), cmap="gray", vmin=0, vmax=vmax)
            if row == 0:
                ax.set_title("Uncorrected" if col == 0 else "Corrected")
            ax.set_ylabel(label if col == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout(pad=0.4)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def corrected_2d_file(root, subject, slice_number):
    return (
        root
        / subject
        / f"Siemens_SingleImage_slice{slice_number:03d}_image01"
        / "GricsRecon.pt"
    )


def corrected_3d_file(root, subject):
    return root / subject / "GricsRecon.pt"


def plot_2d_dataset(root, output_dir, slice_number):
    slice_index = slice_number - 1
    for h5_file in sorted(root.glob("*.h5")):
        subject = h5_file.stem
        corrected_file = corrected_2d_file(root, subject, slice_number)
        if not corrected_file.is_file():
            print(f"Missing corrected 2D file: {corrected_file}")
            continue

        uncorrected = load_uncorrected_2d(h5_file, slice_index)
        corrected = load_corrected(corrected_file)
        output_file = output_dir / f"{subject}_slice{slice_number:03d}_uncorrected_corrected.png"
        plot_2d_pair(uncorrected, corrected, output_file, f"{subject} slice {slice_number:03d}")
        print(f"Saved {output_file}")


def plot_3d_dataset(root, output_dir):
    for h5_file in sorted(root.glob("*.h5")):
        subject = h5_file.stem
        corrected_file = corrected_3d_file(root, subject)
        if not corrected_file.is_file():
            print(f"Missing corrected 3D file: {corrected_file}")
            continue

        uncorrected = load_uncorrected_3d(h5_file)
        corrected = load_corrected(corrected_file)
        output_file = output_dir / f"{subject}_uncorrected_corrected_views.png"
        plot_3d_views(uncorrected, corrected, output_file, subject)
        print(f"Saved {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot uncorrected and GRICS-torch-corrected article dataset images."
    )
    parser.add_argument("--root-2d", type=Path, default=DEFAULT_2D_ROOT)
    parser.add_argument("--root-3d", type=Path, default=DEFAULT_3D_ROOT)
    parser.add_argument("--output-2d", type=Path, default=DEFAULT_2D_OUTPUT)
    parser.add_argument("--output-3d", type=Path, default=DEFAULT_3D_OUTPUT)
    parser.add_argument("--slice-number", type=int, default=DEFAULT_SLICE_NUMBER)
    parser.add_argument("--only", choices=["2d", "3d", "both"], default="both")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.only in {"2d", "both"}:
        plot_2d_dataset(args.root_2d, args.output_2d, args.slice_number)

    if args.only in {"3d", "both"}:
        plot_3d_dataset(args.root_3d, args.output_3d)


if __name__ == "__main__":
    main()

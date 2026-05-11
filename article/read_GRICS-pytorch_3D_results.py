import re
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results_helpers import append_subject_metric, save_alpha_maps_if_available
from sharpness_index import sharpness_index


GRICS_torch_path = "/home/pyuser/wkdir/data/GRICS-torch/article_dataset_3D/"
no_correction_path = "/home/pyuser/wkdir/data/GRICS-torch/article_dataset_3D_nomoco/"

reconstruction_times_file = Path(GRICS_torch_path) / "reconstruction_times.txt"
sharpness_enhansement_file = Path(GRICS_torch_path) / "sharpness_enhansement.txt"
sharpness_corrected_file = Path(GRICS_torch_path) / "sharpness_corrected.txt"


def extract_torch_reconstruction_time(log_text):
    patterns = [
        r"Total time of reconstruction run:\s*([\d.]+)\s*s?",
        r"Elapsed time:\s*([\d.]+)\s*s",
        r"Reconstruction time\s*=\s*([\d.]+)",
        r"Total elapsed time\s*=\s*([\d.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, log_text)
        if match:
            return float(match.group(1))

    return None


def load_torch_recon(file_recon):
    """
    Loads GricsRecon.pt and returns a 3D numpy volume.

    Expected tensor shape:
      [1, Nx, Ny, Nz]
      [Nex, Nx, Ny, Nz]
      [Nx, Ny, Nz]

    If multiple Nex volumes are present, they are averaged.
    """
    recon = torch.load(file_recon, map_location="cpu")

    if not isinstance(recon, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(recon)}")

    recon = recon.detach().cpu()

    if recon.ndim == 4:
        if recon.shape[0] == 1:
            recon = recon[0]
        else:
            recon = recon.mean(dim=0)

    recon = recon.squeeze()

    if recon.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeeze, got shape {recon.shape}")

    return recon.numpy()


def mean_sharpness_metrics(recon_torch, recon_nomoco):
    if recon_torch.shape != recon_nomoco.shape:
        raise ValueError(
            "Corrected and NoMoCo volumes have different shapes: "
            f"{recon_torch.shape} vs {recon_nomoco.shape}"
        )

    sharpness_enhansement = []
    sharpness_corrected = []

    for i_slice in range(recon_torch.shape[2]):
        sharpness_idc_torch = sharpness_index(
            torch.from_numpy(np.abs(recon_torch[:, :, i_slice]))
        )
        sharpness_idc_nomoco = sharpness_index(
            torch.from_numpy(np.abs(recon_nomoco[:, :, i_slice]))
        )

        enhancement = (
            200
            * (sharpness_idc_torch - sharpness_idc_nomoco)
            / (sharpness_idc_torch + sharpness_idc_nomoco)
        )

        sharpness_corrected.append(float(sharpness_idc_torch))
        sharpness_enhansement.append(float(enhancement))

    return float(np.mean(sharpness_enhansement)), float(np.mean(sharpness_corrected))


# Clear output files
reconstruction_times_file.write_text("")
sharpness_enhansement_file.write_text("")
sharpness_corrected_file.write_text("")


for subject_dir in sorted(Path(GRICS_torch_path).iterdir()):
    if not subject_dir.is_dir():
        continue

    print(f"Processing subject: {subject_dir.name}")

    file_log = subject_dir / "joint_reconstruction.log"
    file_grics_torch = subject_dir / "GricsRecon.pt"
    file_alpha_torch = subject_dir / "GricsAlphaMaps.pt"
    file_no_moco = Path(no_correction_path) / subject_dir.name / "GricsRecon.pt"

    total_time = None

    if file_log.exists():
        log = file_log.read_text()
        total_time = extract_torch_reconstruction_time(log)
    else:
        print(f"Missing log: {file_log}")

    if not file_grics_torch.exists():
        print(f"Missing recon: {file_grics_torch}")
        append_subject_metric(reconstruction_times_file, subject_dir.name, "nan")
        append_subject_metric(sharpness_enhansement_file, subject_dir.name, "nan")
        append_subject_metric(sharpness_corrected_file, subject_dir.name, "nan")
        continue

    if not file_no_moco.exists():
        print(f"Missing NoMoCo file: {file_no_moco}")
        append_subject_metric(
            reconstruction_times_file,
            subject_dir.name,
            total_time if total_time is not None else "nan",
        )
        append_subject_metric(sharpness_enhansement_file, subject_dir.name, "nan")
        append_subject_metric(sharpness_corrected_file, subject_dir.name, "nan")
        continue

    GricsRecon_torch = load_torch_recon(file_grics_torch)
    GricsRecon_nomoco = load_torch_recon(file_no_moco)

    output_file = subject_dir / "reconstructed_image.png"
    central_slice = GricsRecon_torch[:, :, GricsRecon_torch.shape[2] // 2]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.imshow(np.abs(central_slice), cmap="gray")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    save_alpha_maps_if_available(
        file_alpha_torch,
        GricsRecon_torch,
        subject_dir,
    )

    sharpness_enhansement, sharpness_corrected = mean_sharpness_metrics(
        GricsRecon_torch,
        GricsRecon_nomoco,
    )

    append_subject_metric(
        reconstruction_times_file,
        subject_dir.name,
        total_time if total_time is not None else "nan",
    )
    append_subject_metric(
        sharpness_enhansement_file,
        subject_dir.name,
        sharpness_enhansement,
    )
    append_subject_metric(
        sharpness_corrected_file,
        subject_dir.name,
        sharpness_corrected,
    )

import re
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from results_helpers import append_subject_metric, save_alpha_maps_if_available
from sharpness_index import sharpness_index


GRICS_torch_path = "/home/pyuser/wkdir/data/GRICS-torch/article_dataset/"
no_correction_path = "/home/pyuser/wkdir/data/GRICS-torch/article_dataset_nomoco/"

reconstruction_times_file = Path(GRICS_torch_path) / "reconstruction_times.txt"
sharpness_enhansement_file = Path(GRICS_torch_path) / "sharpness_enhansement.txt"
sharpness_corrected_file = Path(GRICS_torch_path) / "sharpness_corrected.txt"
 
Nsli_max = 120


def extract_torch_reconstruction_time(log_text):
    """
    Extract elapsed reconstruction time from joint_reconstruction.log.
    Adjust regex if your log uses another wording.
    """
    patterns = [
        r"Elapsed time:\s*([\d.]+)\s*s",
        r"Reconstruction time\s*=\s*([\d.]+)",
        r"Total elapsed time\s*=\s*([\d.]+)",
        r"Total time of reconstruction run:\s*([\d.]+)\s*s?",
    ]

    for pattern in patterns:
        match = re.search(pattern, log_text)
        if match:
            return float(match.group(1))

    return None


def load_bin(file, data_type, output_size=None, endian="<"):
    file = Path(file)

    if data_type == "complex-float":
        raw = np.fromfile(file, dtype=endian + "f4")
        data = raw[0::2] + 1j * raw[1::2]

    elif data_type == "complex-double":
        raw = np.fromfile(file, dtype=endian + "f8")
        data = raw[0::2] + 1j * raw[1::2]

    else:
        data = np.fromfile(file, dtype=endian + data_type)

    if output_size is not None:
        data = data.reshape(output_size)

    return data


def load_torch_recon(file_recon):
    """
    Loads GricsRecon.pt and returns a 2D numpy magnitude image.
    Expected tensor shape: [1, Nx, Ny] or [Nx, Ny].
    """
    recon = torch.load(file_recon, map_location="cpu")

    if isinstance(recon, torch.Tensor):
        recon = recon.detach().cpu()
    else:
        raise TypeError(f"Expected torch.Tensor, got {type(recon)}")

    # Remove singleton dimensions, e.g. [1, 320, 448] -> [320, 448]
    recon = recon.squeeze()

    if recon.ndim != 2:
        raise ValueError(f"Expected 2D image after squeeze, got shape {recon.shape}")

    return recon.numpy()


# Clear output files
reconstruction_times_file.write_text("")
sharpness_enhansement_file.write_text("")
sharpness_corrected_file.write_text("")


for subject_dir in sorted(Path(GRICS_torch_path).iterdir()):
    if not subject_dir.is_dir():
        continue

    total_times = []
    sharpness_enhansement = []
    sharpness_corrected = []

    print(f"Processing subject: {subject_dir.name}")

    for i in range(1, Nsli_max + 1):
        folder_name = f"Siemens_SingleImage_slice{i:03d}_image01"
        slice_dir = subject_dir / folder_name

        if not slice_dir.is_dir():
            break

        file_log = slice_dir / "joint_reconstruction.log"
        file_grics_torch = slice_dir / "GricsRecon.pt"
        file_alpha_torch = slice_dir / "GricsAlphaMaps.pt"

        if not file_log.exists():
            print(f"Missing log: {file_log}")
            continue

        if not file_grics_torch.exists():
            print(f"Missing recon: {file_grics_torch}")
            continue

        # Read log
        log = file_log.read_text()
        total_time = extract_torch_reconstruction_time(log)

        if total_time is not None:
            total_times.append(total_time)

        # Read GRICS-torch reconstructed image
        GricsRecon_torch = load_torch_recon(file_grics_torch)

        # Read NoMoCo image
        file_no_moco = (
            Path(no_correction_path)
            / subject_dir.name
            / folder_name
            / "GricsRecon.pt"
        )

        if not file_no_moco.exists():
            print(f"Missing NoMoCo file: {file_no_moco}")
            continue

        GricsRecon_nomoco = load_torch_recon(file_no_moco)

        # Save reconstructed image PNG
        output_file = slice_dir / "reconstructed_image.png"

        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        ax.imshow(np.abs(GricsRecon_torch), cmap="gray")
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(output_file, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        save_alpha_maps_if_available(
            file_alpha_torch,
            GricsRecon_torch,
            slice_dir,
        )

        # Sharpness index
        sharpness_idc_torch = sharpness_index(
            torch.from_numpy(np.abs(GricsRecon_torch))
        )

        sharpness_idc_nomoco = sharpness_index(
            torch.from_numpy(np.abs(GricsRecon_nomoco))
        )

        enhancement = (
            200
            * (sharpness_idc_torch - sharpness_idc_nomoco)
            / (sharpness_idc_torch + sharpness_idc_nomoco)
        )

        sharpness_corrected.append(float(sharpness_idc_torch))
        sharpness_enhansement.append(float(enhancement))

    if total_times:
        append_subject_metric(reconstruction_times_file, subject_dir.name, max(total_times))
    else:
        append_subject_metric(reconstruction_times_file, subject_dir.name, "nan")

    if sharpness_enhansement:
        append_subject_metric(
            sharpness_enhansement_file,
            subject_dir.name,
            np.mean(sharpness_enhansement),
        )
    else:
        append_subject_metric(sharpness_enhansement_file, subject_dir.name, "nan")

    if sharpness_corrected:
        append_subject_metric(
            sharpness_corrected_file,
            subject_dir.name,
            np.mean(sharpness_corrected),
        )
    else:
        append_subject_metric(sharpness_corrected_file, subject_dir.name, "nan")

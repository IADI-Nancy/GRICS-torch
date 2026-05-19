import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from skimage.metrics import (
    normalized_root_mse,
    peak_signal_noise_ratio,
    structural_similarity,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if "__file__" in globals():
    ARTICLE_DIR = Path(__file__).resolve().parent
    REPO_ROOT = ARTICLE_DIR.parents[0]
else:
    ARTICLE_DIR = Path.cwd() / "article"
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ARTICLE_DIR))

from sharpness_index import sharpness_index
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime


DEFAULT_DATASET_ROOT = Path("/home/pyuser/wkdir/data/GRICS-torch/article_dataset_3D")
DEFAULT_REFERENCE_ROOT = Path("/home/pyuser/wkdir/data/GRICS-torch/article_dataset_3D_nomoco")
DEFAULT_OUTPUT_ROOT = Path("/home/pyuser/wkdir/data/GRICS-torch/article_simulated_nonrigid_3D")
GRICS_plusplus_path = "/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/GRICS-BELT-3D/"

T1_SUBJECT_PATTERN = re.compile(r"^\d{4}_T1_[sm]$")

METRIC_NAMES = [
    "sharpness_uncorrected",
    "sharpness_corrected",
    "sharpness_ground_truth",
    "uncorrected_ssim",
    "uncorrected_psnr",
    "uncorrected_nrmse",
    "corrected_ssim",
    "corrected_psnr",
    "corrected_nrmse",
    "ground_truth_ssim",
    "ground_truth_psnr",
    "ground_truth_nrmse",
    "elapsed_time_s",
]


def subject_motion_overrides(subject_index, base_seed):
    centered = (subject_index % 7) - 3
    cycles_min = 5 + (subject_index % 3)

    return {
        "seed": int(base_seed + subject_index),
        "nonrigid_motion_amplitude": 5.0 * (1.0 + 0.04 * centered),
        "nonrigid_discrete_s_scale": 4.0 * (1.0 + 0.03 * ((subject_index % 5) - 2)),
        "nonrigid_resp_cycles_min": cycles_min,
        "nonrigid_resp_cycles_max": cycles_min + 1,
        "nonrigid_diaphragm_level": 0.2 + 0.015 * centered,
        "nonrigid_diaphragm_sharpness": 10.0 * (1.0 + 0.02 * ((subject_index % 5) - 2)),
        "nonrigid_lateral_sigma_lr": 0.45 * (1.0 + 0.03 * ((subject_index % 4) - 1.5)),
        "nonrigid_lr_fraction": 0.15 * (1.0 + 0.05 * ((subject_index % 5) - 2)),
        "nonrigid_top_decay": 1.0 * (1.0 + 0.04 * ((subject_index % 3) - 1)),
    }


def load_torch_recon_3d(file_recon):
    recon = torch.load(file_recon, map_location="cpu")

    if not isinstance(recon, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(recon)}")

    recon = recon.detach().cpu()

    if recon.ndim == 4:
        if recon.shape[0] == 1:
            recon = recon[0]
        else:
            recon = recon.mean(dim=0)

    recon = torch.abs(recon.squeeze())

    if recon.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeeze, got shape {tuple(recon.shape)}")

    return recon.numpy().astype(np.float64, copy=False)


def as_3d_magnitude(image):
    if torch.is_tensor(image):
        out = image.detach().cpu()
    else:
        out = torch.as_tensor(image)

    out = torch.abs(out).squeeze()

    if out.ndim == 4:
        if out.shape[0] == 1:
            out = out[0]
        else:
            out = out.mean(dim=0)

    if out.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeeze, got shape {tuple(out.shape)}")

    return out


def as_metric_volume(image):
    return as_3d_magnitude(image).numpy().astype(np.float64, copy=False)


def compute_mean_slice_sharpness(volume):
    vol = as_3d_magnitude(volume)

    values = []
    for iz in range(vol.shape[2]):
        values.append(float(sharpness_index(vol[:, :, iz])))

    return float(np.mean(values))


def image_quality_metrics_3d(image, reference):
    image_np = as_metric_volume(image)
    reference_np = as_metric_volume(reference)

    if image_np.shape != reference_np.shape:
        raise ValueError(
            f"Metric volumes have different shapes: {image_np.shape} vs {reference_np.shape}"
        )

    data_range = float(np.max(reference_np) - np.min(reference_np))
    if data_range <= 0:
        data_range = 1.0

    min_dim = min(reference_np.shape)
    win_size = 7 if min_dim >= 7 else min_dim
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3

    return {
        "ssim": float(
            structural_similarity(
                reference_np,
                image_np,
                data_range=data_range,
                win_size=win_size,
            )
        ),
        "psnr": float(
            peak_signal_noise_ratio(
                reference_np,
                image_np,
                data_range=data_range,
            )
        ),
        "nrmse": float(normalized_root_mse(reference_np, image_np)),
    }


def prefixed_metrics(prefix, metrics):
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def save_central_slice_png(volume, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    vol = as_3d_magnitude(volume).numpy()
    central_slice = vol[:, :, vol.shape[2] // 2]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.imshow(central_slice, cmap="gray")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_visual_comparison_3d(corrupted, corrected, ground_truth, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    volumes = [
        as_3d_magnitude(corrupted).numpy(),
        as_3d_magnitude(corrected).numpy(),
        as_3d_magnitude(ground_truth).numpy(),
    ]

    iz = volumes[2].shape[2] // 2
    images = [vol[:, :, iz] for vol in volumes]

    vmax = max(float(np.max(image)) for image in images)
    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200)

    for ax, image, title in zip(
        axes,
        images,
        ["Corrupted", "Corrected", "Ground truth"],
    ):
        ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout(pad=0.2)
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def build_params(
    runtime_output_dir,
    subject_index,
    base_seed,
    runtime_device,
    n_motion_states,
):
    overrides = {
        "clean_output_folders_before_run": False,
        "debug_flag": False,
        "verbose": False,
        "print_to_console": False,
        "jupyter_notebook_flag": False,
        "runtime_device": runtime_device,
        "flip_for_display": True,
        "logs_folder": str(runtime_output_dir / "logs") + "/",
        "results_folder": str(runtime_output_dir / "results") + "/",
        "debug_folder": str(runtime_output_dir / "debug_outputs") + "/",
        "initial_data_folder": str(runtime_output_dir / "initial_data") + "/",
        "acs": 8,
        "kernel_width": 4,
        "espirit_max_iter": 10,
        "N_motion_states": int(n_motion_states),
    }

    overrides.update(subject_motion_overrides(subject_index, base_seed))

    return load_config(
        data_type="real-world",
        reconstruction_config="config/reconstruction/nonrigid_3d.toml",
        motion_simulation_config="config/motion_simulation/nonrigid_3d.toml",
        motion_state_mode="realistic",
        overrides=overrides,
    )


def reference_volume_file(reference_root, subject):
    return Path(reference_root) / subject / "GricsRecon.pt"

def _read_h5_kspace_shape(h5_file):
    with h5py.File(h5_file, "r") as f:
        return tuple(f["kspace"].shape)

def load_grics_plusplus_sensitivity_maps(grics_plusplus_root, h5_file, device):
    subject = Path(h5_file).stem

    sensitivity_file = (
        Path(grics_plusplus_root)
        / subject
        / "SensitivityMaps.dat"
    )

    if not sensitivity_file.is_file():
        raise FileNotFoundError(f"Missing GRICS++ sensitivity maps: {sensitivity_file}")

    Ncoils, _, Nx, Ny, Nz = _read_h5_kspace_shape(h5_file)

    raw = np.fromfile(sensitivity_file, dtype="<f8")
    expected_size = 2 * Ny * Nx * Nz * Ncoils

    if raw.size != expected_size:
        raise ValueError(
            f"{sensitivity_file} contains {raw.size} floats, expected "
            f"{expected_size} for [2, Ny={Ny}, Nx={Nx}, Nz={Nz}, Ncoils={Ncoils}]."
        )

    raw_complex = raw[0::2] + 1j * raw[1::2]

    # GRICS++ writes: c, e2, e0, e1, with e1 fastest.
    maps_np = raw_complex.reshape((Ncoils, Nz, Nx, Ny), order="C")
    maps_np = np.transpose(maps_np, (0, 2, 3, 1))  # [Ncoils, Nx, Ny, Nz]


    

    return torch.from_numpy(np.ascontiguousarray(maps_np)).to(
        device=device,
        dtype=torch.complex128,
    )


def run_subject(
    realworld_file,
    reference_file,
    subject,
    subject_index,
    output_root,
    runtime_device,
    n_motion_states,
    base_seed,
):
    binary_dir = output_root / "binary_results" / subject
    visual_dir = output_root / "visual_results"
    runtime_output_dir = output_root / "runtime_outputs" / subject

    for folder in (binary_dir, visual_dir, runtime_output_dir):
        folder.mkdir(parents=True, exist_ok=True)

    params = build_params(
        runtime_output_dir=runtime_output_dir,
        subject_index=subject_index,
        base_seed=base_seed,
        runtime_device=runtime_device,
        n_motion_states=n_motion_states,
    )

    sp_device, t_device = initialize_runtime(params)

    external_smaps = load_grics_plusplus_sensitivity_maps(
        GRICS_plusplus_path,
        str(realworld_file),
        t_device,
    )

    data = DataLoader(
        params=params,
        t_device=t_device,
        sp_device=sp_device,
        filename=str(realworld_file),
        external_smaps=external_smaps,
    )

    data = DataLoader(
        params=params,
        t_device=t_device,
        sp_device=sp_device,
        filename=str(realworld_file),
    )

    reconstructor = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        motion_plot_context=data.motion_plot_context,
    )

    t0 = time.time()
    corrected_image, alpha_rec = reconstructor.run()
    elapsed_time = time.time() - t0

    torch.save(data.image_ground_truth, binary_dir / "GroundTruth.pt")
    torch.save(data.image_no_moco, binary_dir / "Corrupted.pt")
    torch.save(corrected_image, binary_dir / "Corrected.pt")
    torch.save(alpha_rec, binary_dir / "PredictedAlphaMaps.pt")

    if hasattr(data, "alpha_maps_true"):
        torch.save(data.alpha_maps_true, binary_dir / "SimulatedAlphaMaps.pt")

    if reference_file is not None and reference_file.is_file():
        torch.save(
            torch.from_numpy(load_torch_recon_3d(reference_file)),
            binary_dir / "SourceCorrectedVolume.pt",
        )

    save_central_slice_png(data.image_ground_truth[0], binary_dir / "ground_truth_central_slice.png")
    save_central_slice_png(data.image_no_moco[0], binary_dir / "corrupted_central_slice.png")
    save_central_slice_png(corrected_image[0], binary_dir / "corrected_central_slice.png")

    save_visual_comparison_3d(
        data.image_no_moco[0],
        corrected_image[0],
        data.image_ground_truth[0],
        visual_dir / f"{subject}_corrupted_corrected_ground_truth_central_slice.png",
    )

    uncorrected_quality = image_quality_metrics_3d(
        data.image_no_moco[0],
        data.image_ground_truth[0],
    )
    corrected_quality = image_quality_metrics_3d(
        corrected_image[0],
        data.image_ground_truth[0],
    )
    ground_truth_quality = image_quality_metrics_3d(
        data.image_ground_truth[0],
        data.image_ground_truth[0],
    )

    return {
        "subject": subject,
        "elapsed_time_s": elapsed_time,
        "sharpness_uncorrected": compute_mean_slice_sharpness(data.image_no_moco[0]),
        "sharpness_corrected": compute_mean_slice_sharpness(corrected_image[0]),
        "sharpness_ground_truth": compute_mean_slice_sharpness(data.image_ground_truth[0]),
        **prefixed_metrics("uncorrected", uncorrected_quality),
        **prefixed_metrics("corrected", corrected_quality),
        **prefixed_metrics("ground_truth", ground_truth_quality),
        **subject_motion_overrides(subject_index, base_seed),
    }


def write_results_csv(rows, output_root):
    output_file = output_root / "simulated_nonrigid_3d_metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else ["subject", *METRIC_NAMES]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    return output_file


def write_root_text_metrics(rows, output_root):
    written = []

    for metric_name in METRIC_NAMES:
        output_file = output_root / f"{metric_name}.txt"

        with open(output_file, "w") as f:
            for row in rows:
                if metric_name in row:
                    f.write(f"{row['subject']}\t{row[metric_name]}\n")

        written.append(output_file)

    return written


def write_root_metrics(rows, output_root):
    csv_file = write_results_csv(rows, output_root)
    text_files = write_root_text_metrics(rows, output_root)
    return csv_file, text_files


def discover_realworld_subjects(dataset_root, reference_root=None, requested_subjects=None):
    requested = set(requested_subjects or [])
    rows = []

    for realworld_file in sorted(Path(dataset_root).glob("*.h5")):
        subject = realworld_file.stem

        if not T1_SUBJECT_PATTERN.match(subject):
            continue

        if requested and subject not in requested:
            continue

        reference_file = (
            reference_volume_file(reference_root, subject)
            if reference_root is not None
            else None
        )
        rows.append((subject, realworld_file, reference_file))

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use real-world-format 3D files as motion-free references, "
            "simulate non-rigid 3D respiratory motion, reconstruct, and calculate "
            "sharpness, SSIM, PSNR, and NRMSE."
        )
    )

    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runtime-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--n-motion-states", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--subject", action="append", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    subjects = discover_realworld_subjects(
        args.dataset_root,
        reference_root=args.reference_root,
        requested_subjects=args.subject,
    )

    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]

    args.output_root.mkdir(parents=True, exist_ok=True)

    for subfolder in ("binary_results", "visual_results", "runtime_outputs"):
        (args.output_root / subfolder).mkdir(parents=True, exist_ok=True)

    rows = []
    write_root_metrics(rows, args.output_root)

    for subject_index, (subject, realworld_file, reference_file) in enumerate(subjects):
        print(f"Processing {subject}")

        try:
            row = run_subject(
                realworld_file=realworld_file,
                reference_file=reference_file,
                subject=subject,
                subject_index=subject_index,
                output_root=args.output_root,
                runtime_device=args.runtime_device,
                n_motion_states=args.n_motion_states,
                base_seed=args.base_seed,
            )
        except Exception as exc:
            print(f"Failed {subject}: {exc}")
            continue

        rows.append(row)

        print(
            f"{subject}: "
            f"uncorrected_sharpness={row['sharpness_uncorrected']:.6g}, "
            f"corrected_sharpness={row['sharpness_corrected']:.6g}, "
            f"ground_truth_sharpness={row['sharpness_ground_truth']:.6g}, "
            f"corrected_ssim={row['corrected_ssim']:.6g}, "
            f"corrected_psnr={row['corrected_psnr']:.6g}, "
            f"corrected_nrmse={row['corrected_nrmse']:.6g}, "
            f"time={row['elapsed_time_s']:.2f}s"
        )

        write_root_metrics(rows, args.output_root)

    results_file, metric_files = write_root_metrics(rows, args.output_root)

    print(f"Saved {results_file}")
    for metric_file in metric_files:
        print(f"Saved {metric_file}")

    if not rows:
        print("No subjects completed.")


if __name__ == "__main__":
    main()

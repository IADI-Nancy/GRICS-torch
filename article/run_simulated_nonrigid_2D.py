import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

if "__file__" in globals():
    ARTICLE_DIR = Path(__file__).resolve().parent
    REPO_ROOT = ARTICLE_DIR.parents[0]
else:
    ARTICLE_DIR = Path.cwd() / "article"
    REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ARTICLE_DIR))

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

from sharpness_index import sharpness_index
from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime


DEFAULT_DATASET_ROOT = Path("/home/pyuser/wkdir/data/GRICS-torch/article_dataset")
DEFAULT_OUTPUT_ROOT = Path("/home/pyuser/wkdir/data/GRICS-torch/article_simulated_nonrigid_2D")
DEFAULT_SAMPLING_CONFIG = Path("config/sampling_simulation/linear.toml")
DEFAULT_IMAGE_RESIZE_FACTOR = 0.5
DEFAULT_N_MOTION_STATES = 8
DEFAULT_QUIVER_DISPLAY_SCALE = 1.0
T2_SUBJECT_PATTERN = re.compile(r"^\d{4}_T2_[sm]$")
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
        "nonrigid_motion_amplitude": 5.0 * (1.0 + 0.04 * centered), #2
        "nonrigid_discrete_s_scale": 4.0 * (1.0 + 0.03 * ((subject_index % 5) - 2)),
        "nonrigid_resp_cycles_min": cycles_min,
        "nonrigid_resp_cycles_max": cycles_min + 1,
        "nonrigid_diaphragm_level": 0.2 + 0.015 * centered,
        "nonrigid_diaphragm_sharpness": 10.0 * (1.0 + 0.02 * ((subject_index % 5) - 2)),
        "nonrigid_lateral_sigma_lr": 0.45 * (1.0 + 0.03 * ((subject_index % 4) - 1.5)),
        "nonrigid_lr_fraction": 0.15 * (1.0 + 0.05 * ((subject_index % 5) - 2)),
        "nonrigid_top_decay": 1.0 * (1.0 + 0.04 * ((subject_index % 3) - 1)),
    }


def as_2d_magnitude(image):
    if torch.is_tensor(image):
        out = image.detach().cpu()
    else:
        out = torch.as_tensor(image)
    out = torch.abs(out)
    out = out.squeeze()
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeeze, got shape {tuple(out.shape)}")
    return out


def compute_sharpness(image):
    return float(sharpness_index(as_2d_magnitude(image)))


def as_metric_array(image):
    return as_2d_magnitude(image).numpy().astype(np.float64, copy=False)


def image_quality_metrics(image, reference):
    image_np = as_metric_array(image)
    reference_np = as_metric_array(reference)
    data_range = float(np.max(reference_np) - np.min(reference_np))
    if data_range <= 0:
        data_range = 1.0

    return {
        "ssim": float(
            structural_similarity(
                reference_np,
                image_np,
                data_range=data_range,
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


def save_image_png(image, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.imshow(as_2d_magnitude(image).numpy(), cmap="gray")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_visual_comparison(corrupted, corrected, ground_truth, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    images = [
        as_2d_magnitude(corrupted).numpy(),
        as_2d_magnitude(corrected).numpy(),
        as_2d_magnitude(ground_truth).numpy(),
    ]
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


def _alpha_axis_components(alpha_maps):
    alpha = alpha_maps.detach().cpu()
    if torch.is_complex(alpha):
        alpha = alpha.real
    if alpha.ndim != 3 or alpha.shape[0] < 2:
        raise ValueError(f"Expected alpha maps with shape [2, Nx, Ny], got {tuple(alpha.shape)}")
    return alpha[0], alpha[1]


def _flip_for_display(*items, flip_vertical=True):
    if not flip_vertical:
        return items
    return tuple(torch.flip(item, dims=[0]) for item in items)


def _quiver_fields(alpha_axis0, alpha_axis1, divisor=32):
    nx, ny = alpha_axis0.shape
    step = max(1, min(nx, ny) // divisor)
    yy, xx = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing="ij")
    amp = torch.sqrt(alpha_axis0 * alpha_axis0 + alpha_axis1 * alpha_axis1)
    return (
        xx[::step, ::step].numpy(),
        yy[::step, ::step].numpy(),
        (-alpha_axis1[::step, ::step]).numpy(),
        (alpha_axis0[::step, ::step]).numpy(),
        amp[::step, ::step].numpy(),
    )


def save_alpha_quiver_comparison(
    simulated_alpha,
    predicted_alpha,
    image,
    output_file,
    flip_vertical=True,
    quiver_display_scale=DEFAULT_QUIVER_DISPLAY_SCALE,
):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sim_a0, sim_a1 = _alpha_axis_components(simulated_alpha)
    pred_a0, pred_a1 = _alpha_axis_components(predicted_alpha)
    img = as_2d_magnitude(image)

    sim_a0, sim_a1, pred_a0, pred_a1, img = _flip_for_display(
        sim_a0,
        sim_a1,
        pred_a0,
        pred_a1,
        img,
        flip_vertical=flip_vertical,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    amp_max = max(
        float(torch.max(torch.sqrt(sim_a0 * sim_a0 + sim_a1 * sim_a1)).item()),
        float(torch.max(torch.sqrt(pred_a0 * pred_a0 + pred_a1 * pred_a1)).item()),
        1e-12,
    )

    for ax, a0, a1, title in zip(
        axes,
        [sim_a0, pred_a0],
        [sim_a1, pred_a1],
        ["Simulated", "Predicted"],
    ):
        xx, yy, ux, uy, amp = _quiver_fields(a0, a1)
        ux = ux * float(quiver_display_scale)
        uy = uy * float(quiver_display_scale)
        q = ax.quiver(
            xx,
            yy,
            ux,
            uy,
            amp,
            angles="xy",
            scale_units="xy",
            scale=1,
            cmap="viridis",
            clim=(0, amp_max),
        )
        ax.contour(
            torch.arange(img.shape[1]).numpy(),
            torch.arange(img.shape[0]).numpy(),
            img.numpy(),
            levels=8,
            colors="k",
            linewidths=0.7,
            alpha=0.8,
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

    fig.colorbar(q, ax=axes.ravel().tolist(), label="|u|")
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def build_params(
    runtime_output_dir,
    subject_index,
    base_seed,
    runtime_device,
    n_motion_states,
    sampling_config,
    image_resize_factor,
):
    overrides = {
        "clean_output_folders_before_run": False,
        "debug_flag": False,
        "verbose": False,
        "print_to_console": False,
        "jupyter_notebook_flag": False,
        "runtime_device": runtime_device,
        "logs_folder": str(runtime_output_dir / "logs") + "/",
        "results_folder": str(runtime_output_dir / "results") + "/",
        "debug_folder": str(runtime_output_dir / "debug_outputs") + "/",
        "initial_data_folder": str(runtime_output_dir / "initial_data") + "/",
        "N_motion_states": int(n_motion_states),
        "image_resize_factor": float(image_resize_factor),
    }
    overrides.update(subject_motion_overrides(subject_index, base_seed))

    return load_config(
        data_type="from_image",
        reconstruction_config="config/reconstruction/nonrigid_2d.toml",
        from_image_config="config/from_image.toml",
        sampling_config="config/sampling_simulation/interleaved.toml",
        motion_simulation_config="config/motion_simulation/nonrigid_2d.toml",
        overrides=overrides,
    )


def corrected_slice_file(corrected_root, subject, slice_idx):
    return (
        Path(corrected_root)
        / subject
        / f"Siemens_SingleImage_slice{slice_idx + 1:03d}_image01"
        / "GricsRecon.pt"
    )


def load_corrected_reference_image(corrected_file, flip_vertical=True):
    if not corrected_file.is_file():
        raise FileNotFoundError(f"Missing corrected slice image: {corrected_file}")
    image = torch.load(corrected_file, map_location="cpu")
    image = as_2d_magnitude(image).numpy()
    if flip_vertical:
        image = np.flipud(image)
    return np.ascontiguousarray(image)


def save_reference_image_for_loader(corrected_file, output_file):
    image = load_corrected_reference_image(corrected_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, image.astype(np.float64, copy=False))
    return output_file


def run_subject(
    corrected_file,
    subject,
    subject_index,
    slice_idx,
    output_root,
    runtime_device,
    n_motion_states,
    base_seed,
    sampling_config,
    image_resize_factor,
    quiver_display_scale,
):
    slice_name = f"slice{slice_idx + 1:03d}"
    binary_dir = output_root / "binary_results" / subject / slice_name
    visual_dir = output_root / "visual_results"
    alpha_dir = output_root / "alpha_quivers"
    runtime_output_dir = output_root / "runtime_outputs" / subject / slice_name

    for folder in (binary_dir, visual_dir, alpha_dir, runtime_output_dir):
        folder.mkdir(parents=True, exist_ok=True)

    params = build_params(
        runtime_output_dir=runtime_output_dir,
        subject_index=subject_index,
        base_seed=base_seed,
        runtime_device=runtime_device,
        n_motion_states=n_motion_states,
        sampling_config=sampling_config,
        image_resize_factor=image_resize_factor,
    )
    sp_device, t_device = initialize_runtime(params)
    reference_npy = save_reference_image_for_loader(
        corrected_file,
        runtime_output_dir / "corrected_reference_slice.npy",
    )

    data = DataLoader(
        params=params,
        t_device=t_device,
        sp_device=sp_device,
        filename=str(reference_npy),
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
    torch.save(
        torch.from_numpy(load_corrected_reference_image(corrected_file)),
        binary_dir / "SourceCorrectedSlice.pt",
    )
    torch.save(data.image_no_moco, binary_dir / "Corrupted.pt")
    torch.save(corrected_image, binary_dir / "Corrected.pt")
    torch.save(alpha_rec, binary_dir / "PredictedAlphaMaps.pt")
    if hasattr(data, "alpha_maps_true"):
        torch.save(data.alpha_maps_true, binary_dir / "SimulatedAlphaMaps.pt")

    save_image_png(data.image_ground_truth[0], binary_dir / "ground_truth.png")
    save_image_png(data.image_no_moco[0], binary_dir / "corrupted.png")
    save_image_png(corrected_image[0], binary_dir / "corrected.png")
    save_visual_comparison(
        data.image_no_moco[0],
        corrected_image[0],
        data.image_ground_truth[0],
        visual_dir / f"{subject}_{slice_name}_corrupted_corrected_ground_truth.png",
    )

    if hasattr(data, "alpha_maps_true"):
        save_alpha_quiver_comparison(
            data.alpha_maps_true,
            alpha_rec,
            corrected_image[0],
            alpha_dir / f"{subject}_{slice_name}_alpha_quiver_simulated_predicted.png",
            flip_vertical=params.flip_for_display,
            quiver_display_scale=quiver_display_scale,
        )

    uncorrected_quality = image_quality_metrics(
        data.image_no_moco[0],
        data.image_ground_truth[0],
    )
    corrected_quality = image_quality_metrics(
        corrected_image[0],
        data.image_ground_truth[0],
    )
    ground_truth_quality = image_quality_metrics(
        data.image_ground_truth[0],
        data.image_ground_truth[0],
    )

    return {
        "subject": subject,
        "slice": slice_idx + 1,
        "elapsed_time_s": elapsed_time,
        "sharpness_uncorrected": compute_sharpness(data.image_no_moco[0]),
        "sharpness_corrected": compute_sharpness(corrected_image[0]),
        "sharpness_ground_truth": compute_sharpness(data.image_ground_truth[0]),
        **prefixed_metrics("uncorrected", uncorrected_quality),
        **prefixed_metrics("corrected", corrected_quality),
        **prefixed_metrics("ground_truth", ground_truth_quality),
        **subject_motion_overrides(subject_index, base_seed),
    }


def write_results_csv(rows, output_root):
    output_file = output_root / "slice15_simulated_nonrigid_metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else ["subject", "slice", *METRIC_NAMES]
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use corrected 2D GRICS-torch slice images as motion-free references, "
            "simulate non-rigid respiratory motion, reconstruct, and calculate "
            "sharpness and image-quality metrics."
        )
    )
    parser.add_argument(
        "--corrected-root",
        "--dataset-root",
        dest="corrected_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing corrected per-subject 2D GricsRecon.pt outputs.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sampling-config", type=Path, default=DEFAULT_SAMPLING_CONFIG)
    parser.add_argument("--slice", type=int, default=15, help="One-based slice number.")
    parser.add_argument("--runtime-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--n-motion-states", type=int, default=DEFAULT_N_MOTION_STATES)
    parser.add_argument("--image-resize-factor", type=float, default=DEFAULT_IMAGE_RESIZE_FACTOR)
    parser.add_argument("--quiver-display-scale", type=float, default=DEFAULT_QUIVER_DISPLAY_SCALE)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--subject", action="append", default=None)
    return parser.parse_args()


def discover_corrected_subjects(corrected_root, slice_idx, requested_subjects=None):
    requested = set(requested_subjects or [])
    rows = []

    for subject_dir in sorted(Path(corrected_root).iterdir()):
        if not subject_dir.is_dir() or not T2_SUBJECT_PATTERN.match(subject_dir.name):
            continue
        if requested and subject_dir.name not in requested:
            continue
        corrected_file = corrected_slice_file(corrected_root, subject_dir.name, slice_idx)
        if corrected_file.is_file():
            rows.append((subject_dir.name, corrected_file))
        else:
            print(f"Missing corrected slice image: {corrected_file}")

    return rows


def main():
    args = parse_args()
    slice_idx = args.slice - 1
    if slice_idx < 0:
        raise ValueError("--slice must be one-based and >= 1.")

    subjects = discover_corrected_subjects(
        args.corrected_root,
        slice_idx,
        requested_subjects=args.subject,
    )
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]

    args.output_root.mkdir(parents=True, exist_ok=True)
    for subfolder in ("binary_results", "visual_results", "alpha_quivers", "runtime_outputs"):
        (args.output_root / subfolder).mkdir(parents=True, exist_ok=True)
    rows = []
    write_root_metrics(rows, args.output_root)

    for subject_index, (subject, corrected_file) in enumerate(subjects):
        print(f"Processing {subject} slice {args.slice}")
        try:
            row = run_subject(
                corrected_file=corrected_file,
                subject=subject,
                subject_index=subject_index,
                slice_idx=slice_idx,
                output_root=args.output_root,
                runtime_device=args.runtime_device,
                n_motion_states=args.n_motion_states,
                base_seed=args.base_seed,
                sampling_config=args.sampling_config,
                image_resize_factor=args.image_resize_factor,
                quiver_display_scale=args.quiver_display_scale,
            )
        except Exception as exc:
            print(f"Failed {subject}: {exc}")
            continue
        rows.append(row)
        print(
            f"{subject}: "
            f"uncorrected={row['sharpness_uncorrected']:.6g}, "
            f"corrected={row['sharpness_corrected']:.6g}, "
            f"ground_truth={row['sharpness_ground_truth']:.6g}, "
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

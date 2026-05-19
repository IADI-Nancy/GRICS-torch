import csv
import sys
import time
from pathlib import Path

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


OUTPUT_ROOT = Path(
    "/home/pyuser/wkdir/data/GRICS-torch/shepp_logan_simulated_rigid_3D_20_variants"
)

RUNTIME_DEVICE = "gpu"
N_MOTION_STATES = 8
N_VARIANTS = 20
BASE_SEED = 1000

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


def variant_motion_overrides(variant_index):
    centered = (variant_index % 9) - 4
    slow_variation = (variant_index % 5) - 2
    event_variation = variant_index % 3

    amp_scale_factor = 1.0 + 0.04 * centered
    trans_factor = 1.0 + 0.03 * slow_variation
    rot_factor = 1.0 + 0.025 * centered
    center_factor = 1.0 + 0.02 * slow_variation

    return {
        "seed": int(BASE_SEED + variant_index),
        "num_motion_events": int(3 + event_variation),
        "motion_tau": float(10.0 * (1.0 + 0.05 * slow_variation)),
        "rigid_motion_amplitude_scale": float(2.0 * amp_scale_factor),
        "max_tx_3d": float(0.165 * trans_factor),
        "max_ty_3d": float(1.867 * trans_factor),
        "max_tz_3d": float(0.347 * trans_factor),
        "max_rx_3d": float(0.219 * rot_factor),
        "max_ry_3d": float(0.378 * rot_factor),
        "max_rz_3d": float(0.190 * rot_factor),
        "max_center_x_3d": 0.0,
        "max_center_y_3d": float(-10.0 * center_factor),
        "max_center_z_3d": float(-25.0 * center_factor),
    }


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


def compute_mean_slice_sharpness(volume):
    vol = as_3d_magnitude(volume)
    values = [float(sharpness_index(vol[:, :, iz])) for iz in range(vol.shape[2])]
    return float(np.mean(values))


def image_quality_metrics_3d(image, reference):
    image_np = as_3d_magnitude(image).numpy().astype(np.float64, copy=False)
    reference_np = as_3d_magnitude(reference).numpy().astype(np.float64, copy=False)

    data_range = float(np.max(reference_np) - np.min(reference_np))
    if data_range <= 0:
        data_range = 1.0

    return {
        "ssim": float(
            structural_similarity(
                reference_np,
                image_np,
                data_range=data_range,
                win_size=7,
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


def build_params(variant_index, runtime_output_dir):
    overrides = {
        "clean_output_folders_before_run": False,
        "debug_flag": False,
        "verbose": False,
        "print_to_console": False,
        "jupyter_notebook_flag": False,
        "runtime_device": RUNTIME_DEVICE,
        "logs_folder": str(runtime_output_dir / "logs") + "/",
        "results_folder": str(runtime_output_dir / "results") + "/",
        "debug_folder": str(runtime_output_dir / "debug_outputs") + "/",
        "initial_data_folder": str(runtime_output_dir / "initial_data") + "/",
        "flip_for_display": True,
        "Nex": 2,
        "acs": 8,
        "kernel_width": 4,
        "espirit_max_iter": 10,
        "N_motion_states": N_MOTION_STATES,
    }

    overrides.update(variant_motion_overrides(variant_index))

    return load_config(
        data_type="shepp-logan",
        shepp_logan_config="config/shepp_logan_3d.toml",
        reconstruction_config="config/reconstruction/rigid_3d.toml",
        sampling_config="config/sampling_simulation/interleaved.toml",
        motion_simulation_config="config/motion_simulation/rigid_3d.toml",
        motion_state_mode="realistic",
        N_motion_states=N_MOTION_STATES,
        overrides=overrides,
    )


def run_variant(variant_index):
    variant_name = f"variant_{variant_index + 1:02d}"

    binary_dir = OUTPUT_ROOT / "binary_results" / variant_name
    visual_dir = OUTPUT_ROOT / "visual_results"
    runtime_output_dir = OUTPUT_ROOT / "runtime_outputs" / variant_name

    for folder in (binary_dir, visual_dir, runtime_output_dir):
        folder.mkdir(parents=True, exist_ok=True)

    params = build_params(variant_index, runtime_output_dir)
    sp_device, t_device = initialize_runtime(params)

    data = DataLoader(params=params, t_device=t_device, sp_device=sp_device)

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
    torch.save(alpha_rec, binary_dir / "PredictedRigidMotion.pt")

    if hasattr(data, "alpha_maps_true"):
        torch.save(data.alpha_maps_true, binary_dir / "SimulatedRigidMotion.pt")

    save_central_slice_png(data.image_ground_truth[0], binary_dir / "ground_truth_central_slice.png")
    save_central_slice_png(data.image_no_moco[0], binary_dir / "corrupted_central_slice.png")
    save_central_slice_png(corrected_image[0], binary_dir / "corrected_central_slice.png")

    save_visual_comparison_3d(
        data.image_no_moco[0],
        corrected_image[0],
        data.image_ground_truth[0],
        visual_dir / f"{variant_name}_corrupted_corrected_ground_truth_central_slice.png",
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
        "variant": variant_name,
        "elapsed_time_s": elapsed_time,
        "sharpness_uncorrected": compute_mean_slice_sharpness(data.image_no_moco[0]),
        "sharpness_corrected": compute_mean_slice_sharpness(corrected_image[0]),
        "sharpness_ground_truth": compute_mean_slice_sharpness(data.image_ground_truth[0]),
        **prefixed_metrics("uncorrected", uncorrected_quality),
        **prefixed_metrics("corrected", corrected_quality),
        **prefixed_metrics("ground_truth", ground_truth_quality),
        **variant_motion_overrides(variant_index),
    }


def write_results_csv(rows):
    output_file = OUTPUT_ROOT / "shepp_logan_simulated_rigid_3d_metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else ["variant", *METRIC_NAMES]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    return output_file


def write_root_text_metrics(rows):
    written = []

    for metric_name in METRIC_NAMES:
        output_file = OUTPUT_ROOT / f"{metric_name}.txt"

        with open(output_file, "w") as f:
            for row in rows:
                if metric_name in row:
                    f.write(f"{row['variant']}\t{row[metric_name]}\n")

        written.append(output_file)

    return written


def write_motion_parameter_file(rows):
    motion_keys = [
        "seed",
        "num_motion_events",
        "motion_tau",
        "rigid_motion_amplitude_scale",
        "max_tx_3d",
        "max_ty_3d",
        "max_tz_3d",
        "max_rx_3d",
        "max_ry_3d",
        "max_rz_3d",
        "max_center_x_3d",
        "max_center_y_3d",
        "max_center_z_3d",
    ]

    output_file = OUTPUT_ROOT / "motion_parameters.txt"

    with open(output_file, "w") as f:
        f.write("variant\t" + "\t".join(motion_keys) + "\n")

        for row in rows:
            values = [str(row.get(key, "nan")) for key in motion_keys]
            f.write(row["variant"] + "\t" + "\t".join(values) + "\n")

    return output_file


def write_root_metrics(rows):
    csv_file = write_results_csv(rows)
    text_files = write_root_text_metrics(rows)
    motion_file = write_motion_parameter_file(rows)
    return csv_file, text_files, motion_file


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for subfolder in ("binary_results", "visual_results", "runtime_outputs"):
        (OUTPUT_ROOT / subfolder).mkdir(parents=True, exist_ok=True)

    rows = []
    write_root_metrics(rows)

    for variant_index in range(N_VARIANTS):
        variant_name = f"variant_{variant_index + 1:02d}"
        print(f"Processing {variant_name}")

        try:
            row = run_variant(variant_index)
        except Exception as exc:
            print(f"Failed {variant_name}: {exc}")
            continue

        rows.append(row)

        print(
            f"{variant_name}: "
            f"uncorrected_sharpness={row['sharpness_uncorrected']:.6g}, "
            f"corrected_sharpness={row['sharpness_corrected']:.6g}, "
            f"ground_truth_sharpness={row['sharpness_ground_truth']:.6g}, "
            f"corrected_ssim={row['corrected_ssim']:.6g}, "
            f"corrected_psnr={row['corrected_psnr']:.6g}, "
            f"corrected_nrmse={row['corrected_nrmse']:.6g}, "
            f"time={row['elapsed_time_s']:.2f}s"
        )

        write_root_metrics(rows)

    results_file, metric_files, motion_file = write_root_metrics(rows)

    print(f"Saved {results_file}")
    print(f"Saved {motion_file}")

    for metric_file in metric_files:
        print(f"Saved {metric_file}")

    if not rows:
        print("No variants completed.")


if __name__ == "__main__":
    main()
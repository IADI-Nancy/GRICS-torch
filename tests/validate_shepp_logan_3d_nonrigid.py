"""Run 20 3D non-rigid Shepp-Logan validation simulations.

Motion is sampled independently per shot. Each run uses random k-space sampling
and varies temporal displacement scale and two spatial-envelope parameters.

Run from the repository root:
    python tests/validate_shepp_logan_3d_nonrigid.py --runs 20 --device cpu

Each invocation groups its simulations under
runs/simulation_validation/shepp_logan_3d_nonrigid/<batch timestamp>/.
Pipeline messages, warnings, and device fallback notices are saved in each
simulation's console log. The terminal prints one reconstruction-start line
per simulation, then SSIM and NRMSE median [Q1, Q3], relative improvement,
and paired two-sided Wilcoxon p-values. Individual results stay in subfolders.
SSIM and NRMSE compare image magnitudes without registration. Completion checks
finite outputs and valid shapes; quality improvement is reported separately.
"""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.DataPreparer import DataPreparer
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.output_layout import execution_scope, write_json, write_reconstruction
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from validation_metrics import image_metrics, summarize_residuals, summarize_similarity, print_similarity_summary


def sample_parameters(rng):
    return {"nonrigid_discrete_s_scale": round(float(rng.uniform(3.0, 5.0)), 4),
            "nonrigid_diaphragm_level": round(float(rng.uniform(0.1, 0.3)), 4),
            "nonrigid_lateral_sigma_ap": round(float(rng.uniform(0.25, 0.4)), 4)}


def build_config(run_number, device, output_root, sampled_parameters):
    """Load this case's TOML files and explicit validation settings."""
    # Select "odille-spline" or "espirit" via coil_sensitivity_config in this file.
    # Tune spline settings in config/coil_sensitivity_and_calibration/odille_spline.toml.
    reconstruction_config = ROOT / "config/reconstruction/nonrigid_3d.toml"
    with reconstruction_config.open("rb") as config_file:
        coil_sensitivity_config = tomllib.load(config_file)["reconstruction"]["coil_sensitivity_config"]
    # Compact acquisition: four coils, one repetition, four random-order shots.
    # The short two-level solver schedule is a functional check, not a tuned
    # image-quality benchmark. Preserve it when comparing different devices.
    overrides = {
        "workflow_label": f"simulation_{run_number:03d}", "output_root": str(output_root),
        "runtime_device": device, "seed_enabled": True, "seed": 20261101 + run_number,
        "use_deterministic_algorithms": True, "save_debug_plots": False,
        "save_reconstruction_logs": True, "save_reconstruction_tensors": True,
        "verbose": False, "print_to_console": False, "jupyter_notebook_flag": False,
        "N_SheppLogan": 32, "Nz_SheppLogan": 32,
        "Nex": 1, "NshotsPerNex": 4, "calibration_lines": 24,
        "ResolutionLevels": [0.5, 1.0], "GN_iterations_per_level": [2, 3],
        "max_iter_recon": 12, "max_iter_motion": 8, "cg_reg_scale_num_probes": 2,
        **sampled_parameters,
    }
    # These compact-phantom settings apply only to ESPIRiT.
    # Odille-spline loads its own settings without ESPIRiT overrides.
    if coil_sensitivity_config == "espirit":
        overrides.update(espirit_calibration_width=24, espirit_kernel_width=6)
    return load_config(
        data_type="shepp-logan",
        shepp_logan_config=str(ROOT / "config/synthetic_data/shepp_logan_3d.toml"),
        reconstruction_config=str(reconstruction_config),
        sampling_config=str(ROOT / "config/sampling_simulation/random.toml"),
        motion_simulation_config=str(ROOT / "config/motion_simulation/nonrigid_per_shot_3d.toml"),
        overrides=overrides,
    )


def run_simulation(run_number, device, output_root, console):
    """Run one simulated acquisition and return its objective diagnostics."""
    seed = 20261101 + run_number
    sampled_parameters = sample_parameters(np.random.default_rng(seed))
    with execution_scope():
        params = build_config(run_number, device, output_root, sampled_parameters)

        sp_device, t_device = initialize_runtime(params)

        data = DataPreparer(params=params, t_device=t_device, sp_device=sp_device)
        data.load_input_data()

        data.prepare_reconstructor_inputs()
        if torch.linalg.vector_norm(data.image_ground_truth).item() == 0:
            raise AssertionError("Coil combination produced an empty ground-truth image")

        reconstructor = JointReconstructor(
            data.kspace, data.smaps, data.sampling_idx, motion_signal=data.motion_signal,
            params=params, motion_plot_context=data.motion_plot_context,
            calibration_image_prior=data.grics_reference_image,
        )
        print(f"Simulation {run_number:03d}: reconstruction starting", file=console, flush=True)
        corrected_image, reconstructed_motion = reconstructor.run()

        if not (torch.isfinite(corrected_image).all() and torch.isfinite(reconstructed_motion).all()):
            raise AssertionError("Reconstructed image or motion model contains non-finite values")
        # Histories compare endpoints only at the final spatial resolution.
        # Image: ||measured - predicted k-space|| / ||measured k-space||.
        # Motion: ||motion update|| / (||updated motion|| + 1e-12).
        # The motion diagnostic measures update size, not ground-truth error.
        logger = reconstructor._run_logger.run_log
        # The reference is the coil-combined image actually used by the motion
        # simulator. Keep all images in that same normalized intensity scale.
        uncorrected_metrics = image_metrics(data.image_ground_truth, data.image_no_moco)
        corrected_metrics = image_metrics(data.image_ground_truth, corrected_image)
        metrics = {
            "case": "nonrigid_3d", "run": run_number, "seed": seed, "device": str(t_device),
            "motion_type": "non-rigid-per-shot", "sampled_parameters": sampled_parameters,
            "image_residual_final_level": summarize_residuals(logger["recon_residuals_by_level"][-1]),
            "motion_residual_final_level": summarize_residuals(logger["motion_residuals_by_level"][-1]),
            "uncorrected_vs_ground_truth": uncorrected_metrics,
            "corrected_vs_ground_truth": corrected_metrics,
            "quality_change": {"nrmse_improved": corrected_metrics["nrmse"] < uncorrected_metrics["nrmse"],
                               "ssim_improved": corrected_metrics["ssim"] > uncorrected_metrics["ssim"]},
        }
        if not metrics["motion_residual_final_level"]["count"]:
            raise AssertionError("Final resolution had no motion update")
        write_reconstruction(params, corrected_image, reconstructed_motion)
        write_json(Path(params.run_folder) / "simulation_metrics.json", metrics)
        reconstructor.finish_run()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20, help="Number of simulations (default: 20)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    # One batch folder contains all simulation outputs and the final summary.
    batch_folder = args.output_root / "simulation_validation" / "shepp_logan_3d_nonrigid" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    batch_folder.mkdir(parents=True, exist_ok=False)
    results = []
    console = sys.stdout
    for run_number in range(1, args.runs + 1):
        log_path = batch_folder / f"simulation_{run_number:03d}.log"
        try:
            # Preserve library progress, warnings, and fallback notices on disk.
            with log_path.open("w") as log, redirect_stdout(log), redirect_stderr(log):
                results.append(run_simulation(run_number, args.device, batch_folder, console))
        except Exception as error:
            raise RuntimeError(f"Simulation {run_number} failed: {error}\nDetails: {log_path}") from error

    summary = summarize_similarity(results)
    write_json(batch_folder / "similarity_summary.json", summary)
    print_similarity_summary(summary)


if __name__ == "__main__":
    main()

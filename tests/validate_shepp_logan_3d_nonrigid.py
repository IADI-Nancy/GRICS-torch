"""Run 20 3D non-rigid Shepp-Logan validation simulations.

Motion is sampled independently per shot. Each run uses random k-space sampling
and varies temporal displacement scale and two spatial-envelope parameters.

Run from the repository root:
    python tests/validate_shepp_logan_3d_nonrigid.py --runs 20 --device cpu

Each invocation groups its simulations under
runs/simulation_validation/shepp_logan_3d_nonrigid/<batch timestamp>/.
Pipeline messages, warnings, and device fallback notices are saved in each
simulation's console log. Each progress line reports reconstruction SUCCESS
and evaluation/output PASSED. The final summary prints SSIM and NRMSE median
[Q1, Q3], relative improvement, and paired two-sided Wilcoxon p-values with
reference limits and verdicts. Limits are derived and documented in
utils/validation_quality.py; a reference quality verdict requires 20 simulations
and matching reference settings. Individual results stay in subfolders.
SSIM and NRMSE compare image magnitudes without registration. Completion checks
finite outputs and valid shapes; quality improvement is reported separately.
Each simulation also saves image_comparison.png: corrupted, corrected, ground
truth, with three central orthogonal planes for 3D and a single row for 2D.
"""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.DataPreparer import DataPreparer
from src.utils.notebook_display import save_image_comparison
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.output_layout import execution_scope, write_json, write_reconstruction
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from utils.validation_metrics import image_metrics, summarize_residuals, summarize_similarity, print_similarity_summary
from utils.validation_quality import assess_quality, validation_settings


def sample_parameters(rng):
    return {"nonrigid_discrete_s_scale": round(float(rng.uniform(3.0, 5.0)), 4),
            "nonrigid_diaphragm_level": round(float(rng.uniform(0.1, 0.3)), 4),
            "nonrigid_lateral_sigma_ap": round(float(rng.uniform(0.25, 0.4)), 4)}


def build_config(run_number, device, output_root, sampled_parameters):
    """Load this case's TOML files and explicit validation settings."""
    # Select "odille-spline" or "espirit" via coil_sensitivity_config in this file.
    # Tune spline settings in config/coil_sensitivity_and_calibration/odille_spline.toml.
    reconstruction_config = ROOT / "config/reconstruction/nonrigid_3d.toml"
    # Phantom dimensions, coil count, and fill fraction come from the demo TOML.
    # Coil calibration also uses its selected TOML without compact-size overrides.
    # Reconstruction and sampling settings also come unchanged from their TOMLs.
    # Only randomized motion, seeds, and batch execution/output controls are overridden.
    overrides = {
        "workflow_label": f"simulation_{run_number:03d}", "output_root": str(output_root),
        "runtime_device": device, "seed_enabled": True, "seed": 20261101 + run_number,
        "save_debug_plots": False,
        "save_reconstruction_logs": True, "save_reconstruction_tensors": True,
        "verbose": False, "print_to_console": False, "jupyter_notebook_flag": False,
        **sampled_parameters,
    }
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

        reconstructor = JointReconstructor.from_prepared_data(data)
        print(f"Simulation {run_number:03d}: reconstruction starting ... ", end="", file=console, flush=True)
        corrected_image, reconstructed_motion = reconstructor.run()

        if not (torch.isfinite(corrected_image).all() and torch.isfinite(reconstructed_motion).all()):
            raise AssertionError("Reconstructed image or motion model contains non-finite values")
        if corrected_image.shape != data.image_ground_truth.shape:
            raise AssertionError("Reconstructed image shape differs from ground truth")
        print("SUCCESS; evaluation and output ... ", end="", file=console, flush=True)
        # Histories compare endpoints only at the final spatial resolution.
        # Image: ||measured - predicted k-space|| / ||measured k-space||.
        # Motion: ||motion update|| / (||updated motion|| + 1e-12).
        # The motion diagnostic measures update size, not ground-truth error.
        logger = reconstructor._run_logger.run_log
        # The reference is the coil-combined image actually used by the motion
        # simulator. All comparison images and outputs use restored acquisition units.
        uncorrected_metrics = image_metrics(data.image_ground_truth, data.image_no_moco)
        corrected_metrics = image_metrics(data.image_ground_truth, corrected_image)
        metrics = {
            "validation_settings": validation_settings(params),
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
        # Save the same central-plane comparison used by the demos, even when
        # debug plots are disabled. 2D cases use one row of three images.
        save_image_comparison(data.image_no_moco, corrected_image, data.image_ground_truth,
                              Path(params.run_folder) / "image_comparison.png")
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
            print("PASSED", file=console, flush=True)
        except Exception as error:
            print(f"FAILED (simulation {run_number:03d})", file=console, flush=True)
            raise RuntimeError(f"Simulation {run_number} failed: {error}\nDetails: {log_path}") from error

    summary = summarize_similarity(results)
    assess_quality(summary, results, "nonrigid_3d")
    write_json(batch_folder / "similarity_summary.json", summary)
    print_similarity_summary(summary)


if __name__ == "__main__":
    main()

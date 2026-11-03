"""Validate the demo B free-breathing breast MRI reconstruction.

Run from the repository root:
    python tests/validate_demo_b_freebreathing.py --device cpu

The local HDF5 file is the same input used by the demo B notebook. This dataset
has no known ground truth: image quality metrics against ground truth are
explicitly unavailable. The check records image shape and final-level residual
ratios and verifies finite output tensors.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.DataPreparer import DataPreparer
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.output_layout import execution_scope, write_json, write_reconstruction
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from validation_metrics import print_step, summarize_residuals


def run_reconstruction(data_file, device, output_root):
    """Run demo B and report residuals; real-world data have no ground truth."""
    if not data_file.is_file():
        raise FileNotFoundError(f"Demo B input missing: {data_file}. Download it using demo_b_2D_nonrigid_realworld_freebreathing.ipynb")
    label = "demo_b"
    with execution_scope():
        print_step(label, "BEGIN", "configuration")
        params = load_config(
            data_type="preprocessed-real",
            reconstruction_config=str(ROOT / "config/reconstruction/nonrigid_2d.toml"),
            overrides={"workflow_label": "manual_demo_b", "output_root": str(output_root),
                       "runtime_device": device, "save_debug_plots": False,
                       "save_reconstruction_logs": True, "save_reconstruction_tensors": True,
                       "verbose": False, "print_to_console": False, "jupyter_notebook_flag": False},
        )
        print_step(label, "END", "configuration")
        print_step(label, "BEGIN", "runtime initialization")
        sp_device, t_device = initialize_runtime(params)
        print_step(label, "END", "runtime initialization")
        print_step(label, "BEGIN", "real-world data loading")
        data = DataPreparer(params=params, t_device=t_device, sp_device=sp_device, filename=str(data_file))
        data.load_input_data()
        print_step(label, "END", "real-world data loading")
        print_step(label, "BEGIN", "coil maps and motion binning")
        data.prepare_reconstructor_inputs()
        print_step(label, "END", "coil maps and motion binning")
        print_step(label, "BEGIN", "joint image and motion reconstruction")
        reconstructor = JointReconstructor(
            data.kspace, data.smaps, data.sampling_idx, motion_signal=data.motion_signal,
            params=params, motion_plot_context=data.motion_plot_context,
            calibration_image_prior=data.grics_reference_image,
        )
        image, motion = reconstructor.run()
        print_step(label, "END", "joint image and motion reconstruction")
        print_step(label, "BEGIN", "metrics and output")
        if image.shape != data.image_no_moco.shape or not (torch.isfinite(image).all() and torch.isfinite(motion).all()):
            raise AssertionError("Demo B returned an invalid image shape or non-finite output")
        logger = reconstructor._run_logger.run_log
        metrics = {"case": label, "device": str(t_device), "input_file": str(data_file),
                   "image_shape": list(image.shape),
                   "image_residual_final_level": summarize_residuals(logger["recon_residuals_by_level"][-1]),
                   "motion_residual_final_level": summarize_residuals(logger["motion_residuals_by_level"][-1]),
                   "ground_truth_metrics": "unavailable for real-world data"}
        write_reconstruction(params, image, motion)
        write_json(Path(params.run_folder) / "manual_validation.json", metrics)
        reconstructor.finish_run()
        print(json.dumps(metrics, indent=2), flush=True)
        print_step(label, "END", "metrics and output")
    print("=" * 72, flush=True)
    print("PASS: demo B completed with finite outputs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--data-file", type=Path, default=ROOT / "demos_data/demo_breast_data.h5")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    run_reconstruction(args.data_file, args.device, args.output_root)


if __name__ == "__main__":
    main()

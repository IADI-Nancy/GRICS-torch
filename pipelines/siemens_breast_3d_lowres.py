#!/usr/bin/env python
"""Full-volume GRICS reconstruction for one low-resolution Siemens breast 3D acquisition.

Examples:
    python pipelines/siemens_breast_3d_lowres.py subject.dat subject.saec
    python pipelines/siemens_breast_3d_lowres.py subject.h5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# Pipeline-level settings. Reconstruction parameters live in the TOML file below.
OUTPUT_ROOT = Path("runs/siemens_breast_3d_lowres")
RECONSTRUCTION_CONFIG = "config/reconstruction/nonrigid_3d_breast.toml"
RUNTIME_DEVICE = "gpu"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py

from src.preprocessing.DataLoader import DataLoader
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from src.runtime.output_layout import managed_execution
from pipelines._execution import (
    reconstruction_overrides, timed_reconstruction, synchronize,
    export_reconstruction, finish_run,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_data_file", type=Path,
        help="One Siemens .dat, ISMRMRD .h5/.mrd, or preprocessed .h5 volume.",
    )
    parser.add_argument(
        "saec_file", type=Path, nargs="?",
        help="SAEC physiological file; required for Siemens/ISMRMRD input, omitted for preprocessed HDF5.",
    )
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default=RUNTIME_DEVICE)
    parser.add_argument('--save-reconstruction-tensors', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--save-reconstruction-logs', action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args(argv)


def require_existing_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")


def data_type_from_raw_data_file(raw_data_file: Path, saec_file: Path | None) -> str:
    require_existing_file(raw_data_file, "raw_data_file")
    suffix = raw_data_file.suffix.lower()
    if suffix not in {".dat", ".h5", ".mrd"}:
        raise ValueError("raw_data_file must be a Siemens .dat or HDF5/ISMRMRD .h5/.mrd file.")
    if suffix in {".h5", ".mrd"}:
        with h5py.File(raw_data_file, "r") as source:
            if "kspace" in source:
                if saec_file is not None:
                    raise ValueError("Preprocessed HDF5 already contains physiology; omit saec_file.")
                required = {"kspace", "motion_data", "idx_ky", "idx_kz", "idx_nex"}
                missing = required - set(source)
                if missing:
                    raise ValueError(f"Preprocessed HDF5 is missing datasets: {sorted(missing)}")
                shape = source["kspace"].shape
                if len(shape) != 5 or any(size < 1 for size in shape) or shape[-1] <= 1:
                    raise ValueError(f"Expected 3D kspace [coils, repetitions, Nx, Ny, Nz], got {shape}.")
                return "preprocessed-real"
    if saec_file is None:
        raise ValueError("saec_file is required for Siemens/ISMRMRD raw input.")
    require_existing_file(saec_file, "saec_file")
    return "siemens-saec" if suffix == ".dat" else "ismrmrd-saec"


def load_volume(raw_data_file: Path, saec_file: Path | None = None, *,
                output_root=OUTPUT_ROOT, device=RUNTIME_DEVICE,
                reconstruction_config=RECONSTRUCTION_CONFIG, overrides=None,
                    save_reconstruction_logs=None, save_reconstruction_tensors=None) -> DataLoader:
    """Load one acquisition and estimate coil maps with the Odille spline method."""
    data_type = data_type_from_raw_data_file(raw_data_file, saec_file)
    raw_input = data_type != "preprocessed-real"
    params = load_config(
        data_type=data_type,
        reconstruction_config=REPO_ROOT / reconstruction_config,
        coil_sensitivity_config=REPO_ROOT / "config/coil_sensitivity/odille_spline.toml",
        real_data_config=REPO_ROOT / "config/real_data/saec.toml" if raw_input else None,
        ismrmrd_reader_config=REPO_ROOT / "config/real_data/ismrmrd_reader.toml" if raw_input else None,
        overrides=reconstruction_overrides(
            output_root, device, overrides, save_reconstruction_logs=save_reconstruction_logs,
            save_reconstruction_tensors=save_reconstruction_tensors),
    )
    sp_device, t_device = initialize_runtime(params)
    params._run_outputs.manifest["inputs"] = {
        "raw_data_file": str(raw_data_file.resolve()),
        "saec_file": str(saec_file.resolve()) if saec_file is not None else None,
    }
    params._run_outputs.flush()
    data = DataLoader(
        params=params, t_device=t_device, sp_device=sp_device,
        filename=(str(raw_data_file), str(saec_file)) if raw_input else str(raw_data_file),
        run_pipeline=False,
    )
    data.load_data()
    if int(data.Nz) <= 1:
        raise ValueError("The 3D pipeline requires more than one encoded partition.")
    data.run_slice_pipeline()
    return data


@managed_execution
def run_pipeline(raw_data_file, saec_file=None, *, output_root=OUTPUT_ROOT,
                 device=RUNTIME_DEVICE, reconstruction_config=RECONSTRUCTION_CONFIG,
                 overrides=None, save_reconstruction_logs=None, save_reconstruction_tensors=None, return_tensors=True) -> dict:
    """Reconstruct one volume without changing module globals or reading sys.argv.

    Returns run_folder, timings, and a one-element reconstructions list containing
    CPU image/motion tensors, output paths and reconstruction_seconds.
    Common save_reconstruction_logs/tensors flags control output independently.
    None uses the TOML/overrides value; metadata is always saved.
    Text logging is included in solver timing. Plotting and intermediate tensor
    exports are disabled. Other validated reconstruction overrides are accepted.
    """
    started = time.perf_counter()
    raw_data_file = Path(raw_data_file)
    saec_file = Path(saec_file) if saec_file is not None else None
    data = load_volume(raw_data_file, saec_file, output_root=output_root, device=device,
                       reconstruction_config=reconstruction_config, overrides=overrides,
                       save_reconstruction_logs=save_reconstruction_logs,
                       save_reconstruction_tensors=save_reconstruction_tensors)
    synchronize(data.kspace.device)
    preprocessing_seconds = time.perf_counter() - started
    image, motion, reconstruction_seconds = timed_reconstruction(data)
    transfer_started = time.perf_counter()
    result = {
        'image': image.detach().cpu(), 'motion': motion.detach().cpu(),
        'reconstruction_seconds': reconstruction_seconds,
        'image_stage': 'before_postprocessing',
    }
    transfer_seconds = time.perf_counter() - transfer_started
    export_started = time.perf_counter()
    export_reconstruction(data.params, result)
    timings = {
        'preprocessing_seconds': preprocessing_seconds,
        'reconstruction_seconds': reconstruction_seconds,
        'transfer_seconds': transfer_seconds,
        'export_seconds': time.perf_counter() - export_started,
        'pipeline_seconds': time.perf_counter() - started,
    }
    return finish_run(data, [result], timings,
                      return_tensors=return_tensors)


def main(argv=None) -> dict:
    args = parse_args(argv)
    result = run_pipeline(args.raw_data_file, args.saec_file, output_root=args.output_root,
                          device=args.device, save_reconstruction_tensors=args.save_reconstruction_tensors,
                          save_reconstruction_logs=args.save_reconstruction_logs,
                          return_tensors=False)
    print(f"[run] Solver: {result['timings']['reconstruction_seconds']:.2f} s. "
          f"Run: {result['run_folder']}")
    return result


if __name__ == "__main__":
    main()

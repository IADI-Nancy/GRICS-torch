#!/usr/bin/env python
"""Full-volume GRICS reconstruction for one low-resolution Siemens breast 3D acquisition.

Examples:
    python pipelines/siemens_breast_3d_lowres.py subject.dat subject.saec
    python pipelines/siemens_breast_3d_lowres.py subject.h5

Preprocessed input with raw fallback:
    python pipelines/siemens_breast_3d_lowres.py subject.dat subject.saec --preprocessed-file prepared.h5
    python pipelines/siemens_breast_3d_lowres.py subject.mrd subject.saec --preprocessed-file prepared.h5
Preprocessed input only:
    python pipelines/siemens_breast_3d_lowres.py prepared.h5
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

from pipelines._inputs import data_type_from_raw_data_file as classify_input, resolve_input_files

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
        "raw_data_file", type=Path, nargs="?",
        help="One Siemens .dat, ISMRMRD .h5/.mrd, or preprocessed .h5 volume.",
    )
    parser.add_argument(
        "saec_file", type=Path, nargs="?",
        help="SAEC physiological file; required for Siemens/ISMRMRD input, omitted for preprocessed HDF5.",
    )
    parser.add_argument('--preprocessed-file', type=Path, help='Preferred preprocessed HDF5; use raw_data_file and SAEC if missing.')
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default=RUNTIME_DEVICE)
    parser.add_argument('--save-reconstruction-tensors', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--save-reconstruction-logs', action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args(argv)


def data_type_from_raw_data_file(raw_data_file: Path, saec_file: Path | None = None) -> str:
    return classify_input(raw_data_file, saec_file, dimension="3D")


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
def run_pipeline(raw_data_file=None, saec_file=None, *, preprocessed_file=None, output_root=OUTPUT_ROOT,
                 device=RUNTIME_DEVICE, reconstruction_config=RECONSTRUCTION_CONFIG,
                 overrides=None, save_reconstruction_logs=None, save_reconstruction_tensors=None, return_tensors=True) -> dict:
    """Reconstruct one volume without changing module globals or reading sys.argv.

    preprocessed_file is preferred when present; otherwise raw_data_file and SAEC are used.
    Returns run_folder, timings, and a one-element reconstructions list containing
    CPU image/motion tensors, output paths and reconstruction_seconds.
    Common save_reconstruction_logs/tensors flags control output independently.
    None uses the TOML/overrides value; metadata is always saved.
    Text logging is included in solver timing. Plotting and intermediate tensor
    exports are disabled. Other validated reconstruction overrides are accepted.
    """
    started = time.perf_counter()
    raw_data_file, saec_file = resolve_input_files(raw_data_file, saec_file, preprocessed_file)
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
    result = run_pipeline(args.raw_data_file, args.saec_file, preprocessed_file=args.preprocessed_file, output_root=args.output_root,
                          device=args.device, save_reconstruction_tensors=args.save_reconstruction_tensors,
                          save_reconstruction_logs=args.save_reconstruction_logs,
                          return_tensors=False)
    print(f"[run] Solver: {result['timings']['reconstruction_seconds']:.2f} s. "
          f"Run: {result['run_folder']}")
    return result


if __name__ == "__main__":
    main()

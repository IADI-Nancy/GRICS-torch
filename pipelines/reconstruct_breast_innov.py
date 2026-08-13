#!/usr/bin/env python
"""Batch-reconstruct paired Breast-INNOV Siemens/SAEC T2 acquisitions.

The four outputs follow the existing article dataset layout:
  article_dataset_nomoco_cpp_schedule       uncorrected (one motion state)
  article_dataset_cpp_schedule              BELT-corrected
  article_dataset_1_marmot_cpp_schedule     best single-MARMOT-corrected
  article_dataset_all_marmots_cpp_schedule  all-MARMOT-corrected
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

# Match the known-working article runner: limits must be set before importing
# NumPy, PyTorch, SigPy, or SimpleITK.
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
    "SimpleITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
    "KMP_BLOCKTIME": "0",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}
for _key, _value in _THREAD_ENV.items():
    os.environ[_key] = _value

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.siemens_breast_T2 as breast_pipeline
from src.preprocessing.RespiratoryDataReader import RespiratoryDataReader
from src.reconstruction.JointReconstructor import JointReconstructor

PAIR_PATTERN = re.compile(r"^(?P<subject>\d{4}_T2_[sm])\.dat$", re.IGNORECASE)
MODES = (
    ("uncorrected", None, 1, "article_dataset_nomoco_cpp_schedule"),
    ("belt", "BELT", None, "article_dataset_cpp_schedule"),
    ("one_marmot", "1MARMOT", None, "article_dataset_1_marmot_cpp_schedule"),
    ("all_marmots", "ALL_MARMOTS", None, "article_dataset_all_marmots_cpp_schedule"),
)

MODE_SOURCE_DATA = None
MODE_SUBJECT_DIR = None


def parse_args() -> argparse.Namespace:
    default_database = REPO_ROOT.parent / "data" / "Breast-INNOV_GRICS_database"
    default_output = REPO_ROOT.parent / "data" / "GRICS-torch"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-root", type=Path, default=default_database)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument(
        "--dataset-name",
        help=(
            "Override the mode-specific dataset directory name below output-root. "
            "Useful when replacing an existing article dataset in place."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=tuple(mode[0] for mode in MODES),
        default=None,
        help="Run only one reconstruction mode (default: run all modes).",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--saec-dir", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run a mode whose subject manifest already reports success.",
    )
    return parser.parse_args()


def discover_pairs(raw_dir: Path, saec_dir: Path) -> list[tuple[str, Path, Path]]:
    pairs = []
    for raw_file in sorted(raw_dir.glob("*.dat")):
        match = PAIR_PATTERN.fullmatch(raw_file.name)
        if match is None:
            continue
        subject = match.group("subject")
        saec_file = saec_dir / f"{subject}.h5"
        if saec_file.is_file():
            pairs.append((subject, raw_file, saec_file))
        else:
            print(f"[skip] {raw_file.name}: no matching {saec_file.name}", flush=True)
    return pairs


def available_sensors(saec_file: Path) -> dict[str, bool]:
    available = {}
    for sensor in ("BELT", "1MARMOT", "ALL_MARMOTS"):
        try:
            RespiratoryDataReader._read_and_process_data(str(saec_file), sensor)
        except Exception as error:
            available[sensor] = False
            print(f"[physio] {saec_file.name}: {sensor} unavailable ({error})", flush=True)
        else:
            available[sensor] = True
    return available


def choose_uncorrected_sensor(available: dict[str, bool]) -> str | None:
    # Physiology is still used to order/bin the acquired data; one motion state
    # disables motion correction while retaining the available acquisition timing.
    if available["BELT"]:
        return "BELT"
    if available["ALL_MARMOTS"]:
        return "ALL_MARMOTS"
    return None


def output_overrides(folder: Path, sensor: str, motion_states: int | None) -> dict:
    overrides = breast_pipeline.output_overrides(folder)
    overrides["rawdata_sensor_type"] = sensor
    if motion_states is not None:
        overrides["N_motion_states"] = motion_states
    return overrides


def load_acquisition(raw_file: Path, saec_file: Path, output_dir: Path,
                     sensor: str, motion_states: int | None):
    breast_pipeline.require_existing_file(raw_file, "raw_data_file")
    breast_pipeline.require_existing_file(saec_file, "saec_file")
    params = breast_pipeline.load_config(
        data_type="siemens-saec",
        reconstruction_config=breast_pipeline.RECONSTRUCTION_CONFIG,
        overrides=output_overrides(output_dir / "load", sensor, motion_states),
    )
    sp_device, t_device = breast_pipeline.initialize_runtime(params)
    data = breast_pipeline.DataLoader(
        params=params,
        t_device=t_device,
        sp_device=sp_device,
        filename=(str(raw_file), str(saec_file)),
        run_pipeline=False,
    )
    data.load_data()
    return data


def to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_physiology(output_dir: Path, data) -> None:
    attributes = {
        "motion_curve_chronological": "_motion_curve_for_binning",
        "motion_signal_reconstruction": "motion_signal",
        "motion_labels": "motion_labels",
        "ky_idx_chronological": "ky_idx_chronological",
        "kz_idx_chronological": "kz_idx_chronological",
        "nex_idx_chronological": "nex_idx_chronological",
    }
    tensors = {name: getattr(data, attr, None) for name, attr in attributes.items()}
    arrays = {name: array for name, value in tensors.items()
              if (array := to_numpy(value)) is not None}
    np.savez(output_dir / "PhysiologicalData_torch.npz", **arrays)
    torch.save(tensors, output_dir / "PhysiologicalData_torch.pt")
    curve = arrays.get("motion_curve_chronological")
    if curve is not None:
        np.asarray(curve, dtype="<f4").reshape(1, -1).tofile(
            output_dir / "ModelInputs_torch.dat"
        )


def reconstruct_slice(source_data, slice_idx: int, subject_dir: Path) -> dict:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(
        f"[worker pid={os.getpid()}] slice {slice_idx + 1:03d} "
        f"torch_threads={torch.get_num_threads()}",
        flush=True,
    )
    output_dir = subject_dir / f"Siemens_SingleImage_slice{slice_idx + 1:03d}_image01"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = copy.copy(source_data)
    data.params = copy.copy(source_data.params)
    for key, value in breast_pipeline.output_overrides(output_dir).items():
        setattr(data.params, key, value)
    data.run_slice_pipeline(slice_idx=slice_idx)
    save_physiology(output_dir, data)
    reconstructor = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=data.params,
        kspace_scale=data.kspace_scale,
        motion_plot_context=data.motion_plot_context,
    )
    started = time.time()
    image, alpha = reconstructor.run()
    torch.save(image, output_dir / "GricsRecon.pt")
    torch.save(alpha, output_dir / "GricsAlphaMaps.pt")
    return {"slice_idx": slice_idx, "slice_number": slice_idx + 1,
            "elapsed_s": time.time() - started, "output_dir": str(output_dir)}


def reconstruct_mode_slice(slice_idx: int) -> dict:
    return reconstruct_slice(MODE_SOURCE_DATA, slice_idx, MODE_SUBJECT_DIR)


def run_mode(subject: str, raw_file: Path, saec_file: Path, subject_dir: Path,
             sensor: str, motion_states: int | None, max_workers: int | None) -> None:
    subject_dir.mkdir(parents=True, exist_ok=True)
    source_data = load_acquisition(
        raw_file, saec_file, subject_dir, sensor, motion_states
    )
    slice_indices = list(range(int(source_data.Nz)))
    workers = max_workers or min(len(slice_indices), os.cpu_count() or 1)

    # Reuse the pipeline's fork-based executor by installing this acquisition as
    # shared read-only state and a mode-specific worker callable.
    global MODE_SOURCE_DATA, MODE_SUBJECT_DIR
    MODE_SOURCE_DATA = source_data
    MODE_SUBJECT_DIR = subject_dir
    breast_pipeline.LOADED_DATA = source_data
    original_worker = breast_pipeline.reconstruct_slice
    original_max_workers = breast_pipeline.MAX_WORKERS
    breast_pipeline.MAX_WORKERS = workers
    breast_pipeline.reconstruct_slice = reconstruct_mode_slice
    try:
        results, workers = breast_pipeline.reconstruct_slices_in_parallel(slice_indices)
    finally:
        breast_pipeline.reconstruct_slice = original_worker
        breast_pipeline.MAX_WORKERS = original_max_workers
        MODE_SOURCE_DATA = None
        MODE_SUBJECT_DIR = None

    manifest = {
        "status": "complete",
        "subject": subject,
        "raw_data_file": str(raw_file),
        "saec_file": str(saec_file),
        "sensor_type": sensor,
        "motion_states": int(source_data.params.N_motion_states),
        "nslices": int(source_data.Nz),
        "max_workers": workers,
        "slice_results": results,
    }
    with (subject_dir / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)


def is_complete(subject_dir: Path) -> bool:
    manifest_file = subject_dir / "run_manifest.json"
    try:
        return json.loads(manifest_file.read_text(encoding="utf-8"))["status"] == "complete"
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir or args.database_root / "Siemens_RAW"
    saec_dir = args.saec_dir or args.database_root / "SAEC"
    pairs = discover_pairs(raw_dir, saec_dir)
    if not pairs:
        raise SystemExit(f"No matching T2 Siemens/SAEC pairs in {raw_dir} and {saec_dir}")

    for subject, raw_file, saec_file in pairs:
        available = available_sensors(saec_file)
        for mode, requested_sensor, motion_states, dataset_name in MODES:
            if args.mode is not None and mode != args.mode:
                continue
            sensor = requested_sensor or choose_uncorrected_sensor(available)
            if sensor is None or not available[sensor]:
                print(f"[skip] {subject} {mode}: physiological data unavailable", flush=True)
                continue
            destination_dataset = args.dataset_name or dataset_name
            subject_dir = args.output_root / destination_dataset / subject
            if is_complete(subject_dir) and not args.overwrite:
                print(f"[skip] {subject} {mode}: already complete", flush=True)
                continue
            print(f"[run] {subject} {mode}: sensor={sensor}, output={subject_dir}", flush=True)
            if not args.dry_run:
                run_mode(subject, raw_file, saec_file, subject_dir, sensor,
                         motion_states, args.max_workers)


if __name__ == "__main__":
    main()

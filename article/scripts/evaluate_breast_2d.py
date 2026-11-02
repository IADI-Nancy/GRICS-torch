#!/usr/bin/env python
"""Evaluate breast 2D GRICS-torch reconstruction time and sharpness."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from article.scripts.sharpness_index import sharpness_index
from pipelines.siemens_breast_T2 import RECONSTRUCTION_CONFIG, run_pipeline
from src.preprocessing.MotionBinner import ConstantPhysiologicalSignalError
from src.runtime.runtime_config import load_config

DATABASE = Path("/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database")
DATASET = Path("/home/pyuser/wkdir/data/GRICS-torch/article_dataset_2D")
SUBJECT_PATTERN = re.compile(r"^00\d{2}_T2_[sm]$", re.I)


def validate_prepared_acquisition(prepared: Path) -> None:
    """Reject incomplete prepared inputs before starting reconstruction."""
    with h5py.File(prepared, "r") as source:
        required = ("kspace", "motion_data", "idx_ky", "idx_kz", "idx_nex", "ismrmrd_header")
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"{prepared}: missing prepared datasets {missing}.")
        shape = tuple(source["kspace"].shape)
    if len(shape) != 5 or shape[-1] < 1:
        raise ValueError(f"{prepared}: expected five-dimensional multislice k-space, got {shape}.")


def score_image(image: torch.Tensor) -> float:
    """Complex-average repetitions and score one native-grid 2D image."""
    image = torch.as_tensor(image)
    magnitude = image.mean(dim=0).abs() if image.ndim == 3 else image.abs()
    if magnitude.ndim != 2:
        raise ValueError(f"Expected a 2D image or repetition-stacked 2D image, got {tuple(image.shape)}.")
    score = float(sharpness_index(magnitude))
    if not np.isfinite(score):
        raise ValueError("Sharpness calculation produced a non-finite value.")
    return score


def max_slice_reconstruction_seconds(row: dict) -> float:
    """Return the longest saved slice solver time for an acquisition."""
    saved = row.get("slice_reconstruction_seconds")
    if saved is None:
        manifest = json.loads((Path(row["run_folder"]) / "manifest.json").read_text())
        by_slice = {int(item["slice_number"]): float(item["reconstruction_seconds"])
                    for item in manifest["reconstructions"].values()}
        if sorted(by_slice) != list(range(1, int(row["slice_count"]) + 1)):
            raise ValueError(f"Missing per-slice timing for {row['subject']}.")
        saved = list(by_slice.values())
    if len(saved) != int(row["slice_count"]) or any(
            not np.isfinite(value) or value <= 0 for value in saved):
        raise ValueError(f"Invalid per-slice timing for {row['subject']}.")
    return max(float(value) for value in saved)


def save_results(rows: list[dict], folder: Path) -> None:
    """Checkpoint measurements and regenerate paired acquisition-level plots."""
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "measurements.json.tmp"
    temporary.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    temporary.replace(folder / "measurements.json")
    fields = ["subject", "mode", "motion_states", "slice_count", "mean_sharpness",
              "reconstruction_seconds_sum", "reconstruction_seconds_max",
              "compute_wall_seconds", "run_folder"]
    with (folder / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    paired = [name for name in dict.fromkeys(row["subject"] for row in rows)
              if {row["mode"] for row in rows if row["subject"] == name}
              == {"corrected", "nomoco"}]
    if not paired:
        return
    by_key = {(row["subject"], row["mode"]): row for row in rows}
    for metric, ylabel, filename in (
        ("mean_sharpness", "Mean slice sharpness index", "sharpness_boxplot.png"),
        ("reconstruction_seconds_max", "Longest slice solver time per subject (s)", "time_boxplot.png"),
    ):
        values = []
        for mode in ("corrected", "nomoco"):
            values.append([
                max_slice_reconstruction_seconds(by_key[subject, mode])
                if metric == "reconstruction_seconds_max" else by_key[subject, mode][metric]
                for subject in paired])
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.boxplot(values)
        axis.set_xticks([1, 2], ["Motion corrected", "One state"])
        for corrected, nomoco in zip(*values):
            axis.plot([1, 2], [corrected, nomoco], "o-", color="gray", alpha=.4, markersize=3)
        axis.set_ylabel(ylabel)
        axis.set_title(f"Breast 2D: {len(paired)} paired acquisitions")
        figure.tight_layout()
        figure.savefig(folder / filename, dpi=200)
        plt.close(figure)


def read_records(path: Path, resume: bool) -> list[dict]:
    records = json.loads(path.read_text()) if resume and path.exists() else []
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"Invalid checkpoint: {path}")
    return records


def find_subjects(ismrmrd_dir: Path, saec_dir: Path, requested: set[str]):
    raw_files = sorted(path for path in ismrmrd_dir.iterdir()
                       if path.is_file() and SUBJECT_PATTERN.fullmatch(path.stem)
                       and path.suffix.lower() in {".h5", ".mrd"})
    if requested:
        raw_files = [path for path in raw_files if path.stem in requested]
        found = {path.stem for path in raw_files}
        if found != requested:
            raise FileNotFoundError(f"Requested subjects are unavailable: {sorted(requested - found)}")
    if not raw_files:
        raise ValueError(f"No matching T2 acquisitions found in {ismrmrd_dir}.")
    missing = [path for path in raw_files if not (saec_dir / path.name).is_file()]
    if missing:
        print("[skip] missing matching SAEC files:\n" +
              "\n".join(str(saec_dir / path.name) for path in missing), flush=True)
    return [path for path in raw_files if path not in missing], missing


def evaluate(ismrmrd_dir=DATABASE / "ISMRMRD", saec_dir=DATABASE / "SAEC",
             dataset_dir=DATASET, reconstruction_config=ROOT / RECONSTRUCTION_CONFIG,
             device="cpu", max_workers=None, resume=True,
             modes=("corrected", "nomoco"), rerun=False, subjects=()) -> list[dict]:
    """Run selected corrected and one-state reconstruction pairs."""
    modes = tuple(modes)
    if not modes or any(mode not in {"corrected", "nomoco"} for mode in modes):
        raise ValueError("modes must contain corrected, nomoco, or both.")
    ismrmrd_dir, saec_dir, dataset_dir = map(Path, (ismrmrd_dir, saec_dir, dataset_dir))
    config_path = Path(reconstruction_config).resolve()
    params = load_config(data_type="preprocessed-real", reconstruction_config=config_path)
    if params.reconstruction_dimension != "2D" or params.N_motion_states <= 1:
        raise ValueError("The corrected configuration must be 2D with more than one motion state.")
    requested = set()
    for subject in subjects:
        if SUBJECT_PATTERN.fullmatch(subject):
            requested.add(subject)
        elif re.fullmatch(r"00\d{2}", subject):
            requested.add(f"{subject}_T2_s")
        else:
            raise ValueError(f"Invalid subject selector: {subject!r}.")
    raw_files, missing = find_subjects(ismrmrd_dir, saec_dir, requested)
    results = dataset_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    rows = read_records(results / "measurements.json", resume)
    selected_names = {f"{raw.stem}.h5" for raw in raw_files}
    rows = [row for row in rows if row.get("sharpness_image_stage") == "native_solver_image"]
    if rerun:
        rows = [row for row in rows
                if row["subject"] not in selected_names or row["mode"] not in modes]
    completed = {(row["subject"], row["mode"]) for row in rows}
    if len(completed) != len(rows):
        raise ValueError("Duplicate subject/mode measurements in checkpoint.")
    exclusions = {
        "constant": read_records(results / "excluded_constant_physiological_signal.json", resume),
        "respiratory": read_records(results / "excluded_missing_respiratory_data.json", resume),
        "truncated": read_records(results / "excluded_truncated_files.json", resume),
    }
    (results / "excluded_missing_saec.txt").write_text("".join(f"{path.name}\n" for path in missing))
    skipped = {row["subject"] for records in exclusions.values() for row in records}
    for raw in raw_files:
        subject_name = f"{raw.stem}.h5"
        if raw.name in skipped or subject_name in skipped:
            print(f"[resume] {raw.name}: previously excluded", flush=True)
            continue
        prepared = dataset_dir / subject_name
        if prepared.exists():
            validate_prepared_acquisition(prepared)
        for mode in modes:
            if (subject_name, mode) in completed:
                print(f"[resume] {subject_name}: {mode} complete", flush=True)
                continue
            print(f"[reconstruct] {subject_name}: {mode}", flush=True)
            overrides = {} if mode == "corrected" else {"N_motion_states": 1}
            if prepared.exists():
                input_file, physiological_file, prepared_output = prepared, None, None
            else:
                input_file, physiological_file, prepared_output = raw, saec_dir / raw.name, prepared
            try:
                result = run_pipeline(
                    input_file, physiological_file,
                    output_root=results / "runs" / raw.stem / mode,
                    reconstruction_config=config_path, device=device, max_workers=max_workers,
                    overrides=overrides, save_reconstruction_logs=True,
                    save_reconstruction_tensors=True, return_tensors=True,
                    prepared_output_file=prepared_output, export_dicom=False)
            except ConstantPhysiologicalSignalError as error:
                print(f"[skip] {subject_name}: {error}", flush=True)
                exclusions["constant"].append({"subject": subject_name, "mode": mode, "reason": str(error)})
                rows = [row for row in rows if row["subject"] != subject_name]
                break
            except ValueError as error:
                if not str(error).startswith("No respiratory data found in SAEC file "):
                    raise
                print(f"[skip] {subject_name}: {error}", flush=True)
                exclusions["respiratory"].append({"subject": raw.name, "reason": str(error)})
                break
            except OSError as error:
                if "truncated file:" not in str(error).lower():
                    raise
                print(f"[skip] {subject_name}: {error}", flush=True)
                exclusions["truncated"].append({"subject": raw.name, "reason": str(error)})
                break
            validate_prepared_acquisition(prepared)
            slice_scores = [score_image(item["native_image"])
                            for item in result["reconstructions"]]
            slice_times = [item["reconstruction_seconds"]
                           for item in result["reconstructions"]]
            resolved = json.loads((result["run_folder"] / "config_resolved.json").read_text())
            rows.append({
                "subject": subject_name,
                "mode": mode,
                "motion_states": resolved["N_motion_states"],
                "slice_count": len(slice_scores),
                "mean_sharpness": float(np.mean(slice_scores)),
                "slice_sharpness": slice_scores,
                "sharpness_image_stage": "native_solver_image",
                "reconstruction_seconds_sum": result["timings"]["reconstruction_seconds_sum"],
                "reconstruction_seconds_max": max(slice_times),
                "slice_reconstruction_seconds": slice_times,
                "compute_wall_seconds": result["timings"]["compute_wall_seconds"],
                "run_folder": str(result["run_folder"]),
                "timings": result["timings"],
            })
            completed.add((subject_name, mode))
            save_results(rows, results)
            del result
    for name, records in exclusions.items():
        filename = {"constant": "excluded_constant_physiological_signal.json",
                    "respiratory": "excluded_missing_respiratory_data.json",
                    "truncated": "excluded_truncated_files.json"}[name]
        (results / filename).write_text(json.dumps(records, indent=2) + "\n")
    save_results(rows, results)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ismrmrd-dir", type=Path, default=DATABASE / "ISMRMRD")
    parser.add_argument("--saec-dir", type=Path, default=DATABASE / "SAEC")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--reconstruction-config", type=Path, default=ROOT / RECONSTRUCTION_CONFIG)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Fixed worker count for comparable 2D timing.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--modes", nargs="+", choices=("corrected", "nomoco"),
                        default=("corrected", "nomoco"))
    parser.add_argument("--rerun", action="store_true",
                        help="Reconstruct selected subject/mode pairs even when checkpointed.")
    parser.add_argument("--subject", action="append", default=[],
                        help="Optional ID such as 0079 or 0079_T2_s; repeat to select several.")
    args = parser.parse_args(argv)
    subjects = args.subject
    del args.subject
    evaluate(**vars(args), subjects=subjects)


if __name__ == "__main__":
    main()

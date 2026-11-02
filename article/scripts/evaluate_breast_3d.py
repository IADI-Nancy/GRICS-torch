#!/usr/bin/env python
"""Evaluate breast 3D GRICS-torch reconstruction time and sharpness."""
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
from pipelines.siemens_breast_3d_lowres import RECONSTRUCTION_CONFIG, run_pipeline
from src.runtime.runtime_config import load_config

DATABASE = Path("/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database")
DATASET = Path("/home/pyuser/wkdir/data/GRICS-torch/article_dataset_3D")
SUBJECT_PATTERN = re.compile(r"^\d{4}_T1_[sm]$")


def validate_prepared_volume(prepared: Path) -> None:
    """Reject incomplete prepared inputs before starting a long reconstruction."""
    with h5py.File(prepared, "r") as source:
        required = ("kspace", "motion_data", "idx_ky", "idx_kz", "idx_nex")
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"{prepared}: missing prepared datasets {missing}.")
        shape = tuple(source["kspace"].shape)
    if len(shape) != 5 or shape[-1] <= 1:
        raise ValueError(f"{prepared}: expected five-dimensional 3D k-space, got {shape}.")


def score_volume(image: torch.Tensor) -> list[float]:
    """Score the magnitude of every native axial partition."""
    image = torch.as_tensor(image).detach().cpu()
    if image.ndim == 4:
        image = image.mean(dim=0)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image or repetition-stacked 3D image, got {tuple(image.shape)}.")
    magnitude = image.abs()
    scores = [float(sharpness_index(magnitude[..., partition]))
              for partition in range(magnitude.shape[-1])]
    if not np.isfinite(scores).all():
        raise ValueError("Sharpness calculation produced a non-finite value.")
    return scores


def save_results(rows: list[dict], folder: Path) -> None:
    """Checkpoint measurements and regenerate subject-level comparison plots."""
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "measurements.json.tmp"
    temporary.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    temporary.replace(folder / "measurements.json")
    fields = ["subject", "mode", "motion_states", "mean_sharpness",
              "reconstruction_seconds", "run_folder"]
    with (folder / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    paired = [name for name in dict.fromkeys(row["subject"] for row in rows)
              if {row["mode"] for row in rows if row["subject"] == name}
              == {"corrected", "nomoco"}]
    if not paired:
        return
    for metric, ylabel, filename in (
        ("mean_sharpness", "Mean axial-partition sharpness index", "sharpness_boxplot.png"),
        ("reconstruction_seconds", "Reconstruction time including logs (s)", "time_boxplot.png"),
    ):
        values = [[next(row[metric] for row in rows
                        if row["subject"] == name and row["mode"] == mode)
                   for name in paired] for mode in ("corrected", "nomoco")]
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.boxplot(values)
        axis.set_xticks([1, 2], ["Motion corrected", "One state"])
        for corrected, nomoco in zip(*values):
            axis.plot([1, 2], [corrected, nomoco], "o-", color="gray", alpha=.4, markersize=3)
        axis.set_ylabel(ylabel)
        axis.set_title(f"Breast 3D: {len(paired)} paired acquisitions")
        figure.tight_layout()
        figure.savefig(folder / filename, dpi=200)
        plt.close(figure)


def find_subjects(ismrmrd_dir: Path, saec_dir: Path, requested: set[str]) -> list[tuple[Path, Path]]:
    raw_files = sorted(path for path in ismrmrd_dir.iterdir()
                       if path.is_file() and SUBJECT_PATTERN.fullmatch(path.stem)
                       and path.suffix.lower() in {".h5", ".mrd"})
    if requested:
        raw_files = [path for path in raw_files if path.stem in requested]
        found = {path.stem for path in raw_files}
        if found != requested:
            raise FileNotFoundError(f"Requested subjects are unavailable: {sorted(requested - found)}")
    if not raw_files:
        raise ValueError(f"No matching T1 acquisitions found in {ismrmrd_dir}.")
    pairs = [(raw, saec_dir / raw.name) for raw in raw_files]
    missing = [str(saec) for _, saec in pairs if not saec.is_file()]
    if missing:
        raise FileNotFoundError("Missing matching SAEC files:\n" + "\n".join(missing))
    return pairs


def evaluate(ismrmrd_dir=DATABASE / "ISMRMRD", saec_dir=DATABASE / "SAEC",
             dataset_dir=DATASET, reconstruction_config=ROOT / RECONSTRUCTION_CONFIG,
             device="gpu", subjects=()) -> list[dict]:
    """Run corrected and one-state reconstruction for every selected acquisition."""
    ismrmrd_dir, saec_dir, dataset_dir = map(Path, (ismrmrd_dir, saec_dir, dataset_dir))
    config_path = Path(reconstruction_config).resolve()
    params = load_config(data_type="preprocessed-real", reconstruction_config=config_path)
    if params.reconstruction_dimension != "3D" or params.N_motion_states <= 1:
        raise ValueError("The corrected configuration must be 3D with more than one motion state.")
    requested = {subject if SUBJECT_PATTERN.fullmatch(subject) else f"{subject}_T1_s"
                 for subject in subjects}
    pairs = find_subjects(ismrmrd_dir, saec_dir, requested)
    rows = []
    results_folder = dataset_dir / "results"
    for raw, saec in pairs:
        prepared = dataset_dir / f"{raw.stem}.h5"
        if prepared.exists():
            validate_prepared_volume(prepared)
        for mode in ("corrected", "nomoco"):
            print(f"[reconstruct] {raw.stem}: {mode}", flush=True)
            overrides = {} if mode == "corrected" else {"N_motion_states": 1}
            if prepared.exists():
                input_file, physiological_file, prepared_output = prepared, None, None
            else:
                input_file, physiological_file, prepared_output = raw, saec, prepared
            result = run_pipeline(
                input_file, physiological_file,
                output_root=results_folder / "runs" / raw.stem / mode,
                reconstruction_config=config_path, device=device, overrides=overrides,
                save_reconstruction_logs=True, save_reconstruction_tensors=True,
                return_tensors=True, prepared_output_file=prepared_output, export_dicom=False)
            validate_prepared_volume(prepared)
            image = result["reconstructions"][0]["image"]
            scores = score_volume(image)
            resolved = json.loads((result["run_folder"] / "config_resolved.json").read_text())
            rows.append({
                "subject": prepared.name,
                "mode": mode,
                "motion_states": resolved["N_motion_states"],
                "mean_sharpness": float(np.mean(scores)),
                "slice_sharpness": scores,
                "reconstruction_seconds": result["timings"]["reconstruction_seconds"],
                "run_folder": str(result["run_folder"]),
                "timings": result["timings"],
            })
            save_results(rows, results_folder)
            del result
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ismrmrd-dir", type=Path, default=DATABASE / "ISMRMRD")
    parser.add_argument("--saec-dir", type=Path, default=DATABASE / "SAEC")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--reconstruction-config", type=Path, default=ROOT / RECONSTRUCTION_CONFIG)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--subject", action="append", default=[],
                        help="Optional subject ID such as 0079 or 0079_T1_s; repeat to select several.")
    args = parser.parse_args(argv)
    subjects = args.subject
    del args.subject
    evaluate(**vars(args), subjects=subjects)


if __name__ == "__main__":
    main()

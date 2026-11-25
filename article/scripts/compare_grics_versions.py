#!/usr/bin/env python
"""Compare saved GRICS++, GRICS-torch 2.0 and current sharpness and calculation times.

Run from the repository root:
    python article/scripts/compare_grics_versions.py --v3-2d /path/to/current/2d/measurements.json --v3-3d /path/to/current/3d/measurements.json

The --v2-2d and --v2-3d options select historical measurements; --v3-2d
and --v3-3d select current measurements. Defaults retain the original article
run locations. Use --plusplus-2d and --plusplus-3d to select GRICS++ data,
--dimensions to select 2d or 3d, and --output-dir to select the destination.
The output includes comparison.svg, comparison.png, and JSON/CSV measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from article.scripts.compare_grics_reconstructions import (
    DATABASE, DATASET, read_measurements, read_plusplus_image, reconstruction_seconds, slice_folders,
)
from article.scripts.sharpness_index import sharpness_index

IMPLEMENTATIONS = ("plusplus", "v2", "v3")
LABELS = ("GRICS++", "GRICS-torch:2.0", "GIRCS-torch:current")
COLORS = ("#5d85ad", "#db9857", "#6a9f78")


def validate_torch_acquisition(row, shapes, dimension):
    """Require complete native-grid sharpness measurements covering the same slices."""
    run_folder = Path(row["run_folder"])
    manifest = json.loads((run_folder / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"{run_folder}: incomplete reconstruction.")
    items = list(manifest["reconstructions"].values())
    if dimension == "2d":
        by_slice = {int(item["slice_number"]): item for item in items}
        if len(by_slice) != len(items) or set(by_slice) != set(shapes):
            raise ValueError(f"{run_folder}: GRICS++ and torch slice sets differ.")
        if row["sharpness_image_stage"] != "native_solver_image":
            raise ValueError(f"{run_folder}: sharpness must use native solver images.")
        for number, item in by_slice.items():
            if tuple(item["native_image_shape"]) != shapes[number][1:]:
                raise ValueError(f"{run_folder}: native grid differs for slice {number}.")
        count = len(items)
    else:
        if len(items) != 1 or tuple(items[0]["native_image_shape"]) != (shapes[0][1], shapes[0][2], shapes[0][0]):
            raise ValueError(f"{run_folder}: native volume grids differ.")
        count = shapes[0][0]
    scores = np.asarray(row["slice_sharpness"], dtype=float)
    if len(scores) != count or not np.isfinite(scores).all():
        raise ValueError(f"{run_folder}: invalid native sharpness measurements.")
    if not np.isclose(float(row["mean_sharpness"]), scores.mean(), rtol=1e-10, atol=1e-12):
        raise ValueError(f"{run_folder}: mean sharpness does not match per-slice scores.")
    return float(scores.mean()), count


def compare_dimension(torch_v2, torch_v3, corrected, timing, dimension, time_mode):
    records = {"v2": read_measurements(torch_v2), "v3": read_measurements(torch_v3)}
    names = {key: {subject for subject, mode in rows if mode == "corrected"} for key, rows in records.items()}
    names["plusplus"] = {path.name for path in corrected.iterdir() if path.is_dir()}
    subjects = sorted(set.intersection(*names.values()))
    if not subjects:
        raise ValueError(f"{dimension}: no acquisitions shared by all three implementations.")
    excluded = {key: sorted(values - set(subjects)) for key, values in names.items()}
    print(f"[{dimension}] {len(subjects)} shared acquisitions; excluded: {excluded}", flush=True)
    rows = []
    for subject in subjects:
        folder = corrected / subject
        folders = slice_folders(folder) if dimension == "2d" else {0: folder}
        shapes, scores = {}, []
        for number, path in folders.items():
            image = read_plusplus_image(path)
            shapes[number] = image.shape
            scores.extend(float(sharpness_index(torch.from_numpy(plane.copy()))) for plane in image)
        if not np.isfinite(scores).all():
            raise ValueError(f"{folder}: non-finite GRICS++ sharpness.")
        row = {"subject": subject, "sharpness_plusplus": float(np.mean(scores))}
        for key, by_key in records.items():
            saved = by_key[subject, "corrected"]
            if saved["motion_states"] <= 1:
                raise ValueError(f"{subject}: corrected measurement must have multiple motion states.")
            sharpness, count = validate_torch_acquisition(saved, shapes, dimension)
            first, second = reconstruction_seconds(by_key[subject, time_mode], timing / subject, dimension, count)
            row.update({f"sharpness_{key}": sharpness, f"seconds_{key}": second, f"run_folder_{key}": saved["run_folder"]})
            row["seconds_plusplus"] = first
        rows.append(row)
        print(f"[{dimension}] {subject}: times = {[round(row[f'seconds_{key}'], 2) for key in IMPLEMENTATIONS]} s", flush=True)
    return {"excluded": excluded, "acquisitions": rows}


def plot_comparison(results, output_dir):
    figure, axes = plt.subplots(2, len(results), figsize=(6 * len(results), 9.6), squeeze=False)
    for index, (dimension, result) in enumerate(results.items()):
        rows = result["acquisitions"]
        for row_index, (metric, ylabel) in enumerate((("sharpness", "Mean native slice sharpness index"), ("seconds", "Calculation time (s)"))):
            axis = axes[row_index, index]
            values = [[row[f"{metric}_{key}"] for row in rows] for key in IMPLEMENTATIONS]
            boxes = axis.boxplot(values, patch_artist=True, widths=.5, showfliers=False)
            for box, color in zip(boxes["boxes"], COLORS):
                box.set_facecolor(color)
                box.set_alpha(.65)
            for acquisition in zip(*values):
                axis.plot([1, 2, 3], acquisition, "o-", color="0.45", alpha=.35, markersize=3, linewidth=.7)
            axis.set_xticks([1, 2, 3], LABELS)
            axis.set_ylabel(ylabel)
            scope = "acquisition compute time" if dimension == "2d" else "volume solver time"
            title = "sharpness" if metric == "sharpness" else scope
            axis.set_title(f"{dimension.upper()} {title} ({len(rows)} acquisitions)")
            axis.grid(axis="y", alpha=.2)
    figure.tight_layout()
    for suffix in ("svg", "png"):
        figure.savefig(output_dir / f"comparison.{suffix}", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", choices=("2d", "3d"), default=("2d", "3d"))
    for dimension, suffix in (("2d", ""), ("3d", "-3D")):
        parser.add_argument(f"--v2-{dimension}", type=Path, default=DATASET / f"article_dataset_{dimension.upper()}/results/measurements.json", help="Saved GRICS 2.0 measurements; defaults to the historical article dataset.")
        default_v3 = "breast_2d_gpu_sequential" if dimension == "2d" else "breast_3d_article_reader"
        parser.add_argument(f"--v3-{dimension}", type=Path, default=ROOT / f"runs/article/{default_v3}/measurements.json")
        parser.add_argument(f"--plusplus-{dimension}", type=Path, default=DATABASE / f"GRICS-BELT{suffix}")
        parser.add_argument(f"--plusplus-nomoco-{dimension}", type=Path, default=DATABASE / f"GRICS-BELT{suffix}_nomoco")
    parser.add_argument("--time-mode", choices=("corrected", "nomoco"), default="corrected", help="Mode used for timing; sharpness always uses corrected images.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/article/comparison_three_versions")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 11})
    print("[timing] 2D: torch compute_wall_seconds; GRICS++ batch estimate from preserved log timestamps. 3D: solver seconds. Saved runs retain their original hardware and execution settings.", flush=True)
    results = {}
    inputs = {}
    for dimension in dict.fromkeys(args.dimensions):
        v2, v3, corrected = (getattr(args, f"{prefix}_{dimension}") for prefix in ("v2", "v3", "plusplus"))
        timing = corrected if args.time_mode == "corrected" else getattr(args, f"plusplus_nomoco_{dimension}")
        inputs[dimension] = {"v2": str(v2.resolve()), "v3": str(v3.resolve()), "plusplus": str(corrected.resolve()), "plusplus_timing": str(timing.resolve())}
        results[dimension] = compare_dimension(v2, v3, corrected, timing, dimension, args.time_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, args.output_dir)
    report = {"column_order": LABELS, "image_mode": "corrected", "time_mode": args.time_mode,
              "timing_2d": "torch acquisition compute wall time; GRICS++ estimated batch elapsed time from preserved log mtimes",
              "timing_3d": "volume solver time", "inputs": inputs, "dimensions": results}
    (args.output_dir / "comparison_measurements.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = [dict(dimension=dimension, **row) for dimension, result in results.items() for row in result["acquisitions"]]
    with (args.output_dir / "comparison_measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved comparison.svg, comparison.png and JSON/CSV measurements to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Compare saved GRICS++ and GRICS-torch reconstructions in SVGs and subject PNGs.

Run from the repository root:
    python article/scripts/compare_grics_reconstructions.py --output-dir article/figures/comparison

Sharpness uses native, motion-corrected magnitude images, with one observation
per acquisition (mean over all slices/partitions). Repetitions are complex-averaged.
Similarity uses the same corrected images and least-squares intensity scaling
of torch to GRICS++, matching the subject comparisons: mean slice SSIM/NRMSE
in 2D, whole-volume SSIM/NRMSE in 3D. Insets show the median and interquartile
range across acquisitions. No registration, resampling, masking, or automatic
orientation search is performed.

Timing uses corrected runs by default. In 2D, torch compute_wall_seconds spans
slice worker startup, preprocessing, reconstruction, and result collection.
GRICS++ batch elapsed time is ESTIMATED from preserved grics.log.0 modification
times: max(completion) - min(completion - Total elapsed time). This requires
original file timestamps, preserved by the acquisition runner's copy2/copytree;
it is not a directly recorded batch timer. In 3D, solver durations are used.
All timing values are seconds. Hardware and settings belong to the original runs.

Subject PNGs use the corrected acquisition selected by --subject/--sequence
(default 0079/s, matching tests/run_real_2d_subject.py and run_real_3d_subject.py).
They use the saved article dataset runs, not a new reconstruction. The 2D slice
is at one quarter of the stack, displayed in torch orientation with a horizontal
flip for the requested left-right view; the 3D
planes and orientations match the test scripts. Columns are uncorrected torch,
corrected GRICS++, and corrected torch, without gaps or metric overlays. PNG similarity metrics use least-squares intensity scaling, as in the
tests: slice-wise means and standard deviations in 2D, whole-volume in 3D.

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import torch
from skimage.metrics import structural_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from article.scripts.sharpness_index import sharpness_index

DATABASE = Path("/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database")
DATASET = Path("/home/pyuser/wkdir/data/GRICS-torch")
SLICE_PATTERN = re.compile(r"Siemens_SingleImage_slice(\d+)_image01")


def read_measurements(path: Path) -> dict:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of measurements.")
    by_key = {}
    for row in rows:
        key = (Path(row["subject"]).stem, row["mode"])
        if key in by_key:
            raise ValueError(f"{path}: duplicate acquisition/mode {key}.")
        by_key[key] = row
    return by_key


def read_log(folder: Path) -> str:
    path = folder / "grics.log.0"
    text = path.read_text()
    if "RECONSTRUCTION COMPLETE" not in text:
        raise ValueError(f"{path}: reconstruction did not complete.")
    return text


def read_plusplus_image(folder: Path) -> np.ndarray:
    text = read_log(folder)
    sizes = []
    for name in ("Nx", "Ny", "Nz"):
        matches = re.findall(rf"^\s*{name}\s*=\s*(\d+)\s*$", text, re.M)
        if not matches or len(set(matches)) != 1 or int(matches[0]) < 1:
            raise ValueError(f"{folder}: missing or inconsistent {name}.")
        sizes.append(int(matches[0]))
    nx, ny, nz = sizes
    image = np.fromfile(folder / "GricsRecon.dat.0000", dtype="<c8")
    if image.size != nx * ny * nz:
        raise ValueError(f"{folder}: binary image size does not match {(nx, ny, nz)}.")
    # These breast exports store planes as (Nx, Ny), with Nz outermost.
    return np.abs(image.reshape(nz, nx, ny)).astype(np.float64)


def read_torch_image(item: dict, run_folder: Path, dimension: str) -> np.ndarray:
    path = Path(item["native_image_file"])
    if not path.is_absolute():
        path = run_folder / path
    # Pipeline exports can contain a NumPy array or a torch tensor. Keep the
    # restricted loader, explicitly allowing only the numeric NumPy types used.
    with torch.serialization.safe_globals([
        np._core.multiarray._reconstruct, np.ndarray, np.dtype,
        np.dtypes.Complex64DType, np.dtypes.Complex128DType,
        np.dtypes.Float32DType, np.dtypes.Float64DType,
    ]):
        image = torch.as_tensor(torch.load(path, map_location="cpu", weights_only=True))
    ndim = 2 if dimension == "2d" else 3
    if image.ndim == ndim + 1:
        image = image.mean(dim=0)
    if image.ndim != ndim:
        raise ValueError(f"{path}: expected {ndim} spatial axes, got {tuple(image.shape)}.")
    magnitude = image.abs().double().numpy()
    return magnitude[None] if ndim == 2 else np.moveaxis(magnitude, -1, 0)


def slice_folders(folder: Path) -> dict[int, Path]:
    slices = {}
    for path in folder.iterdir():
        match = SLICE_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            index = int(match[1])
            if index in slices:
                raise ValueError(f"{folder}: duplicate slice {index}.")
            slices[index] = path
    if not slices or sorted(slices) != list(range(1, len(slices) + 1)):
        raise ValueError(f"{folder}: missing slices or non-contiguous slice numbering.")
    return dict(sorted(slices.items()))


def image_metrics(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float, float]:
    if first.shape != second.shape or first.ndim != 2 or min(first.shape) < 7:
        raise ValueError(f"Incompatible slice grids: {first.shape} versus {second.shape} (minimum 7 pixels).")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Images contain non-finite values.")
    scores = [float(sharpness_index(torch.from_numpy(image.copy()))) for image in (first, second)]
    if not np.isfinite(scores).all():
        raise ValueError("Sharpness produced non-finite values.")
    if first.max() <= 0 or second.max() <= 0:
        raise ValueError("Cannot normalize an empty magnitude image for similarity metrics.")
    metrics = scaled_similarity(first, second)
    return *scores, metrics["ssim"], metrics["nrmse"]


def acquisition_metrics(row: dict, folder: Path, dimension: str) -> dict:
    run_folder = Path(row["run_folder"])
    manifest = json.loads((run_folder / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"{run_folder}: incomplete torch run.")
    items = list(manifest["reconstructions"].values())
    if dimension == "2d":
        folders = slice_folders(folder)
        by_slice = {int(item["slice_number"]): item for item in items}
        if len(by_slice) != len(items) or set(by_slice) != set(folders) or len(items) != row["slice_count"]:
            raise ValueError(f"{folder.name}: GRICS++ and torch slice sets differ.")
        pairs = [(path, by_slice[index]) for index, path in folders.items()]
    else:
        if len(items) != 1:
            raise ValueError(f"{run_folder}: expected exactly one volume.")
        pairs = [(folder, items[0])]
    values = []
    for path, item in pairs:
        first = read_plusplus_image(path)
        second = read_torch_image(item, run_folder, dimension)
        if first.shape != second.shape:
            raise ValueError(f"{path}: native grids differ: {first.shape} versus {second.shape}.")
        values.extend(image_metrics(a, b) for a, b in zip(first, second))
    means = np.mean(values, axis=0)
    result = dict(zip(("sharpness_plusplus", "sharpness_torch", "ssim", "nrmse"), map(float, means)),
                  slice_count=len(values))
    if dimension == "3d":
        metrics = scaled_similarity(np.moveaxis(first, 0, -1), np.moveaxis(second, 0, -1))
        result.update(ssim=metrics["ssim"], nrmse=metrics["nrmse"])
    return result


def reconstruction_seconds(row: dict, folder: Path, dimension: str, slice_count: int) -> tuple[float, float]:
    if dimension == "2d":
        folders = list(slice_folders(folder).values())
        saved = row["slice_reconstruction_seconds"]
        if len(folders) != slice_count or len(saved) != slice_count or row["slice_count"] != slice_count:
            raise ValueError(f"{folder.name}: timing does not cover the full image slice set.")
    else:
        folders, saved = [folder], [row["reconstruction_seconds"]]
    times = []
    for path in folders:
        matches = re.findall(r"^Reconstruction time\s*=\s*([\d.eE+-]+)\s*$", read_log(path), re.M)
        if len(matches) != 1:
            raise ValueError(f"{path}: expected one Reconstruction time entry.")
        times.append(float(matches[0]))
    if any(not np.isfinite(value) or value <= 0 for value in [*times, *saved]):
        raise ValueError(f"{folder.name}: invalid reconstruction times.")
    if dimension == "2d":
        completed, started = [], []
        for path in folders:
            matches = re.findall(r"^Total elapsed time\s*=\s*([\d.eE+-]+)\s*$", read_log(path), re.M)
            if len(matches) != 1 or not np.isfinite(float(matches[0])) or float(matches[0]) <= 0:
                raise ValueError(f"{path}: invalid Total elapsed time.")
            end = (path / "grics.log.0").stat().st_mtime
            completed.append(end)
            started.append(end - float(matches[0]))
        first, second = max(completed) - min(started), float(row["compute_wall_seconds"])
        if not np.isfinite(second) or second <= 0:
            raise ValueError(f"{folder.name}: invalid compute_wall_seconds.")
        return first, second
    return times[0], float(saved[0])


def compare_dimension(measurements: Path, corrected: Path, timing: Path, dimension: str, time_mode: str) -> dict:
    by_key = read_measurements(measurements)
    torch_names = {subject for subject, mode in by_key if mode == "corrected"}
    pp_names = {path.name for path in corrected.iterdir() if path.is_dir()}
    subjects = sorted(torch_names & pp_names)
    excluded = {"only_torch": sorted(torch_names - pp_names), "only_plusplus": sorted(pp_names - torch_names)}
    print(f"[{dimension}] Unmatched acquisitions: {excluded}", flush=True)
    if not subjects:
        raise ValueError(f"{dimension}: no matching corrected acquisitions.")
    rows = []
    for subject in subjects:
        print(f"[{dimension}] {subject}", flush=True)
        if (subject, time_mode) not in by_key:
            raise ValueError(f"{subject}: missing torch {time_mode} timing measurement.")
        if by_key[subject, "corrected"]["motion_states"] <= 1:
            raise ValueError(f"{subject}: corrected torch measurement must use multiple motion states.")
        row = acquisition_metrics(by_key[subject, "corrected"], corrected / subject, dimension)
        first, second = reconstruction_seconds(by_key[subject, time_mode], timing / subject, dimension, row["slice_count"])
        rows.append(dict(subject=subject, **row, seconds_plusplus=first, seconds_torch=second))
    return {"image_mode": "corrected", "similarity_scope": "mean_slice" if dimension == "2d" else "whole_volume",
            "excluded": excluded, "acquisitions": rows}


def save_boxplot(rows: list[dict], metric: str, ylabel: str, path: Path) -> None:
    values = [[row[f"{metric}_{implementation}"] for row in rows] for implementation in ("plusplus", "torch")]
    figure, axis = plt.subplots(figsize=(5.5, 5))
    boxes = axis.boxplot(values, patch_artist=True, widths=.5, showfliers=False)
    for box, color in zip(boxes["boxes"], ("#5d85ad", "#db9857")):
        box.set_facecolor(color)
        box.set_alpha(.65)
    for first, second in zip(*values):
        axis.plot([1, 2], [first, second], "o-", color="0.45", alpha=.4, markersize=3, linewidth=.7)
    axis.set_xticks([1, 2], ["GRICS++", "GRICS-torch"])
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(True)
    axis.grid(axis="y", alpha=.2)
    if metric == "sharpness":
        lines = [f"n = {len(rows)} acquisitions", "Median [Q1, Q3]"]
        for key, label in (("ssim", "SSIM"), ("nrmse", "NRMSE")):
            q1, median, q3 = np.quantile([row[key] for row in rows], [.25, .5, .75])
            lines.append(f"{label}: {median:.3f} [{q1:.3f}, {q3:.3f}]")
        # Reserve space above the data for the similarity inset, without a title.
        lower, upper = axis.get_ylim()
        axis.set_ylim(lower, upper + .42 * (upper - lower))
        # Size the frame independently of the text so its extra space is visible.
        frame = FancyBboxPatch((.43, .805), .55, .175, transform=axis.transAxes,
                               boxstyle="round,pad=0.005,rounding_size=0.025", facecolor="white", edgecolor="0.8", zorder=4)
        axis.add_patch(frame)
        axis.text(.89, .95, "\n".join(lines), transform=axis.transAxes, ha="right", va="top", multialignment="left",
                  fontsize=9, zorder=5)
    figure.tight_layout()
    figure.savefig(path, format="svg")
    plt.close(figure)


def scaled_similarity(first: np.ndarray, second: np.ndarray) -> dict:
    """Match the real-subject tests' least-squares scaling of torch to GRICS++."""
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Subject images must have matching finite native grids.")
    denominator, reference = np.sum(second ** 2), np.sum(first ** 2)
    if denominator <= 0 or reference <= 0:
        raise ValueError("Subject images have zero magnitude.")
    scale = float(np.sum(first * second) / denominator)
    aligned = second * scale
    data_range = float(max(first.max(), aligned.max()) - min(first.min(), aligned.min()))
    if data_range <= 0:
        raise ValueError("Subject images have zero comparison range.")
    return {"scale_torch_to_grics_cpp": scale,
            "nrmse": float(np.sqrt(np.sum((aligned - first) ** 2) / reference)),
            "ssim": float(structural_similarity(first, aligned, data_range=data_range))}


def save_subject_comparison(measurements: Path, cpp_root: Path, dimension: str, subject: str,
                            sequence: str, output_dir: Path) -> dict:
    """Save uncorrected torch, corrected GRICS++, and corrected torch without gaps."""
    if not re.fullmatch(r"\d{4}", subject):
        raise ValueError("Subject must be a four-digit identifier.")
    acquisition = f"{subject}_{'T2' if dimension == '2d' else 'T1'}_{sequence}"
    by_key = read_measurements(measurements)
    row = by_key[acquisition, "corrected"]
    nomoco_row = by_key[acquisition, "nomoco"]
    if nomoco_row["motion_states"] != 1:
        raise ValueError(f"{acquisition}: uncorrected image must use one motion state.")
    nomoco_folder = Path(nomoco_row["run_folder"])
    nomoco_manifest = json.loads((nomoco_folder / "manifest.json").read_text())
    if nomoco_manifest.get("status") != "complete":
        raise ValueError(f"{nomoco_folder}: incomplete uncorrected reconstruction.")
    nomoco_items = list(nomoco_manifest["reconstructions"].values())
    run_folder = Path(row["run_folder"])
    manifest = json.loads((run_folder / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"{run_folder}: incomplete subject reconstruction.")
    items = list(manifest["reconstructions"].values())
    folder = cpp_root / acquisition
    if dimension == "2d":
        folders = slice_folders(folder)
        by_slice = {int(item["slice_number"]): item for item in items}
        if len(by_slice) != len(items) or set(by_slice) != set(folders):
            raise ValueError(f"{acquisition}: subject slice sets differ.")
        nomoco_by_slice = {int(item["slice_number"]): item for item in nomoco_items}
        if len(nomoco_by_slice) != len(nomoco_items) or set(nomoco_by_slice) != set(folders):
            raise ValueError(f"{acquisition}: uncorrected slice set differs.")
        central_slice = sorted(folders)[len(folders) // 4]
        metrics = []
        for number, path in folders.items():
            first = read_plusplus_image(path)[0]
            second = read_torch_image(by_slice[number], run_folder, dimension)[0]
            metric = scaled_similarity(first, second)
            metrics.append(dict(slice_number=number, **metric))
            if number == central_slice:
                nomoco = read_torch_image(nomoco_by_slice[number], nomoco_folder, dimension)[0]
                nomoco_metric = scaled_similarity(first, nomoco)
                displayed = [nomoco * nomoco_metric["scale_torch_to_grics_cpp"], first,
                             second * metric["scale_torch_to_grics_cpp"]]
                flip_for_display = by_slice[number]["configuration"]["flip_for_display"]
                if flip_for_display:
                    displayed = [np.flipud(image) for image in displayed]
                displayed = [np.fliplr(image) for image in displayed]
        summary = {f"{stat}_{key}": float(func([item[key] for item in metrics]))
                   for key in ("nrmse", "ssim") for stat, func in (("mean", np.mean), ("std", np.std))}
        planes = [displayed]
        vmax = max(float(np.quantile(image, .995)) for image in displayed)
        origin = "upper"
        summary.update(slice_number=central_slice, per_slice=metrics, flip_for_display=flip_for_display,
                       flip_horizontal=True, nomoco_metrics=nomoco_metric)
    else:
        if len(items) != 1 or len(nomoco_items) != 1:
            raise ValueError(f"{acquisition}: expected one corrected and one uncorrected volume.")
        first = np.moveaxis(read_plusplus_image(folder), 0, -1)
        second = np.moveaxis(read_torch_image(items[0], run_folder, dimension), 0, -1)
        summary = scaled_similarity(first, second)
        second = second * summary["scale_torch_to_grics_cpp"]
        nomoco = np.moveaxis(read_torch_image(nomoco_items[0], nomoco_folder, dimension), 0, -1)
        nomoco_metric = scaled_similarity(first, nomoco)
        nomoco = nomoco * nomoco_metric["scale_torch_to_grics_cpp"]
        summary["nomoco_metrics"] = nomoco_metric
        nx, ny, nz = first.shape
        if 3 * nx // 4 - 8 < 0:
            raise ValueError("Volume is too small for the test script's coronal plane.")
        planes = []
        for index in range(3):
            displayed = []
            for image in (nomoco, first, second):
                partition = (image[:, :, nz // 2], image[:, 3 * ny // 4, :], image[3 * nx // 4 - 8, :, :])[index]
                oriented = np.flipud(partition.T)
                displayed.append(np.rot90(oriented) if index == 0 else np.flipud(oriented))
            planes.append(displayed)
        vmax = max(float(np.quantile(image, .995)) for image in (nomoco, first, second))
        origin = "upper"
        summary["plane_indices_zero_based"] = {"axial_z": nz // 2, "sagittal_y": 3 * ny // 4, "coronal_x": 3 * nx // 4 - 8}
    # Each row has its own image aspect ratio. Matching the figure and GridSpec
    # geometry to these ratios removes gaps without stretching image pixels.
    ratios = [displayed[0].shape[0] / displayed[0].shape[1] for displayed in planes]
    figure, axes = plt.subplots(len(planes), 3, figsize=(15, 5 * sum(ratios)), squeeze=False,
                                gridspec_kw={"height_ratios": ratios, "wspace": 0, "hspace": 0})
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for index, displayed in enumerate(planes):
        for column, image in enumerate(displayed):
            axis = axes[index, column]
            axis.imshow(image, cmap="gray", origin=origin, vmin=0, vmax=vmax)
            axis.axis("off")
    path = output_dir / f"subject_{acquisition}_corrected_comparison_{dimension}.png"
    figure.savefig(path, dpi=180, pad_inches=0)
    plt.close(figure)
    print(f"[subject] {acquisition}: {run_folder} -> {path}", flush=True)
    return {"subject": acquisition, "run_folder": str(run_folder), "grics_cpp_folder": str(folder),
            "nomoco_run_folder": str(nomoco_folder),
            "column_order": ["Uncorrected (GRICS-torch)", "GRICS++", "GRICS-torch"],
            "comparison_image": str(path), **summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for dimension, suffix in (("2d", ""), ("3d", "-3D")):
        parser.add_argument(f"--torch-{dimension}", type=Path,
                            default=DATASET / f"article_dataset_{dimension.upper()}/results/measurements.json")
        parser.add_argument(f"--plusplus-{dimension}", type=Path, default=DATABASE / f"GRICS-BELT{suffix}")
        parser.add_argument(f"--plusplus-nomoco-{dimension}", type=Path, default=DATABASE / f"GRICS-BELT{suffix}_nomoco")
    parser.add_argument("--subject", default="0079", help="Subject selected by the real-subject test scripts.")
    parser.add_argument("--sequence", choices=("s", "m"), default="s")
    parser.add_argument("--time-mode", choices=("corrected", "nomoco"), default="corrected")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "article/figures/comparison")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 11})
    print(f"Timing mode: {args.time_mode}; 2D = batch elapsed time; 3D = volume solver time (seconds).", flush=True)
    print("[timing estimate] GRICS++ 2D uses preserved log mtimes minus per-slice total durations; "
          "torch uses recorded compute_wall_seconds, including worker/preprocessing overhead.", flush=True)
    results = {"image_mode": "corrected", "time_mode": args.time_mode, "method": __doc__}
    for dimension in ("2d", "3d"):
        corrected = getattr(args, f"plusplus_{dimension}")
        timing = corrected if args.time_mode == "corrected" else getattr(args, f"plusplus_nomoco_{dimension}")
        results[dimension] = compare_dimension(getattr(args, f"torch_{dimension}"), corrected, timing, dimension, args.time_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dimension in ("2d", "3d"):
        rows = results[dimension]["acquisitions"]
        save_boxplot(rows, "sharpness", "Mean slice sharpness index", args.output_dir / f"sharpness_corrected_{dimension}.svg")
        ylabel = "Reconstruction time"
        save_boxplot(rows, "seconds", ylabel, args.output_dir / f"reconstruction_time_{args.time_mode}_{dimension}.svg")
    results["subject_comparisons"] = {}
    for dimension in ("2d", "3d"):
        results["subject_comparisons"][dimension] = save_subject_comparison(
            getattr(args, f"torch_{dimension}"), getattr(args, f"plusplus_{dimension}"),
            dimension, args.subject, args.sequence, args.output_dir)
    (args.output_dir / "comparison_measurements.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"Saved four SVG boxplots, two subject PNGs, and comparison_measurements.json to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

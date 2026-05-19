import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FONT_SIZE = 20

DEFAULT_GRICS_PP_2D = Path("/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/GRICS-BELT")
DEFAULT_GRICS_TORCH_2D = Path("/home/pyuser/wkdir/data/GRICS-torch/article_dataset")
DEFAULT_GRICS_PP_3D = Path("/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/GRICS-BELT-3D")
DEFAULT_GRICS_TORCH_3D = Path("/home/pyuser/wkdir/data/GRICS-torch/article_dataset_3D")
DEFAULT_OUTPUT_DIR = Path("/home/pyuser/wkdir/data/GRICS-torch/article_metric_comparison_plots")


RECONSTRUCTION_TIME_METRIC = {
    "file": "reconstruction_times.txt",
    "ylabel": "Time (s)",
}

SHARPNESS_METRIC = {
    "uncorrected_file": "sharpness_uncorrected.txt",
    "corrected_file": "sharpness_corrected.txt",
    "ylabel": "Sharpness index",
}


def _parse_float(value):
    try:
        out = float(value)
    except ValueError:
        return math.nan
    return out


def read_subject_metric_file(file_path):
    """
    Return rows keyed by (subject_number, occurrence_index).

    The metric files store the numeric subject id as the first column. For 2D,
    the same subject id can appear more than once, so duplicate ids are paired
    by occurrence order.
    """
    rows = {}
    occurrence_counts = defaultdict(int)

    with open(file_path, "r") as f:
        for row_index, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                subject = parts[0]
                value = _parse_float(parts[1])
            else:
                subject = f"row_{row_index:04d}"
                value = _parse_float(parts[0])

            occurrence_counts[subject] += 1
            key = (subject, occurrence_counts[subject])
            rows[key] = value

    return rows


def paired_metric_values(grics_pp_file, grics_torch_file):
    grics_pp = read_subject_metric_file(grics_pp_file)
    grics_torch = read_subject_metric_file(grics_torch_file)
    common_keys = sorted(set(grics_pp) & set(grics_torch))

    subjects = []
    pp_values = []
    torch_values = []

    for subject, occurrence in common_keys:
        pp_value = grics_pp[(subject, occurrence)]
        torch_value = grics_torch[(subject, occurrence)]
        if not np.isfinite(pp_value) or not np.isfinite(torch_value):
            continue
        subjects.append(subject if occurrence == 1 else f"{subject}.{occurrence}")
        pp_values.append(pp_value)
        torch_values.append(torch_value)

    return subjects, np.asarray(pp_values, dtype=float), np.asarray(torch_values, dtype=float)


def paired_wilcoxon(first_values, second_values):
    if len(first_values) == 0:
        return math.nan, math.nan

    differences = second_values - first_values
    if np.allclose(differences, 0):
        return 0.0, 1.0

    result = wilcoxon(
        first_values,
        second_values,
        alternative="two-sided",
        zero_method="zsplit",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def plot_paired_boxplot(
    first_values,
    second_values,
    ylabel,
    output_dir,
    output_stem,
    xlabels,
    ylim=None,
):
    fig, ax = plt.subplots(figsize=(6.4, 6.2), dpi=200)
    positions = [1, 2]
    data = [first_values, second_values]

    ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"facecolor": "#d8e6f3", "edgecolor": "#4f5d6b", "linewidth": 1.1},
        whiskerprops={"color": "#4f5d6b", "linewidth": 1.0},
        capprops={"color": "#4f5d6b", "linewidth": 1.0},
    )

    rng = np.random.default_rng(42)
    for first_value, second_value in zip(first_values, second_values):
        jitter = rng.uniform(-0.045, 0.045)
        color = "#666666"
        ax.plot(
            [positions[0] + jitter, positions[1] + jitter],
            [first_value, second_value],
            color=color,
            linewidth=0.8,
            alpha=0.35,
            zorder=3,
        )
        ax.scatter(
            [positions[0] + jitter, positions[1] + jitter],
            [first_value, second_value],
            color=color,
            s=12,
            alpha=0.65,
            zorder=4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    png_path = output_dir / f"{output_stem}.png"
    svg_path = output_dir / f"{output_stem}.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, svg_path


def shared_limits(value_pairs):
    values = []
    for first_values, second_values in value_pairs:
        if len(first_values):
            values.append(first_values)
        if len(second_values):
            values.append(second_values)

    if not values:
        return None

    all_values = np.concatenate(values)
    lower = float(np.min(all_values))
    upper = float(np.max(all_values))
    if np.isclose(lower, upper):
        padding = 0.05 * abs(lower) if lower != 0 else 1.0
        return lower - padding, upper + padding
    return lower, upper


def summarize_metric(
    dimension,
    comparison,
    subjects,
    first_label,
    second_label,
    first_values,
    second_values,
):
    statistic, p_value = paired_wilcoxon(first_values, second_values)
    diff = second_values - first_values
    return {
        "dimension": dimension,
        "comparison": comparison,
        "n_pairs": len(first_values),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "first_label": first_label,
        "second_label": second_label,
        "first_mean": float(np.mean(first_values)) if len(first_values) else math.nan,
        "second_mean": float(np.mean(second_values)) if len(second_values) else math.nan,
        "first_median": float(np.median(first_values)) if len(first_values) else math.nan,
        "second_median": float(np.median(second_values)) if len(second_values) else math.nan,
        "paired_difference_mean_second_minus_first": float(np.mean(diff)) if len(diff) else math.nan,
        "paired_difference_median_second_minus_first": float(np.median(diff)) if len(diff) else math.nan,
        "paired_subjects": ";".join(subjects),
    }


def write_wilcoxon_results(rows, output_dir):
    csv_path = output_dir / "wilcoxon_results.csv"
    fieldnames = [
        "dimension",
        "comparison",
        "n_pairs",
        "wilcoxon_statistic",
        "p_value",
        "first_label",
        "second_label",
        "first_mean",
        "second_mean",
        "first_median",
        "second_median",
        "paired_difference_mean_second_minus_first",
        "paired_difference_median_second_minus_first",
        "paired_subjects",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    txt_path = output_dir / "wilcoxon_results.txt"
    with open(txt_path, "w") as f:
        for row in rows:
            f.write(
                f"{row['dimension']} {row['comparison']}: "
                f"n={row['n_pairs']}, W={row['wilcoxon_statistic']:.6g}, "
                f"p={row['p_value']:.6g}, "
                f"median({row['first_label']})={row['first_median']:.6g}, "
                f"median({row['second_label']})={row['second_median']:.6g}\n"
            )

    return csv_path, txt_path


def run_comparison(paths, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    dimensions = {
        "2D": (paths["grics_pp_2d"], paths["grics_torch_2d"]),
        "3D": (paths["grics_pp_3d"], paths["grics_torch_3d"]),
    }

    summary_rows = []

    for dimension, (grics_pp_root, grics_torch_root) in dimensions.items():
        subjects, pp_values, torch_values = paired_metric_values(
            grics_pp_root / RECONSTRUCTION_TIME_METRIC["file"],
            grics_torch_root / RECONSTRUCTION_TIME_METRIC["file"],
        )

        if len(pp_values) == 0:
            print(f"No paired finite values found for {dimension} reconstruction_time")
        else:
            png_path, svg_path = plot_paired_boxplot(
                pp_values,
                torch_values,
                RECONSTRUCTION_TIME_METRIC["ylabel"],
                output_dir,
                f"{dimension.lower()}_reconstruction_time",
                ["GRICS++", "GRICS-torch"],
            )
            print(f"Saved {png_path}")
            print(f"Saved {svg_path}")

            summary_rows.append(
                summarize_metric(
                    dimension,
                    "reconstruction_time",
                    subjects,
                    "GRICS++",
                    "GRICS-torch",
                    pp_values,
                    torch_values,
                )
            )

        sharpness_rows = []
        for code_key, code_label, code_root in [
            ("grics_pp", "GRICS++", grics_pp_root),
            ("grics_torch", "GRICS-torch", grics_torch_root),
        ]:
            subjects, uncorrected_values, corrected_values = paired_metric_values(
                code_root / SHARPNESS_METRIC["uncorrected_file"],
                code_root / SHARPNESS_METRIC["corrected_file"],
            )
            sharpness_rows.append(
                (code_key, code_label, subjects, uncorrected_values, corrected_values)
            )

        sharpness_ylim = shared_limits(
            (uncorrected_values, corrected_values)
            for _, _, _, uncorrected_values, corrected_values in sharpness_rows
        )

        for code_key, code_label, subjects, uncorrected_values, corrected_values in sharpness_rows:

            if len(uncorrected_values) == 0:
                print(f"No paired finite values found for {dimension} {code_label} sharpness")
                continue

            png_path, svg_path = plot_paired_boxplot(
                uncorrected_values,
                corrected_values,
                SHARPNESS_METRIC["ylabel"],
                output_dir,
                f"{dimension.lower()}_{code_key}_sharpness_index",
                ["Uncorrected", "Corrected"],
                ylim=sharpness_ylim,
            )
            print(f"Saved {png_path}")
            print(f"Saved {svg_path}")

            summary_rows.append(
                summarize_metric(
                    dimension,
                    f"{code_label}_sharpness_index",
                    subjects,
                    "Uncorrected",
                    "Corrected",
                    uncorrected_values,
                    corrected_values,
                )
            )

    csv_path, txt_path = write_wilcoxon_results(summary_rows, output_dir)
    print(f"Saved {csv_path}")
    print(f"Saved {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot reconstruction-time code comparisons and paired uncorrected "
            "versus corrected sharpness-index comparisons."
        )
    )
    parser.add_argument("--grics-pp-2d", type=Path, default=DEFAULT_GRICS_PP_2D)
    parser.add_argument("--grics-torch-2d", type=Path, default=DEFAULT_GRICS_TORCH_2D)
    parser.add_argument("--grics-pp-3d", type=Path, default=DEFAULT_GRICS_PP_3D)
    parser.add_argument("--grics-torch-3d", type=Path, default=DEFAULT_GRICS_TORCH_3D)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = {
        "grics_pp_2d": args.grics_pp_2d,
        "grics_torch_2d": args.grics_torch_2d,
        "grics_pp_3d": args.grics_pp_3d,
        "grics_torch_3d": args.grics_torch_3d,
    }
    run_comparison(paths, args.output_dir)


if __name__ == "__main__":
    main()

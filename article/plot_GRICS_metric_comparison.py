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


METRICS = {
    "reconstruction_time": {
        "file": "reconstruction_times.txt",
        "title": "Reconstruction Time",
        "ylabel": "Time (s)",
    },
    "sharpness": {
        "file": "sharpness_corrected.txt",
        "title": "Sharpness",
        "ylabel": "Sharpness index",
    },
    "sharpness_enhancement": {
        "file": "sharpness_enhansement.txt",
        "title": "Sharpness Enhancement",
        "ylabel": "Enhancement (%)",
    },
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


def paired_wilcoxon(grics_pp_values, grics_torch_values):
    if len(grics_pp_values) == 0:
        return math.nan, math.nan

    differences = grics_torch_values - grics_pp_values
    if np.allclose(differences, 0):
        return 0.0, 1.0

    result = wilcoxon(
        grics_pp_values,
        grics_torch_values,
        alternative="two-sided",
        zero_method="zsplit",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def plot_paired_boxplot(pp_values, torch_values, dimension, metric_name, metric_config, output_dir):
    fig, ax = plt.subplots(figsize=(6.4, 6.2), dpi=200)
    positions = [1, 2]
    data = [pp_values, torch_values]

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
    for pair_idx, (pp_value, torch_value) in enumerate(zip(pp_values, torch_values)):
        jitter = rng.uniform(-0.045, 0.045)
        color = "#666666"
        ax.plot(
            [positions[0] + jitter, positions[1] + jitter],
            [pp_value, torch_value],
            color=color,
            linewidth=0.8,
            alpha=0.35,
            zorder=1,
        )
        ax.scatter(
            [positions[0] + jitter, positions[1] + jitter],
            [pp_value, torch_value],
            color=color,
            s=12,
            alpha=0.65,
            zorder=2,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(["GRICS++", "GRICS-torch"], fontsize=FONT_SIZE)
    ax.set_ylabel(metric_config["ylabel"], fontsize=FONT_SIZE)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    base_name = f"{dimension.lower()}_{metric_name}"
    png_path = output_dir / f"{base_name}.png"
    svg_path = output_dir / f"{base_name}.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, svg_path


def summarize_metric(dimension, metric_name, subjects, pp_values, torch_values):
    statistic, p_value = paired_wilcoxon(pp_values, torch_values)
    diff = torch_values - pp_values
    return {
        "dimension": dimension,
        "metric": metric_name,
        "n_pairs": len(pp_values),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "grics_pp_mean": float(np.mean(pp_values)) if len(pp_values) else math.nan,
        "grics_torch_mean": float(np.mean(torch_values)) if len(torch_values) else math.nan,
        "grics_pp_median": float(np.median(pp_values)) if len(pp_values) else math.nan,
        "grics_torch_median": float(np.median(torch_values)) if len(torch_values) else math.nan,
        "paired_difference_mean_torch_minus_pp": float(np.mean(diff)) if len(diff) else math.nan,
        "paired_difference_median_torch_minus_pp": float(np.median(diff)) if len(diff) else math.nan,
        "paired_subjects": ";".join(subjects),
    }


def write_wilcoxon_results(rows, output_dir):
    csv_path = output_dir / "wilcoxon_results.csv"
    fieldnames = [
        "dimension",
        "metric",
        "n_pairs",
        "wilcoxon_statistic",
        "p_value",
        "grics_pp_mean",
        "grics_torch_mean",
        "grics_pp_median",
        "grics_torch_median",
        "paired_difference_mean_torch_minus_pp",
        "paired_difference_median_torch_minus_pp",
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
                f"{row['dimension']} {row['metric']}: "
                f"n={row['n_pairs']}, W={row['wilcoxon_statistic']:.6g}, "
                f"p={row['p_value']:.6g}, "
                f"median(GRICS++)={row['grics_pp_median']:.6g}, "
                f"median(GRICS-torch)={row['grics_torch_median']:.6g}\n"
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
        for metric_name, metric_config in METRICS.items():
            grics_pp_file = grics_pp_root / metric_config["file"]
            grics_torch_file = grics_torch_root / metric_config["file"]
            subjects, pp_values, torch_values = paired_metric_values(
                grics_pp_file,
                grics_torch_file,
            )

            if len(pp_values) == 0:
                print(f"No paired finite values found for {dimension} {metric_name}")
                continue

            png_path, svg_path = plot_paired_boxplot(
                pp_values,
                torch_values,
                dimension,
                metric_name,
                metric_config,
                output_dir,
            )
            print(f"Saved {png_path}")
            print(f"Saved {svg_path}")

            summary_rows.append(
                summarize_metric(dimension, metric_name, subjects, pp_values, torch_values)
            )

    csv_path, txt_path = write_wilcoxon_results(summary_rows, output_dir)
    print(f"Saved {csv_path}")
    print(f"Saved {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot paired GRICS++ vs GRICS-torch metric comparisons."
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

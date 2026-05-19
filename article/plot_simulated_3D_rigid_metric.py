import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon


FONT_SIZE = 20

DEFAULT_OUTPUT_ROOT = Path(
    "/home/pyuser/wkdir/data/GRICS-torch/shepp_logan_simulated_rigid_3D_20_variants"
)

PLOTS = {
    "sharpness_index": {
        "files": [
            "sharpness_uncorrected.txt",
            "sharpness_corrected.txt",
            "sharpness_ground_truth.txt",
        ],
        "labels": ["Uncorrected", "Corrected", "Ground truth"],
        "ylabel": "Sharpness index",
        "output": "sharpness_index.svg",
    },
    "ssim": {
        "files": [
            "uncorrected_ssim.txt",
            "corrected_ssim.txt",
        ],
        "labels": ["Uncorrected", "Corrected"],
        "ylabel": "SSIM",
        "output": "ssim.svg",
    },
    "psnr": {
        "files": [
            "uncorrected_psnr.txt",
            "corrected_psnr.txt",
        ],
        "labels": ["Uncorrected", "Corrected"],
        "ylabel": "PSNR",
        "output": "psnr.svg",
    },
    "nrmse": {
        "files": [
            "uncorrected_nrmse.txt",
            "corrected_nrmse.txt",
        ],
        "labels": ["Uncorrected", "Corrected"],
        "ylabel": "NRMSE",
        "output": "nrmse.svg",
    },
}


def paired_wilcoxon(values_list, labels):
    rows = []

    for i in range(len(values_list)):
        for j in range(i + 1, len(values_list)):
            x = np.asarray(values_list[i], dtype=float)
            y = np.asarray(values_list[j], dtype=float)

            n = min(len(x), len(y))
            x = x[:n]
            y = y[:n]

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            if len(x) == 0:
                statistic = math.nan
                p_value = math.nan
            elif np.allclose(y - x, 0):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="zsplit",
                    method="auto",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            rows.append(
                {
                    "comparison": f"{labels[i]} vs {labels[j]}",
                    "n_pairs": len(x),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "median_1": float(np.median(x)) if len(x) else math.nan,
                    "median_2": float(np.median(y)) if len(y) else math.nan,
                    "median_difference_2_minus_1": (
                        float(np.median(y - x)) if len(x) else math.nan
                    ),
                }
            )

    return rows


def _parse_float(value):
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_metric_file(file_path):
    values = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.replace(",", " ").split()
            value = _parse_float(parts[-1])

            if np.isfinite(value):
                values.append(value)

    return np.asarray(values, dtype=float)


def plot_metric_boxplot(values_list, labels, ylabel, output_path):
    fig, ax = plt.subplots(figsize=(6.4, 6.2), dpi=200)

    positions = np.arange(1, len(values_list) + 1)

    ax.boxplot(
        values_list,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"facecolor": "#d8e6f3", "edgecolor": "#4f5d6b", "linewidth": 1.1},
        whiskerprops={"color": "#4f5d6b", "linewidth": 1.0},
        capprops={"color": "#4f5d6b", "linewidth": 1.0},
    )

    finite_lengths = [len(values) for values in values_list if len(values)]
    n_variants = min(finite_lengths) if finite_lengths else 0
    rng = np.random.default_rng(42)
    jitters = rng.uniform(-0.045, 0.045, size=n_variants)

    for variant_idx in range(n_variants):
        x_values = positions + jitters[variant_idx]
        y_values = [values[variant_idx] for values in values_list]

        ax.plot(
            x_values,
            y_values,
            color="#666666",
            linewidth=0.8,
            alpha=0.35,
            zorder=3,
        )

        ax.scatter(
            x_values,
            y_values,
            color="#666666",
            s=12,
            alpha=0.65,
            zorder=4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


def run(output_root):
    output_root.mkdir(parents=True, exist_ok=True)

    for plot_name, config in PLOTS.items():
        values_list = []

        for filename in config["files"]:
            file_path = output_root / filename
            if not file_path.is_file():
                print(f"Missing file: {file_path}")
                values = np.asarray([], dtype=float)
            else:
                values = read_metric_file(file_path)

            values_list.append(values)

        output_path = output_root / config["output"]
        plot_metric_boxplot(
            values_list=values_list,
            labels=config["labels"],
            ylabel=config["ylabel"],
            output_path=output_path,
        )
        wilcoxon_rows = paired_wilcoxon(values_list, config["labels"])
        print(f"Saved {output_path}")

        wilcoxon_txt = output_root / f"{plot_name}_wilcoxon.txt"
        with open(wilcoxon_txt, "w") as f:
            for row in wilcoxon_rows:
                f.write(
                    f"{plot_name} {row['comparison']}: "
                    f"n={row['n_pairs']}, "
                    f"W={row['wilcoxon_statistic']:.6g}, "
                    f"p={row['p_value']:.6g}, "
                    f"median_1={row['median_1']:.6g}, "
                    f"median_2={row['median_2']:.6g}, "
                    f"median_difference_2_minus_1="
                    f"{row['median_difference_2_minus_1']:.6g}\n"
                )

        print(f"Saved {wilcoxon_txt}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot simulated rigid 3D reconstruction metrics as SVG box plots."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.output_root)


if __name__ == "__main__":
    main()

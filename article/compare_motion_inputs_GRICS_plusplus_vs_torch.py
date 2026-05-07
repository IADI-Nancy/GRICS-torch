import argparse
import csv
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRICS_PLUSPLUS_ROOT = Path("/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/GRICS-BELT")
GRICS_TORCH_NOMOCO_ROOT = Path("/home/pyuser/wkdir/data/GRICS-torch/article_dataset_nomoco")
DEFAULT_SUBJECT = "0068_T2_m"


def slice_number(slice_dir):
    match = re.search(r"slice(\d+)_image", slice_dir.name)
    return None if match is None else int(match.group(1))


def zscore(signal):
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    std = signal.std()
    if std <= 0:
        return signal - signal.mean()
    return (signal - signal.mean()) / std


def load_model_inputs(path):
    path = Path(path)
    data = np.fromfile(path, dtype="<f4")
    if data.size == 0:
        raise ValueError(f"Empty model input file: {path}")
    return data.astype(np.float64, copy=False).reshape(-1)


def load_torch_npz(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def best_lag_correlation(a, b, max_lag=None):
    a = zscore(a)
    b = zscore(b)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    if max_lag is None:
        max_lag = min(50, max(n - 2, 0))

    best = {"lag": 0, "corr": np.nan}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: n + lag]
        elif lag > 0:
            aa = a[: n - lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b

        if aa.size < 2 or bb.size < 2:
            continue
        corr = float(np.corrcoef(aa, bb)[0, 1])
        if np.isnan(best["corr"]) or abs(corr) > abs(best["corr"]):
            best = {"lag": lag, "corr": corr}

    return best


def plot_slice(output_file, slice_idx, grics_pp, torch_input, torch_npz):
    x_pp = np.arange(grics_pp.size)
    x_torch = np.arange(torch_input.size)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=150, sharex=False)

    axes[0].plot(x_pp, grics_pp, label="GRICS++ ModelInputs.dat", linewidth=1.5)
    axes[0].plot(x_torch, torch_input, label="GRICS-torch ModelInputs_torch.dat", linewidth=1.2)
    axes[0].set_title(f"Slice {slice_idx:03d}: raw motion inputs")
    axes[0].set_ylabel("Signal")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x_pp, zscore(grics_pp), label="GRICS++ z-score", linewidth=1.5)
    axes[1].plot(x_torch, zscore(torch_input), label="GRICS-torch z-score", linewidth=1.2)
    if "motion_signal_reconstruction" in torch_npz:
        motion_signal = np.asarray(torch_npz["motion_signal_reconstruction"]).reshape(-1)
        axes[1].scatter(
            np.linspace(0, max(grics_pp.size, torch_input.size) - 1, motion_signal.size),
            zscore(motion_signal),
            s=16,
            label="GRICS-torch motion states",
            zorder=3,
        )
    axes[1].set_title("Normalized comparison")
    axes[1].set_xlabel("Readout / model-input sample")
    axes[1].set_ylabel("z-score")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def compare_subject(subject, grics_pp_root, grics_torch_nomoco_root, output_dir):
    grics_pp_subject = Path(grics_pp_root) / subject
    torch_subject = Path(grics_torch_nomoco_root) / subject

    if not grics_pp_subject.is_dir():
        raise FileNotFoundError(f"Missing GRICS++ subject folder: {grics_pp_subject}")
    if not torch_subject.is_dir():
        raise FileNotFoundError(f"Missing GRICS-torch no-moco subject folder: {torch_subject}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for pp_slice_dir in sorted(grics_pp_subject.glob("Siemens_SingleImage_slice*_image01")):
        idx = slice_number(pp_slice_dir)
        if idx is None:
            continue

        torch_slice_dir = torch_subject / pp_slice_dir.name
        pp_file = pp_slice_dir / "ModelInputs.dat"
        torch_file = torch_slice_dir / "ModelInputs_torch.dat"
        torch_npz_file = torch_slice_dir / "PhysiologicalData_torch.npz"

        if not pp_file.is_file() or not torch_file.is_file():
            continue

        grics_pp = load_model_inputs(pp_file)
        torch_input = load_model_inputs(torch_file)
        torch_npz = load_torch_npz(torch_npz_file)

        n_common = min(grics_pp.size, torch_input.size)
        pp_z = zscore(grics_pp[:n_common])
        torch_z = zscore(torch_input[:n_common])
        corr = float(np.corrcoef(pp_z, torch_z)[0, 1]) if n_common > 1 else np.nan
        rmse_z = float(np.sqrt(np.mean((pp_z - torch_z) ** 2))) if n_common > 0 else np.nan
        best = best_lag_correlation(grics_pp, torch_input)

        plot_slice(
            output_dir / f"motion_inputs_slice{idx:03d}.png",
            idx,
            grics_pp,
            torch_input,
            torch_npz,
        )

        rows.append(
            {
                "slice": idx,
                "grics_plusplus_samples": grics_pp.size,
                "torch_samples": torch_input.size,
                "common_samples": n_common,
                "corr_zscore_no_lag": corr,
                "rmse_zscore_no_lag": rmse_z,
                "best_abs_corr": best["corr"],
                "best_corr_lag_samples": best["lag"],
                "grics_plusplus_file": str(pp_file),
                "torch_file": str(torch_file),
            }
        )

    rows.sort(key=lambda row: row["slice"])
    summary_file = output_dir / "motion_input_comparison_summary.csv"
    with summary_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["slice"])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        corrs = np.asarray([row["corr_zscore_no_lag"] for row in rows], dtype=np.float64)
        print(f"Compared {len(rows)} slices for {subject}.")
        print(f"Mean no-lag z-score correlation: {np.nanmean(corrs):.4f}")
        print(f"Saved summary: {summary_file}")
        print(f"Saved per-slice plots in: {output_dir}")
    else:
        print(f"No common ModelInputs files found for {subject}.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare GRICS++ and GRICS-torch no-moco physiological model inputs."
    )
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--grics-plusplus-root", type=Path, default=GRICS_PLUSPLUS_ROOT)
    parser.add_argument("--torch-nomoco-root", type=Path, default=GRICS_TORCH_NOMOCO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.torch_nomoco_root / args.subject / "motion_input_comparison"

    compare_subject(
        subject=args.subject,
        grics_pp_root=args.grics_plusplus_root,
        grics_torch_nomoco_root=args.torch_nomoco_root,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

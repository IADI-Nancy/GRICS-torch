"""Generic numerical checks and console markers for manual validation.

This module owns no datasets, configuration files, reconstruction settings, or
pipeline execution. Each validation script describes and runs its own case.
"""

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity
from scipy.stats import wilcoxon


def print_step(label, boundary, step):
    print(f"[{label}] {boundary} {step}", flush=True)


def summarize_similarity(results):
    """Summarize paired metrics, percentage improvement, and two-sided Wilcoxon.

    Improvement is (corrected - uncorrected) / abs(uncorrected) for SSIM,
    and (uncorrected - corrected) / abs(uncorrected) for NRMSE, times 100.
    Zero baselines are excluded only from percentage summaries and counted.
    Wilcoxon uses raw paired differences, discards zero differences, and uses
    SciPy's automatic method. All-identical pairs are reported as p=1.
    """
    summary = {}
    for comparison in ("uncorrected_vs_ground_truth", "corrected_vs_ground_truth"):
        summary[comparison] = {}
        for metric in ("ssim", "nrmse"):
            values = np.asarray([result[comparison][metric] for result in results])
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError("Similarity summary requires nonempty, finite metrics")
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            summary[comparison][metric] = {"median": float(median), "q1": float(q1), "q3": float(q3)}
    summary["paired_statistics"] = {}
    for metric in ("ssim", "nrmse"):
        before = np.asarray([result["uncorrected_vs_ground_truth"][metric] for result in results])
        after = np.asarray([result["corrected_vs_ground_truth"][metric] for result in results])
        differences = after - before
        nonzero_baseline = before != 0
        direction = 1 if metric == "ssim" else -1
        improvement = 100 * direction * differences[nonzero_baseline] / np.abs(before[nonzero_baseline])
        relative = None
        if improvement.size:
            q1, median, q3 = np.quantile(improvement, [0.25, 0.5, 0.75])
            relative = {"median": float(median), "q1": float(q1), "q3": float(q3)}
        # The signed-rank test has no nonzero ranks when every pair is equal.
        p_value = 1.0 if np.all(differences == 0) else float(
            wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
        summary["paired_statistics"][metric] = {
            "relative_improvement_percent": relative,
            "zero_baselines_excluded": int(np.count_nonzero(~nonzero_baseline)),
            "paired_count": len(results), "wilcoxon_p_value": p_value,
            "wilcoxon_alternative": "two-sided",
        }
    return summary


def print_similarity_summary(summary):
    """Print all SSIM results first, then all NRMSE results."""
    for metric in ("ssim", "nrmse"):
        for comparison, label in (("uncorrected_vs_ground_truth", "Uncorrected"), ("corrected_vs_ground_truth", "Corrected")):
            values = summary[comparison][metric]
            print(f"{label} {metric.upper()}: {values['median']:.6f} [{values['q1']:.6f}, {values['q3']:.6f}]", flush=True)
        stats = summary["paired_statistics"][metric]
        values = stats["relative_improvement_percent"]
        improvement = "undefined (all baselines are zero)" if values is None else (
            f"{values['median']:.2f}% [{values['q1']:.2f}%, {values['q3']:.2f}%]")
        excluded = stats["zero_baselines_excluded"]
        note = f" ({excluded} zero baselines excluded)" if excluded else ""
        print(f"{metric.upper()} relative improvement: {improvement}{note}", flush=True)
        print(f"{metric.upper()} paired Wilcoxon p-value (two-sided): {stats['wilcoxon_p_value']:.6g}", flush=True)


def summarize_residuals(values):
    """Return the final level's last/first ratio and its endpoints."""
    if not values:
        return {"start": None, "end": None, "end_over_start": None, "count": 0}
    start, end = float(values[0]), float(values[-1])
    if not (math.isfinite(start) and math.isfinite(end)):
        raise AssertionError("Solver residual history contains a non-finite endpoint")
    return {"start": start, "end": end, "end_over_start": None if start == 0 else end / start, "count": len(values)}


def image_metrics(reference, candidate):
    """Calculate magnitude-image NRMSE and SSIM on the native image grid."""
    reference = torch.abs(reference).detach().cpu().numpy().astype(np.float64)
    candidate = torch.abs(candidate).detach().cpu().numpy().astype(np.float64)
    if reference.shape != candidate.shape:
        raise AssertionError(f"Image shape {candidate.shape} differs from reference {reference.shape}")
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        raise AssertionError("Image contains non-finite values")
    reference_norm = np.linalg.norm(reference)
    data_range = float(np.ptp(reference))
    if reference_norm == 0 or data_range == 0:
        raise AssertionError("Ground-truth phantom has zero norm or contrast")
    return {"nrmse": float(np.linalg.norm(reference - candidate) / reference_norm),
            "ssim": float(structural_similarity(reference, candidate, data_range=data_range))}

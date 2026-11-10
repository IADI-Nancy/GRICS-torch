"""Frozen reference limits for the 20-simulation Shepp-Logan validations."""

import math


# Bounds were computed from the named completed batches, ordered by simulation
# number, with 100,000 paired bootstrap resamples of 20 simulations:
# indices = np.random.default_rng(42).integers(0, 20, size=(100000, 20)).
# Each resample retains the before/after pair. Calculate its corrected median
# and median of per-simulation percentage improvements, NOT a ratio of medians.
# Store the 1st percentile for SSIM/improvements and 99th for NRMSE below.
# These are marginal bootstrap uncertainty bounds, not simultaneous coverage or
# prediction guarantees for future hardware. The additional engineering margins
# are explicit: subtract 0.01 SSIM, allow 10% higher NRMSE, and retain 90% of the
# lower improvement bound. Round outward to 0.001 or 0.01 percentage points.
# Freeze these values: never fit thresholds to the batch being evaluated.
# Recalibrate using independent batches/devices when the reference changes.
REFERENCES = {
    "nonrigid_2d": {
        "batch": "shepp_logan_2d_nonrigid/2026-11-04_09-46-22_879280",
        "bounds": (0.8938390690188893, 0.06945774179931827, 5.56063105855963, 43.844904548125044),
        "settings": (1, "odille-spline", False, 20, 10, 0.05, 10.0),
    },
    "rigid_2d": {
        "batch": "shepp_logan_2d_rigid/2026-11-03_16-33-28_698729",
        "bounds": (0.9111961110270252, 0.07772476529423958, 3.46923133172596, 43.02459270489646),
        "settings": (1, "espirit", True, 160, 128, 0.05, 0.2),
    },
    "nonrigid_3d": {
        "batch": "shepp_logan_3d_nonrigid/2026-11-04_09-55-48_066348",
        "bounds": (0.9748418786175816, 0.13601259624155776, 1.1144462672114082, 28.052692286568792),
        "settings": (64, "espirit", False, 20, 15, 0.05, 10.0),
    },
    "rigid_3d": {
        "batch": "shepp_logan_3d_rigid/2026-11-04_10-20-02_707423",
        "bounds": (0.9412542740370631, 0.21000800496315603, 11.5455971943091, 39.268420291912484),
        "settings": (64, "espirit", True, 160, 128, 0.2, 0.2),
    },
}

COMMON_SETTINGS = {
    "N_SheppLogan": 128, "Nex": 1, "NshotsPerNex": 4,
    "ResolutionLevels": [0.25, 0.5, 1.0], "GN_iterations_per_level": [10, 4, 4],
    "regularization_scaling": "direct", "N_motion_states": 4,
}
CASE_SETTINGS = ("Nz_SheppLogan", "coil_sensitivity_method", "use_calibration_weighted_image",
                 "max_iter_recon", "max_iter_motion", "lambda_r", "lambda_m")


def validation_settings(params):
    """Capture the main reference settings; device is deliberately unrestricted."""
    return {name: getattr(params, name) for name in (*COMMON_SETTINGS, *CASE_SETTINGS)}


def assess_quality(summary, results, case):
    """Attach auditable limits and verdicts without treating quality misses as crashes."""
    reference = REFERENCES[case]
    ssim, nrmse, ssim_imp, nrmse_imp = reference["bounds"]
    limits = {
        "SSIM_ref": math.floor((ssim - 0.01) * 1000) / 1000,
        "NRMSE_ref": math.ceil(nrmse * 1.1 * 1000) / 1000,
        "SSIM_imp_ref": math.floor(ssim_imp * 0.9 * 100) / 100,
        "NRMSE_imp_ref": math.floor(nrmse_imp * 0.9 * 100) / 100,
        # Two paired, two-sided Wilcoxon tests: Bonferroni alpha = 0.05 / 2.
        # Significance is evidence of a difference; positive improvement is also
        # required. This alpha is chosen in advance, not fitted to observed p.
        "p_ref": 0.025,
    }
    expected = dict(COMMON_SETTINGS, **dict(zip(CASE_SETTINGS, reference["settings"])))
    reasons = []
    if len(results) != 20:
        reasons.append("reference limits require exactly 20 simulations")
    changed = sorted({key for result in results for key, value in expected.items()
                      if result.get("validation_settings", {}).get(key) != value})
    if changed:
        reasons.append("reference settings differ: " + ", ".join(changed))
    space = "globally rigid-registered magnitude images" if case.startswith("rigid_") else "raw"
    if summary.get("metric_space", "raw") != space or any(result.get("metric_space", "raw") != space for result in results):
        reasons.append("reference metric space differs")
    checks = {}
    for metric in ("ssim", "nrmse"):
        name = metric.upper()
        value = summary["corrected_vs_ground_truth"][metric]["median"]
        limit = limits[f"{name}_ref"]
        checks[f"{name}_ref"] = math.isfinite(value) and (value >= limit if metric == "ssim" else value <= limit)
        stats = summary["paired_statistics"][metric]
        improvement = stats["relative_improvement_percent"]
        checks[f"{name}_imp_ref"] = (improvement is not None and math.isfinite(improvement["median"])
                                          and improvement["median"] > 0 and improvement["median"] >= limits[f"{name}_imp_ref"])
        checks[f"{name}_p_ref"] = math.isfinite(stats["wilcoxon_p_value"]) and stats["wilcoxon_p_value"] < limits["p_ref"]
    summary["quality_assessment"] = {
        "reference_batch": reference["batch"], "limits": limits, "checks": checks,
        "reference_settings": expected,
        "method": "100000 paired bootstrap resamples, seed 42; 1% lower/99% upper median bounds; -0.01 SSIM, +10% NRMSE, -10% improvement margins; outward rounding; Wilcoxon alpha 0.05/2",
        "reasons": reasons, "status": "INCONCLUSIVE" if reasons else ("PASSED" if all(checks.values()) else "QUALITY CHECK NOT MET"),
    }


def quality_suffix(summary, key, operator, percent=False):
    """Show the acceptance condition even when a batch is not comparable."""
    assessment = summary["quality_assessment"]
    limit = assessment["limits"]["p_ref" if key.endswith("_p_ref") else key]
    status = "INCONCLUSIVE" if assessment["reasons"] else ("PASSED" if assessment["checks"][key] else "NOT MET")
    return f"; criterion: {operator} {key} = {limit:g}{'%' if percent else ''} — {status}"

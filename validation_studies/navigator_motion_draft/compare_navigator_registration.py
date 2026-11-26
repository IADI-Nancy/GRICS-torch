"""Compare translation-only registration with saved non-rigid navigator results.

Run from the repository root:
    python validation_studies/navigator_motion_draft/compare_navigator_registration.py --registration both --set 2
    python validation_studies/navigator_motion_draft/compare_navigator_registration.py --registration both --navigator-sets interleaved --set 2

Input NPZs are produced by compare_mosar_navigators.py. Neither their profiles
nor existing non-rigid results are overwritten. Translation fitting uses the
last profile as reference and no temporal smoothing or amplitude fitting.
Single-set sampling remains the default. Interleaved sampling aligns constant
reference offsets to --set and writes a separate interleaved/ output directory.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from validation_studies.navigator_motion_draft.compare_mosar_navigators import calculate_eigenmode_variants, plot_registration_comparison
from src.utils.ismrmrd_io import ReadOnlyDataset


def register_translations(profiles, max_shift=16):
    """Fit one shift per profile by globally minimizing linearly interpolated SSD.

    Compare a fixed interior ROI valid for every candidate shift. Within each
    integer-shift interval SSD is quadratic; solve its minimum analytically,
    then choose the best interval. Thus local optimizer jumps are avoided.
    Sign convention: registered(x,t) = original(x + shift(t),t).
    """
    if type(max_shift) is not int or max_shift < 1:
        raise ValueError("max_shift must be a positive integer in profile pixels.")
    image = np.asarray(profiles, dtype=np.float64)
    if image.ndim != 2 or not np.isfinite(image).all() or image.shape[0] <= 2 * max_shift + 2:
        raise ValueError("Profiles must be finite [position,time] arrays with enough positions for the shift bound.")
    nx, nt = image.shape
    positions = np.arange(max_shift, nx - max_shift)
    target = image[positions, -1]
    if np.linalg.norm(np.diff(target)) == 0:
        raise ValueError("Reference ROI has no spatial variation; translation cannot be identified.")
    cost_before = np.mean((image[positions] - target[:, None]) ** 2, axis=0)
    best_cost = np.full(nt, np.inf)
    shifts = np.zeros(nt)
    for integer in range(-max_shift, max_shift):
        lower = image[positions + integer]
        gradient = image[positions + integer + 1] - lower
        denominator = np.sum(gradient ** 2, axis=0)
        numerator = np.sum(gradient * (target[:, None] - lower), axis=0)
        fraction = np.zeros(nt)
        np.divide(numerator, denominator, out=fraction, where=denominator > 0)
        fraction = np.clip(fraction, 0., 1.)
        candidate = integer + fraction
        cost = np.mean((lower + gradient * fraction - target[:, None]) ** 2, axis=0)
        # Exact ties use the smaller absolute shift; this does not smooth time.
        selected = (cost < best_cost) | ((cost == best_cost) & (np.abs(candidate) < np.abs(shifts)))
        best_cost[selected] = cost[selected]
        shifts[selected] = candidate[selected]
    registered = np.stack([np.interp(np.arange(nx) + shifts[t], np.arange(nx), image[:, t], left=0., right=0.) for t in range(nt)], axis=1)
    return shifts, registered, cost_before, best_cost


def plot_translation_debug(shifts, before, after, max_shift, output_dir, name):
    """Save every coil's temporal parameter and a common-scale heatmap."""
    ncoils, nt = shifts.shape
    times = np.arange(1, nt + 1)
    curve_limit = max(float(np.abs(shifts).max()) * 1.05, .05)
    fig, axes = plt.subplots(int(np.ceil(ncoils / 4)), 4, figsize=(18, 2 * int(np.ceil(ncoils / 4))), sharex=True, sharey=True, squeeze=False)
    for coil, ax in enumerate(axes.flat):
        if coil >= ncoils:
            ax.axis("off")
            continue
        ax.plot(times, shifts[coil], linewidth=1)
        ax.axhline(0, color="gray", linewidth=.5)
        ax.set_title(f"Coil {coil + 1}")
        ax.set_ylim(-curve_limit, curve_limit)
        ax.grid(alpha=.2)
        ax.set_xlabel("Navigator index")
        ax.set_ylabel("Shift (profile pixels)")
    fig.suptitle(f"{name}: one translation per coil/time; no temporal smoothing; search bound +/-{max_shift} pixels")
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(output_dir / f"{name}_translation_per_coil.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    limit = max(float(np.abs(shifts).max()), 1e-12)
    im = axes[0].imshow(shifts, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit, extent=(.5, nt + .5, ncoils + .5, .5))
    axes[0].set_ylabel("Coil")
    fig.colorbar(im, ax=axes[0], label="Translation (profile pixels)")
    axes[1].plot(times, shifts.T, alpha=.25, linewidth=.7)
    axes[1].plot(times, np.median(shifts, axis=0), color="black", linewidth=2, label="Median across coils (diagnostic only)")
    axes[1].legend()
    axes[1].set_ylabel("Translation (profile pixels)")
    axes[2].plot(times, np.sqrt(before.sum(axis=0)), label="Before")
    axes[2].plot(times, np.sqrt(after.sum(axis=0)), label="After")
    axes[2].legend()
    axes[2].set_ylabel("Fixed-ROI aggregate RMS error")
    axes[2].set_xlabel("Navigator index")
    fig.suptitle(f"{name}: translation diagnostics; reference = final profile")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output_dir / f"{name}_translation_debug.png", dpi=150)
    plt.close(fig)


def align_set_offsets(source, source_times, anchor, anchor_times, excluded_startup=4):
    """Subtract one robust constant per feature, using overlapping measured times.

    Linear interpolation is used only to estimate the reference offset. No
    measured sample is temporally filtered. Non-rigid alignment is an additive
    shared-grid approximation, not exact composition of deformation fields.
    """
    if source.shape[:-1] != anchor.shape[:-1] or not np.isfinite(source).all() or not np.isfinite(anchor).all():
        raise ValueError("Both sets must contain finite fields with matching feature dimensions.")
    source_times, anchor_times = np.asarray(source_times), np.asarray(anchor_times)
    if source_times.shape != (source.shape[-1],) or anchor_times.shape != (anchor.shape[-1],):
        raise ValueError("Set timestamps must match their field time dimensions.")
    if not np.isfinite(source_times).all() or not np.isfinite(anchor_times).all() or np.any(np.diff(source_times) <= 0) or np.any(np.diff(anchor_times) <= 0):
        raise ValueError("Each set must have strictly increasing timestamps.")
    anchor_times = anchor_times[excluded_startup:]
    if anchor_times.size < 2:
        raise ValueError("Not enough anchor profiles after excluding startup replacement.")
    valid = (np.arange(len(source_times)) >= excluded_startup) & (source_times >= anchor_times[0]) & (source_times <= anchor_times[-1])
    if valid.sum() < 2:
        raise ValueError("At least two overlapping source times are required for reference alignment.")
    flat = anchor[..., excluded_startup:].reshape(-1, len(anchor_times))
    interpolated = np.stack([np.interp(source_times[valid], anchor_times, row) for row in flat]).reshape(*anchor.shape[:-1], int(valid.sum()))
    offset = np.median(source[..., valid] - interpolated, axis=-1)
    return source - offset[..., None], offset, valid


def compare_interleaved_registration(input_dir, output_dir, anchor_set=2, registration="both", max_shift=16):
    """Interleave both contrasts after independent within-set registration."""
    metadata = json.loads((input_dir / "comparison.json").read_text())
    startup = metadata["matlab_parameters"]["dummy_profiles"]
    reader = ReadOnlyDataset(metadata["input_file"])
    times = {}
    try:
        origin = reader.read_acquisition(metadata["central_triplet_acquisition_indices"]["0"][0][0]).acquisition_time_stamp
        for set_id in (0, 1):
            groups = metadata["central_triplet_acquisition_indices"][str(set_id)]
            for direction in range(3):
                times[(set_id, direction)] = np.array([(int(reader.read_acquisition(group[direction]).acquisition_time_stamp) - int(origin)) * .0025 for group in groups])
    finally:
        reader.close()
    snapshot_times = np.concatenate([times[(set_id, 0)] for set_id in (0, 1)])
    order = np.argsort(snapshot_times)
    snapshot_times = snapshot_times[order]
    set_ids = np.concatenate([np.full(len(times[(set_id, 0)]), set_id + 1) for set_id in (0, 1)])[order]
    startup_flags = np.concatenate([np.arange(len(times[(set_id, 0)])) < startup for set_id in (0, 1)])[order]
    methods = ("nonrigid", "translation") if registration == "both" else (registration,)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"navigator_sets": "interleaved", "anchor_set": anchor_set, "registration": registration,
              "temporal_smoothing": False, "alignment": "Constant per feature: median(source - linearly interpolated anchor) over overlapping times, excluding replaced startup profiles",
              "time_origin": "First set-1 triplet, first readout", "startup_replaced_profiles_per_set": startup,
              "single_set_median_interval_seconds": float(np.median(np.diff(times[(1, 0)]))),
              "interleaved_median_interval_seconds": float(np.median(np.diff(snapshot_times))),
              "samples": len(snapshot_times), "series": {},
              "notes": ["Non-rigid reference alignment uses additive offsets on a shared grid; exact reference-warp composition is not performed.",
                        "Contrast-dependent bias or different motion sensitivity cannot be corrected by constant offsets.",
                        "Set 1's second readout is unstable; it is retained explicitly and can degrade xyz modes.",
                        "Startup-replaced observations are retained and flagged, but excluded when estimating offsets.",
                        "SVD columns represent triplets at their first-readout timestamps; each direction's actual readout times are also saved.",
                        "Single-set comparison uses the same anchor-set fields on the interleaved time span; no temporal filtering is applied."]}
    anchor_id, source_id = anchor_set - 1, 2 - anchor_set
    for method in methods:
        folder = output_dir / method
        folder.mkdir(exist_ok=True)
        combined_fields = {}
        anchor_fields = {}
        for direction, label in enumerate("xyz"):
            fields, bound_counts = {}, {}
            for set_id in (0, 1):
                with np.load(input_dir / f"nav{set_id + 1}{label}.npz") as data:
                    if method == "nonrigid":
                        fields[set_id] = data["displacement"]
                    else:
                        shifts = np.stack([register_translations(profile, max_shift)[0] for profile in data["profiles"]])
                        fields[set_id] = shifts[:, None, :]
                        bound_counts[str(set_id + 1)] = int(np.isclose(np.abs(shifts), max_shift, atol=1e-6).sum())
            raw_source = fields[source_id].copy()
            fields[source_id], offset, valid = align_set_offsets(fields[source_id], times[(source_id, direction)], fields[anchor_id], times[(anchor_id, direction)], startup)
            combined = np.concatenate([fields[set_id] for set_id in (0, 1)], axis=-1)[..., order]
            combined_fields[(anchor_id, direction)] = combined
            anchor_fields[(anchor_id, direction)] = fields[anchor_id]
            actual_times = np.concatenate([times[(set_id, direction)] for set_id in (0, 1)])[order]
            if np.any(np.diff(actual_times) <= 0):
                raise ValueError("Direction readout times do not preserve triplet chronology.")
            name = f"interleaved_{label}"
            np.savez_compressed(folder / f"{name}.npz", displacement=combined, time_seconds=actual_times, snapshot_time_seconds=snapshot_times, set_ids=set_ids, startup_replaced=startup_flags,
                                anchor=fields[anchor_id], source_before_alignment=raw_source, source_after_alignment=fields[source_id], alignment_offset=offset, alignment_valid=valid)
            summary = combined.mean(axis=1)  # Per-coil diagnostic; SVD retains every spatial feature.
            before = raw_source.mean(axis=1)
            aligned = fields[source_id].mean(axis=1)
            anchor = fields[anchor_id].mean(axis=1)
            rows = int(np.ceil(summary.shape[0] / 4))
            fig, axes = plt.subplots(rows, 4, figsize=(18, 2 * rows), sharex=True, squeeze=False)
            for coil, ax in enumerate(axes.flat):
                if coil >= summary.shape[0]:
                    ax.axis("off")
                    continue
                ax.plot(times[(anchor_id, direction)], anchor[coil], color="C0", label=f"Set {anchor_set} anchor")
                ax.plot(times[(source_id, direction)], before[coil], color="gray", alpha=.45, linestyle=":", label="Other set before alignment")
                ax.plot(times[(source_id, direction)], aligned[coil], ".", color="C1", markersize=3, label="Other set aligned")
                ax.plot(actual_times, summary[coil], color="C1", linewidth=.6, alpha=.6)
                ax.set_title(f"Coil {coil + 1}")
                ax.grid(alpha=.2)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Profile pixels")
            axes[0, 0].legend(fontsize=6)
            description = "translation" if method == "translation" else "spatial mean displacement (SVD uses full fields)"
            fig.suptitle(f"Both sets, {label}: {description}; constant reference alignment, no temporal smoothing")
            fig.tight_layout(rect=(0, 0, 1, .98))
            fig.savefig(folder / f"{name}_per_coil.png", dpi=150)
            plt.close(fig)
            with (folder / f"{name}_per_coil.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("time_seconds", "set", "coil", "translation_or_spatial_mean_profile_pixels", "startup_replaced"))
                for t in range(len(actual_times)):
                    for coil in range(summary.shape[0]):
                        writer.writerow((actual_times[t], int(set_ids[t]), coil + 1, summary[coil, t], bool(startup_flags[t])))
            report["series"][f"{method}_{label}"] = {"alignment_offset_median": float(np.median(offset)), "alignment_offset_max_abs": float(np.abs(offset).max()),
                                                       "alignment_samples": int(valid.sum()), "fits_at_search_bound_per_set": bound_counts}
            print(f"{method} {label}: {valid.sum()} offset-alignment samples; maximum absolute offset {np.abs(offset).max():.3f} profile pixels; bound hits per set {bound_counts}", flush=True)
        report[f"{method}_eigenmodes"] = calculate_eigenmode_variants(combined_fields, anchor_id, folder, snapshot_times, "Both sets interleaved")
        single_folder = folder / "single_set_reference"
        single_folder.mkdir(exist_ok=True)
        calculate_eigenmode_variants(anchor_fields, anchor_id, single_folder, times[(anchor_id, 0)], f"Set {anchor_set} only")
        # This view shows temporal sample density within one common SVD basis.
        fig, axes = plt.subplots(4, 3, figsize=(18, 11), sharex=True)
        for column, (variant, directions) in enumerate({"x": (0,), "z": (2,), "xyz": (0, 1, 2)}.items()):
            with np.load(folder / f"nav_eigenmodes_{variant}.npz") as data:
                modes = data["temporal_modes"]
            mask = set_ids == anchor_set
            for mode in range(4):
                axes[mode, column].plot(snapshot_times, modes[:, mode], ".-", markersize=3, label="Both sets")
                axes[mode, column].plot(snapshot_times[mask], modes[mask, mode], "o--", markersize=3, alpha=.7, label=f"Set {anchor_set} samples")
                axes[mode, column].set_title(f"{variant}, mode {mode + 1}")
                axes[mode, column].grid(alpha=.2)
            axes[-1, column].set_xlabel("Time (s)")
        axes[0, 0].legend()
        fig.suptitle(f"{method}: interleaved modes and anchor observations within the same SVD basis; no temporal filtering")
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(folder / "interleaved_sampling_comparison.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(4, 3, figsize=(18, 11), sharex=True)
        mask = set_ids == anchor_set
        for column, variant in enumerate(("x", "z", "xyz")):
            with np.load(folder / f"nav_eigenmodes_{variant}.npz") as data:
                combined_modes = data["temporal_modes"]
            with np.load(single_folder / f"nav_eigenmodes_{variant}.npz") as data:
                single_modes = data["temporal_modes"]
            for mode in range(4):
                single, combined = single_modes[:, mode], combined_modes[:, mode]
                norm = np.linalg.norm(combined[mask])
                if norm == 0:
                    raise ValueError("Combined mode has no anchor-set component; cannot normalize comparison.")
                sign = -1 if np.dot(single, combined[mask]) < 0 else 1
                axes[mode, column].plot(snapshot_times, sign * combined / norm, ".-", markersize=3, label="Interleaved SVD")
                axes[mode, column].plot(snapshot_times[mask], single, "--", label=f"Set {anchor_set}-only SVD")
                axes[mode, column].set_title(f"{variant}, mode {mode + 1}")
                axes[mode, column].grid(alpha=.2)
            axes[-1, column].set_xlabel("Time (s)")
        axes[0, 0].legend()
        fig.suptitle(f"{method}: separate SVDs; display signs aligned and norms matched on anchor times; modes need not represent the same motion")
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(folder / "single_vs_interleaved_eigenmodes.png", dpi=150)
        plt.close(fig)
        print(f"{method}: {len(snapshot_times)} interleaved samples; median spacing {report['interleaved_median_interval_seconds']:.4f} s", flush=True)
    (output_dir / "interleaved_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare_registration(input_dir, output_dir, set_number=2, registration="both", max_shift=16, navigator_sets="single"):
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Use a separate output directory to preserve the existing registration results.")
    if navigator_sets == "interleaved":
        return compare_interleaved_registration(input_dir, output_dir / "interleaved", set_number, registration, max_shift)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_set = set_number - 1
    fields = {method: {} for method in (("nonrigid", "translation") if registration == "both" else (registration,))}
    report = {"set": set_number, "registration": registration, "reference": "last profile", "temporal_smoothing": False,
              "max_shift_profile_pixels": max_shift, "fit_roi": f"[{max_shift}, Npositions-{max_shift})",
              "input_profiles": "Saved profiles, including their existing first-four startup replacement",
              "nonrigid_source": "Existing saved registration results; not re-estimated or overwritten",
              "notes": ["x/y/z denote triplet ordinal, not independently verified physical directions.",
                        "Translations are in profile pixels, not millimetres or full 3D head displacement.",
                        "SVD uses uncentered displacement matrices and no coil normalization; eigenvector signs are arbitrary.",
                        "Full-profile residuals compare methods; translation optimization uses a fixed interior ROI."], "series": {}}
    for direction, label in enumerate("xyz"):
        name = f"nav{set_number}{label}"
        with np.load(input_dir / f"{name}.npz") as saved:
            profiles = saved["profiles"]
            nonrigid = saved["displacement"]
            nonrigid_registered = saved["registered"]
        reference = profiles[..., -1:]
        baseline = float(np.linalg.norm(profiles - reference))
        metrics = {}
        if "nonrigid" in fields:
            fields["nonrigid"][(selected_set, direction)] = nonrigid
            metrics["nonrigid"] = {"relative_full_profile_residual": float(np.linalg.norm(nonrigid_registered - reference) / baseline),
                                   "temporal_rms_step_profile_pixels": float(np.sqrt(np.mean(np.diff(nonrigid, axis=-1) ** 2)))}
        if "translation" in fields:
            fits = [register_translations(profile, max_shift) for profile in profiles]
            shifts, registered, before, after = [np.asarray(values) for values in zip(*fits)]
            # Singleton spatial dimension keeps one parameter per coil in SVD.
            fields["translation"][(selected_set, direction)] = shifts[:, None, :]
            at_bound = np.isclose(np.abs(shifts), max_shift, atol=1e-6)
            np.savez_compressed(output_dir / f"{name}_translation.npz", shifts=shifts, registered=registered, mse_before=before, mse_after=after, at_search_bound=at_bound)
            with (output_dir / f"{name}_translation.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("navigator_index", "coil", "shift_profile_pixels", "mse_before", "mse_after", "at_search_bound"))
                for coil in range(shifts.shape[0]):
                    for t in range(shifts.shape[1]):
                        writer.writerow((t + 1, coil + 1, shifts[coil, t], before[coil, t], after[coil, t], bool(at_bound[coil, t])))
            plot_translation_debug(shifts, before, after, max_shift, output_dir, name)
            plot_registration_comparison(profiles, registered, np.broadcast_to(shifts[:, None, :], profiles.shape), output_dir / f"{name}_translation_profiles.png", f"{name}, translation only")
            metrics["translation"] = {"relative_full_profile_residual": float(np.linalg.norm(registered - reference) / baseline),
                                      "relative_fixed_roi_residual": float(np.sqrt(after.sum() / before.sum())),
                                      "temporal_rms_step_profile_pixels": float(np.sqrt(np.mean(np.diff(shifts, axis=-1) ** 2))),
                                      "maximum_absolute_shift": float(np.abs(shifts).max()), "fits_at_search_bound": int(at_bound.sum())}
            print(f"{name}: shift range [{shifts.min():.3f}, {shifts.max():.3f}] pixels; {at_bound.sum()} fits at search bound; full-profile residual ratio {metrics['translation']['relative_full_profile_residual']:.3f}", flush=True)
        report["series"][name] = metrics
    modes = {}
    for method, displacement in fields.items():
        folder = output_dir / method
        folder.mkdir(exist_ok=True)
        report[f"{method}_eigenmodes"] = calculate_eigenmode_variants(displacement, selected_set, folder)
        modes[method] = {}
        for variant in ("x", "z", "xyz"):
            with np.load(folder / f"nav_eigenmodes_{variant}.npz") as saved:
                modes[method][variant] = saved["temporal_modes"]
    if registration == "both":
        fig, axes = plt.subplots(4, 3, figsize=(18, 11), sharex=True)
        for column, variant in enumerate(("x", "z", "xyz")):
            for mode in range(4):
                old = modes["nonrigid"][variant][:, mode]
                new = modes["translation"][variant][:, mode]
                sign = -1 if np.dot(old, new) < 0 else 1
                axes[mode, column].plot(np.arange(1, len(old) + 1), old, label="Non-rigid", alpha=.8)
                axes[mode, column].plot(np.arange(1, len(new) + 1), sign * new, label="Translation", alpha=.8)
                axes[mode, column].set_title(f"{variant}, mode {mode + 1}")
                axes[mode, column].grid(alpha=.2)
                axes[mode, column].legend(fontsize=8)
            axes[-1, column].set_xlabel("Navigator index")
        fig.suptitle(f"Set {set_number}: non-rigid versus translation; signs aligned for display only, mode indices need not represent the same motion")
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(output_dir / "eigenmodes_registration_comparison.png", dpi=150)
        plt.close(fig)
    (output_dir / "registration_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "runs/mosar_navigator_comparison")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/mosar_navigator_comparison/registration_methods")
    parser.add_argument("--set", dest="set_number", type=int, choices=(1, 2), default=2, help="Single input set, or reference set for interleaved offset alignment")
    parser.add_argument("--registration", choices=("nonrigid", "translation", "both"), default="both")
    parser.add_argument("--navigator-sets", choices=("single", "interleaved"), default="single", help="Use --set alone, or interleave both sets aligned to --set; default: single")
    parser.add_argument("--max-shift", type=int, default=16, help="Translation search half-width in profile pixels; integer >= 1")
    args = parser.parse_args()
    if args.max_shift < 1:
        parser.error("--max-shift must be >= 1")
    compare_registration(args.input_dir, args.output_dir, args.set_number, args.registration, args.max_shift, args.navigator_sets)

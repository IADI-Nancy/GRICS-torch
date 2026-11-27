"""Compare flagged navigators and MATLAB MOSAR central-line navigator figures.

Run from the repository root:
    python validation_studies/navigator_motion_draft/compare_mosar_navigators.py --input acquisition.mrd

For Siemens inputs, --measurement defaults to 1 (the supplied VB acquisition).
The central-triplet comparison is explicit and never replaces flagged navigators.
"""

import argparse
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path

import ismrmrd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.real_data.ISMRMRDReader import ISMRMRDReader, _is_noise, _is_non_imaging
from src.utils.ismrmrd_io import ReadOnlyDataset
from src.preprocessing.real_data.physiological_data.NavigatorEigenmodeCalculator import NavigatorEigenmodeCalculator


def plot_coils(values, path, title, cmap="jet"):
    """MATLAB-style per-coil profile images with independent color limits."""
    ncoils = values.shape[0]
    fig, axes = plt.subplots(max(4, math.ceil(ncoils / 8)), 8, figsize=(20, 10), squeeze=False)
    for coil, ax in enumerate(axes.flat):
        if coil < ncoils:
            ax.imshow(values[coil], aspect="auto", cmap=cmap, interpolation="nearest")
            ax.set_title(f"coil {coil + 1}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_registration_comparison(profiles, registered, fields, path, name):
    """Show representative coils before/after registration with matched scales."""
    coils = [coil for coil in (0, 2, 8, 15) if coil < profiles.shape[0]]
    fig, axes = plt.subplots(len(coils), 4, figsize=(16, 3 * len(coils)), squeeze=False)
    for row, coil in enumerate(coils):
        original, warped = profiles[coil], registered[coil]
        reference = original[:, -1:]
        maximum = max(original.max(), warped.max())
        axes[row, 0].imshow(original, aspect="auto", cmap="jet", vmin=0, vmax=maximum)
        axes[row, 1].imshow(warped, aspect="auto", cmap="jet", vmin=0, vmax=maximum)
        bound = max(float(np.max(np.abs(fields[coil]))), 1e-12)
        displacement = axes[row, 2].imshow(fields[coil], aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
        fig.colorbar(displacement, ax=axes[row, 2], label="profile pixels")
        axes[row, 3].plot(np.linalg.norm(original - reference, axis=0), label="before")
        axes[row, 3].plot(np.linalg.norm(warped - reference, axis=0), label="after")
        axes[row, 3].legend(fontsize=8)
        axes[row, 0].set_ylabel(f"Coil {coil + 1}\nProfile position")
        for ax in axes[row]:
            ax.set_xlabel("Navigator index")
    for ax, title in zip(axes[0], ("Original magnitude profiles", "Registered profiles (same color scale)", "Estimated displacement", "Mismatch to final reference profile")):
        ax.set_title(title)
    fig.suptitle(f"{name}: registration to the last profile", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def profiles_from_samples(samples, dummy_profiles):
    """Match fftshift(fft(fftshift(...))) along the MATLAB sample dimension."""
    profiles = np.fft.fftshift(np.fft.fft(np.fft.fftshift(samples, axes=1), axis=1), axes=1)
    if profiles.shape[-1] <= dummy_profiles:
        raise ValueError(f"Need more than {dummy_profiles} profiles to apply MATLAB startup replacement.")
    profiles[..., :dummy_profiles] = profiles[..., dummy_profiles:dummy_profiles + 1]
    return profiles


def read_matlab_triplets(path):
    """Select runs of exactly three central lines among retained imaging records.

    This sequence-specific MATLAB comparison uses the entire acquisition and
    actual set counters. Isolated central imaging lines are reported and excluded.
    Triplet ordinal x/y/z follows MATLAB; physical direction is not inferred.
    """
    dset = ReadOnlyDataset(str(path))
    try:
        header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
        enc = header.encoding[0]
        cy = enc.encodingLimits.kspace_encoding_step_1.center
        cz = enc.encodingLimits.kspace_encoding_step_2.center
        oversampling = int(enc.encodedSpace.matrixSize.x) // int(enc.reconSpace.matrixSize.x)
        if oversampling not in (1, 2):
            raise ValueError(f"Unsupported imaging oversampling factor {oversampling}.")
        runs, current = [], []
        for i in range(dset.number_of_acquisitions()):
            acq = dset.read_acquisition(i)
            if _is_noise(acq) or _is_non_imaging(acq):
                continue
            if acq.idx.kspace_encode_step_1 == cy and acq.idx.kspace_encode_step_2 == cz:
                current.append((i, int(acq.idx.set), acq.data.copy(), int(acq.acquisition_time_stamp)))
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
    finally:
        dset.close()
    groups = {}
    rejected = []
    for run in runs:
        if len(run) != 3 or len({record[1] for record in run}) != 1:
            rejected.append({"length": len(run), "acquisition_indices": [record[0] for record in run]})
            continue
        groups.setdefault(run[0][1], []).append(run)
    if not groups:
        raise ValueError("No unambiguous central triplets found; cannot reproduce the MATLAB navigator selection.")
    output = {}
    for contrast, triplets in groups.items():
        for direction in range(3):
            samples = np.stack([triplet[direction][2] for triplet in triplets], axis=-1)
            if oversampling == 2:
                image = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(samples, axes=1), axis=1), axes=1)
                size = samples.shape[1] // 2
                start = samples.shape[1] // 2 - size // 2
                samples = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(image[:, start:start + size], axes=1), axis=1), axes=1)
            output[(contrast, direction)] = samples
    return output, {"central_triplets_per_set": {str(k): len(v) for k, v in groups.items()},
                    "excluded_central_runs": rejected, "imaging_readout_oversampling": oversampling,
                    "central_triplet_acquisition_indices": {str(k): [[r[0] for r in t] for t in v] for k, v in groups.items()}}


def calculate_eigenmode_variants(displacements, selected_set, output_dir, time_seconds=None, source_label=None):
    """Compare uncentered SVD using x, z, or all three navigator series."""
    variants = {"x": (0,), "z": (2,), "xyz": (0, 1, 2)}
    results = {}
    source_label = source_label or f"Set {selected_set + 1}"
    xlabel = "Navigator index" if time_seconds is None else "Time from first set-1 navigator (s)"
    fig, axes = plt.subplots(4, 3, figsize=(18, 11), sharex=True)
    for column, (name, directions) in enumerate(variants.items()):
        fields = [displacements[(selected_set, direction)] for direction in directions]
        if any(field.ndim != 3 or not np.isfinite(field).all() for field in fields):
            raise ValueError("Displacements must be finite arrays shaped [coil, position, time].")
        if len({field.shape[-1] for field in fields}) != 1 or fields[0].shape[-1] < 4:
            raise ValueError("Navigator series must have matching time counts and at least four profiles.")
        matrix = np.concatenate([field.reshape(-1, field.shape[-1]) for field in fields], axis=0)
        times = np.arange(1, matrix.shape[1] + 1) if time_seconds is None else np.asarray(time_seconds)
        if times.shape != (matrix.shape[1],) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Eigenmode times must be finite, strictly increasing, and match the displacement samples.")
        _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        energy = singular_values ** 2
        if energy.sum() == 0:
            raise ValueError(f"Cannot extract motion eigenmodes from zero displacements ({name}).")
        fractions = energy / energy.sum()
        individual, individual_axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
        for mode in range(4):
            title = f"{name}: mode {mode + 1} ({fractions[mode]:.1%} of displacement energy)"
            for ax in (axes[mode, column], individual_axes[mode]):
                ax.plot(times, vh[mode])
                ax.set_title(title)
                ax.grid(alpha=0.2)
        individual_axes[-1].set_xlabel(xlabel)
        individual.suptitle(f"{source_label}: {name} navigator SVD (uncentered)")
        individual.tight_layout(rect=(0, 0, 1, .97))
        individual.savefig(output_dir / f"5.nav_eigenmodes_{name}.png", dpi=150)
        plt.close(individual)
        axes[-1, column].set_xlabel(xlabel)
        np.savez_compressed(output_dir / f"nav_eigenmodes_{name}.npz", singular_values=singular_values, temporal_modes=vh[:4].T, energy_fractions=fractions, sample_coordinates=times)
        results[name] = {"directions": list(name), "matrix_shape": list(matrix.shape),
                         "singular_values": singular_values.tolist(), "energy_fractions": fractions.tolist()}
    fig.suptitle(f"{source_label}: x-only, z-only, and combined navigator SVD; signs are arbitrary")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output_dir / "5.nav_eigenmodes_comparison.png", dpi=150)
    plt.close(fig)
    return results


def run_comparison(input_file, output_dir, measurement=1, dummy_profiles=4):
    with (ROOT / 'config/physiological_data/navigators.toml').open('rb') as stream:
        settings = tomllib.load(stream)['navigators']
    output_dir.mkdir(parents=True, exist_ok=True)
    input_file = input_file.resolve()
    if input_file.suffix.lower() == ".dat":
        converted = output_dir / "acquisition.mrd"
        if converted.exists():
            raise FileExistsError(f"{converted} already exists; pass it as --input to reuse it explicitly.")
        with (output_dir / "conversion.log").open("w") as log:
            subprocess.run(["siemens_to_ismrmrd", "-f", str(input_file), "-z", str(measurement), "-o", str(converted)], stdout=log, stderr=subprocess.STDOUT, check=True)
        input_file = converted
    print("Reading flagged navigators through ISMRMRDReader.read_navigator_data()", flush=True)
    navigators = ISMRMRDReader(str(input_file), read_navigators=True).read_navigator_data()
    if not navigators:
        raise ValueError("No flagged navigators found; central-line selection will not silently replace them.")
    grouped = {}
    for record in navigators:
        key = (record["data"].shape[0], record["data"].shape[1], record["counters"]["set"], tuple(record["read_dir"]))
        grouped.setdefault(key, []).append(record)
    report = {"input_file": str(input_file), "flagged_navigator_count": len(navigators), "flagged_groups": [],
              "matlab_parameters": {"lambda": settings["navigator_regularization"], "V0": "Inf (no temporal penalty)", "loops": settings["navigator_registration_loops"], "levels": settings["navigator_resolution_levels"], "dummy_profiles": dummy_profiles, "reference": "last profile"},
              "comparison_notes": ["Flagged feedback records and MATLAB central-line triplets are separate streams.",
                                   "FFT plots of flagged feedback records are diagnostics; their interpretation as anatomical k-space profiles is unverified.",
                                   "The supplied MOSAR acquisition fails full read_data() on duplicate central parallel-calibration lines; navigator-only decoding avoids imaging gridding.",
                                   "x/y/z labels denote triplet order as assumed by MATLAB; converted read_dir is identical for all three.",
                                   "Both MP2RAGE sets receive the profile FFT, correcting MATLAB's Nset>2 condition.",
                                   "SVD is uncentered, as in the visible MATLAB caller; the supplied SVD helper is unavailable.",
                                   "SVD signs are arbitrary; no numerical MATLAB outputs were supplied for equivalence validation."]}
    for group_index, (key, records) in enumerate(grouped.items(), start=1):
        print(f"Plotting flagged group {group_index}: {key[0]} coils, {key[1]} samples, {len(records)} records", flush=True)
        samples = np.stack([r["data"].numpy() for r in records], axis=-1)
        profiles = profiles_from_samples(samples, dummy_profiles)
        filename = f"flagged_group{group_index}_profiles.png"
        plot_coils(np.abs(profiles), output_dir / filename, f"Flagged navigation/feedback group {group_index}: {key[0]} coils, {key[1]} samples, {len(records)} readouts")
        report["flagged_groups"].append({"coils": key[0], "samples": key[1], "set": key[2], "read_dir": key[3], "count": len(records), "figure": filename})
        del samples, profiles
    del navigators, grouped
    print("Explicit MATLAB comparison: selecting unflagged central triplets", flush=True)
    central, selection = read_matlab_triplets(input_file)
    report.update(selection)
    displacements = {}
    for (contrast, direction), samples in sorted(central.items()):
        name = f"nav{contrast + 1}{'xyz'[direction]}"
        profiles = np.abs(profiles_from_samples(samples, dummy_profiles))
        plot_coils(profiles, output_dir / f"4.{name}.png", f"MATLAB central-triplet comparison: {name}")
        fields, registered, residuals = [], [], []
        for coil in range(profiles.shape[0]):
            field, warped, history = NavigatorEigenmodeCalculator.register_profiles(profiles[coil], levels=settings["navigator_resolution_levels"], loops=settings["navigator_registration_loops"], regularization=settings["navigator_regularization"])
            fields.append(field)
            registered.append(warped)
            residuals.append(history)
        displacements[(contrast, direction)] = np.asarray(fields)
        plot_coils(np.asarray(registered), output_dir / f"{name}_registered.png", f"Registered profiles: {name}")
        plot_coils(np.asarray(fields), output_dir / f"{name}_displacements.png", f"1D displacement in profile pixels: {name}", cmap="coolwarm")
        plot_registration_comparison(profiles, np.asarray(registered), np.asarray(fields), output_dir / f"registration_comparison_{name}.png", name)
        np.savez_compressed(output_dir / f"{name}.npz", profiles=profiles, registered=np.asarray(registered), displacement=np.asarray(fields), residuals=np.asarray(residuals))
        print(f"Saved {name}: {profiles.shape[0]} coils, {profiles.shape[1]} samples, {profiles.shape[2]} profiles", flush=True)
    selected_set = max(key[0] for key in central)
    report["svd_variants"] = calculate_eigenmode_variants(displacements, selected_set, output_dir)
    report["svd_set"] = selected_set
    report["singular_values"] = report["svd_variants"]["xyz"]["singular_values"]
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Figures and comparison metadata saved to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Siemens .dat or converted ISMRMRD file")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/mosar_navigator_comparison")
    parser.add_argument("--measurement", type=int, default=1)
    parser.add_argument("--dummy-profiles", type=int, default=4)
    args = parser.parse_args()
    if args.dummy_profiles < 0:
        parser.error("--dummy-profiles must be nonnegative")
    run_comparison(args.input, args.output_dir, args.measurement, args.dummy_profiles)

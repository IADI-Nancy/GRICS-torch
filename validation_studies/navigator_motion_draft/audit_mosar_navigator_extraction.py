"""Audit MOSAR extraction against original Siemens VB bytes, without MATLAB.

Run with --raw-file, --ismrmrd-file, and --comparison-dir. Only Siemens VB is
supported here: scan DMA lengths are checked against 128-byte channel headers
and interleaved float32 complex samples. This audit does not infer physical
navigator directions from their ordinal positions.

Header layout and flag mapping reference:
https://github.com/ismrmrd/siemens_to_ismrmrd/blob/master/siemensraw.h
https://github.com/ismrmrd/siemens_to_ismrmrd/blob/master/main.cpp
"""

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import h5py
import ismrmrd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.ismrmrd_io import ReadOnlyDataset


def read_vb_headers(path):
    """Walk original DMA boundaries, retaining headers without gridding data."""
    records = []
    with path.open("rb") as stream:
        length, second = struct.unpack("<II", stream.read(8))
        if length < 10000 and second <= 64:
            raise ValueError("This independent audit supports VB files only, not VD/VE multi-measurement files.")
        offset = length
        while True:
            stream.seek(offset)
            header = stream.read(128)
            if len(header) != 128:
                raise ValueError(f"Truncated MDH at byte {offset}.")
            dma = struct.unpack_from("<I", header)[0] & 0x1ffffff
            low, high = struct.unpack_from("<II", header, 20)
            if low & 1:
                break
            if dma < 128:
                raise ValueError(f"Invalid DMA length {dma} at byte {offset}.")
            if low & (1 << 5):
                offset += dma
                continue
            samples, channels = struct.unpack_from("<HH", header, 28)
            expected = channels * (128 + 8 * samples)
            if dma != expected:
                raise ValueError(f"Unsupported packed/padded scan at byte {offset}: DMA={dma}, channel payload total={expected}.")
            records.append({"offset": offset, "scan_counter": struct.unpack_from("<I", header, 8)[0],
                            "timestamp": struct.unpack_from("<I", header, 12)[0],
                            "eval_low": low, "eval_high": high, "samples": samples, "channels": channels,
                            "counters": struct.unpack_from("<14H", header, 32),
                            "cutoff": struct.unpack_from("<2H", header, 60),
                            "center_sample": struct.unpack_from("<H", header, 64)[0],
                            "readout_offcentre": struct.unpack_from("<f", header, 68)[0],
                            "time_since_last_rf": struct.unpack_from("<I", header, 72)[0],
                            "center_line_partition": struct.unpack_from("<2H", header, 76),
                            "ice": struct.unpack_from("<4H", header, 80),
                            "free": struct.unpack_from("<4H", header, 88),
                            "quaternion": struct.unpack_from("<4f", header, 108)})
            offset += dma
    return records


def raw_samples(stream, record):
    """Read every channel header and sample payload, checking channel identity."""
    values, channel_ids = [], []
    for channel in range(record["channels"]):
        stream.seek(record["offset"] + channel * (128 + 8 * record["samples"]))
        header = stream.read(128)
        if struct.unpack_from("<I", header, 8)[0] != record["scan_counter"]:
            raise ValueError("Channel header scan counters differ within a readout.")
        if struct.unpack_from("<HH", header, 28) != (record["samples"], record["channels"]):
            raise ValueError("Channel header dimensions differ within a readout.")
        channel_ids.append(struct.unpack_from("<H", header, 124)[0])
        values.append(np.frombuffer(stream.read(8 * record["samples"]), dtype="<c8").copy())
    if channel_ids != list(range(record["channels"])):
        raise ValueError(f"Channel ordering requires explicit handling: {channel_ids}.")
    return np.stack(values)


def is_imaging(record):
    excluded = sum(1 << bit for bit in (0, 1, 2, 5, 21, 25))
    return not (record["eval_low"] & excluded or record["eval_high"] & (1 << (51 - 32)))


def run_audit(raw_file, ismrmrd_file, comparison_dir):
    records = read_vb_headers(raw_file)
    by_scan = {record["scan_counter"]: record for record in records}
    if len(by_scan) != len(records):
        raise ValueError("Original raw scan counters are not unique.")
    with h5py.File(ismrmrd_file, "r") as handle:
        heads = handle["dataset/data"].fields("head")[:]
    if len(heads) != len(records):
        raise ValueError("Original and converted acquisition counts differ.")
    for head, record in zip(heads, records):
        if int(head["scan_counter"]) != record["scan_counter"] or int(head["acquisition_time_stamp"]) != record["timestamp"]:
            raise ValueError("Conversion reordered readouts or changed timestamps.")
        if int(head["number_of_samples"]) != record["samples"] or int(head["active_channels"]) != record["channels"]:
            raise ValueError("Converted sample/channel dimensions differ from raw headers.")
        names = ("kspace_encode_step_1", "average", "slice", "kspace_encode_step_2", "contrast", "phase", "repetition", "set", "segment")
        if tuple(int(head["idx"][name]) for name in names) != record["counters"][:9]:
            raise ValueError("Converted loop counters differ from raw headers.")
    print(f"Validated metadata for {len(records)} original/converted readouts", flush=True)
    imaging = [record for record in records if is_imaging(record)]
    starts = [0] + [i for i in range(1, len(imaging)) if imaging[i]["counters"][7] != imaging[i - 1]["counters"][7]]
    blocks = [imaging[a:b] for a, b in zip(starts, starts[1:] + [len(imaging)])]
    selected = {0: [], 1: []}
    for block in blocks:
        if len(block) != 179:
            raise ValueError(f"Observed block of {len(block)} readouts, rather than 179; investigate sequence structure.")
        first = block[:3]
        if any((record["counters"][0], record["counters"][3]) != record["center_line_partition"] for record in first):
            raise ValueError("The first three records of a set block are not all at declared central k-space.")
        selected[block[0]["counters"][7]].append(first)
    report = json.loads((comparison_dir / "comparison.json").read_text())
    max_error = 0.
    direct = {}
    dset = ReadOnlyDataset(str(ismrmrd_file))
    try:
        with raw_file.open("rb") as stream:
            for set_id, triplets in selected.items():
                expected = report["central_triplet_acquisition_indices"][str(set_id)]
                if len(expected) != len(triplets):
                    raise ValueError("Comparison triplet count differs from independent raw set-block selection.")
                for ordinal in range(3):
                    values = []
                    for triplet, indices in zip(triplets, expected):
                        record = triplet[ordinal]
                        acquisition = dset.read_acquisition(indices[ordinal])
                        if int(acquisition.scan_counter) != record["scan_counter"]:
                            raise ValueError("Comparison selected a different record or triplet order from the raw set block.")
                        original = raw_samples(stream, record)
                        error = float(np.max(np.abs(original - acquisition.data)))
                        max_error = max(max_error, error)
                        if error != 0:
                            raise ValueError(f"Converted complex samples differ from original bytes in scan {record['scan_counter']}.")
                        values.append(original)
                    direct[(set_id, ordinal)] = np.stack(values, axis=-1)
    finally:
        dset.close()
    summaries = {}
    for (set_id, ordinal), samples in direct.items():
        name = f"nav{set_id + 1}{'xyz'[ordinal]}"
        # Independently reconstruct the same central half-FOV profile from bytes.
        image = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(samples, axes=1), axis=1), axes=1)
        size = samples.shape[1] // 2
        start = samples.shape[1] // 2 - size // 2
        cropped = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(image[:, start:start + size], axes=1), axis=1), axes=1)
        profiles = np.abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(cropped, axes=1), axis=1), axes=1))
        profiles[..., :4] = profiles[..., 4:5]
        with np.load(comparison_dir / f"{name}.npz") as saved:
            profile_error = float(np.max(np.abs(profiles - saved["profiles"])))
            if profile_error != 0:
                raise ValueError(f"Saved profiles differ from independently processed original bytes: {name}.")
        magnitude = np.abs(samples[..., 4:])
        summaries[name] = {"profile_max_abs_error": profile_error,
                           "raw_magnitude_temporal_roughness": float(np.linalg.norm(np.diff(magnitude, axis=-1)) / np.linalg.norm(magnitude))}
    nav_bit = 1 << (ismrmrd.ACQ_IS_NAVIGATION_DATA - 1)
    flagged = [by_scan[int(head["scan_counter"])] for head in heads if int(head["flags"]) & nav_bit]
    metadata = {}
    for set_id, triplets in selected.items():
        for ordinal in range(3):
            chosen = [triplet[ordinal] for triplet in triplets]
            metadata[f"set{set_id}_ordinal{ordinal + 1}"] = {
                key: [list(value) if isinstance(value, tuple) else value for value in sorted({record[key] for record in chosen})]
                for key in ("eval_low", "eval_high", "cutoff", "center_sample", "readout_offcentre", "ice", "free", "quaternion")}
    result = {"original_raw_file": str(raw_file), "ismrmrd_file": str(ismrmrd_file),
              "raw_readouts": len(records), "metadata_matches_all_converted_readouts": True,
              "set_blocks": len(blocks), "set_block_lengths": dict(Counter(len(block) for block in blocks)),
              "triplets_per_set": {str(k): len(v) for k, v in selected.items()},
              "independent_raw_selection_matches_comparison": True,
              "compared_triplet_readouts": sum(3 * len(v) for v in selected.values()),
              "complex_sample_max_abs_error": max_error, "series": summaries,
              "selected_raw_metadata": metadata, "ismrmrd_flagged_navigation_records": len(flagged),
              "flagged_records_with_siemens_rtfeedback_bit": sum(bool(r["eval_low"] & (1 << 1)) for r in flagged),
              "flagged_records_with_siemens_vop_bit53": sum(bool(r["eval_high"] & (1 << (53 - 32))) for r in flagged),
              "limitations": ["First/second/third readout labels do not establish physical x/y/z directions.",
                              "Set-block structure and central repetitions support navigator selection but do not replace sequence source/gradient trajectory confirmation.",
                              "ISMRMRD navigation flags also mark Siemens RT-feedback/VOP data in this acquisition; they are not specific to motion navigation.",
                              "The cause of the first-set second-readout variability is not established by extraction checks."]}
    path = comparison_dir / "extraction_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("selected_raw_metadata", "series")}, indent=2), flush=True)
    print(f"Detailed audit saved to {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", required=True, type=Path)
    parser.add_argument("--ismrmrd-file", required=True, type=Path)
    parser.add_argument("--comparison-dir", type=Path, default=ROOT / "runs/mosar_navigator_comparison")
    args = parser.parse_args()
    run_audit(args.raw_file, args.ismrmrd_file, args.comparison_dir)

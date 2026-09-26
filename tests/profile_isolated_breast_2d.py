#!/usr/bin/env python
"""Profile one isolated 2D breast reconstruction on one CPU thread.

This is an opt-in reconstruction test. --dry-run checks inputs only; a normal
invocation reconstructs exactly one slice from an existing prepared HDF5 and
writes a new run, without touching prior Torch or C++ outputs.

Example:
    python tests/profile_isolated_breast_2d.py --dry-run
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/profile_isolated_breast_2d.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from pipelines.siemens_breast_T2 import run_pipeline
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator

DATASET = Path('/home/pyuser/wkdir/data/GRICS-torch/article_dataset_2D')
CPP_ROOT = Path('/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/GRICS-BELT')
TORCH_MEASUREMENTS = DATASET / 'results/measurements.json'
CONFIG = ROOT / 'config/reconstruction/nonrigid_2d_breast.toml'
OUTPUT = ROOT / 'tests/artifacts/isolated_cpu_timing'


def cpp_reconstruction_seconds(root: Path, subject: str, slice_number: int) -> tuple[Path, float]:
    path = root / subject / f'Siemens_SingleImage_slice{slice_number:03d}_image01/grics.log.0'
    log = path.read_text()
    values = re.findall(r'^Reconstruction time\s*=\s*([0-9.eE+-]+)\s*$', log, re.M)
    if len(values) != 1 or not math.isfinite(float(values[0])) or float(values[0]) <= 0:
        raise ValueError(f'{path}: expected one positive reconstruction time')
    return path, float(values[0])


def previous_parallel_timing(path: Path, subject: str, slice_number: int) -> dict | None:
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())
    matches = [row for row in rows
               if Path(row['subject']).stem == subject and row['mode'] == 'corrected']
    if len(matches) != 1:
        return None
    folder = Path(matches[0]['run_folder'])
    manifest_path = folder / 'manifest.json'
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    record = manifest['reconstructions'].get(f'slice_{slice_number:03d}')
    if record is None:
        return None
    return {
        'run_folder': str(folder),
        'slice_reconstruction_seconds': float(record['reconstruction_seconds']),
        'subject_compute_wall_seconds': float(matches[0]['compute_wall_seconds']),
        'subject_worker_count': int(manifest['max_workers']),
        'runtime_device': record['configuration']['runtime_device'],
    }


@contextmanager
def profile_normal_operators():
    """Time the complete normal-operator calls, grouped by spatial resolution."""
    samples = defaultdict(list)
    originals = []

    def instrument(cls, name):
        original = cls.normal

        @wraps(original)
        def timed(operator, vector):
            shape = operator.smaps.shape if name == 'image' else operator.SensitivityMaps.shape
            label = f'{int(shape[1])}x{int(shape[2])}'
            started = time.perf_counter()
            try:
                return original(operator, vector)
            finally:
                samples[(name, label)].append(time.perf_counter() - started)

        originals.append((cls, original))
        cls.normal = timed

    instrument(EncodingOperator, 'image')
    instrument(MotionPerturbationSimulator, 'motion')
    try:
        yield samples
    finally:
        for cls, original in originals:
            cls.normal = original


def summarize_samples(samples) -> dict:
    by_operator = {'image': {}, 'motion': {}}
    for (name, shape), durations in sorted(samples.items()):
        by_operator[name][shape] = {
            'calls': len(durations),
            'total_seconds': sum(durations),
            'mean_seconds': sum(durations) / len(durations),
            'max_seconds': max(durations),
        }
    return by_operator


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', default='0079_T2_m')
    parser.add_argument('--slice', type=int, default=15, help='One-based slice number.')
    parser.add_argument('--dataset-dir', type=Path, default=DATASET)
    parser.add_argument('--cpp-root', type=Path, default=CPP_ROOT)
    parser.add_argument('--torch-measurements', type=Path, default=None,
                        help='Saved Torch measurements; defaults to DATASET_DIR/results/measurements.json.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[0-9]{4}_T2_[a-z]', args.subject, re.I):
        parser.error('--subject must look like 0079_T2_m')
    if args.slice < 1:
        parser.error('--slice must be positive')
    prepared = args.dataset_dir / f'{args.subject}.h5'
    if not prepared.is_file():
        raise FileNotFoundError(f'{prepared}: this test never prepares raw data')
    cpp_log, cpp_seconds = cpp_reconstruction_seconds(
        args.cpp_root, args.subject, args.slice)
    measurements = (args.torch_measurements if args.torch_measurements is not None
                    else args.dataset_dir / 'results/measurements.json')
    prior = previous_parallel_timing(measurements, args.subject, args.slice)
    print(f'[input] {prepared}; slice {args.slice}', flush=True)
    print('[plan] CPU, one worker, one Torch thread; no raw preparation.', flush=True)
    print(f'[reference] {cpp_log}: {cpp_seconds:.3f} s', flush=True)
    if prior:
        print(f'[previous Torch] {prior["slice_reconstruction_seconds"]:.3f} s '
              f'with {prior["subject_worker_count"]} workers', flush=True)
    if args.dry_run:
        print('[dry-run] No reconstruction or output files written.', flush=True)
        return 0

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError('Could not set one inter-op thread for the isolated CPU test')
    if torch.get_num_threads() != 1:
        raise RuntimeError('Could not set one intra-op thread for the isolated CPU test')

    with profile_normal_operators() as samples:
        result = run_pipeline(
            preprocessed_file=prepared,
            output_root=args.output_dir / 'runs' / args.subject / 'corrected',
            reconstruction_config=CONFIG, device='cpu', max_workers=1,
            slice_start=args.slice - 1, slice_stop=args.slice,
            save_reconstruction_logs=True, save_reconstruction_tensors=False,
            return_tensors=False, export_dicom=False,
        )
    reconstructions = result['reconstructions']
    if len(reconstructions) != 1 or reconstructions[0]['slice_number'] != args.slice:
        raise ValueError('The isolated run did not reconstruct exactly the requested slice')
    seconds = float(reconstructions[0]['reconstruction_seconds'])
    run_folder = Path(result['run_folder'])
    report = {
        'subject': args.subject,
        'slice_number': args.slice,
        'prepared_file': str(prepared),
        'run_folder': str(run_folder),
        'torch_cpu_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'torch_worker_count': 1,
        'isolated_torch_solver_seconds': seconds,
        'cpp_solver_seconds': cpp_seconds,
        'isolated_torch_over_cpp': seconds / cpp_seconds,
        'previous_parallel_torch': prior,
        'operator_normal_calls': summarize_samples(samples),
        'scope': 'single prepared slice, CPU only; no raw preparation or prior-run overwrite',
    }
    if prior:
        report['parallel_torch_over_isolated'] = (
            prior['slice_reconstruction_seconds'] / seconds)
    destination = run_folder / 'isolated_cpu_profile.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'[result] isolated Torch {seconds:.3f} s; C++ {cpp_seconds:.3f} s', flush=True)
    if prior:
        print(f'[result] parallel/isolated Torch: '
              f'{report["parallel_torch_over_isolated"]:.3f}x', flush=True)
    print(f'[result] {destination}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

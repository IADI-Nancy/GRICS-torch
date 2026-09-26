#!/usr/bin/env python
"""Sweep CPU worker counts for the same corrected 2D breast slices.

Opt-in reconstruction test: --dry-run validates inputs without creating output.
Uses only an existing prepared HDF5; does not prepare raw data, export images,
or overwrite previous runs. Saves a checkpointed JSON report after each run.

    python tests/profile_breast_2d_worker_sweep.py --dry-run
    python tests/profile_breast_2d_worker_sweep.py --workers 60 48 32 16 8
    python tests/profile_breast_2d_worker_sweep.py --workers 60 32 16 --repeats 2

C++ values in the report are per-slice solver times, NOT C++ batch wall time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import statistics
import sys
import time
from uuid import uuid4

# Limit native thread pools before importing NumPy/PyTorch or forking workers.
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import torch

from pipelines import siemens_breast_T2 as pipeline

DATASET = Path('/home/pyuser/wkdir/data/GRICS-torch/article_dataset_2D')
CPP_MEASUREMENTS = ROOT / 'article/results_grics_cpp_2d/measurements.json'
CONFIG = ROOT / 'config/reconstruction/nonrigid_2d_breast.toml'
OUTPUT = ROOT / 'tests/artifacts/breast_2d_worker_sweep'
_original_reconstruct_slice = pipeline.reconstruct_slice


def measured_reconstruct_slice(slice_idx, source=None):
    """Instrument one slice without changing its reconstruction."""
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    result = _original_reconstruct_slice(slice_idx, source)
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    result['worker_wall_seconds'] = time.perf_counter() - wall_start
    result['worker_cpu_seconds'] = time.process_time() - cpu_start
    result['voluntary_context_switches'] = usage_end.ru_nvcsw - usage_start.ru_nvcsw
    result['involuntary_context_switches'] = usage_end.ru_nivcsw - usage_start.ru_nivcsw
    return result


def cpp_reference(path: Path, subject: str, numbers: list[int]) -> dict | None:
    """Read saved C++ slice times without launching C++."""
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())
    matches = [row for row in rows
               if row.get('subject') == subject and row.get('mode') == 'corrected']
    if len(matches) != 1:
        raise ValueError(f'{path}: expected one corrected row for {subject}')
    by_number = {int(item['slice_number']): float(item['reconstruction_seconds'])
                 for item in matches[0]['slices']}
    if any(number not in by_number for number in numbers):
        raise ValueError(f'{path}: missing selected C++ slice times')
    selected = {str(number): by_number[number] for number in numbers}
    return {'measurements_file': str(path.resolve()),
            'slice_solver_seconds': selected,
            'max_slice_solver_seconds': max(selected.values())}


def save_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def summarize(result: dict, numbers: list[int], cpp: dict | None) -> dict:
    by_number = {int(item['slice_number']): item for item in result['reconstructions']}
    if sorted(by_number) != numbers:
        raise ValueError(f'Expected slices {numbers}, got {sorted(by_number)}')
    slices = []
    for number in numbers:
        item = by_number[number]
        solver = float(item['reconstruction_seconds'])
        row = {
            'slice_number': number,
            'solver_seconds': solver,
            'worker_wall_seconds': float(item['worker_wall_seconds']),
            'worker_cpu_seconds': float(item['worker_cpu_seconds']),
            'cpu_over_wall': float(item['worker_cpu_seconds'] / item['worker_wall_seconds']),
            'voluntary_context_switches': int(item['voluntary_context_switches']),
            'involuntary_context_switches': int(item['involuntary_context_switches']),
        }
        if cpp:
            row['cpp_solver_seconds'] = cpp['slice_solver_seconds'][str(number)]
            row['torch_over_cpp_solver'] = solver / row['cpp_solver_seconds']
        slices.append(row)
    solvers = [row['solver_seconds'] for row in slices]
    wall = float(result['timings']['compute_wall_seconds'])
    manifest = json.loads((result['run_folder'] / 'manifest.json').read_text())
    return {
        'run_folder': str(result['run_folder']),
        'actual_workers': int(manifest['max_workers']),
        'slice_count': len(slices),
        'compute_wall_seconds': wall,
        'pipeline_seconds': float(result['timings']['pipeline_seconds']),
        'load_seconds': float(result['timings']['load_seconds']),
        'solver_sum_seconds': float(result['timings']['reconstruction_seconds_sum']),
        'max_slice_solver_seconds': max(solvers),
        'median_slice_solver_seconds': statistics.median(solvers),
        'slices_per_compute_second': len(slices) / wall,
        'median_slice_cpu_over_wall': statistics.median(row['cpu_over_wall'] for row in slices),
        'total_involuntary_context_switches':
            sum(row['involuntary_context_switches'] for row in slices),
        'slices': slices,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', default='0079_T2_m')
    parser.add_argument('--dataset-dir', type=Path, default=DATASET)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--cpp-measurements', type=Path, default=CPP_MEASUREMENTS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--slice-start', type=int, default=0,
                        help='Zero-based inclusive; default first slice.')
    parser.add_argument('--slice-stop', type=int, default=None,
                        help='Zero-based exclusive; default all slices.')
    parser.add_argument('--workers', type=int, nargs='+', default=[60, 48, 32, 16, 8])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    if not re.fullmatch(r'[0-9]{4}_T2_[a-z]', args.subject, re.I):
        parser.error('--subject must look like 0079_T2_m')
    if args.repeats < 1 or not args.workers or any(count < 1 for count in args.workers):
        parser.error('--workers and --repeats must be positive')
    if len(set(args.workers)) != len(args.workers):
        parser.error('--workers must be distinct; use --repeats instead')
    prepared = args.dataset_dir / f'{args.subject}.h5'
    if not prepared.is_file():
        raise FileNotFoundError(f'{prepared}: this test never prepares raw data')
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    with h5py.File(prepared, 'r') as handle:
        if 'kspace' not in handle or len(handle['kspace'].shape) != 5:
            raise ValueError(f'{prepared}: expected multislice 2D kspace')
        total_slices = int(handle['kspace'].shape[-1])
    stop = total_slices if args.slice_stop is None else args.slice_stop
    if args.slice_start < 0 or stop <= args.slice_start or stop > total_slices:
        parser.error(f'Invalid slice range [{args.slice_start}, {stop}) for {total_slices} slices')
    numbers = list(range(args.slice_start + 1, stop + 1))
    if max(args.workers) > len(numbers):
        parser.error('Worker count exceeds selected slices and would be silently capped')
    cpp = cpp_reference(args.cpp_measurements, args.subject, numbers)
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    print(f'[input] {prepared}; slices {numbers[0]}-{numbers[-1]} ({len(numbers)})', flush=True)
    print(f'[plan] CPU, one Torch thread per worker; workers={args.workers}; '
          f'repeats={args.repeats}; affinity CPUs={available_cpus}', flush=True)
    print('[plan] Prepared HDF5 only; no raw preparation or image/DICOM exports.', flush=True)
    if cpp:
        print(f'[reference] C++ max selected-slice solver: '
              f'{cpp["max_slice_solver_seconds"]:.3f} s (not batch wall time)', flush=True)
    if args.dry_run:
        print('[dry-run] No reconstruction or output files written.', flush=True)
        return 0

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pipeline.reconstruct_slice = measured_reconstruct_slice
    group = (args.output_dir / args.subject /
             (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '-' + uuid4().hex[:8]))
    group.mkdir(parents=True, exist_ok=False)
    report_path = group / 'worker_sweep.json'
    report = {
        'subject': args.subject,
        'prepared_file': str(prepared.resolve()),
        'config_file': str(args.config.resolve()),
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'slice_numbers': numbers,
        'worker_counts_in_order': args.workers,
        'repeats': args.repeats,
        'affinity_logical_cpus': available_cpus,
        'library_threads_per_worker': 1,
        'cpp_reference': cpp,
        'scope': 'same prepared slices at each count; CPU only; per-slice and batch timers',
        'runs': [],
    }
    save_report(report_path, report)
    try:
        for repeat in range(1, args.repeats + 1):
            # Reverse order on alternate repeats to reveal warm-cache effects.
            order = args.workers if repeat % 2 else list(reversed(args.workers))
            for workers in order:
                print(f'[run] repeat {repeat}/{args.repeats}; {workers} workers', flush=True)
                result = pipeline.run_pipeline(
                    preprocessed_file=prepared,
                    output_root=group / 'runs' / f'workers_{workers:02d}_repeat_{repeat:02d}',
                    reconstruction_config=args.config,
                    device='cpu', max_workers=workers,
                    slice_start=args.slice_start, slice_stop=stop,
                    save_reconstruction_logs=True, save_reconstruction_tensors=False,
                    return_tensors=False, export_dicom=False,
                )
                row = summarize(result, numbers, cpp)
                row['requested_workers'] = workers
                row['repeat'] = repeat
                report['runs'].append(row)
                save_report(report_path, report)
                print(f'[result] workers={workers}: max solver={row["max_slice_solver_seconds"]:.2f} s, '
                      f'median solver={row["median_slice_solver_seconds"]:.2f} s, '
                      f'compute wall={row["compute_wall_seconds"]:.2f} s, '
                      f'median CPU/wall={row["median_slice_cpu_over_wall"]:.2f}', flush=True)
                del result
                gc.collect()
    finally:
        pipeline.reconstruct_slice = _original_reconstruct_slice
    print(f'[report] {report_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

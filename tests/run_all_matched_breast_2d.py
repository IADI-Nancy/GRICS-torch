#!/usr/bin/env python
"""Reconstruct every 2D breast slice with a complete matching reference output.

Only corrected reconstructions are run. Prepared HDF5 inputs are reused;
neither raw-data preparation nor previous reconstruction runs are touched.
Results are checkpointed after each subject for safe resumption.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from fnmatch import fnmatchcase
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from article.evaluate_breast_2d import score_image
from pipelines.siemens_breast_T2 import run_pipeline

DATASET = Path('/home/pyuser/wkdir/data/GRICS-torch/article_dataset_2D')
CPP_ROOT = Path('/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/GRICS-BELT')
CPP_MEASUREMENTS = ROOT / 'article/results_grics_cpp_2d/measurements.json'
CONFIG = ROOT / 'config/reconstruction/nonrigid_2d_breast.toml'
OUTPUT = ROOT / 'tests/artifacts/all_subject_cpp_matched_2d'
SUBJECT_PATTERN = '????_T2_?'


def reference_inventory(cpp_root: Path, measurements: Path, dataset: Path):
    rows = json.loads(measurements.read_text())
    if not isinstance(rows, list):
        raise ValueError(f'{measurements}: expected a list')
    corrected = {}
    for row in rows:
        subject = row['subject']
        if row['mode'] != 'corrected' or not fnmatchcase(subject, SUBJECT_PATTERN):
            continue
        if subject in corrected:
            raise ValueError(f'Duplicate corrected reference for {subject}')
        slices = row['slices']
        numbers = sorted(item['slice_number'] for item in slices)
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f'{subject}: reference slices must start at 1 and be contiguous')
        if any(not Path(item['image_file']).is_file() for item in slices):
            raise FileNotFoundError(f'{subject}: reference image export is missing')
        if not (dataset / f'{subject}.h5').is_file():
            raise FileNotFoundError(f'{subject}: prepared Torch input is missing')
        corrected[subject] = row
    if not corrected:
        raise ValueError('No complete corrected reference acquisitions found')
    excluded = sorted(
        folder.name for folder in cpp_root.iterdir()
        if folder.is_dir() and fnmatchcase(folder.name, SUBJECT_PATTERN)
        and folder.name not in corrected
    )
    return corrected, excluded


def save_report(path: Path, report: dict):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(path)


def completed(row: dict, expected_count: int) -> bool:
    slices = row.get('slices', [])
    return (
        len(slices) == expected_count
        and sorted(item['slice_number'] for item in slices) == list(range(1, expected_count + 1))
        and all(Path(item['native_image_file']).is_file() and
                Path(item['image_file']).is_file() for item in slices)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=DATASET)
    parser.add_argument('--cpp-root', type=Path, default=CPP_ROOT)
    parser.add_argument('--cpp-measurements', type=Path, default=CPP_MEASUREMENTS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--subject', action='append', help='Optional subject filter; repeatable.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help='Rerun matched subjects and replace their report entries; old timestamped runs remain.')
    args = parser.parse_args(argv)

    references, excluded = reference_inventory(
        args.cpp_root, args.cpp_measurements, args.dataset_dir)
    if args.subject:
        unknown = set(args.subject) - set(references)
        if unknown:
            parser.error(f'No complete matched reference for: {sorted(unknown)}')
        references = {name: references[name] for name in sorted(set(args.subject))}
    total_slices = sum(row['slice_count'] for row in references.values())
    print(f'[plan] {len(references)} subjects, {total_slices} corrected slices on GPU', flush=True)
    for name, row in references.items():
        print(f'[match] {name}: slices 1-{row["slice_count"]}', flush=True)
    print(f'[excluded] no complete scored reference: {excluded}', flush=True)
    if args.dry_run:
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no CPU fallback for this batch')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required_bytes = max(2 * 1024**3, total_slices * 10_000_000)
    free_bytes = shutil.disk_usage(args.output_dir).free
    if free_bytes < required_bytes:
        raise OSError(f"Insufficient free space for this batch: {free_bytes / 1024**3:.1f} GiB "
                      f"available, at least {required_bytes / 1024**3:.1f} GiB required.")
    report_path = args.output_dir / 'comparison.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get('configuration') != str(CONFIG.resolve()):
            raise ValueError(f'{report_path}: configuration path differs from this batch')
    else:
        report = {
            'configuration': str(CONFIG.resolve()),
            'reference_measurements': str(args.cpp_measurements.resolve()),
            'subject_pattern': SUBJECT_PATTERN,
            'mode': 'corrected',
            'excluded_reference_subjects': excluded,
            'subjects': {},
            'failures': {},
        }
    failed_this_run = set()
    for subject, reference in references.items():
        previous = report['subjects'].get(subject)
        if previous and not args.overwrite and completed(previous, reference['slice_count']):
            print(f'[resume] {subject}: already complete', flush=True)
            continue
        print(f'[reconstruct] {subject}: {reference["slice_count"]} slices', flush=True)
        try:
            result = run_pipeline(
                preprocessed_file=args.dataset_dir / f'{subject}.h5',
                output_root=args.output_dir / 'runs' / subject / 'corrected',
                reconstruction_config=CONFIG, device='gpu', max_workers=1,
                slice_start=0, slice_stop=reference['slice_count'],
                save_reconstruction_logs=True, save_reconstruction_tensors=True,
                return_tensors=True, export_dicom=False,
            )
            cpp_by_slice = {item['slice_number']: item for item in reference['slices']}
            slices = []
            for item in result['reconstructions']:
                number = item['slice_number']
                native_path = Path(item['output_dir']) / 'image_native.pt'
                torch.save(item['native_image'], native_path)
                slices.append({
                    'slice_number': number,
                    'native_sharpness': score_image(item['native_image']),
                    'reference_sharpness': float(cpp_by_slice[number]['sharpness']),
                    'reconstruction_seconds': float(item['reconstruction_seconds']),
                    'native_image_file': str(native_path),
                    'image_file': str(item['image_file']),
                    'motion_file': str(item['motion_file']),
                })
            if sorted(item['slice_number'] for item in slices) != list(
                range(1, reference['slice_count'] + 1)
            ):
                raise ValueError(f'{subject}: reconstructed slice inventory differs from reference')
            report['subjects'][subject] = {
                'run_folder': str(result['run_folder']),
                'slice_count': len(slices),
                'mean_native_sharpness': sum(item['native_sharpness'] for item in slices) / len(slices),
                'mean_reference_sharpness': float(reference['mean_sharpness']),
                'reconstruction_seconds_sum': float(result['timings']['reconstruction_seconds_sum']),
                'slices': slices,
            }
            report['failures'].pop(subject, None)
            save_report(report_path, report)
            print(f'[complete] {subject}: {len(slices)} slices; report {report_path}', flush=True)
        except Exception as error:
            failed_this_run.add(subject)
            report['failures'][subject] = f'{type(error).__name__}: {error}'
            save_report(report_path, report)
            print(f'[failed] {subject}: {error}', flush=True)
        finally:
            if 'result' in locals():
                del result
            gc.collect()
            torch.cuda.empty_cache()
    remaining = [name for name, row in references.items()
                 if not completed(report['subjects'].get(name, {}), row['slice_count'])]
    print(f'[summary] completed {len(references) - len(remaining)}/{len(references)} subjects; '
          f'remaining={remaining}', flush=True)
    return 1 if remaining or failed_this_run else 0


if __name__ == '__main__':
    raise SystemExit(main())

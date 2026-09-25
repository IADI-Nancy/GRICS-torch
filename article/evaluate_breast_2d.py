#!/usr/bin/env python
"""Evaluate 00XX_T2_Y acquisitions with configured 2D GRICS and one-state reconstruction."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from article.sharpness_index import sharpness_index
from pipelines.siemens_breast_T2 import RECONSTRUCTION_CONFIG, run_pipeline
from src.preprocessing.RawDataPreparer import RawDataPreparer
from src.preprocessing.MotionBinner import ConstantPhysiologicalSignalError
from src.runtime.hdf5_cache import write_tree
from src.runtime.runtime_config import load_config

DATABASE = Path('/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database')
DATASET = Path('/home/pyuser/wkdir/data/GRICS-torch/article_dataset_2D')


def prepare_subject(raw: Path, saec: Path, destination: Path, params) -> Path:
    """Reuse a validated prepared acquisition or atomically cache a new one."""
    if not destination.exists():
        preparer = RawDataPreparer(
            str(raw), str(saec), physiological_format='SAEC',
            sensor_type=params.rawdata_sensor_type, device='cpu',
            print_raw_calibration_lines=params.print_raw_calibration_lines,
            polaris_channel_mode=None)
        arrays = preparer.read_data(cache_h5=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.h5', delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            with h5py.File(temporary, 'w') as handle:
                write_tree(handle, arrays)
                handle.attrs['source_ismrmrd'] = str(raw.resolve())
                handle.attrs['source_saec'] = str(saec.resolve())
                handle.attrs['sensor_type'] = params.rawdata_sensor_type
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    with h5py.File(destination, 'r') as handle:
        for key in ('kspace', 'motion_data', 'idx_ky', 'idx_kz', 'idx_nex', 'ismrmrd_header'):
            if key not in handle:
                raise ValueError(f'{destination}: missing {key}; repair or remove this prepared file.')
        shape = handle['kspace'].shape
        if len(shape) != 5 or shape[-1] < 1:
            raise ValueError(f'{destination}: expected a multislice 2D acquisition, got {shape}.')
        sensor = handle.attrs.get('sensor_type')
        if sensor is not None and sensor != params.rawdata_sensor_type:
            raise ValueError(f'{destination}: cached sensor {sensor} differs from configured {params.rawdata_sensor_type}.')
    return destination


def score_image(image: torch.Tensor) -> float:
    """Complex-average repetitions and score one native-grid 2D image."""
    image = torch.as_tensor(image)
    magnitude = image.mean(dim=0).abs() if image.ndim == 3 else image.abs()
    score = float(sharpness_index(magnitude))
    if not np.isfinite(score):
        raise ValueError('Non-finite sharpness index; results are not silently discarded.')
    return score


def save_results(rows, folder: Path) -> None:
    """Checkpoint measurements and regenerate paired acquisition-level plots."""
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / 'measurements.json.tmp'
    temporary.write_text(json.dumps(rows, indent=2, allow_nan=False))
    temporary.replace(folder / 'measurements.json')
    fields = ['subject', 'mode', 'motion_states', 'slice_count', 'mean_sharpness',
              'reconstruction_seconds_sum', 'compute_wall_seconds', 'run_folder']
    with (folder / 'measurements.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    paired = [name for name in dict.fromkeys(row['subject'] for row in rows)
              if {row['mode'] for row in rows if row['subject'] == name} == {'corrected', 'nomoco'}]
    if not paired:
        return
    for metric, ylabel, filename in (
        ('mean_sharpness', 'Mean slice sharpness index', 'sharpness_boxplot.png'),
        ('reconstruction_seconds_sum', 'Sum of per-slice reconstruction times (s)', 'time_boxplot.png'),
    ):
        values = [[next(row[metric] for row in rows if row['subject'] == subject and row['mode'] == mode)
                   for subject in paired] for mode in ('corrected', 'nomoco')]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot(values)
        ax.set_xticks([1, 2], ['Motion corrected', 'No motion correction'])
        for corrected, nomoco in zip(*values):
            ax.plot([1, 2], [corrected, nomoco], 'o-', color='gray', alpha=.4, markersize=3)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Breast 2D: {len(paired)} paired acquisitions')
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=200)
        plt.close(fig)


def evaluate(ismrmrd_dir=DATABASE/'ISMRMRD', saec_dir=DATABASE/'SAEC', dataset_dir=DATASET,
             reconstruction_config=ROOT/RECONSTRUCTION_CONFIG, device='cpu', max_workers=None,
             resume=True, modes=('corrected', 'nomoco'), rerun=False):
    modes = tuple(modes)
    if not modes or any(mode not in ('corrected', 'nomoco') for mode in modes):
        raise ValueError("modes must contain 'corrected', 'nomoco', or both.")
    ismrmrd_dir, saec_dir, dataset_dir = map(Path, (ismrmrd_dir, saec_dir, dataset_dir))
    config_path = Path(reconstruction_config).resolve()
    params = load_config(
        data_type='ismrmrd-saec', reconstruction_config=config_path,
        coil_sensitivity_config=ROOT/'config/coil_sensitivity/odille_spline.toml',
        real_data_config=ROOT/'config/real_data/saec.toml',
        ismrmrd_reader_config=ROOT/'config/real_data/ismrmrd_reader.toml')
    if params.reconstruction_dimension != '2D' or params.N_motion_states <= 1:
        raise ValueError('The corrected config must be 2D with N_motion_states > 1; config values are not replaced.')
    subjects = sorted(path for path in ismrmrd_dir.iterdir()
                      if path.is_file() and re.fullmatch(r'00[0-9]{2}_T2_[A-Z]', path.stem.upper())
                      and path.suffix.lower() in {'.h5', '.mrd'})
    if not subjects:
        raise ValueError(f'No 00XX_T2_Y acquisitions found in {ismrmrd_dir}.')
    missing = [path for path in subjects if not (saec_dir / path.name).is_file()]
    if missing:
        print('[skip] missing matching SAEC files:\n' + '\n'.join(str(saec_dir / path.name) for path in missing),
              flush=True)
        subjects = [path for path in subjects if path not in missing]
    if not subjects:
        raise FileNotFoundError(f'No T2 acquisitions with matching SAEC files in {ismrmrd_dir}.')
    results = dataset_dir / 'results'
    results.mkdir(parents=True, exist_ok=True)
    def previous_records(filename):
        path = results / filename
        records = json.loads(path.read_text()) if resume and path.exists() else []
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise ValueError(f'Invalid checkpoint: {path}')
        return records

    rows = previous_records('measurements.json')
    selected_names = {f'{path.stem}.h5' for path in subjects}
    rows = [row for row in rows if row['subject'] in selected_names]
    # Older checkpoints scored the postprocessed (zero-filled) image. They
    # cannot be mixed with native-grid scores, so recreate selected measurements.
    rows = [row for row in rows
            if row['mode'] not in modes or row.get('sharpness_image_stage') == 'native_solver_image']
    if rerun:
        rows = [row for row in rows if row['mode'] not in modes]
    completed = {(row['subject'], row['mode']) for row in rows}
    if len(completed) != len(rows):
        raise ValueError('Duplicate subject/mode measurements in checkpoint.')
    constant_signals = previous_records('excluded_constant_physiological_signal.json')
    constant_signal_file = results / 'excluded_constant_physiological_signal.json'
    constant_signal_file.write_text(json.dumps(constant_signals, indent=2))
    (results / 'excluded_missing_saec.txt').write_text(
        ''.join(f'{path.name}\n' for path in missing))
    excluded = previous_records('excluded_missing_respiratory_data.json')
    excluded_file = results / 'excluded_missing_respiratory_data.json'
    excluded_file.write_text(json.dumps(excluded, indent=2))
    truncated = previous_records('excluded_truncated_files.json')
    truncated_file = results / 'excluded_truncated_files.json'
    truncated_file.write_text(json.dumps(truncated, indent=2))
    skipped = {row['subject'] for row in excluded + truncated + constant_signals}
    prepared = []
    for raw in subjects:
        name = f'{raw.stem}.h5'
        if raw.name in skipped or name in skipped:
            print(f'[resume] {raw.name}: previously excluded', flush=True)
            continue
        if all((name, mode) in completed for mode in modes):
            print(f'[resume] {raw.name}: selected modes complete', flush=True)
            continue
        print(f'[prepare] {raw.name}', flush=True)
        try:
            subject = prepare_subject(raw, saec_dir / raw.name, dataset_dir / f'{raw.stem}.h5', params)
        except ValueError as error:
            if not str(error).startswith('No respiratory data found in SAEC file '):
                raise
            print(f'[skip] {raw.name}: {error}', flush=True)
            excluded.append(dict(subject=raw.name, reason=str(error)))
            excluded_file.write_text(json.dumps(excluded, indent=2))
            continue
        except OSError as error:
            if 'truncated file:' not in str(error).lower():
                raise
            print(f'[skip] {raw.name}: {error}', flush=True)
            truncated.append(dict(subject=raw.name, reason=str(error)))
            truncated_file.write_text(json.dumps(truncated, indent=2))
            continue
        prepared.append(subject)
    for subject in prepared:
        for mode in modes:
            if (subject.name, mode) in completed:
                print(f'[resume] {subject.name}: {mode} complete', flush=True)
                continue
            print(f'[reconstruct] {subject.name}: {mode}', flush=True)
            overrides = {} if mode == 'corrected' else {
                'N_motion_states': 1, 'N_motion_states_per_level': 'full'}
            try:
                result = run_pipeline(
                    subject, output_root=results/'runs'/subject.stem/mode,
                    reconstruction_config=config_path, device=device, max_workers=max_workers,
                    overrides=overrides, save_reconstruction_logs=True, return_tensors=True,
                    export_dicom=False)
            except ConstantPhysiologicalSignalError as error:
                print(f'[skip] {subject.name}: {error}', flush=True)
                constant_signals.append(dict(subject=subject.name, mode=mode, reason=str(error)))
                constant_signal_file.write_text(json.dumps(constant_signals, indent=2))
                # Exclude the whole acquisition from this paired comparison.
                rows = [row for row in rows if row['subject'] != subject.name]
                save_results(rows, results)
                break
            slice_scores = [score_image(item['native_image']) for item in result['reconstructions']]
            resolved = json.loads((result['run_folder']/'config_resolved.json').read_text())
            rows.append(dict(
                subject=subject.name, mode=mode, motion_states=resolved['N_motion_states'],
                slice_count=len(slice_scores), mean_sharpness=float(np.mean(slice_scores)),
                slice_sharpness=slice_scores,
                sharpness_image_stage='native_solver_image',
                reconstruction_seconds_sum=result['timings']['reconstruction_seconds_sum'],
                compute_wall_seconds=result['timings']['compute_wall_seconds'],
                run_folder=str(result['run_folder']), timings=result['timings']))
            save_results(rows, results)
            del result
    save_results(rows, results)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ismrmrd-dir', type=Path, default=DATABASE/'ISMRMRD')
    parser.add_argument('--saec-dir', type=Path, default=DATABASE/'SAEC')
    parser.add_argument('--dataset-dir', type=Path, default=DATASET)
    parser.add_argument('--reconstruction-config', type=Path, default=ROOT/RECONSTRUCTION_CONFIG)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True,
                        help='Resume saved measurements and exclusions (default); --no-resume starts fresh.')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Fixed worker count for comparable 2D timing; default uses available CPUs.')
    parser.add_argument('--modes', nargs='+', choices=('corrected', 'nomoco'),
                        default=('corrected', 'nomoco'), help='Modes to evaluate (default: both).')
    parser.add_argument('--rerun', action='store_true',
                        help='Discard completed measurements for the selected modes and reconstruct them again.')
    evaluate(**vars(parser.parse_args(argv)))


if __name__ == '__main__':
    main()

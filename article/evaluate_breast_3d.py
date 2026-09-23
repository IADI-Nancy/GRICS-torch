#!/usr/bin/env python
"""Prepare all T1 acquisitions and compare configured 3D GRICS against one-state reconstruction."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from article.sharpness_index import sharpness_index
from pipelines.siemens_breast_3d_lowres import RECONSTRUCTION_CONFIG, run_pipeline
from src.preprocessing.RawDataPreparer import RawDataPreparer
from src.runtime.hdf5_cache import write_tree, read_tree
from src.runtime.runtime_config import load_config

DATABASE = Path('/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database')
DATASET = Path('/home/pyuser/wkdir/data/GRICS-torch/article_dataset_3D')


def prepare_subject(raw: Path, saec: Path, destination: Path, params):
    """Reuse an existing prepared volume, otherwise publish a complete HDF5 atomically."""
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
        for key in ('kspace', 'motion_data', 'idx_ky', 'idx_kz', 'idx_nex'):
            if key not in handle:
                raise ValueError(f'{destination}: missing {key}; repair or remove this prepared file.')
        shape = handle['kspace'].shape
        if len(shape) != 5 or shape[-1] <= 1:
            raise ValueError(f'{destination}: expected a 3D volume, got {shape}.')
        sensor = handle.attrs.get('sensor_type')
        if sensor is not None and sensor != params.rawdata_sensor_type:
            raise ValueError(f'{destination}: cached sensor {sensor} differs from configured {params.rawdata_sensor_type}.')
        if 'slice_geometry' in handle:
            geometry = next(iter(read_tree(handle['slice_geometry']).values()))
            direction = np.asarray(geometry['slice_dir'], dtype=float)
        else:
            # Legacy article files lack geometry; obtain only the direction from the source.
            import ismrmrd
            source = ismrmrd.Dataset(str(raw), 'dataset', create_if_needed=False)
            try:
                direction = np.asarray(source.read_acquisition(0).slice_dir, dtype=float)
            finally:
                source.close()
    # Native axial (including mildly oblique axial) partitions; do not silently score sagittal slices.
    if not np.isfinite(direction).all() or abs(direction[2]) <= max(abs(direction[0]), abs(direction[1])):
        raise ValueError(f'{raw}: partitions are not axial; resampling would be required.')
    return destination


def score_volume(image):
    """Complex-average repetitions, then score magnitude of every native axial partition."""
    magnitude = image.detach().cpu().mean(dim=0).abs()
    scores = [float(sharpness_index(magnitude[..., z])) for z in range(magnitude.shape[-1])]
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite sharpness index; results are not silently discarded.')
    return scores


def save_results(rows, folder):
    """Checkpoint measurements and regenerate subject-level comparison plots."""
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / 'measurements.json.tmp'
    temporary.write_text(json.dumps(rows, indent=2, allow_nan=False))
    temporary.replace(folder / 'measurements.json')
    fields = ['subject', 'mode', 'motion_states', 'mean_sharpness', 'reconstruction_seconds', 'run_folder']
    with (folder / 'measurements.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    paired = [name for name in dict.fromkeys(r['subject'] for r in rows)
              if {r['mode'] for r in rows if r['subject'] == name} == {'corrected', 'nomoco'}]
    if not paired:
        return
    for metric, ylabel, filename in (
        ('mean_sharpness', 'Mean axial-slice sharpness index', 'sharpness_boxplot.png'),
        ('reconstruction_seconds', 'Reconstruction time including logs (s)', 'time_boxplot.png'),
    ):
        values = [[next(r[metric] for r in rows if r['subject'] == name and r['mode'] == mode)
                   for name in paired] for mode in ('corrected', 'nomoco')]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot(values)
        ax.set_xticks([1, 2], ['Motion corrected', 'No motion correction'])
        for a, b in zip(*values):
            ax.plot([1, 2], [a, b], 'o-', color='gray', alpha=.4, markersize=3)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Breast 3D: {len(paired)} paired acquisitions')
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=200)
        plt.close(fig)


def evaluate(ismrmrd_dir=DATABASE/'ISMRMRD', saec_dir=DATABASE/'SAEC', dataset_dir=DATASET,
             reconstruction_config=ROOT/RECONSTRUCTION_CONFIG, device='gpu'):
    ismrmrd_dir, saec_dir, dataset_dir = map(Path, (ismrmrd_dir, saec_dir, dataset_dir))
    config_path = Path(reconstruction_config).resolve()
    params = load_config(data_type='ismrmrd-saec', reconstruction_config=config_path,
        coil_sensitivity_config=ROOT/'config/coil_sensitivity/odille_spline.toml',
        real_data_config=ROOT/'config/real_data/saec.toml',
        ismrmrd_reader_config=ROOT/'config/real_data/ismrmrd_reader.toml')
    if params.reconstruction_dimension != '3D' or params.N_motion_states <= 1:
        raise ValueError('The corrected config must be 3D with N_motion_states > 1; config values are not replaced.')
    subjects = sorted(p for p in ismrmrd_dir.iterdir()
                      if p.is_file() and '_T1_' in p.stem.upper() and p.suffix.lower() in {'.h5', '.mrd'})
    if not subjects:
        raise ValueError(f'No T1 acquisitions found in {ismrmrd_dir}.')
    missing = [str(saec_dir / p.name) for p in subjects if not (saec_dir / p.name).is_file()]
    if missing:
        raise FileNotFoundError('Missing matching SAEC files:\n' + '\n'.join(missing))
    prepared = []
    for raw in subjects:
        print(f'[prepare] {raw.name}', flush=True)
        prepared.append(prepare_subject(raw, saec_dir/raw.name, dataset_dir/(raw.stem+'.h5'), params))
    rows = []
    results = dataset_dir/'results'
    for subject in prepared:
        for mode in ('corrected', 'nomoco'):
            print(f'[reconstruct] {subject.name}: {mode}', flush=True)
            # Preserve all solver settings; a per-level state schedule must also collapse to one.
            overrides = {} if mode == 'corrected' else {'N_motion_states': 1, 'N_motion_states_per_level': 'full'}
            result = run_pipeline(subject, output_root=results/'runs'/subject.stem/mode,
                reconstruction_config=config_path, device=device, overrides=overrides,
                save_reconstruction_logs=True, return_tensors=True, export_dicom=False)
            scores = score_volume(result['reconstructions'][0]['image'])
            resolved = json.loads((result['run_folder']/'config_resolved.json').read_text())
            rows.append(dict(subject=subject.name, mode=mode, motion_states=resolved['N_motion_states'],
                mean_sharpness=float(np.mean(scores)), slice_sharpness=scores,
                reconstruction_seconds=result['timings']['reconstruction_seconds'],
                run_folder=str(result['run_folder']), timings=result['timings']))
            save_results(rows, results)
            del result
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ismrmrd-dir', type=Path, default=DATABASE/'ISMRMRD')
    parser.add_argument('--saec-dir', type=Path, default=DATABASE/'SAEC')
    parser.add_argument('--dataset-dir', type=Path, default=DATASET)
    parser.add_argument('--reconstruction-config', type=Path, default=ROOT/RECONSTRUCTION_CONFIG)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
    evaluate(**vars(parser.parse_args(argv)))


if __name__ == '__main__':
    main()

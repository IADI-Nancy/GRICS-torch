#!/usr/bin/env python
"""Score existing GRICS++ 00XX_T2_Y outputs and compare paired solver times."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from article.sharpness_index import sharpness_index

DATA = Path('/home/pyuser/wkdir/data')
SUBJECT = re.compile(r'00[0-9]{2}_T2_[a-z]', re.I)
SLICE = re.compile(r'Siemens_SingleImage_slice([0-9]+)_image01')


def read_slice(folder):
    """Read GRICS++ complex-float export (Ny,Nx,Nz in column-major storage)."""
    xml = ET.parse(folder / 'ParamGRICS++_TSE_Breast.xml').getroot()
    dims = xml.find('./PreProcessing/Dimensions')
    nx, ny, nz = (int(dims.attrib[key]) for key in ('Nx', 'Ny', 'Nz'))
    if nz != 1:
        raise ValueError(f'{folder}: expected a 2D slice, Nz={nz}')
    filename = xml.find('./PostProcessing/ReconstructedImage').attrib['FileName']
    image_file = folder / (filename + '.0000')
    if image_file.stat().st_size != nx * ny * 8:
        raise ValueError(f'{image_file}: binary size differs from XML dimensions')
    image = np.fromfile(image_file, dtype='<c8').reshape(nx, ny)
    score = float(sharpness_index(torch.from_numpy(image)))
    log = (folder / 'grics.log.0').read_text()
    if 'RECONSTRUCTION COMPLETE' not in log:
        raise ValueError(f'{folder}: incomplete reconstruction log')
    def seconds(label):
        values = re.findall(r'^' + re.escape(label) + r'\s*=\s*([0-9.eE+\-]+)\s*$', log, re.M)
        if len(values) != 1:
            raise ValueError(f'{folder}: expected exactly one {label}')
        value = float(values[0])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{folder}: invalid {label}')
        return value
    if not np.isfinite(score):
        raise ValueError(f'{folder}: nonfinite sharpness')
    threads = re.search(r'^OMP_NUM_THREADS\s*=\s*(\d+)', log, re.M)
    states = xml.find('./Reconstruction/MotionStatesClustering')
    return dict(slice_number=int(SLICE.fullmatch(folder.name)[1]), sharpness=score,
                reconstruction_seconds=seconds('Reconstruction time'),
                total_elapsed_seconds=seconds('Total elapsed time'),
                motion_states=int(states.attrib['Nclusters']),
                omp_threads=int(threads[1]) if threads else None,
                image_shape=[nx, ny], image_file=str(image_file), log_file=str(folder/'grics.log.0'))


def collect(root, mode):
    rows, excluded = [], []
    if not root.is_dir():
        raise FileNotFoundError(root)
    for subject in sorted(root.iterdir()):
        if not subject.is_dir() or not SUBJECT.fullmatch(subject.name):
            continue
        try:
            folders = sorted(p for p in subject.iterdir() if p.is_dir() and SLICE.fullmatch(p.name))
            if not folders:
                raise ValueError('No slice directories')
            slices = [read_slice(p) for p in folders]
            numbers = sorted(s['slice_number'] for s in slices)
            if numbers != list(range(1, len(slices) + 1)):
                raise ValueError('Slice numbers must be contiguous starting at 1')
            if any((s['motion_states'] != 1 if mode == 'nomoco' else s['motion_states'] <= 1) for s in slices):
                raise ValueError(f'Motion-state configuration does not match {mode}')
            rows.append(dict(subject=subject.name, mode=mode, slice_count=len(slices),
                             mean_sharpness=float(np.mean([s['sharpness'] for s in slices])),
                             reconstruction_seconds_sum=sum(s['reconstruction_seconds'] for s in slices),
                             total_elapsed_seconds_sum=sum(s['total_elapsed_seconds'] for s in slices),
                             slices=slices))
            print(f'[score] {subject.name}: {mode}, {len(slices)} slices', flush=True)
        except (OSError, ValueError, ET.ParseError, KeyError, AttributeError) as error:
            excluded.append(dict(subject=subject.name, mode=mode, reason=str(error)))
            print(f'[skip] {subject.name}: {error}', flush=True)
    return rows, excluded


def boxplot(values, labels, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(values)
    ax.set_xticks(range(1, len(labels)+1), labels)
    if len(values) == 2:
        for a, b in zip(*values):
            ax.plot([1, 2], [a, b], 'o-', color='gray', alpha=.4, markersize=3)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cross_implementation_plots(cpp_rows, torch_rows, output_dir):
    """Plot matched 2D native-grid quality and corrected solver maxima."""
    cpp = {(row['subject'], row['mode']): row for row in cpp_rows}
    tor = {(Path(row['subject']).stem, row['mode']): row for row in torch_rows}
    subjects_by_mode = {}
    for mode in ('nomoco', 'corrected'):
        subjects = sorted({subject for subject, row_mode in cpp if row_mode == mode} &
                          {subject for subject, row_mode in tor if row_mode == mode})
        subjects = [subject for subject in subjects
                    if cpp[subject, mode]['slice_count'] == tor[subject, mode]['slice_count']]
        if any(tor[subject, mode].get('sharpness_image_stage') != 'native_solver_image'
               for subject in subjects):
            raise ValueError('Torch measurements must be regenerated with native_solver_image sharpness.')
        subjects_by_mode[mode] = subjects
    if not any(subjects_by_mode.values()):
        return []

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for axis, mode, title in zip(axes, ('nomoco', 'corrected'),
                                 ('No motion correction', 'Motion corrected')):
        subjects = subjects_by_mode[mode]
        if subjects:
            values = [[cpp[subject, mode]['mean_sharpness'] for subject in subjects],
                      [tor[subject, mode]['mean_sharpness'] for subject in subjects]]
            axis.boxplot(values)
            for cpp_value, torch_value in zip(*values):
                axis.plot([1, 2], [cpp_value, torch_value], 'o-', color='gray', alpha=.4, markersize=3)
        else:
            axis.set_xlim(0.5, 2.5)
            axis.text(0.5, 0.5, 'Results pending', transform=axis.transAxes,
                      ha='center', va='center', color='gray')
        axis.set_xticks([1, 2], ['GRICS++', 'GRICS-torch'])
        axis.set_title(f'{title} (n={len(subjects)})')
    axes[0].set_ylabel('Mean native-grid slice sharpness index')
    fig.suptitle('Breast 2D sharpness')
    fig.tight_layout()
    fig.savefig(output_dir/'sharpness_cpp_vs_torch_by_correction.png', dpi=200)
    plt.close(fig)

    corrected = subjects_by_mode['corrected']
    if corrected:
        maxima = [max(cpp[subject, 'corrected']['reconstruction_seconds_sum'] for subject in corrected),
                  max(tor[subject, 'corrected']['reconstruction_seconds_sum'] for subject in corrected)]
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.bar(['GRICS++', 'GRICS-torch'], maxima)
        axis.set_ylabel('Maximum corrected sum of per-slice solver times (s)')
        axis.set_title(f'Breast 2D corrected calculation time: {len(corrected)} matched acquisitions')
        fig.tight_layout()
        fig.savefig(output_dir/'max_corrected_solver_time_cpp_vs_torch.png', dpi=200)
        plt.close(fig)
    return sorted(set().union(*subjects_by_mode.values()))

def evaluate(cpp_root, torch_measurements, output_dir, nomoco_root=None):
    rows, excluded = collect(cpp_root, 'corrected')
    output_dir.mkdir(parents=True, exist_ok=True)
    if nomoco_root is None:
        candidate = cpp_root.with_name(cpp_root.name + '_nomoco')
        if candidate.is_dir():
            nomoco_root = candidate
        else:
            print(f'[baseline pending] {candidate}', flush=True)
    if nomoco_root:
        other, skipped = collect(nomoco_root, 'nomoco')
        rows += other
        excluded += skipped
    (output_dir/'measurements.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    with (output_dir/'measurements.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject', 'mode', 'slice_count', 'mean_sharpness',
                                            'reconstruction_seconds_sum', 'total_elapsed_seconds_sum'],
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (output_dir/'excluded.json').write_text(json.dumps(excluded, indent=2))
    by_mode = {mode: {r['subject']: r for r in rows if r['mode'] == mode}
               for mode in ('corrected', 'nomoco')}
    pairs = sorted(set(by_mode['corrected']) & set(by_mode['nomoco']))
    pairs = [s for s in pairs if
             [(x['slice_number'], x['image_shape']) for x in by_mode['corrected'][s]['slices']] ==
             [(x['slice_number'], x['image_shape']) for x in by_mode['nomoco'][s]['slices']]]
    for metric, ylabel, filename in (
        ('mean_sharpness', 'Mean native-grid slice sharpness index', 'sharpness_boxplot.png'),
        ('reconstruction_seconds_sum', 'Sum of per-slice solver times (s)', 'time_boxplot.png')):
        if pairs:
            values = [[by_mode[mode][s][metric] for s in pairs] for mode in ('corrected', 'nomoco')]
            labels = ['GRICS++ corrected', 'GRICS++ one-state']
            title = f'GRICS++: {len(pairs)} paired acquisitions'
        elif by_mode['corrected']:
            values = [[r[metric] for r in by_mode['corrected'].values()]]
            labels = ['GRICS++ corrected']
            title = f'GRICS++: {len(values[0])} acquisitions (no paired baseline)'
        else:
            continue
        boxplot(values, labels, ylabel, title, output_dir/filename)
    torch_rows = json.loads(torch_measurements.read_text())
    comparison, unmatched = [], []
    for cpp in rows:
        matches = [r for r in torch_rows if Path(r['subject']).stem == cpp['subject'] and r['mode'] == cpp['mode']]
        if len(matches) != 1 or matches[0]['slice_count'] != cpp['slice_count']:
            unmatched.append(dict(subject=cpp['subject'], mode=cpp['mode'], reason='Missing/duplicate Torch measurement or unequal slice count'))
            continue
        t = matches[0]
        seconds = float(t['reconstruction_seconds_sum'])
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Invalid Torch timing: {cpp['subject']}")
        comparison.append(dict(subject=cpp['subject'], mode=cpp['mode'], slice_count=cpp['slice_count'],
                               cpp_solver_seconds=cpp['reconstruction_seconds_sum'], torch_solver_seconds=seconds,
                               cpp_over_torch=cpp['reconstruction_seconds_sum']/seconds,
                               torch_run_folder=t['run_folder']))
    (output_dir/'unmatched.json').write_text(json.dumps(unmatched, indent=2))
    with (output_dir/'time_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject', 'mode', 'slice_count', 'cpp_solver_seconds',
                                            'torch_solver_seconds', 'cpp_over_torch', 'torch_run_folder'])
        writer.writeheader()
        writer.writerows(comparison)
    for mode in ('corrected', 'nomoco'):
        paired = [r for r in comparison if r['mode'] == mode]
        if paired:
            boxplot([[r[key] for r in paired] for key in ('cpp_solver_seconds', 'torch_solver_seconds')],
                    ['GRICS++', 'GRICS-torch'], 'Sum of per-slice solver times (s)',
                    f'{mode}: {len(paired)} paired acquisitions', output_dir/f'time_cpp_vs_torch_{mode}.png')
    print(f'[results] {output_dir}: {len(rows)} GRICS++ measurements, {len(comparison)} timing pairs')
    cross_implementation_plots(rows, torch_rows, output_dir)
    return rows, comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpp-root', type=Path, default=DATA/'Breast-INNOV_GRICS_database/GRICS-BELT')
    parser.add_argument('--nomoco-root', type=Path, help='One-state outputs; defaults to CPP_ROOT + _nomoco when present')
    parser.add_argument('--torch-measurements', type=Path, default=DATA/'GRICS-torch/article_dataset_2D/results/measurements.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'article/results_grics_cpp_2d')
    args = parser.parse_args()
    torch.set_num_threads(1)
    evaluate(**vars(args))

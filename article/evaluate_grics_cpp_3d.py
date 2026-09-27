#!/usr/bin/env python
"""Score existing GRICS++ 3D volumes and compare them with paired Torch results."""
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from article.sharpness_index import sharpness_index

DATA = Path('/home/pyuser/wkdir/data')
SUBJECT = re.compile(r'00[0-9]{2}_T1_[a-z]', re.I)
XML_NAME = 'ParamGRICS++_GRE_3D_LR.xml'


def seconds(log, label, folder):
    values = re.findall(r'^' + re.escape(label) + r'\s*=\s*([0-9.eE+\-]+)\s*$', log, re.M)
    if len(values) != 1:
        raise ValueError(f'{folder}: expected exactly one {label}')
    value = float(values[0])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{folder}: invalid {label}')
    return value


def read_volume(folder):
    """Read complex-float voxels with y varying fastest, then x, then z."""
    xml = ET.parse(folder / XML_NAME).getroot()
    dimensions = xml.find('./PreProcessing/Dimensions')
    nx, ny, nz = (int(dimensions.attrib[key]) for key in ('Nx', 'Ny', 'Nz'))
    if min(nx, ny) <= 1 or nz <= 1:
        raise ValueError(f'{folder}: invalid 3D dimensions {(nx, ny, nz)}')
    filename = xml.find('./PostProcessing/ReconstructedImage').attrib['FileName']
    image_file = folder / (filename + '.0000')
    if image_file.stat().st_size != nx * ny * nz * np.dtype('<c8').itemsize:
        raise ValueError(f'{image_file}: binary size differs from XML dimensions')
    # GRICS++ stores each native axial partition as a contiguous (Nx, Ny) slab.
    image = np.memmap(image_file, dtype='<c8', mode='r', shape=(nz, nx, ny))
    scores = [float(sharpness_index(torch.from_numpy(np.asarray(image[z]).copy())))
              for z in range(nz)]
    if not np.isfinite(scores).all():
        raise ValueError(f'{folder}: nonfinite sharpness')
    log = (folder / 'grics.log.0').read_text()
    if 'RECONSTRUCTION COMPLETE' not in log:
        raise ValueError(f'{folder}: incomplete reconstruction log')
    states = int(xml.find('./Reconstruction/MotionStatesClustering').attrib['Nclusters'])
    threads = re.search(r'^OMP_NUM_THREADS\s*=\s*(\d+)', log, re.M)
    return dict(subject=folder.name, motion_states=states, image_shape=[nx, ny, nz],
                slice_sharpness=scores, mean_sharpness=float(np.mean(scores)),
                reconstruction_seconds=seconds(log, 'Reconstruction time', folder),
                total_elapsed_seconds=seconds(log, 'Total elapsed time', folder),
                omp_threads=int(threads[1]) if threads else None,
                image_file=str(image_file), log_file=str(folder / 'grics.log.0'))


def collect(root, mode):
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows, excluded = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not SUBJECT.fullmatch(folder.name):
            continue
        try:
            row = read_volume(folder)
            if (row['motion_states'] == 1) != (mode == 'nomoco'):
                raise ValueError(f'Motion-state configuration does not match {mode}')
            row['mode'] = mode
            rows.append(row)
            print(f'[score] {folder.name}: {mode}, {row["image_shape"][2]} partitions', flush=True)
        except (OSError, ValueError, ET.ParseError, KeyError, AttributeError, TypeError) as error:
            excluded.append(dict(subject=folder.name, mode=mode, reason=str(error)))
            print(f'[skip] {folder.name}: {error}', flush=True)
    return rows, excluded


def boxplot(values, labels, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(values)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    if len(values) == 2:
        for a, b in zip(*values):
            ax.plot([1, 2], [a, b], 'o-', color='gray', alpha=.4, markersize=3)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)



def load_cpp_image(row):
    nx, ny, nz = row['image_shape']
    image = np.memmap(row['image_file'], dtype='<c8', mode='r', shape=(nz, nx, ny))
    return np.abs(np.asarray(image).transpose(1, 2, 0)).copy()


def load_torch_image(row):
    path = Path(row['run_folder']) / 'reconstructions/volume_001/results/image_reconstructed.pt'
    image = torch.as_tensor(torch.load(path, map_location='cpu', weights_only=True))
    if image.ndim == 4:
        image = image.mean(dim=0)
    if image.ndim != 3 or min(image.shape) <= 1:
        raise ValueError(f'{path}: expected a 3D image or repetition-stacked 3D image, got {tuple(image.shape)}')
    return image.abs().numpy()


def selected_planes(image):
    nx, ny, nz = image.shape
    # x follows the anterior-posterior direction; y follows left-right.
    return (image[:, :, nz // 2], image[:, 3 * ny // 4, :], image[3 * nx // 4 - 8, :, :])


def oriented_plane(partition, plane_number):
    # Use an upper-origin array equivalent to the original lower-origin view,
    # then apply the requested rotation or vertical flip in screen coordinates.
    displayed = np.flipud(partition.T)
    return np.rot90(displayed) if plane_number == 0 else np.flipud(displayed)


def image_panels(cpp_by_key, torch_by_key, output_dir):
    """Save one 3-by-4 panel per subject and one all-subject contact sheet."""
    subjects = sorted(subject for subject, mode in cpp_by_key if mode == 'corrected')
    columns = [('GRICS++', 'nomoco'), ('GRICS++', 'corrected'),
               ('GRICS-torch', 'nomoco'), ('GRICS-torch', 'corrected')]
    planes = ('Axial (z=1/2)', 'Sagittal (y=3/4)', 'Coronal (x=3/4-10)')
    missing = []
    overview, grid = plt.subplots(len(subjects), 12, figsize=(27, max(2.5, 2.2 * len(subjects))),
                                  squeeze=False) if subjects else (None, None)
    for row_number, subject in enumerate(subjects):
        images = {}
        for implementation, mode in columns:
            key = (subject, mode)
            try:
                if implementation == 'GRICS++':
                    record = cpp_by_key.get(key)
                    if record is None:
                        raise FileNotFoundError('GRICS++ volume is unavailable')
                    images[implementation, mode] = load_cpp_image(record)
                else:
                    matches = torch_by_key.get(key, [])
                    if len(matches) != 1:
                        raise ValueError('Missing or duplicate Torch measurement')
                    images[implementation, mode] = load_torch_image(matches[0])
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                missing.append(dict(subject=subject, implementation=implementation,
                                    mode=mode, reason=str(error)))
                images[implementation, mode] = None
        scales = {}
        for implementation in ('GRICS++', 'GRICS-torch'):
            available = [images[implementation, mode] for mode in ('nomoco', 'corrected')
                         if images[implementation, mode] is not None]
            if available:
                high = max(float(np.percentile(image[np.isfinite(image)], 99.5))
                           for image in available)
                scales[implementation] = max(high, np.finfo(float).eps)
        fig, axes = plt.subplots(3, 4, figsize=(13, 9), squeeze=False)
        for plane_number, plane in enumerate(planes):
            for column, (implementation, mode) in enumerate(columns):
                axis = axes[plane_number, column]
                image = images[implementation, mode]
                if image is None:
                    axis.text(.5, .5, 'Unavailable', ha='center', va='center', transform=axis.transAxes)
                else:
                    partition = oriented_plane(selected_planes(image)[plane_number], plane_number)
                    axis.imshow(partition, cmap='gray', origin='upper', vmin=0,
                                vmax=scales[implementation], aspect='auto')
                if plane_number == 0:
                    axis.set_title(f'{implementation}\n{("One-state" if mode == "nomoco" else "Corrected")}')
                if column == 0:
                    axis.set_ylabel(plane)
                axis.set_xticks([])
                axis.set_yticks([])
                overview_axis = grid[row_number, plane_number * 4 + column]
                if image is None:
                    overview_axis.text(.5, .5, 'Unavailable', fontsize=7, ha='center',
                                       va='center', transform=overview_axis.transAxes)
                else:
                    overview_axis.imshow(partition, cmap='gray', origin='upper', vmin=0,
                                         vmax=scales[implementation], aspect='auto')
                if row_number == 0:
                    overview_axis.set_title(f'{plane}\n{implementation}\n'
                                            f'{("One-state" if mode == "nomoco" else "Corrected")}', fontsize=8)
                if plane_number == 0 and column == 0:
                    overview_axis.set_ylabel(subject, fontsize=8)
                overview_axis.set_xticks([])
                overview_axis.set_yticks([])
        fig.suptitle(f'{subject}: axial center; sagittal at 3/4; coronal at 3/4-10, magnitude',
                     fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / f'{subject}_central_planes.png', dpi=180)
        plt.close(fig)
        print(f'[image] {subject}', flush=True)
    if overview is not None:
        overview.suptitle('Breast 3D: axial center; sagittal at 3/4; coronal at 3/4-10',
                          fontsize=16)
        overview.tight_layout(rect=(0, 0, 1, .965))
        overview.savefig(output_dir / 'all_subjects_central_planes.png', dpi=150)
        plt.close(overview)
    (output_dir / 'image_plot_missing.json').write_text(json.dumps(missing, indent=2))
    return missing


def evaluate(cpp_root, torch_measurements, output_dir, nomoco_root=None):
    cpp_root, torch_measurements, output_dir = map(Path, (cpp_root, torch_measurements, output_dir))
    rows, excluded = collect(cpp_root, 'corrected')
    if nomoco_root is None:
        candidate = cpp_root.with_name(cpp_root.name + '_nomoco')
        if candidate.is_dir():
            nomoco_root = candidate
        else:
            print(f'[baseline pending] {candidate}', flush=True)
    if nomoco_root is not None:
        baseline, skipped = collect(Path(nomoco_root), 'nomoco')
        rows.extend(baseline)
        excluded.extend(skipped)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'measurements.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output_dir / 'excluded.json').write_text(json.dumps(excluded, indent=2))
    write_csv(output_dir / 'measurements.csv', rows,
              ['subject', 'mode', 'motion_states', 'mean_sharpness',
               'reconstruction_seconds', 'total_elapsed_seconds', 'omp_threads'])

    by_mode = {mode: {r['subject']: r for r in rows if r['mode'] == mode}
               for mode in ('corrected', 'nomoco')}
    pairs = sorted(set(by_mode['corrected']) & set(by_mode['nomoco']))
    pairs = [s for s in pairs if by_mode['corrected'][s]['image_shape'] ==
             by_mode['nomoco'][s]['image_shape']]
    for metric, ylabel, filename in (
        ('mean_sharpness', 'Mean native axial-partition sharpness index', 'sharpness_boxplot.png'),
        ('reconstruction_seconds', 'Volume solver time (s)', 'time_boxplot.png')):
        if pairs:
            values = [[by_mode[m][s][metric] for s in pairs] for m in ('corrected', 'nomoco')]
            labels = ['GRICS++ corrected', 'GRICS++ one-state']
            title = f'GRICS++ 3D: {len(pairs)} paired acquisitions'
        elif by_mode['corrected']:
            values = [[r[metric] for r in by_mode['corrected'].values()]]
            labels = ['GRICS++ corrected']
            title = f'GRICS++ 3D: {len(values[0])} acquisitions (no paired baseline)'
        else:
            continue
        boxplot(values, labels, ylabel, title, output_dir / filename)

    torch_rows = json.loads(torch_measurements.read_text())
    torch_by_key = {}
    for row in torch_rows:
        key = (Path(row['subject']).stem, row['mode'])
        torch_by_key.setdefault(key, []).append(row)
    comparison, unmatched = [], []
    for cpp in rows:
        key = (cpp['subject'], cpp['mode'])
        matches = torch_by_key.get(key, [])
        if len(matches) != 1:
            unmatched.append(dict(subject=key[0], mode=key[1], reason='Missing or duplicate Torch measurement'))
            continue
        tor = matches[0]
        try:
            if len(tor['slice_sharpness']) != cpp['image_shape'][2]:
                raise ValueError('Unequal axial-partition counts')
            if (int(tor['motion_states']) == 1) != (cpp['mode'] == 'nomoco'):
                raise ValueError('Torch motion-state configuration differs from mode')
            sharpness = float(tor['mean_sharpness'])
            timing = float(tor['reconstruction_seconds'])
            if not np.isfinite([sharpness, timing]).all() or timing <= 0:
                raise ValueError('Invalid Torch score or reconstruction time')
        except (KeyError, TypeError, ValueError) as error:
            unmatched.append(dict(subject=key[0], mode=key[1], reason=str(error)))
            continue
        comparison.append(dict(subject=key[0], mode=key[1], partition_count=cpp['image_shape'][2],
                               cpp_native_sharpness=cpp['mean_sharpness'],
                               torch_pipeline_sharpness=sharpness,
                               cpp_solver_seconds=cpp['reconstruction_seconds'],
                               torch_reconstruction_seconds=timing,
                               cpp_over_torch_seconds=cpp['reconstruction_seconds'] / timing,
                               torch_run_folder=tor['run_folder']))
    (output_dir / 'unmatched.json').write_text(json.dumps(unmatched, indent=2))
    write_csv(output_dir / 'comparison.csv', comparison,
              ['subject', 'mode', 'partition_count', 'cpp_native_sharpness',
               'torch_pipeline_sharpness', 'cpp_solver_seconds',
               'torch_reconstruction_seconds', 'cpp_over_torch_seconds', 'torch_run_folder'])
    for mode in ('corrected', 'nomoco'):
        paired = [r for r in comparison if r['mode'] == mode]
        if not paired:
            continue
        boxplot([[r[key] for r in paired] for key in ('cpp_solver_seconds', 'torch_reconstruction_seconds')],
                ['GRICS++', 'GRICS-torch'], 'Volume reconstruction time (s)',
                f'Breast 3D {mode}: {len(paired)} paired acquisitions',
                output_dir / f'time_cpp_vs_torch_{mode}.png')
        boxplot([[r[key] for r in paired] for key in ('cpp_native_sharpness', 'torch_pipeline_sharpness')],
                ['GRICS++ native', 'Torch pipeline'], 'Mean axial-partition sharpness index',
                f'Breast 3D {mode}: {len(paired)} paired acquisitions',
                output_dir / f'sharpness_cpp_vs_torch_{mode}.png')
    image_panels({(row['subject'], row['mode']): row for row in rows},
                 torch_by_key, output_dir)
    unique_subjects = len({row['subject'] for row in rows})
    corrected_count = sum(row['mode'] == 'corrected' for row in rows)
    nomoco_count = sum(row['mode'] == 'nomoco' for row in rows)
    print(f'[results] {output_dir}: {unique_subjects} acquisitions, '
          f'{corrected_count} corrected and {nomoco_count} one-state volumes, '
          f'{len(comparison)} Torch pairs')
    return rows, comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpp-root', type=Path, default=DATA / 'Breast-INNOV_GRICS_database/GRICS-BELT-3D')
    parser.add_argument('--nomoco-root', type=Path, help='One-state outputs; defaults to CPP_ROOT + _nomoco when present')
    parser.add_argument('--torch-measurements', type=Path,
                        default=DATA / 'GRICS-torch/article_dataset_3D/results/measurements.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'article/results_grics_cpp_3d')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    evaluate(**vars(args))


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Rerun one 2D breast subject with the new motion solver and compare saved runs.

Default: slice 15 of 0079_T2_m, reconstructed on GPU. The prepared HDF5 must exist;
this script never extracts raw ISMRMRD/SAEC data or modifies baseline runs.

Example:
    python tests/compare_one_subject_motion_correction.py --dry-run
    python tests/compare_one_subject_motion_correction.py
    python tests/compare_one_subject_motion_correction.py --slice 15
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from article.evaluate_breast_2d import score_image
from pipelines.siemens_breast_T2 import run_pipeline
from src.runtime.runtime_config import load_config

DATASET = Path('/home/pyuser/wkdir/data/GRICS-torch/article_dataset_2D')
CPP_ROOT = Path('/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/GRICS-BELT')
CPP_MEASUREMENTS = ROOT / 'article/results_grics_cpp_2d/measurements.json'
CONFIG = ROOT / 'config/reconstruction/nonrigid_2d_breast.toml'


def _saved_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f'{path}: expected a list of measurements')
    return rows


def _row(rows: list[dict], subject: str, mode: str) -> dict:
    matches = [r for r in rows if Path(r['subject']).stem == subject and r['mode'] == mode]
    if len(matches) != 1:
        raise ValueError(f'Expected one saved {mode} row for {subject}; found {len(matches)}')
    return matches[0]


def _torch_magnitude(run_folder: Path, slice_number: int) -> np.ndarray:
    path = run_folder / 'reconstructions' / f'slice_{slice_number:03d}' / 'results/image_reconstructed.pt'
    image = torch.as_tensor(torch.load(path, map_location='cpu', weights_only=True))
    if image.ndim == 3:
        image = image.mean(dim=0)
    if image.ndim != 2:
        raise ValueError(f'{path}: expected a 2D image, got {tuple(image.shape)}')
    return image.abs().numpy()


def _cpp_magnitude(subject: str, slice_number: int, cpp_root: Path) -> np.ndarray:
    folder = cpp_root / subject / f'Siemens_SingleImage_slice{slice_number:03d}_image01'
    xml = ET.parse(folder / 'ParamGRICS++_TSE_Breast.xml').getroot()
    dims = xml.find('./PreProcessing/Dimensions')
    nx, ny, nz = (int(dims.attrib[name]) for name in ('Nx', 'Ny', 'Nz'))
    if nz != 1:
        raise ValueError(f'{folder}: expected a 2D slice')
    filename = xml.find('./PostProcessing/ReconstructedImage').attrib['FileName']
    path = folder / (filename + '.0000')
    if path.stat().st_size != nx * ny * np.dtype('<c8').itemsize:
        raise ValueError(f'{path}: size does not match XML dimensions')
    return np.abs(np.fromfile(path, dtype='<c8').reshape(nx, ny))


def _plot_comparison(subject: str, slice_number: int, images: list[np.ndarray],
                     scores: dict[str, float], destination: Path) -> None:
    labels = ('Torch no correction', 'Torch previous corrected',
              'Torch updated corrected', 'GRICS++ corrected')
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, label, image in zip(axes, labels, images):
        if not np.isfinite(image).all():
            raise ValueError(f'{label} image contains non-finite values')
        # Display normalization is per panel; SI below is from the native grid.
        upper = float(np.percentile(image, 99.5))
        display = image / upper if upper > 0 else image
        ax.imshow(display.T, cmap='gray', vmin=0, vmax=1, origin='lower', aspect='auto')
        ax.set_title(f'{label}\nNative SI: {scores[label]:.1f}', fontsize=10)
        ax.axis('off')
    fig.suptitle(f'{subject} / slice {slice_number} — display intensities normalized per panel')
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', default='0079_T2_m')
    parser.add_argument('--slice', type=int, default=15,
                        help='One-based slice number to reconstruct (default: 15).')
    parser.add_argument('--dataset-dir', type=Path, default=DATASET)
    parser.add_argument('--cpp-root', type=Path, default=CPP_ROOT)
    parser.add_argument('--cpp-measurements', type=Path, default=CPP_MEASUREMENTS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'tests/artifacts/one_subject_motion_comparison')
    parser.add_argument('--dry-run', action='store_true', help='Check inputs and print the planned run only.')
    args = parser.parse_args(argv)

    if not re.fullmatch(r'00[0-9]{2}_T2_[ms]', args.subject, re.I):
        parser.error('--subject must look like 0068_T2_m')
    subject = args.subject
    prepared = args.dataset_dir / f'{subject}.h5'
    if not prepared.is_file():
        raise FileNotFoundError(f'Prepared subject not found: {prepared}; this script will not prepare raw data.')
    torch_rows = _saved_rows(args.dataset_dir / 'results/measurements.json')
    cpp_rows = _saved_rows(args.cpp_measurements)
    old_corrected = _row(torch_rows, subject, 'corrected')
    old_nomoco = _row(torch_rows, subject, 'nomoco')
    cpp_corrected = _row(cpp_rows, subject, 'corrected')
    old_folder = Path(old_corrected['run_folder'])
    no_folder = Path(old_nomoco['run_folder'])
    if not old_folder.is_dir() or not no_folder.is_dir():
        raise FileNotFoundError('A saved Torch comparison run is missing.')
    n_slices = min(len(old_corrected['slice_sharpness']), len(old_nomoco['slice_sharpness']),
                   len(cpp_corrected['slices']))
    if n_slices < 1:
        raise ValueError('No paired slices in saved measurements.')
    if not 1 <= args.slice <= n_slices:
        parser.error(f'--slice must be between 1 and {n_slices}')
    chosen = [args.slice]
    params = load_config(
        data_type='preprocessed-real', reconstruction_config=CONFIG,
        coil_sensitivity_config=ROOT / 'config/coil_sensitivity/odille_spline.toml')
    if (not params.use_motion_preconditioner or not params.image_only_last_iteration_per_level
            or params.motion_signal_normalization != 'acquisition_zscore'):
        raise ValueError('The selected configuration must enable the motion solver changes and acquisition z-score.')
    print(f'[input] {prepared} / slices {chosen[0]}–{chosen[-1]}')
    print(f'[baseline] Torch corrected: {old_folder}')
    print(f'[baseline] Torch no correction: {no_folder}')
    print(f'[baseline] GRICS++ corrected: {args.cpp_root / subject}')
    if args.dry_run:
        print('[dry-run] No reconstruction or files written.')
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError('GPU reconstruction requested, but PyTorch CUDA is unavailable; no CPU fallback is allowed.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(
        preprocessed_file=prepared, output_root=args.output_dir / 'updated_corrected',
        reconstruction_config=CONFIG, device='gpu', max_workers=1,
        slice_start=chosen[0] - 1, slice_stop=chosen[-1],
        save_reconstruction_logs=True, save_reconstruction_tensors=True,
        return_tensors=True, export_dicom=False)
    by_slice = {item['slice_number']: item for item in result['reconstructions']}
    if sorted(by_slice) != chosen:
        raise ValueError(f'Reconstructed slices {sorted(by_slice)} differ from requested {chosen}')
    cpp_by_slice = {item['slice_number']: item for item in cpp_corrected['slices']}
    comparison = []
    for number in chosen:
        native_si = score_image(by_slice[number]['native_image'])
        comparison.append({
            'slice_number': number, 'torch_nomoco_native_si': float(old_nomoco['slice_sharpness'][number - 1]),
            'torch_previous_corrected_native_si': float(old_corrected['slice_sharpness'][number - 1]),
            'torch_updated_corrected_native_si': native_si,
            'cpp_corrected_native_si': float(cpp_by_slice[number]['sharpness']),
            'updated_reconstruction_seconds': float(by_slice[number]['reconstruction_seconds']),
            'updated_minus_previous_native_si': native_si - float(old_corrected['slice_sharpness'][number - 1]),
        })
    plot_slice = args.slice
    row = next(item for item in comparison if item['slice_number'] == plot_slice)
    scores = {
        'Torch no correction': row['torch_nomoco_native_si'],
        'Torch previous corrected': row['torch_previous_corrected_native_si'],
        'Torch updated corrected': row['torch_updated_corrected_native_si'],
        'GRICS++ corrected': row['cpp_corrected_native_si'],
    }
    torch_images = [
        _torch_magnitude(no_folder, plot_slice),
        _torch_magnitude(old_folder, plot_slice),
        torch.as_tensor(by_slice[plot_slice]['image']).mean(dim=0).abs().numpy(),
    ]
    if any(image.shape != torch_images[0].shape for image in torch_images[1:]):
        raise ValueError('Torch display grids differ; refusing a misleading side-by-side plot.')
    cpp_image = _cpp_magnitude(subject, plot_slice, args.cpp_root)
    run_name = Path(result['run_folder']).name
    figure = args.output_dir / f'{subject}_slice{plot_slice:03d}_{run_name}_comparison.png'
    report = args.output_dir / f'{subject}_{run_name}_comparison.json'
    _plot_comparison(subject, plot_slice, [*torch_images, cpp_image], scores, figure)
    payload = {
        'subject': subject, 'prepared_file': str(prepared),
        'new_run_folder': str(result['run_folder']),
        'previous_torch_corrected_run': str(old_folder),
        'previous_torch_nomoco_run': str(no_folder),
        'cpp_corrected_folder': str(args.cpp_root / subject),
        'sharpness_stage': 'native_solver_image',
        'display_stage': 'Torch postprocessed exports; GRICS++ native export',
        'slices': comparison, 'plot_file': str(figure),
        'summary': {
            'mean_updated_minus_previous_native_si': float(np.mean(
                [row['updated_minus_previous_native_si'] for row in comparison])),
            'slices_improved_vs_previous': sum(
                row['updated_minus_previous_native_si'] > 0 for row in comparison),
            'slice_count': len(comparison),
        },
    }
    report.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f'[result] {figure}')
    print(f'[result] {report}')
    print(f'[slice {plot_slice}] native SI: old Torch {scores["Torch previous corrected"]:.1f}, '
          f'updated Torch {scores["Torch updated corrected"]:.1f}, '
          f'GRICS++ {scores["GRICS++ corrected"]:.1f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

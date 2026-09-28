#!/usr/bin/env python
"""Run one 2D breast subject and compare every reconstructed slice to GRICS++.

This opt-in integration test uses the prepared 0079 T2 acquisition by default.
It does not run as part of unittest discovery.

Example:
    python tests/run_real_2d_subject.py --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity

from pipelines.siemens_breast_T2 import run_pipeline


DATA = Path('/home/pyuser/wkdir/data')
DATABASE = DATA / 'Breast-INNOV_GRICS_database'
DATASET = DATA / 'GRICS-torch/article_dataset_2D'
CPP_ROOT = DATABASE / 'GRICS-BELT'
SLICE_NAME = re.compile(r'Siemens_SingleImage_slice(\d+)_image01$')


def _cpp_slice_folders(subject_root: Path) -> dict[int, Path]:
    folders = {}
    for folder in subject_root.iterdir():
        match = SLICE_NAME.fullmatch(folder.name)
        if match and folder.is_dir():
            folders[int(match.group(1))] = folder
    expected = list(range(1, len(folders) + 1))
    if sorted(folders) != expected:
        raise ValueError(f'{subject_root}: GRICS++ slice folders must be contiguous; got {sorted(folders)}.')
    return folders


def _load_cpp_slice(folder: Path) -> np.ndarray:
    xml = ET.parse(folder / 'ParamGRICS++_TSE_Breast.xml').getroot()
    dimensions = xml.find('./PreProcessing/Dimensions')
    nx, ny, nz = (int(dimensions.attrib[key]) for key in ('Nx', 'Ny', 'Nz'))
    if nz != 1:
        raise ValueError(f'{folder}: expected a 2D GRICS++ image, got Nz={nz}.')
    name = xml.find('./PostProcessing/ReconstructedImage').attrib['FileName']
    image_path = folder / f'{name}.0000'
    image = np.fromfile(image_path, dtype='<c8')
    if image.size != nx * ny:
        raise ValueError(f'{image_path}: expected {nx * ny} complex values, got {image.size}.')
    return np.abs(image.reshape(nx, ny)).copy()


def _torch_magnitude(image: torch.Tensor) -> np.ndarray:
    image = torch.as_tensor(image).detach().cpu()
    if image.ndim == 3:
        image = image.mean(dim=0)
    if image.ndim != 2:
        raise ValueError(f'Expected [NEX, Nx, Ny] or [Nx, Ny], got {tuple(image.shape)}.')
    return image.abs().numpy()


def _metrics(torch_image: np.ndarray, cpp_image: np.ndarray) -> dict[str, float]:
    if torch_image.shape != cpp_image.shape:
        raise ValueError(f'Torch/GRICS++ shape mismatch: {torch_image.shape} versus {cpp_image.shape}.')
    denominator = float(np.vdot(torch_image.ravel(), torch_image.ravel()).real)
    if denominator <= 0:
        raise ValueError('Torch image has zero magnitude.')
    scale = float(np.vdot(torch_image.ravel(), cpp_image.ravel()).real / denominator)
    aligned = torch_image * scale
    reference_norm = float(np.linalg.norm(cpp_image))
    if reference_norm <= 0:
        raise ValueError('GRICS++ image has zero magnitude.')
    data_range = float(max(aligned.max(), cpp_image.max()) - min(aligned.min(), cpp_image.min()))
    if data_range <= 0:
        raise ValueError('Images have zero comparison range.')
    return {
        'scale_torch_to_grics_cpp': scale,
        'nrmse': float(np.linalg.norm(aligned - cpp_image) / reference_norm),
        'ssim': float(structural_similarity(cpp_image, aligned, data_range=data_range)),
    }



def _metric_label(summary: dict[str, float]) -> str:
    return (f"Axial slices (n={summary['slice_count']})\n"
            f"NRMSE {summary['mean_nrmse']:.4f} ± {summary['std_nrmse']:.4f}\n"
            f"SSIM {summary['mean_ssim']:.4f} ± {summary['std_ssim']:.4f}")


def _write_comparison_image(rows: list[dict], torch_image: np.ndarray, cpp_image: np.ndarray,
                            central_slice: int, run_folder: Path) -> Path:
    """Save the central axial slice, with display scaling shared with GRICS++."""
    central = next(row for row in rows if row['slice_number'] == central_slice)
    aligned_torch = torch_image * float(central['scale_torch_to_grics_cpp'])
    vmax = max(float(np.quantile(aligned_torch, .995)), float(np.quantile(cpp_image, .995)))
    summary = {
        'slice_count': len(rows),
        'mean_nrmse': float(np.mean([row['nrmse'] for row in rows])),
        'std_nrmse': float(np.std([row['nrmse'] for row in rows])),
        'mean_ssim': float(np.mean([row['ssim'] for row in rows])),
        'std_ssim': float(np.std([row['ssim'] for row in rows])),
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for axis, name, image in zip(axes, ('GRICS-torch', 'GRICS++'), (aligned_torch, cpp_image)):
        axis.imshow(image.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
        axis.set_title(f'Axial slice {central_slice}: {name}')
        axis.axis('off')
        axis.text(.02, .02, _metric_label(summary), transform=axis.transAxes,
                  va='bottom', ha='left', fontsize=9,
                  bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
    output = run_folder / 'comparison_vs_grics_cpp.png'
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def _metric_image(item: dict, cpp_image: np.ndarray) -> tuple[np.ndarray, str]:
    """Prefer the native solver image; accept main's matching pipeline image."""
    native_path = item.get('native_image_file')
    if native_path is not None and Path(native_path).is_file():
        return (_torch_magnitude(torch.load(native_path, map_location='cpu', weights_only=True)),
                str(native_path))
    image = _torch_magnitude(item['image'])
    if image.shape != cpp_image.shape:
        raise ValueError(
            'The main-branch pipeline did not export a native image and its returned image shape '
            f'{image.shape} differs from GRICS++ {cpp_image.shape}. '
            'Apply the native-image export change before using pre-zero-fill metrics.'
        )
    return image, 'pipeline returned image'

def run_subject(subject: str, *, sequence: str = 's', dataset_root: Path = DATASET,
                cpp_root: Path = CPP_ROOT, output_root: Path = ROOT / 'runs/test_real_2d_subject',
                device: str = 'cpu', max_workers: int | None = None,
                slice_start: int = 0, slice_stop: int | None = None) -> dict:
    """Reconstruct one subject and compare all of its axial slices with GRICS++."""
    if not re.fullmatch(r'\d{4}', subject):
        raise ValueError(f'Subject must be a four-digit identifier, got {subject!r}.')
    if sequence not in ('s', 'm'):
        raise ValueError(f'Sequence must be "s" or "m", got {sequence!r}.')
    acquisition = f'{subject}_T2_{sequence}'
    prepared = Path(dataset_root) / f'{acquisition}.h5'
    cpp_subject = Path(cpp_root) / acquisition
    if not prepared.is_file():
        raise FileNotFoundError(f'Prepared input is required: {prepared}')
    cpp_folders = _cpp_slice_folders(cpp_subject)

    result = run_pipeline(
        prepared, output_root=output_root, device=device,
        max_workers=max_workers, slice_start=slice_start,
        slice_stop=slice_stop, return_tensors=True, export_dicom=False,
    )
    rows = []
    central_slice = sorted(cpp_folders)[len(cpp_folders) // 4]
    central_images = None
    for item in result['reconstructions']:
        number = int(item['slice_number'])
        if number not in cpp_folders:
            raise ValueError(f'No GRICS++ image for Torch slice {number}.')
        cpp_image = _load_cpp_slice(cpp_folders[number])
        torch_image, source = _metric_image(item, cpp_image)
        rows.append({'slice_number': number, 'torch_metric_image_source': source,
                     **_metrics(torch_image, cpp_image)})
        if number == central_slice:
            central_images = torch_image, cpp_image
    if central_images is None:
        raise ValueError(f'Central GRICS++ slice {central_slice} was not reconstructed.')

    summary = {
        'subject': acquisition,
        'run_folder': str(result['run_folder']),
        'grics_cpp_subject_folder': str(cpp_subject),
        'slice_count': len(rows),
        'mean_nrmse': float(np.mean([row['nrmse'] for row in rows])),
        'std_nrmse': float(np.std([row['nrmse'] for row in rows])),
        'mean_ssim': float(np.mean([row['ssim'] for row in rows])),
        'std_ssim': float(np.std([row['ssim'] for row in rows])),
        'max_nrmse': float(np.max([row['nrmse'] for row in rows])),
        'min_ssim': float(np.min([row['ssim'] for row in rows])),
        'per_slice': rows,
    }
    comparison_image = _write_comparison_image(
        rows, *central_images, central_slice, Path(result['run_folder']))
    summary['comparison_image'] = str(comparison_image)
    output = Path(result['run_folder']) / 'comparison_vs_grics_cpp.json'
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'[comparison] NRMSE={summary["mean_nrmse"]:.6f} ± {summary["std_nrmse"]:.6f}; '
          f'SSIM={summary["mean_ssim"]:.6f} ± {summary["std_ssim"]:.6f}; {output}', flush=True)
    return summary


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', default='0079', help='Four-digit subject identifier.')
    parser.add_argument('--sequence', choices=('s', 'm'), default='s',
                        help='0079 T2 acquisition: s is the default.')
    parser.add_argument('--dataset-root', type=Path, default=DATASET)
    parser.add_argument('--cpp-root', type=Path, default=CPP_ROOT)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs/test_real_2d_subject')
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--slice-start', type=int, default=0)
    parser.add_argument('--slice-stop', type=int, default=None)
    args = parser.parse_args(argv)

    return run_subject(
        args.subject, sequence=args.sequence, dataset_root=args.dataset_root,
        cpp_root=args.cpp_root, output_root=args.output_root, device=args.device,
        max_workers=args.max_workers, slice_start=args.slice_start, slice_stop=args.slice_stop,
    )


if __name__ == '__main__':
    main()

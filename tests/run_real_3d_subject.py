#!/usr/bin/env python
"""Run the configured 3D breast reconstruction for 0079 and compare it with GRICS++.

The reconstruction parameters come exclusively from
``config/reconstruction/nonrigid_3d_breast.toml``.  The script only selects the
input, runtime device, and output folder.  It writes the pipeline outputs plus
``comparison_vs_grics_cpp.json`` and ``comparison_vs_grics_cpp.png``.

Example:
    python tests/run_real_3d_subject.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity

from pipelines.siemens_breast_3d_lowres import run_pipeline

DATA = Path('/home/pyuser/wkdir/data')
DATASET_ROOT = DATA / 'GRICS-torch/article_dataset_3D'
RAW_ROOT = DATA / 'Breast-INNOV_GRICS_database'
XML_NAME = 'ParamGRICS++_GRE_3D_LR.xml'


def _load_cpp_volume(folder: Path) -> np.ndarray:
    xml = ET.parse(folder / XML_NAME).getroot()
    dimensions = xml.find('./PreProcessing/Dimensions')
    nx, ny, nz = (int(dimensions.attrib[key]) for key in ('Nx', 'Ny', 'Nz'))
    name = xml.find('./PostProcessing/ReconstructedImage').attrib['FileName']
    path = folder / f'{name}.0000'
    values = np.fromfile(path, dtype='<c8')
    if values.size != nx * ny * nz:
        raise ValueError(f'{path}: expected {nx * ny * nz} complex values, got {values.size}.')
    # GRICS++ writes native axial partitions, then transpose into [x, y, z].
    return np.abs(values.reshape(nz, nx, ny).transpose(1, 2, 0)).copy()


def _torch_magnitude(image: torch.Tensor) -> np.ndarray:
    image = torch.as_tensor(image).detach().cpu()
    if image.ndim != 3:
        raise ValueError(f'Expected spatial image [Nx, Ny, Nz], got {tuple(image.shape)}.')
    return image.abs().numpy()


def _slice_metrics(torch_slice: np.ndarray, cpp_slice: np.ndarray) -> dict[str, float]:
    denominator = float(np.vdot(torch_slice.ravel(), torch_slice.ravel()).real)
    reference_norm = float(np.linalg.norm(cpp_slice))
    if denominator <= 0 or reference_norm <= 0:
        raise ValueError('A comparison slice has zero magnitude.')
    scale = float(np.vdot(torch_slice.ravel(), cpp_slice.ravel()).real / denominator)
    aligned = torch_slice * scale
    data_range = float(max(aligned.max(), cpp_slice.max()) - min(aligned.min(), cpp_slice.min()))
    if data_range <= 0:
        raise ValueError('A comparison slice has zero range.')
    return {
        'scale_torch_to_grics_cpp': scale,
        'nrmse': float(np.linalg.norm(aligned - cpp_slice) / reference_norm),
        'ssim': float(structural_similarity(cpp_slice, aligned, data_range=data_range)),
    }


def _selected_planes(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the slices used by article/evaluate_grics_cpp_3d.py."""
    nx, ny, nz = image.shape
    return image[:, :, nz // 2], image[:, 3 * ny // 4, :], image[3 * nx // 4 - 8, :, :]


def _oriented_plane(partition: np.ndarray, plane_number: int) -> np.ndarray:
    """Match the article evaluator display orientation."""
    displayed = np.flipud(partition.T)
    return np.rot90(displayed) if plane_number == 0 else np.flipud(displayed)


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        'axial_slice_count': len(rows),
        'mean_nrmse': float(np.mean([row['nrmse'] for row in rows])),
        'std_nrmse': float(np.std([row['nrmse'] for row in rows])),
        'mean_ssim': float(np.mean([row['ssim'] for row in rows])),
        'std_ssim': float(np.std([row['ssim'] for row in rows])),
    }


def _metric_label(metrics: dict[str, float]) -> str:
    return ("Whole volume\n"
            f"NRMSE {metrics['nrmse']:.4f}\n"
            f"SSIM {metrics['ssim']:.4f}")


def _write_comparison(torch_image: np.ndarray, cpp_image: np.ndarray, run_folder: Path,
                      cpp_folder: Path) -> dict:
    if torch_image.shape != cpp_image.shape:
        raise ValueError(f'Torch/GRICS++ shape mismatch: {torch_image.shape} versus {cpp_image.shape}.')
    rows = [{'axial_slice': z + 1, **_slice_metrics(torch_image[:, :, z], cpp_image[:, :, z])}
            for z in range(torch_image.shape[2])]
    summary = _summary(rows)
    volume_scale = float(np.vdot(torch_image.ravel(), cpp_image.ravel()).real /
                         np.vdot(torch_image.ravel(), torch_image.ravel()).real)
    aligned_torch = torch_image * volume_scale
    volume_range = float(max(aligned_torch.max(), cpp_image.max()) -
                         min(aligned_torch.min(), cpp_image.min()))
    if volume_range <= 0:
        raise ValueError('The comparison volumes have zero range.')
    volume_metrics = {
        'nrmse': float(np.linalg.norm(aligned_torch - cpp_image) / np.linalg.norm(cpp_image)),
        'ssim': float(structural_similarity(cpp_image, aligned_torch, data_range=volume_range)),
    }
    folder = run_folder / 'comparison'
    folder.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 2, figsize=(10, 13), constrained_layout=True)
    planes = ('Axial (z=1/2)', 'Sagittal (y=3/4)', 'Coronal (x=3/4-10)')
    images = (('GRICS-torch', aligned_torch), ('GRICS++', cpp_image))
    vmax = max(float(np.quantile(image, .995)) for _, image in images)
    for row, plane_name in enumerate(planes):
        for column, (name, image) in enumerate(images):
            axis = axes[row, column]
            axis.imshow(_oriented_plane(_selected_planes(image)[row], row), cmap='gray',
                        origin='upper', vmin=0, vmax=vmax)
            axis.set_title(f'{plane_name}: {name}')
            axis.axis('off')
            if row == 0:
                axis.text(.02, .02, _metric_label(volume_metrics), transform=axis.transAxes,
                          va='bottom', ha='left', fontsize=9,
                          bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
    figure.savefig(folder / 'comparison_vs_grics_cpp.png', dpi=180)
    plt.close(figure)
    result = {
        'grics_cpp_folder': str(cpp_folder),
        'volume_scale_torch_to_grics_cpp': volume_scale,
        'volume_metrics': volume_metrics,
        'axial_metrics': summary,
        'per_axial_slice': rows,
        'comparison_image': str(folder / 'comparison_vs_grics_cpp.png'),
    }
    (folder / 'comparison_vs_grics_cpp.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def run_subject(subject: str, *, sequence: str = 's', dataset_root: Path = DATASET_ROOT,
                raw_root: Path = RAW_ROOT, output_root: Path = REPO_ROOT / 'runs/test_real_3d_subject',
                device: str = 'gpu') -> dict:
    """Reconstruct one 3D subject and compare it with its GRICS++ result."""
    if not re.fullmatch(r'\d{4}', subject):
        raise ValueError(f'Subject must be a four-digit identifier, got {subject!r}.')
    if sequence not in ('s', 'm'):
        raise ValueError(f'Sequence must be "s" or "m", got {sequence!r}.')
    acquisition = f'{subject}_T1_{sequence}'
    prepared = Path(dataset_root) / f'{acquisition}.h5'
    raw_root = Path(raw_root)
    ismrmrd_file = raw_root / 'ISMRMRD' / f'{acquisition}.h5'
    saec_file = raw_root / 'SAEC' / f'{acquisition}.h5'
    cpp_folder = raw_root / 'GRICS-BELT-3D' / acquisition
    if not prepared.is_file():
        for path in (ismrmrd_file, saec_file):
            if not path.is_file():
                raise FileNotFoundError(path)
    if prepared.is_file():
        with h5py.File(prepared, 'r') as source:
            if source['kspace'].ndim != 5:
                raise ValueError('Prepared 3D k-space must have five dimensions.')
    # A prepared HDF5 is itself a valid pipeline input. Passing it directly
    # works with the main-branch API that predates ``preprocessed_file`` and
    # with the current API.
    input_file, input_saec_file = (prepared, None) if prepared.is_file() else (ismrmrd_file, saec_file)
    result = run_pipeline(
        input_file, input_saec_file, output_root=output_root, device=device,
        return_tensors=True, export_dicom=False,
        prepared_output_file=prepared if not prepared.is_file() else None,
    )
    volume = result['reconstructions'][0]
    comparison = _write_comparison(_torch_magnitude(volume['image']), _load_cpp_volume(cpp_folder),
                                   Path(result['run_folder']), cpp_folder)
    print(json.dumps(comparison['volume_metrics'], indent=2))
    print(f"[outputs] {result['run_folder']}")
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', default='0079', help='Four-digit subject identifier.')
    parser.add_argument('--sequence', choices=('s', 'm'), default='s')
    parser.add_argument('--dataset-root', type=Path, default=DATASET_ROOT)
    parser.add_argument('--raw-root', type=Path, default=RAW_ROOT)
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--output-root', type=Path, default=REPO_ROOT / 'runs/test_real_3d_subject')
    args = parser.parse_args(argv)
    return run_subject(args.subject, sequence=args.sequence, dataset_root=args.dataset_root,
                       raw_root=args.raw_root, output_root=args.output_root, device=args.device)


if __name__ == '__main__':
    main()

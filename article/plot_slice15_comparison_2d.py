#!/usr/bin/env python
"""Plot slice 15, uncorrected versus corrected, for C++ and Torch 2D runs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/pyuser/wkdir/data')
SUBJECT = re.compile(r'00[0-9]{2}_T2_[a-z]', re.I)
SLICE_NAME = 'Siemens_SingleImage_slice015_image01'


def read_cpp_image(folder: Path) -> np.ndarray:
    xml = ET.parse(folder / 'ParamGRICS++_TSE_Breast.xml').getroot()
    dims = xml.find('./PreProcessing/Dimensions')
    nx, ny, nz = (int(dims.attrib[key]) for key in ('Nx', 'Ny', 'Nz'))
    if nz != 1:
        raise ValueError(f'{folder}: expected Nz=1')
    filename = xml.find('./PostProcessing/ReconstructedImage').attrib['FileName']
    path = folder / f'{filename}.0000'
    image = np.fromfile(path, dtype='<c8').reshape(nx, ny)
    return np.abs(image)


def read_torch_image(run_folder: Path) -> np.ndarray:
    path = run_folder / 'reconstructions' / 'slice_015' / 'results' / 'image_reconstructed.pt'
    image = torch.load(path, map_location='cpu', weights_only=True)
    image = torch.as_tensor(image)
    if image.ndim == 3:
        image = image.mean(dim=0)
    elif image.ndim != 2:
        raise ValueError(f'{path}: expected 2D or repetition-stacked 2D image, got {tuple(image.shape)}')
    return image.abs().numpy()


def plot(rows, title, output: Path):
    values = [image for row in rows for image in row['images']]
    finite = np.concatenate([image[np.isfinite(image)].ravel() for image in values])
    low, high = np.percentile(finite, (1, 99.5))
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, max(3, 2.8 * len(rows))), squeeze=False)
    for index, row in enumerate(rows):
        for column, (axis, image, mode) in enumerate(zip(axes[index], row['images'], ('Uncorrected', 'Corrected'))):
            axis.imshow(image.T, cmap='gray', vmin=low, vmax=high, origin='lower', aspect='auto')
            axis.set_title(mode if index == 0 else '')
            axis.axis('off')
        axes[index, 0].set_ylabel(row['subject'], rotation=90, size=9, labelpad=12)
    fig.suptitle(f'{title}: slice 15 magnitude', y=.995)
    fig.subplots_adjust(left=.08, right=.99, bottom=.01, top=.96, wspace=.02, hspace=.08)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpp-root', type=Path, default=DATA / 'Breast-INNOV_GRICS_database/GRICS-BELT')
    parser.add_argument('--cpp-nomoco-root', type=Path, default=DATA / 'Breast-INNOV_GRICS_database/GRICS-BELT_nomoco')
    parser.add_argument('--torch-measurements', type=Path, default=DATA / 'GRICS-torch/article_dataset_2D/results/measurements.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'article')
    parser.add_argument('--cpp-output-name', default='slice15_grics_cpp_2d.png')
    parser.add_argument('--torch-output-name', default='slice15_grics_torch_2d.png')
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cpp_rows = []
    for subject_dir in sorted(args.cpp_root.iterdir()):
        if not subject_dir.is_dir() or not SUBJECT.fullmatch(subject_dir.name):
            continue
        corrected = subject_dir / SLICE_NAME
        uncorrected = args.cpp_nomoco_root / subject_dir.name / SLICE_NAME
        if corrected.is_dir() and uncorrected.is_dir():
            cpp_rows.append({'subject': subject_dir.name,
                             'images': [read_cpp_image(uncorrected), read_cpp_image(corrected)]})

    measurements = json.loads(args.torch_measurements.read_text())
    by_key = {(Path(row['subject']).stem, row['mode']): row for row in measurements}
    torch_rows = []
    subjects = sorted({subject for subject, mode in by_key if mode == 'corrected'})
    for subject in subjects:
        if (subject, 'nomoco') not in by_key:
            continue
        uncorrected = read_torch_image(Path(by_key[subject, 'nomoco']['run_folder']))
        corrected = read_torch_image(Path(by_key[subject, 'corrected']['run_folder']))
        torch_rows.append({'subject': subject, 'images': [uncorrected, corrected]})

    cpp_output = args.output_dir / args.cpp_output_name
    torch_output = args.output_dir / args.torch_output_name
    plot(cpp_rows, 'GRICS++', cpp_output)
    plot(torch_rows, 'GRICS-torch', torch_output)
    print(f'GRICS++: {len(cpp_rows)} subjects -> {cpp_output}')
    print(f'GRICS-torch: {len(torch_rows)} subjects -> {torch_output}')


if __name__ == '__main__':
    main()

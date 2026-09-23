#!/usr/bin/env python
"""Opt-in integration test: reconstruct one real 3D subject through the Siemens pipeline.

The article branch's run_dataset_reconstruction_GRICS-torch_3D.py selects
NNNN_T1_[sm].h5 files from the default dataset directory below. This test uses
the current pipeline, including Odille spline sensitivity maps and its full
reconstruction settings. If the preprocessed file is absent, the matching
ISMRMRD and SAEC files are used. It is excluded from unittest discovery.

Examples (from the repository root):
    python tests/run_real_3d_subject.py
    python tests/run_real_3d_subject.py --dataset-root /path/to/article_dataset_3D
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import torch

from pipelines.siemens_breast_3d_lowres import run_pipeline

DEFAULT_DATASET_ROOT = Path('/home/pyuser/wkdir/data/GRICS-torch/article_dataset_3D')
# Fixed subject for reproducible integration runs; change it here when needed.
SUBJECT_FILENAME = '0072_T1_s.h5'
RAW_DATA_ROOT = Path('/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database')
DEFAULT_ISMRMRD_FILE = RAW_DATA_ROOT / 'ISMRMRD' / SUBJECT_FILENAME
DEFAULT_SAEC_FILE = RAW_DATA_ROOT / 'SAEC' / SUBJECT_FILENAME


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run_test(subject_file: Path, *, device: str, output_root: Path,
             ismrmrd_file: Path = DEFAULT_ISMRMRD_FILE,
             saec_file: Path = DEFAULT_SAEC_FILE) -> dict:
    shape = None
    if subject_file.is_file():
        # Read only the shape here; the pipeline loads the volume once.
        with h5py.File(subject_file, 'r') as source:
            shape = source['kspace'].shape
        require(len(shape) == 5 and shape[-1] > 1,
                f'Expected kspace [coils, repetitions, x, y, z] with z > 1, got {shape}.')
    selected_input = subject_file if subject_file.is_file() else ismrmrd_file
    print(f'[test] Subject: {selected_input}', flush=True)
    print('[test] Running full 3D pipeline configuration with Odille spline coil maps.', flush=True)
    result = run_pipeline(
        ismrmrd_file, saec_file, preprocessed_file=subject_file,
        output_root=output_root, device=device,
        save_reconstruction_logs=True, save_reconstruction_tensors=True,
        return_tensors=True, export_dicom=True,
    )
    require(len(result['reconstructions']) == 1, 'Expected exactly one reconstructed volume.')
    volume = result['reconstructions'][0]
    image, motion = volume['image'], volume['motion']
    require(image.ndim == 4 and image.shape[-1] > 1,
            f'Expected image [repetitions, x, y, z], got {tuple(image.shape)}.')
    if shape is not None:
        require(tuple(image.shape) == tuple(shape[1:]),
                f'Image shape {tuple(image.shape)} does not match input {shape[1:]}.')
    require(torch.is_complex(image), 'Expected a complex reconstructed image.')
    require(bool(torch.isfinite(image).all()), 'Image contains non-finite values.')
    require(bool(torch.count_nonzero(image)), 'Reconstructed image is entirely zero.')
    require(motion.ndim == 5 and tuple(motion.shape[:4]) == (3, *image.shape[1:])
            and motion.shape[-1] > 0,
            f'Unexpected nonrigid motion shape: {tuple(motion.shape)}.')
    require(bool(torch.isfinite(motion).all()), 'Motion parameters contain non-finite values.')
    for key in ('image_file', 'motion_file', 'log_file'):
        path = volume[key]
        require(path is not None and path.is_file() and path.stat().st_size > 0,
                f'Missing or empty {key}: {path}')
    require(len(volume['dicom_files']) == image.shape[-1], 'Expected one DICOM per partition.')
    require(all(path.is_file() for path in volume['dicom_files']), 'Missing DICOM export files.')
    run_folder = result['run_folder']
    manifest = json.loads((run_folder / 'manifest.json').read_text())
    config = json.loads((run_folder / 'config_resolved.json').read_text())
    require(manifest['status'] == 'complete', 'Run manifest is not complete.')
    require(manifest['inputs']['raw_data_file'] == str(selected_input.resolve()),
            'Run provenance points to a different subject.')
    require(config['coil_sensitivity_method'] == 'odille-spline', 'Unexpected coil sensitivity method.')
    require(config['data_dimension'] == '3D', 'Expected a 3D reconstruction.')
    require(result['timings']['reconstruction_seconds'] > 0, 'Missing solver timing.')
    print(f"[PASS] {subject_file.name}; actual device: {config['runtime_device']}")
    print(f"Image: {tuple(image.shape)}; motion: {tuple(motion.shape)}")
    print(f"Reconstruction including logs: {result['timings']['reconstruction_seconds']:.3f} s")
    print(f'Outputs: {run_folder}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT,
                        help='Directory of preprocessed NNNN_T1_[sm].h5 volumes.')
    parser.add_argument('--ismrmrd-file', type=Path, default=DEFAULT_ISMRMRD_FILE,
                        help='ISMRMRD acquisition used if the preprocessed file is missing.')
    parser.add_argument('--saec-file', type=Path, default=DEFAULT_SAEC_FILE,
                        help='Matching SAEC physiology used with the ISMRMRD acquisition.')
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--output-root', type=Path, default=REPO_ROOT / 'runs/test_real_3d_subject')
    args = parser.parse_args(argv)
    subject_file = args.dataset_root.expanduser() / SUBJECT_FILENAME
    if not subject_file.is_file():
        for path in (args.ismrmrd_file.expanduser(), args.saec_file.expanduser()):
            if not path.is_file():
                parser.error(f'Fallback input does not exist: {path}')
    run_test(subject_file, device=args.device, output_root=args.output_root.expanduser(),
             ismrmrd_file=args.ismrmrd_file.expanduser(), saec_file=args.saec_file.expanduser())


if __name__ == '__main__':
    main()

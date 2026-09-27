#!/usr/bin/env python
"""Compare the corrected 3D state hierarchy with a saved 0079 baseline.

By default, this runner reconstructs the acquisition once with the current
configuration and compares it with the saved pre-virtual-state shared-image
reconstruction.  ``--nex-comparison`` retains the earlier two-run independent
versus shared NEX experiment.

Examples (from the repository root):

    python tests/run_real_3d_subject.py
    python tests/run_real_3d_subject.py --baseline-run-folder /path/to/old/run
    python tests/run_real_3d_subject.py --nex-comparison
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from article.evaluate_breast_3d import (
    PHYSIOLOGY_PREPROCESSING, prepare_subject,
)
from pipelines.siemens_breast_3d_lowres import RECONSTRUCTION_CONFIG, run_pipeline
from src.runtime.runtime_config import load_config

DEFAULT_DATASET_ROOT = Path('/home/pyuser/wkdir/data/GRICS-torch/article_dataset_3D')
SUBJECT_FILENAME = '0079_T1_s.h5'
RAW_DATA_ROOT = Path('/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database')
DEFAULT_ISMRMRD_FILE = RAW_DATA_ROOT / 'ISMRMRD' / SUBJECT_FILENAME
DEFAULT_SAEC_FILE = RAW_DATA_ROOT / 'SAEC' / SUBJECT_FILENAME
DEFAULT_BASELINE_RUN_FOLDER = (
    DEFAULT_DATASET_ROOT / 'results/0079_independent_vs_shared_nex/shared'
    / '20260926T145743965727-06ccc999'
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def ensure_current_preprocessing(subject_file, *, ismrmrd_file, saec_file):
    """Rebuild the prepared volume when its physiology predates the IADI port."""
    current = False
    if subject_file.is_file():
        with h5py.File(subject_file, 'r') as source:
            current = source.attrs.get('physiology_preprocessing') == PHYSIOLOGY_PREPROCESSING
    if current:
        print(f'[prepare] Using {PHYSIOLOGY_PREPROCESSING}: {subject_file}', flush=True)
        return subject_file
    print(f'[prepare] Rebuilding {subject_file} with {PHYSIOLOGY_PREPROCESSING}', flush=True)
    params = load_config(
        data_type='ismrmrd-saec',
        reconstruction_config=REPO_ROOT / RECONSTRUCTION_CONFIG,
        coil_sensitivity_config=REPO_ROOT / 'config/coil_sensitivity/odille_spline.toml',
        real_data_config=REPO_ROOT / 'config/real_data/saec.toml',
        ismrmrd_reader_config=REPO_ROOT / 'config/real_data/ismrmrd_reader.toml',
    )
    return prepare_subject(ismrmrd_file, saec_file, subject_file, params)


def magnitude_image(image):
    require(image.ndim == 4, f'Expected [image unknown, x, y, z], got {tuple(image.shape)}.')
    return image.abs().mean(dim=0)


def selected_planes(image):
    nx, ny, nz = image.shape
    return (
        image[:, :, nz // 2],
        image[:, 3 * ny // 4, :],
        image[3 * nx // 4 - 10, :, :],
    )


def oriented_plane(partition, plane_number):
    displayed = np.flipud(partition.T)
    return np.rot90(displayed) if plane_number == 0 else np.flipud(displayed)


def normalized(image):
    scale = torch.quantile(image.flatten(), 0.995)
    require(torch.isfinite(scale) and scale > 0, 'Cannot normalize a nonpositive image.')
    return image / scale, float(scale)


def save_comparison(independent, shared, output_root, *, first_label, second_label, stem, title):
    independent_norm, independent_scale = normalized(independent)
    shared_norm, shared_scale = normalized(shared)
    fit_scale = float(
        torch.sum(independent_norm * shared_norm)
        / torch.sum(shared_norm.square()).clamp_min(torch.finfo(shared_norm.dtype).eps)
    )
    shared_matched = shared_norm * fit_scale
    difference = (independent_norm - shared_matched).abs()

    first = independent_norm.flatten()
    second = shared_matched.flatten()
    correlation = float(torch.corrcoef(torch.stack((first, second)))[0, 1])
    relative_l2 = float(torch.linalg.norm(difference) / torch.linalg.norm(independent_norm))
    comparison = {
        'first_p995': independent_scale,
        'second_p995': shared_scale,
        'second_to_first_scale': fit_scale,
        'magnitude_correlation': correlation,
        'magnitude_relative_l2_difference': relative_l2,
    }

    planes = ('Axial: centre', 'Sagittal: 3/4', 'Coronal: 3/4 - 10')
    figure, axes = plt.subplots(3, 3, figsize=(11, 11), squeeze=False)
    for row, label in enumerate(planes):
        independent_plane = oriented_plane(selected_planes(independent_norm.numpy())[row], row)
        shared_plane = oriented_plane(selected_planes(shared_matched.numpy())[row], row)
        difference_plane = oriented_plane(selected_planes(difference.numpy())[row], row)
        vmax = max(np.percentile(independent_plane, 99.5), np.percentile(shared_plane, 99.5))
        dmax = max(np.percentile(difference_plane, 99.5), np.finfo(float).eps)
        for col, (panel, title, maximum) in enumerate((
            (independent_plane, first_label, vmax),
            (shared_plane, second_label, vmax),
            (difference_plane, 'Absolute difference', dmax),
        )):
            axis = axes[row, col]
            axis.imshow(panel, cmap='gray', origin='upper', vmin=0, vmax=maximum, aspect='auto')
            if row == 0:
                axis.set_title(title)
            if col == 0:
                axis.set_ylabel(label)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        f'0079: {title}\n'
        f'magnitude correlation={correlation:.5f}; relative L2 difference={relative_l2:.5f}'
    )
    figure.tight_layout()
    image_path = output_root / f'{stem}_planes.png'
    figure.savefig(image_path, dpi=180)
    plt.close(figure)
    return comparison, image_path


def run_one(subject_file, *, ismrmrd_file, saec_file, output_root, device, model, input_shape,
            grics_cpp_sensitivity_maps=None):
    result = run_pipeline(
        ismrmrd_file, saec_file, preprocessed_file=subject_file,
        output_root=output_root / model, device=device,
        overrides={'repetition_image_model': model},
        save_reconstruction_logs=True, save_reconstruction_tensors=True,
        return_tensors=True, export_dicom=False,
        grics_cpp_sensitivity_maps=grics_cpp_sensitivity_maps,
    )
    require(len(result['reconstructions']) == 1, 'Expected exactly one reconstructed volume.')
    volume = result['reconstructions'][0]
    image, motion = volume['image'], volume['motion']
    expected_images = input_shape[1] if model == 'independent' else 1
    expected_shape = (expected_images, *input_shape[2:])
    require(tuple(image.shape) == expected_shape,
            f'{model}: image shape {tuple(image.shape)} differs from {expected_shape}.')
    require(torch.is_complex(image) and bool(torch.isfinite(image).all()),
            f'{model}: reconstructed image is invalid.')
    require(bool(torch.count_nonzero(image)), f'{model}: reconstructed image is zero.')
    require(motion.ndim == 5 and tuple(motion.shape[:4]) == (3, *image.shape[1:])
            and motion.shape[-1] > 0 and bool(torch.isfinite(motion).all()),
            f'{model}: invalid nonrigid motion shape {tuple(motion.shape)}.')
    for key in ('image_file', 'motion_file', 'log_file'):
        value = volume[key]
        require(value is not None and value.is_file() and value.stat().st_size > 0,
                f'{model}: missing or empty {key}: {value}')
    config = json.loads((result['run_folder'] / 'config_resolved.json').read_text())
    require(config['Nex'] == input_shape[1], f'{model}: acquired NEX changed unexpectedly.')
    require(config['repetition_image_model'] == model,
            f'{model}: resolved image model is {config["repetition_image_model"]!r}.')
    return result


def run_test(subject_file: Path, *, device: str, output_root: Path,
             ismrmrd_file: Path = DEFAULT_ISMRMRD_FILE,
             saec_file: Path = DEFAULT_SAEC_FILE,
             grics_cpp_sensitivity_maps: Path | None = None) -> dict:
    require(subject_file.is_file(), f'Missing prepared subject: {subject_file}')
    with h5py.File(subject_file, 'r') as source:
        input_shape = source['kspace'].shape
    require(len(input_shape) == 5 and input_shape[-1] > 1,
            f'Expected kspace [coils, repetitions, x, y, z], got {input_shape}.')
    require(input_shape[1] == 2,
            f'0079 comparison expects two acquired repetitions, got {input_shape[1]}.')

    output_root.mkdir(parents=True, exist_ok=True)
    print(f'[test] Subject: {subject_file}', flush=True)
    independent = run_one(
        subject_file, ismrmrd_file=ismrmrd_file, saec_file=saec_file,
        output_root=output_root, device=device, model='independent', input_shape=input_shape,
        grics_cpp_sensitivity_maps=grics_cpp_sensitivity_maps,
    )
    shared = run_one(
        subject_file, ismrmrd_file=ismrmrd_file, saec_file=saec_file,
        output_root=output_root, device=device, model='shared', input_shape=input_shape,
        grics_cpp_sensitivity_maps=grics_cpp_sensitivity_maps,
    )
    metrics, image_path = save_comparison(
        magnitude_image(independent['reconstructions'][0]['image']),
        magnitude_image(shared['reconstructions'][0]['image']),
        output_root,
        first_label='Independent NEX images, averaged', second_label='One shared image',
        stem='independent_vs_shared_nex', title='independent versus shared NEX image',
    )
    summary = {
        'subject': subject_file.name,
        'input_kspace_shape': list(input_shape),
        'independent_run_folder': str(independent['run_folder']),
        'shared_run_folder': str(shared['run_folder']),
        'comparison_image': str(image_path),
        **metrics,
    }
    summary_path = output_root / 'independent_vs_shared_nex_metrics.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'[PASS] {subject_file.name}', flush=True)
    print(f'[comparison] {image_path}', flush=True)
    print(f'[metrics] {summary_path}', flush=True)
    return summary


def run_saved_baseline_test(subject_file: Path, *, device: str, output_root: Path,
                            baseline_run_folder: Path,
                            ismrmrd_file: Path = DEFAULT_ISMRMRD_FILE,
                            saec_file: Path = DEFAULT_SAEC_FILE,
                            grics_cpp_sensitivity_maps: Path | None = None) -> dict:
    require(subject_file.is_file(), f'Missing prepared subject: {subject_file}')
    baseline_image_file = baseline_run_folder / 'reconstructions/volume_001/results/image_reconstructed.pt'
    baseline_config_file = baseline_run_folder / 'config_resolved.json'
    require(baseline_image_file.is_file(), f'Missing saved baseline image: {baseline_image_file}')
    require(baseline_config_file.is_file(), f'Missing saved baseline config: {baseline_config_file}')
    baseline_config = json.loads(baseline_config_file.read_text())
    require(baseline_config['repetition_image_model'] == 'shared',
            'The saved baseline must be a shared-image reconstruction.')
    with h5py.File(subject_file, 'r') as source:
        input_shape = source['kspace'].shape
    require(len(input_shape) == 5 and input_shape[1] == 2,
            f'Expected 0079 kspace [coils, 2, x, y, z], got {input_shape}.')

    output_root.mkdir(parents=True, exist_ok=True)
    print(f'[test] Subject: {subject_file}', flush=True)
    print(f'[baseline] {baseline_run_folder}', flush=True)
    current = run_one(
        subject_file, ismrmrd_file=ismrmrd_file, saec_file=saec_file,
        output_root=output_root, device=device, model='shared', input_shape=input_shape,
        grics_cpp_sensitivity_maps=grics_cpp_sensitivity_maps,
    )
    current_config = json.loads((current['run_folder'] / 'config_resolved.json').read_text())
    require(current_config['kspace_energy_level_source'] == 'quantized_virtual_states',
            'The new run did not use quantized virtual states at each resolution.')
    baseline_image = torch.load(baseline_image_file, map_location='cpu', weights_only=True)
    require(tuple(baseline_image.shape) == tuple(current['reconstructions'][0]['image'].shape),
            f'Baseline image shape {tuple(baseline_image.shape)} does not match the new run.')
    require(torch.is_complex(baseline_image) and bool(torch.isfinite(baseline_image).all()),
            'The saved baseline image is invalid.')
    metrics, image_path = save_comparison(
        magnitude_image(baseline_image),
        magnitude_image(current['reconstructions'][0]['image'].detach().cpu()),
        output_root,
        first_label='Saved baseline: selected states',
        second_label='New: quantized virtual states',
        stem='selected_vs_quantized_virtual_states',
        title='selected states versus quantized virtual states',
    )
    summary = {
        'subject': subject_file.name,
        'input_kspace_shape': list(input_shape),
        'baseline_run_folder': str(baseline_run_folder),
        'new_run_folder': str(current['run_folder']),
        'comparison_image': str(image_path),
        **metrics,
    }
    summary_path = output_root / 'selected_vs_quantized_virtual_states_metrics.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'[PASS] {subject_file.name}', flush=True)
    print(f'[comparison] {image_path}', flush=True)
    print(f'[metrics] {summary_path}', flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--ismrmrd-file', type=Path, default=DEFAULT_ISMRMRD_FILE)
    parser.add_argument('--saec-file', type=Path, default=DEFAULT_SAEC_FILE)
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--output-root', type=Path,
                        default=REPO_ROOT / 'runs/0079_quantized_virtual_state_comparison')
    parser.add_argument('--baseline-run-folder', type=Path, default=DEFAULT_BASELINE_RUN_FOLDER,
                        help='Saved pre-fix shared-image run used by the default one-run comparison.')
    parser.add_argument('--nex-comparison', action='store_true',
                        help='Run the historical two-reconstruction independent versus shared NEX experiment.')
    parser.add_argument('--grics-cpp-sensitivity-maps', type=Path,
                        help='Optional GRICS++ SensitivityMaps.dat to use for the new reconstruction.')
    args = parser.parse_args(argv)
    common = dict(
        device=args.device, output_root=args.output_root.expanduser(),
        ismrmrd_file=args.ismrmrd_file.expanduser(), saec_file=args.saec_file.expanduser(),
        grics_cpp_sensitivity_maps=(args.grics_cpp_sensitivity_maps.expanduser()
                                    if args.grics_cpp_sensitivity_maps is not None else None),
    )
    subject_file = args.dataset_root.expanduser() / SUBJECT_FILENAME
    subject_file = ensure_current_preprocessing(
        subject_file, ismrmrd_file=common['ismrmrd_file'], saec_file=common['saec_file'])
    if args.nex_comparison:
        run_test(subject_file, **common)
    else:
        run_saved_baseline_test(
            subject_file, baseline_run_folder=args.baseline_run_folder.expanduser(), **common,
        )


if __name__ == '__main__':
    main()

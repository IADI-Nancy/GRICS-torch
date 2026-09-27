#!/usr/bin/env python
"""Reconstruct 0079 with the article-validation revision and compare it to release 1.

This runner deliberately uses the historical 3D pipeline unchanged.  Its defaults
are the exact preprocessed input and saved 0079 corrected volume produced by the
article-validation/release-1 workflow:

* ``article_dataset_3D/0079_T1_s.h5``;
* Odille spline sensitivity maps; and
* the historical ``nonrigid_3d_breast.toml`` settings (including 15 motion-CG
  iterations).

It writes the normal reconstruction outputs plus ``comparison.json`` and
three separate three-plane plots below the new run directory.  The shared
NEX result from the ``test`` branch is compared against three release-1
references: the mean of both NEX images, NEX 1, and NEX 2.  It reports errors
without imposing a similarity threshold, because repeated GPU runs need not be
bitwise identical.

Run from this historical worktree:

    python tests/reproduce_release1_3d.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.siemens_breast_3d_lowres import run_pipeline


DATA_ROOT = Path('/home/pyuser/wkdir/data/GRICS-torch')
DEFAULT_INPUT = DATA_ROOT / 'article_dataset_3D/0079_T1_s.h5'
DEFAULT_BASELINE = (DATA_ROOT / 'article_dataset_3D_old/results/runs/0079_T1_s/corrected/'
                    '20260923T114822516683-f18c5a40/reconstructions/volume_001/results/'
                    'image_reconstructed.pt')
DEFAULT_TEST_BRANCH_IMAGE = (ROOT / 'runs/0079_quantized_virtual_state_comparison/shared/'
                             '20260927T073511398806-bc42ff2c/reconstructions/volume_001/results/'
                             'image_reconstructed.pt')
DEFAULT_OUTPUT_ROOT = ROOT / 'runs/release1_3d_reproduction'

# These are recorded in the reference run's config_resolved.json.  They guard
# against silently comparing an altered historical configuration.
REFERENCE_CONFIG = {
    'coil_sensitivity_method': 'odille-spline',
    'N_motion_states': 16,
    'N_motion_states_per_level': 'full',
    'GN_iterations_per_level': [8, 8, 2],
    'ResolutionLevels': [0.25, 0.5, 1.0],
    'lambda_r': 0.05,
    'lambda_m': 10.0,
    'max_iter_recon': 10,
    'max_iter_motion': 15,
    'tol_recon': 0.001,
    'tol_motion': 0.01,
    'motion_binning_mode': 'kspace_energy',
    'motion_quantization_bins': 256,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help=f'Preprocessed release-1 input (default: {DEFAULT_INPUT}).')
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE,
                        help=f'Saved release-1 corrected image (default: {DEFAULT_BASELINE}).')
    parser.add_argument('--test-branch-image', type=Path, default=DEFAULT_TEST_BRANCH_IMAGE,
                        help=f'Saved 0079 image from the test branch (default: {DEFAULT_TEST_BRANCH_IMAGE}).')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help=f"Parent for this run's outputs (default: {DEFAULT_OUTPUT_ROOT}).")
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='gpu',
                        help='Reconstruction device (default: gpu).')
    args = parser.parse_args(argv)
    for name in ('input', 'baseline', 'test_branch_image'):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            parser.error(f'--{name} does not exist: {path}')
        setattr(args, name, path)
    args.output_root = args.output_root.expanduser().resolve()
    return args


def validate_reference_config(run_folder: Path):
    config = json.loads((run_folder / 'config_resolved.json').read_text())
    mismatches = {key: {'expected': expected, 'actual': config.get(key)}
                  for key, expected in REFERENCE_CONFIG.items() if config.get(key) != expected}
    if mismatches:
        raise AssertionError(f'Historical release-1 configuration changed: {mismatches}')
    return config


def load_tensor(path: Path):
    try:
        value = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        value = torch.load(path, map_location='cpu')
    if not torch.is_tensor(value) or value.ndim != 4 or not torch.is_complex(value):
        raise AssertionError(f'{path}: expected complex image [nex, x, y, z], got {type(value)} {getattr(value, "shape", None)}.')
    if not torch.isfinite(value).all():
        raise AssertionError(f'{path}: image contains non-finite values.')
    return value


def image_metrics(image: torch.Tensor, baseline: torch.Tensor, label: str):
    if image.shape != baseline.shape:
        raise AssertionError(f'Image shape changed for {label}: {tuple(image.shape)}; release 1: {tuple(baseline.shape)}.')
    reference_norm = torch.linalg.vector_norm(baseline)
    if reference_norm == 0:
        raise AssertionError('Saved release-1 image is zero.')
    # Fit one complex gain so the metric measures spatial/reconstruction change,
    # rather than a harmless global phase or scale difference.
    gain = torch.vdot(baseline.flatten(), image.flatten()) / torch.vdot(baseline.flatten(), baseline.flatten())
    aligned = image / gain if gain != 0 else image
    return {
        'shape': list(image.shape),
        'relative_complex_l2': float(torch.linalg.vector_norm(image - baseline) / reference_norm),
        'relative_magnitude_l2': float(torch.linalg.vector_norm(image.abs() - baseline.abs()) / reference_norm),
        'relative_complex_l2_after_global_gain': float(torch.linalg.vector_norm(aligned - baseline) / reference_norm),
        'complex_gain_new_over_release1': {'real': float(gain.real), 'imag': float(gain.imag)},
    }


def plane_views(image: torch.Tensor):
    volume = image.abs().mean(dim=0).numpy()  # [x, y, z]
    x, y, z = volume.shape
    return {
        'axial (z center)': volume[:, :, z // 2].T,
        'sagittal (x center)': volume[x // 2, :, :].T,
        'coronal (y center)': volume[:, y // 2, :].T,
    }


def write_figure(image: torch.Tensor, reference: torch.Tensor, output: Path, label: str):
    image_views = plane_views(image)
    reference_views = plane_views(reference)
    vmax = max(float(np.quantile(view, 0.995))
               for view in [*image_views.values(), *reference_views.values()])
    figure, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for row, name in enumerate(image_views):
        expected, actual = reference_views[name], image_views[name]
        difference = np.abs(actual - expected)
        for axis, plane, title, cmap, limit in (
            (axes[row, 0], expected, f'release 1: {label}', 'gray', vmax),
            (axes[row, 1], actual, 'test branch shared NEX', 'gray', vmax),
            (axes[row, 2], difference, '|test - release 1|', 'magma', float(np.quantile(difference, 0.995))),
        ):
            axis.imshow(plane, cmap=cmap, vmin=0, vmax=limit)
            axis.set_title(f'{name}: {title}')
            axis.axis('off')
    figure.savefig(output / f'comparison_{label}.png', dpi=160)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    # Passing only preprocessed_file guarantees this run does not convert raw
    # data or invoke current-worktree preprocessing.  The imported pipeline is
    # from this article-validation checkout.
    result = run_pipeline(
        preprocessed_file=args.input, output_root=args.output_root, device=args.device,
        save_reconstruction_logs=True, save_reconstruction_tensors=True,
        return_tensors=True, export_dicom=False,
    )
    run_folder = Path(result['run_folder'])
    config = validate_reference_config(run_folder)
    current = result['reconstructions'][0]['image'].detach().cpu()
    baseline = load_tensor(args.baseline)
    test_branch = load_tensor(args.test_branch_image)
    release1_references = {
        'mean_nex': baseline.mean(dim=0, keepdim=True),
        'nex_1': baseline[0:1],
        'nex_2': baseline[1:2],
    }
    metrics = {
        'historical_rerun_vs_release1': image_metrics(current, baseline, 'historical rerun'),
        'test_branch_vs_release1': {
            label: image_metrics(test_branch, reference, f'test branch image vs {label}')
            for label, reference in release1_references.items()
        },
    }
    comparison_folder = run_folder / 'comparison'
    comparison_folder.mkdir()
    for label, reference in release1_references.items():
        write_figure(test_branch, reference, comparison_folder, label)
    (comparison_folder / 'comparison.json').write_text(json.dumps({
        'new_run': str(run_folder), 'input': str(args.input), 'baseline': str(args.baseline),
        'test_branch_image': str(args.test_branch_image),
        'historical_code_revision': json.loads((run_folder / 'manifest.json').read_text()).get('code_revision'),
        'reference_config': {key: config[key] for key in REFERENCE_CONFIG},
        'metrics': metrics,
    }, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
    print(f'Comparison artifacts: {comparison_folder}')


if __name__ == '__main__':
    main()

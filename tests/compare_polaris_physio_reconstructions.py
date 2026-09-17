#!/usr/bin/env python
"""Opt-in integration test: reconstruct demo E using Polaris, text, and NPY inputs.

Run from any directory:
    python tests/compare_polaris_physio_reconstructions.py

Defaults come from MedUniVienna:demo_e_Vienna_motion.ipynb, with all XYZ tracks.
This performs three full reconstructions; it is not part of unittest discovery.
Generic readers do not filter/normalize, so export the filtered Polaris signals
with the same full-acquisition centering/scaling used by its prepare_motion().
Normalization is affine and therefore commutes with linear interpolation, apart
from floating-point round-off because Polaris normalizes after interpolation.
Real timestamps exercise end alignment and interpolation in both generic modes.
No image rescaling, registration, or phase alignment is used in the comparison.
"""

import argparse
import gc
import itertools
import json
import os
from pathlib import Path
import random
import sys
import tempfile

# Set before importing numerical libraries. GPU is required by default for this comparison.
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.output_layout import execution_scope
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import cleanup_runtime, initialize_runtime

VIENNA = ROOT.parent / 'data' / 'GRICS-torch' / 'Vienna'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-file', type=Path, default=VIENNA / 'raw_data' /
                        'meas_MID01133_FID01857_AX_T2_TSE_HR_2NEX_SAT_MOTION.dat')
    parser.add_argument('--polaris-file', type=Path, default=VIENNA / 'motion' / 'R1.tsv')
    parser.add_argument('--slice-idx', type=int, default=15, help='Zero-based source slice (demo E: 15).')
    parser.add_argument('--reconstruction-config', type=Path,
                        default=ROOT / 'config/reconstruction/nonrigid_2d.toml')
    parser.add_argument('--coil-sensitivity-config', type=Path,
                        default=ROOT / 'config/coil_sensitivity/odille_spline.toml')
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--rtol', type=float, default=1e-5,
                        help='Maximum relative L2 error in complex and magnitude images.')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs/physio_format_comparison')
    args = parser.parse_args()
    for name in ('raw_file', 'polaris_file', 'reconstruction_config', 'coil_sensitivity_config'):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            parser.error(f'{name}: file does not exist: {path}')
        setattr(args, name, path)
    if args.slice_idx < 0 or not np.isfinite(args.rtol) or args.rtol <= 0:
        parser.error('--slice-idx must be nonnegative and --rtol must be finite and positive.')
    return args


def export_physiology(preparer, folder):
    """Export all filtered tracks on their original physiological sample grid."""
    if preparer.selected_polaris_channels != ['Tx', 'Ty', 'Tz']:
        raise AssertionError('Reference run must use all Polaris translation tracks.')
    sync = preparer.synchronization
    timestamps = np.asarray(sync['physiological_time_seconds'], dtype=np.float64)
    np.testing.assert_array_equal(timestamps, np.broadcast_to(timestamps[0], timestamps.shape))
    filtered = np.column_stack(sync['physiological_values'])
    interpolated = sync['acquisition_values']
    mean = interpolated.mean(axis=0)
    scale = interpolated.std(axis=0).max()
    values = filtered - mean
    if scale > 0:
        values /= scale
    folder.mkdir(parents=True)
    time_file, values_file = folder / 'timestamps.npy', folder / 'values.npy'
    np.save(time_file, timestamps[0][None, :, None], allow_pickle=False)
    np.save(values_file, values[None, :, :], allow_pickle=False)
    text_file = folder / 'physio.txt'
    count = values.shape[0]
    rows = np.column_stack((np.zeros(count, dtype=int), timestamps[0], values))
    # 17 significant digits preserve float64 values through a decimal round trip.
    np.savetxt(text_file, rows, fmt=['%d', '%.17g', '%.17g', '%.17g', '%.17g'],
               header='SENSOR TIMESTAMP VALUE1 VALUE2 VALUE3', comments='')
    (folder / 'export_metadata.json').write_text(json.dumps({
        'tracks': ['Tx', 'Ty', 'Tz'], 'timestamp_units': 'seconds',
        'sequence_end_timestamp': 0.0, 'filter': 'Polaris reader low-pass 1 Hz',
        'full_acquisition_mean': mean.tolist(), 'full_acquisition_scale': float(scale),
        'timestamps_shape': [1, count, 1], 'values_shape': [1, count, 3],
    }, indent=2) + '\n')
    return {'physio_text': {'physio_file': str(text_file)},
            'physio_array': {'physio_timestamps_file': str(time_file), 'physio_values_file': str(values_file)}}


def seed_everything(seed, device, sp_device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    if sp_device.id >= 0:
        import cupy
        with sp_device:
            cupy.random.seed(seed)


def compare_images(images, output, tolerance):
    metrics = {}
    for left, right in itertools.combinations(images, 2):
        a, b = images[left], images[right]
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError(f'Invalid reconstructed images: {left}, {right}.')
        norm = torch.linalg.vector_norm(a).item()
        if norm == 0:
            raise AssertionError(f'{left}: a zero reconstruction cannot validate equivalence.')
        complex_error = torch.linalg.vector_norm(a - b).item() / norm
        magnitude_error = torch.linalg.vector_norm(a.abs() - b.abs()).item() / norm
        metrics[f'{left}_vs_{right}'] = {
            'relative_complex_l2': complex_error, 'relative_magnitude_l2': magnitude_error,
            'max_absolute_complex_error': (a - b).abs().max().item(),
            'passed': complex_error <= tolerance and magnitude_error <= tolerance,
        }
    (output / 'comparison.json').write_text(json.dumps({
        'relative_l2_tolerance': tolerance, 'comparisons': metrics,
    }, indent=2) + '\n')
    # Preview averages repetition magnitudes; numerical comparisons use every complex voxel.
    previews = {name: image.abs().reshape(-1, *image.shape[-2:]).mean(0).numpy()
                for name, image in images.items()}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    vmax = max(float(value.max()) for value in previews.values())
    baseline = previews['polaris']
    for column, (name, preview) in enumerate(previews.items()):
        axes[0, column].imshow(preview, cmap='gray', vmin=0, vmax=vmax)
        axes[0, column].set_title(name)
        artist = axes[1, column].imshow(np.abs(preview - baseline), cmap='magma')
        axes[1, column].set_title('Absolute magnitude difference vs Polaris')
        fig.colorbar(artist, ax=axes[1, column], shrink=0.7)
        axes[0, column].axis('off')
        axes[1, column].axis('off')
    fig.savefig(output / 'comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)
    if not all(value['passed'] for value in metrics.values()):
        raise AssertionError(f'Reconstructed images differ beyond rtol={tolerance}; see {output}.')


def main():
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='comparison_', dir=args.output_root))
    print(f'Comparison artifacts: {output}', flush=True)
    (output / 'inputs.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    torch.set_num_threads(1)
    images, exports = {}, {}
    reference_motion = reference_labels = None
    # Keep conversion leases for the three runs, and release them together at the end.
    with execution_scope():
        for fmt in ('polaris', 'physio_text', 'physio_array'):
            print(f'\nStarting siemens-{fmt} reconstruction...', flush=True)
            overrides = dict(output_root=str(output), workflow_label=fmt, runtime_device=args.device,
                             seed_enabled=True, seed=args.seed, use_deterministic_algorithms=True,
                             jupyter_notebook_flag=False, save_debug_plots=False,
                             cache_preprocessed_data=False, remove_temporary_data_after_run=True)
            extra = {}
            if fmt == 'polaris':
                overrides['polaris_channel_mode'] = 'all'
                extra['polaris_config'] = ROOT / 'config/real_data/polaris.toml'
            params = load_config(
                data_type=f'siemens-{fmt}', reconstruction_config=args.reconstruction_config,
                coil_sensitivity_config=args.coil_sensitivity_config,
                ismrmrd_reader_config=ROOT / 'config/real_data/ismrmrd_reader.toml',
                overrides=overrides, **extra)
            if params.data_dimension != '2D':
                raise ValueError('This demo E comparison requires a 2D reconstruction config.')
            sp_device, device = initialize_runtime(params)
            physiology = {'polaris_file': str(args.polaris_file)} if fmt == 'polaris' else exports[fmt]
            data = DataLoader(params, sp_device=sp_device, t_device=device,
                              filename={'siemens_raw_file': str(args.raw_file), **physiology},
                              slice_idx=args.slice_idx, run_pipeline=False)
            data.load_data()
            motion = data._source_motion_data.detach().cpu().clone()
            if fmt == 'polaris':
                exports = export_physiology(data.raw_data_preparer, output / 'exports')
                reference_motion = motion
            else:
                # Polaris normalizes after interpolation, whereas the generic
                # export normalizes source samples before interpolation. These
                # are mathematically equivalent; allow only float64 round-off.
                torch.testing.assert_close(motion, reference_motion, rtol=1e-9, atol=1e-10,
                                           msg=f'{fmt}: full-acquisition physiological inputs differ beyond '
                                               'the expected interpolation round-off.')
            seed_everything(args.seed, device, sp_device)
            data.run_slice_pipeline()
            labels = data.motion_labels.detach().cpu().clone()
            if fmt == 'polaris':
                reference_labels = labels
            else:
                torch.testing.assert_close(labels, reference_labels, rtol=0, atol=0,
                                           msg=f'{fmt}: motion bins differ from Polaris.')
            recon = JointReconstructor(data.kspace, data.smaps, data.sampling_idx,
                                       motion_signal=data.motion_signal, params=params,
                                       kspace_scale=data.kspace_scale,
                                       motion_plot_context=data.motion_plot_context)
            image, motion_model = recon.run()
            images[fmt] = image.detach().cpu().clone()
            torch.save(images[fmt], output / f'{fmt}_image.pt')
            del image, motion_model, recon, data, motion, labels
            gc.collect()
            cleanup_runtime()
        compare_images(images, output, args.rtol)
    print(f'PASS: all three reconstructed images agree. Results: {output}', flush=True)


if __name__ == '__main__':
    main()

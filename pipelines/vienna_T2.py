#!/usr/bin/env python
"""All-slice Vienna T2/Polaris reconstruction with sharpness-based clock fitting.

python pipelines/vienna_T2.py \\
  ../data/GRICS-torch/Vienna/raw_data/meas_MID01133_FID01857_AX_T2_TSE_HR_2NEX_SAT_MOTION.dat \\
  ../data/GRICS-torch/Vienna/motion/R1.tsv
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set numerical-library limits before importing torch/numpy (CPU fork workers).
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
              'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'SimpleITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
    os.environ[_name] = '1'
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['MKL_DYNAMIC'] = 'FALSE'

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from pipelines.siemens_breast_T2 import grics_zero_fill_shapes, zero_fill_loaded_reconstruction
from src.preprocessing.DataLoader import DataLoader
from src.preprocessing.RawDataPreparer import RawDataPreparer
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.output_layout import bind_output_paths, managed_execution, record_reconstruction, write_json
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from src.utils.clock_optimization import optimize_clock_correction
from src.utils.dicom_export import write_reconstruction_dicom
from src.utils.sharpness_index import sharpness_index
from src.utils.plotting import show_and_save_image

LOADED_DATA = None
EVALUATION_FOLDER = None
FINAL_EVALUATION = False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_data_file', type=Path, help='Siemens .dat or ISMRMRD .h5/.mrd.')
    parser.add_argument('polaris_file', type=Path, help='Single-tool Polaris TSV, as in demo E.')
    parser.add_argument('--output-root', type=Path, default=REPO_ROOT / 'runs/vienna_T2')
    parser.add_argument('--reconstruction-config', type=Path,
                        default=REPO_ROOT / 'config/reconstruction/nonrigid_2d.toml')
    parser.add_argument('--reader-config', type=Path,
                        default=REPO_ROOT / 'config/real_data/ismrmrd_reader.toml')
    parser.add_argument('--polaris-channel-mode', choices=['all', 'largest-amplitude'],
                        default='largest-amplitude')
    parser.add_argument('--clock-optimization', action=argparse.BooleanOptionalAction, default=True,
                        help='Optimize the clock correction (default: enabled); disable to use the configured or initial correction.')
    parser.add_argument('--reuse-clock-run', help='Previous run directory, or latest: reuse its best clock correction without optimization.')
    parser.add_argument('--flip-for-display', action=argparse.BooleanOptionalAction, default=True,
                        help='Flip PNG previews vertically (default: enabled, as in the breast pipeline).')
    parser.add_argument('--initial-correction', type=float, default=None,
                        help='Seconds; defaults to physio_clock_drift_seconds in Polaris config.')
    parser.add_argument('--clock-bounds', type=float, nargs=2, default=(-1.0, 1.0),
                        metavar=('LOWER', 'UPPER'), help='Seconds; intersected with recording coverage limits.')
    parser.add_argument('--difference-step', type=float, default=0.05, help='Finite-difference spacing in seconds.')
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--max-step', type=float, default=0.2, help='Maximum descent step in seconds.')
    parser.add_argument('--clock-tolerance', type=float, default=0.001, help='Stopping tolerance in seconds.')
    parser.add_argument('--max-iterations', type=int, default=8)
    parser.add_argument('--max-backtracks', type=int, default=8)
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Optional worker cap; defaults to one worker per slice.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dicom-header-dir', type=Path, default=None)
    parser.add_argument('--dicom-series-number', type=int, default=1001)
    args = parser.parse_args(argv)
    for name in ('raw_data_file', 'polaris_file', 'reconstruction_config', 'reader_config'):
        if not getattr(args, name).is_file():
            parser.error(f'{name} does not exist: {getattr(args, name)}')
    if args.raw_data_file.suffix.lower() not in {'.dat', '.h5', '.mrd'}:
        parser.error('Raw input must be .dat, .h5 or .mrd.')
    if args.max_workers is not None and args.max_workers < 1:
        parser.error('--max-workers must be positive.')
    if args.seed < 0:
        parser.error('--seed must be nonnegative.')
    args.reused_clock_result = None
    if args.reuse_clock_run:
        args.clock_optimization = False
        if args.initial_correction is not None:
            parser.error('--initial-correction cannot be combined with --reuse-clock-run.')
        try:
            source, summary = load_previous_clock_result(args.reuse_clock_run, args.output_root)
            manifest_path = source / 'manifest.json'
            if manifest_path.is_file():
                inputs = json.loads(manifest_path.read_text()).get('inputs', {})
                for name in ('raw_data_file', 'polaris_file'):
                    if name in inputs and Path(inputs[name]).resolve() != getattr(args, name).resolve():
                        raise ValueError(f'Previous run uses a different {name}.')
            args.reused_clock_result = {**summary, 'reused_from': str(source)}
            args.initial_correction = float(summary['correction_seconds'])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        return args
    # Validate optimization settings before loading a large acquisition.
    initial = args.initial_correction
    if initial is not None and not math.isfinite(initial):
        parser.error('--initial-correction must be finite.')
    if not args.clock_optimization:
        return args
    try:
        optimize_clock_correction(lambda _: 0.0,
            initial=initial if initial is not None else sum(args.clock_bounds) / 2,
            **optimizer_options(args, args.clock_bounds))
    except ValueError as exc:
        parser.error(str(exc))
    return args


def load_previous_clock_result(selection, output_root):
    """Resolve the latest saved optimizer result before creating a new run."""
    if selection == 'latest':
        candidates = sorted(Path(output_root).glob('*/optimization/result.json'))
        if not candidates:
            raise ValueError(f'No saved clock optimization found under {output_root}.')
        source = candidates[-1].parents[1]
    else:
        source = Path(selection).expanduser()
    summary = json.loads((source / 'optimization/result.json').read_text())
    correction = summary['correction_seconds']
    if type(correction) not in (int, float) or not math.isfinite(correction):
        raise ValueError('Saved clock correction must be a finite number.')
    return source.resolve(), summary


def optimizer_options(args, bounds):
    return dict(bounds=bounds, difference_step=args.difference_step,
                learning_rate=args.learning_rate, max_step=args.max_step,
                tolerance=args.clock_tolerance, max_iterations=args.max_iterations,
                max_backtracks=args.max_backtracks)


def load_all_slices(args):
    overrides = dict(workflow_label=args.output_root.name,
        output_root=str(args.output_root.resolve().parent), runtime_device='cpu',
        jupyter_notebook_flag=False, save_debug_plots=False, flip_for_display=args.flip_for_display,
        polaris_channel_mode=args.polaris_channel_mode,
        seed_enabled=True, seed=args.seed, use_deterministic_algorithms=True,
        verbose=False, print_to_console=False)
    if args.initial_correction is not None:
        overrides['physio_clock_drift_seconds'] = args.initial_correction
    params = load_config(
        data_type='siemens-polaris' if args.raw_data_file.suffix.lower() == '.dat' else 'ismrmrd-polaris',
        reconstruction_config=args.reconstruction_config,
        coil_sensitivity_config=REPO_ROOT / 'config/coil_sensitivity/odille_spline.toml',
        ismrmrd_reader_config=args.reader_config,
        polaris_config=REPO_ROOT / 'config/real_data/polaris.toml', overrides=overrides)
    if params.reconstruction_dimension != '2D':
        raise ValueError('Vienna T2 pipeline requires a 2D reconstruction configuration.')
    sp_device, t_device = initialize_runtime(params)
    data = DataLoader(params=params, t_device=t_device, sp_device=sp_device,
        filename=(str(args.raw_data_file), str(args.polaris_file)), run_pipeline=False)
    data.load_data()
    return data


def clock_bounds(sync, initial_correction, requested):
    """Intersect requested bounds with <=1 second missing coverage per edge/channel."""
    target = sync['acquisition_time_seconds']
    lower, upper = requested
    for times in sync['physiological_time_seconds']:
        unshifted = np.asarray(times) - initial_correction
        lower = max(lower, float(np.max(target) - unshifted[-1] - 1.0))
        upper = min(upper, float(np.min(target) - unshifted[0] + 1.0))
    if lower >= upper:
        raise ValueError('No clock search interval remains within the one-second extrapolation limit.')
    return float(lower), float(upper)


def apply_clock_correction(data, original_sync, initial_correction, correction):
    """Reuse MRI arrays and filtered traces; recompute full-sequence alignment and channels."""
    times = [np.asarray(t) - initial_correction for t in original_sync['physiological_time_seconds']]
    values = original_sync['physiological_values']
    aligned = RawDataPreparer._synchronize_to_sequence_end(
        times, values, original_sync['acquisition_time_seconds'], source_sequence_end=0.0,
        bounds='autoregression', physio_clock_drift_seconds=correction)
    preparer = data.raw_data_preparer
    motion = preparer.physiological_reader.prepare_motion(aligned)
    slices = original_sync['slice_indices']
    # Match RawDataPreparer's per-slice acquisition ordering before binning.
    motion = np.stack([motion[slices == index] for index in range(int(data.Nz))])
    data._source_motion_data = torch.as_tensor(motion, device=data.t_device)
    data.params.physio_clock_drift_seconds = correction
    preparer.physio_clock_drift_seconds = correction
    preparer.synchronization = {**original_sync,
        'physiological_time_seconds': [t + correction for t in times],
        'acquisition_values': aligned}


def initialize_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def reconstruct_slice(slice_idx):
    data = copy.copy(LOADED_DATA)
    data.params = copy.copy(LOADED_DATA.params)
    # Fixed per-slice randomness across all clock candidates and the final export.
    torch.manual_seed(data.params.seed + slice_idx)
    np.random.seed((data.params.seed + slice_idx) % 2**32)
    # Retain images and previews for visual comparison of every tested correction.
    data.params.save_reconstruction_outputs = True
    folder = EVALUATION_FOLDER / f'slice_{slice_idx + 1:03d}'
    bind_output_paths(data.params, folder)
    started = time.perf_counter()
    data.run_slice_pipeline(slice_idx=slice_idx, output_folder=folder)
    reconstructor = JointReconstructor(data.kspace, data.smaps, data.sampling_idx,
        motion_signal=data.motion_signal, params=data.params, kspace_scale=data.kspace_scale,
        motion_plot_context=data.motion_plot_context)
    image, _ = reconstructor.run()
    # Save explicitly: comparisons must not depend on reconstructor logging flags.
    results_folder = Path(data.params.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    torch.save(image.detach().cpu(), results_folder / 'image_reconstructed.pt')
    show_and_save_image(image.mean(dim=0), 'image_reconstructed', str(results_folder),
                        flip_for_display=getattr(data.params, 'flip_for_display', False))
    uncorrected = data.image_no_moco.detach().cpu() * data.kspace_scale
    if uncorrected.ndim == 4 and uncorrected.shape[-1] == 1:
        uncorrected = uncorrected[..., 0]
    torch.save(uncorrected, results_folder / 'image_no_motion_correction.pt')
    # Native reconstruction grid, magnitude of complex repetition mean, without
    # breast-specific reference normalization or display interpolation.
    score = float(sharpness_index(image.mean(dim=0).abs()).item())
    if not math.isfinite(score):
        raise ValueError(f'Nonfinite sharpness for slice {slice_idx + 1}.')
    result = dict(slice_idx=slice_idx, slice_number=slice_idx + 1, sharpness_index=score,
                  physio_clock_drift_seconds=data.params.physio_clock_drift_seconds,
                  elapsed_s=time.perf_counter() - started)
    if FINAL_EVALUATION:
        target, encoded = LOADED_DATA.zero_fill_shapes
        exported = zero_fill_loaded_reconstruction(image, target, encoded)
        dicom_path = Path(data.params.run_folder) / 'exports/dicom' / f'slice_{slice_idx + 1:03d}.dcm'
        write_reconstruction_dicom(exported, dicom_path, raw_data=LOADED_DATA,
            slice_index=slice_idx, series_description='GRICS Vienna T2 RESEARCH ONLY',
            images_in_acquisition=int(LOADED_DATA.Nz), series_number=LOADED_DATA.export_series_number,
            reference_dicom_path=LOADED_DATA.export_reference, **LOADED_DATA.export_uids)
        result['dicom_file'] = str(dicom_path.relative_to(data.params.run_folder))
    record_reconstruction(data.params, status='complete', **result)
    return result


def reconstruct_all_slices(data, folder, max_workers, *, final=False):
    global LOADED_DATA, EVALUATION_FOLDER, FINAL_EVALUATION
    LOADED_DATA, EVALUATION_FOLDER, FINAL_EVALUATION = data, Path(folder), final
    nslices = int(data.Nz)
    workers = min(max_workers, nslices) if max_workers is not None else nslices
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=initialize_worker) as executor:
        futures = [executor.submit(reconstruct_slice, index) for index in range(nslices)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"  slice {result['slice_number']:03d}: sharpness={result['sharpness_index']:.6g}", flush=True)
    return sorted(results, key=lambda result: result['slice_idx'])


def save_comparisons(root, nslices, correction, *, flip_for_display=False,
                     optimized_root=None, baseline_root=None):
    """Save three-way comparisons with identical geometry and intensity windows."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    rows = []
    for index in range(nslices):
        slice_name = f'slice_{index + 1:03d}'
        optimized_folder = (optimized_root or root / 'reconstructions') / slice_name / 'results'
        baseline_folder = (baseline_root or root / 'baselines/zero_clock') / slice_name / 'results'
        images = {
            'no_motion_correction': torch.load(baseline_folder / 'image_no_motion_correction.pt',
                                               map_location='cpu', weights_only=True),
            'motion_correction_zero_clock': torch.load(baseline_folder / 'image_reconstructed.pt',
                                                       map_location='cpu', weights_only=True),
            'motion_correction_optimized_clock': torch.load(optimized_folder / 'image_reconstructed.pt',
                                                            map_location='cpu', weights_only=True),
        }
        if len({tuple(image.shape) for image in images.values()}) != 1:
            raise ValueError(f'{slice_name}: comparison images must have matching shapes.')
        folder = root / 'comparisons' / slice_name
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(images, folder / 'images.pt')
        magnitudes = [image.mean(dim=0).abs().numpy() for image in images.values()]
        # One common display window for all three images of this slice.
        vmax = float(np.percentile(np.stack(magnitudes), 99))
        vmax = max(vmax, np.finfo(float).eps)
        scores = {key: float(sharpness_index(image.mean(dim=0).abs()))
                  for key, image in images.items()}
        titles = ['No motion correction', 'Motion correction, clock = 0 s',
                  f'Motion correction, clock = {correction:+.4f} s']
        fig = Figure(figsize=(15, 5), layout='constrained')
        FigureCanvasAgg(fig)
        for ax, magnitude, title, score in zip(fig.subplots(1, 3), magnitudes, titles, scores.values()):
            ax.imshow(np.flipud(magnitude) if flip_for_display else magnitude,
                      cmap='gray', vmin=0, vmax=vmax, interpolation='nearest')
            ax.set_title(f'{title}\nSharpness = {score:.5g}')
            ax.axis('off')
        fig.suptitle(f'Slice {index + 1:03d}')
        fig.savefig(folder / 'comparison.png', dpi=150)
        fig.clear()
        row = dict(slice_idx=index, optimized_clock_seconds=correction, sharpness=scores,
                   display_window=[0, vmax], image=str((folder / 'comparison.png').relative_to(root)))
        write_json(folder / 'scores.json', row)
        rows.append(row)
    write_json(root / 'comparisons/scores.json', rows)
    return rows


@managed_execution
def main(argv=None):
    args = parse_args(argv)
    # Avoid forking after parent-side multithreaded numerical work.
    torch.set_num_threads(1)
    data = load_all_slices(args)
    original_sync = data.raw_data_preparer.synchronization
    initial = data.params.physio_clock_drift_seconds
    bounds = clock_bounds(original_sync, initial, args.clock_bounds) if args.clock_optimization else None
    root = Path(data.params.run_folder)
    outputs = data.params._run_outputs
    outputs.manifest.update(inputs={'raw_data_file': str(args.raw_data_file.resolve()),
                                   'polaris_file': str(args.polaris_file.resolve())},
        pipeline='vienna_T2', slice_indices=list(range(int(data.Nz))),
        clock_optimization_enabled=args.clock_optimization,
        optimization_settings=optimizer_options(args, bounds) if args.clock_optimization else None,
        objective='maximize mean per-slice sharpness index (minimize its negative)',
        sharpness_representation='magnitude of repetition mean on native grid; article pmode=3',
        max_workers=min(args.max_workers, int(data.Nz)) if args.max_workers is not None else int(data.Nz))
    outputs.flush()
    trials = []

    def objective(correction):
        trial_dir = root / 'optimization' / f'evaluation_{len(trials):03d}'
        print(f'[clock] Evaluating {correction:+.6f} s on all {data.Nz} slices', flush=True)
        apply_clock_correction(data, original_sync, initial, correction)
        results = reconstruct_all_slices(data, trial_dir, args.max_workers)
        mean = float(np.mean([item['sharpness_index'] for item in results]))
        trial = dict(correction_seconds=correction, mean_sharpness_index=mean,
                     objective=-mean, slice_results=results)
        trials.append(trial)
        write_json(trial_dir / 'scores.json', trial)
        write_json(root / 'optimization/history.json', trials)
        print(f'[clock] Mean sharpness={mean:.6g}', flush=True)
        return -mean

    if args.reused_clock_result:
        summary = args.reused_clock_result
        best_correction = float(summary['correction_seconds'])
        print(f"[clock] Reusing {best_correction:+.6f} s from {summary['reused_from']}; skipping optimization", flush=True)
    elif not args.clock_optimization:
        best_correction = initial
        summary = dict(correction_seconds=initial, initial_correction_seconds=initial,
                       iterations=0, evaluations=[], stop_reason='clock optimization disabled')
        print(f'[clock] Optimization disabled; using {best_correction:+.6f} s', flush=True)
    else:
        result = optimize_clock_correction(objective, initial, **optimizer_options(args, bounds))
        best_correction = result.correction_seconds
        summary = {**asdict(result), 'mean_sharpness_index': -result.objective,
                   'initial_correction_seconds': initial, 'effective_bounds_seconds': bounds}
        print(f'[clock] Best correction: {best_correction:+.6f} s; '
              f'mean sharpness={-result.objective:.6g}; {result.stop_reason}', flush=True)
    write_json(root / 'optimization/result.json', summary)
    outputs.manifest['clock_optimization'] = summary
    outputs.flush()

    # Reconstruct the best evaluated offset with full outputs and one DICOM series.
    apply_clock_correction(data, original_sync, initial, best_correction)
    from pydicom.uid import generate_uid
    data.zero_fill_shapes = grics_zero_fill_shapes(data)
    data.export_uids = {key: generate_uid() for key in
        ('study_instance_uid', 'series_instance_uid', 'frame_of_reference_uid')}
    data.export_reference = args.dicom_header_dir
    data.export_series_number = args.dicom_series_number
    final = reconstruct_all_slices(data, root / 'reconstructions', args.max_workers, final=True)
    write_json(root / 'sharpness.json', final)
    print('[compare] Reconstructing all slices at zero clock correction', flush=True)
    apply_clock_correction(data, original_sync, initial, 0.0)
    baseline = reconstruct_all_slices(data, root / 'baselines/zero_clock', args.max_workers)
    write_json(root / 'baselines/zero_clock/scores.json', baseline)
    comparisons = save_comparisons(root, int(data.Nz), best_correction,
        flip_for_display=getattr(data.params, 'flip_for_display', False))
    # Keep the chosen clock correction in the final resolved configuration.
    apply_clock_correction(data, original_sync, initial, best_correction)
    outputs.manifest['comparisons'] = comparisons
    outputs.manifest['no_motion_baseline'] = 'Coil-combined inverse FFT, without GRICS motion correction'
    outputs.manifest.update(slice_results=final, dicom_uids=data.export_uids,
        dicom_header_dir=str(args.dicom_header_dir) if args.dicom_header_dir else None,
        dicom_series_number=args.dicom_series_number,
        final_mean_sharpness_index=float(np.mean([row['sharpness_index'] for row in final])))
    outputs.snapshot(data.params)
    outputs.flush()
    print(f'[run] Complete: {root}', flush=True)


if __name__ == '__main__':
    main()

"""Pipeline timing, configuration, and final output coordination."""
from pathlib import Path
import time

import torch

from src.reconstruction.JointReconstructor import JointReconstructor
from src.utils.ismrmrd_io import acquisition_header
from src.runtime.output_layout import write_reconstruction


def reconstruction_overrides(output_root, device, overrides=None, *,
                             save_reconstruction_logs=None, save_reconstruction_tensors=None):
    output_root = Path(output_root)
    settings = dict(overrides or {})
    settings.update(
        output_root=str(output_root.parent), workflow_label=output_root.name,
        runtime_device=device, jupyter_notebook_flag=False, flip_for_display=True,
        save_debug_plots=False,
        verbose=False, print_to_console=False,
    )
    for key, value in [('save_reconstruction_logs', save_reconstruction_logs),
                       ('save_reconstruction_tensors', save_reconstruction_tensors)]:
        if value is not None:
            settings[key] = value
    settings.setdefault('use_deterministic_algorithms', False)
    return settings


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def timed_reconstruction(data):
    """Time construction, optimization and text logs; exclude tensor/DICOM export."""
    synchronize(data.kspace.device)
    started = time.perf_counter()
    voxel_spacing_mm = None
    if data.params.regularization_scaling == "grics_cpp":
        encoding = acquisition_header(data).encoding[0].encodedSpace
        matrix, fov = encoding.matrixSize, encoding.fieldOfView_mm
        axes = ("x", "y", "z") if data.params.data_dimension == "3D" else ("x", "y")
        voxel_spacing_mm = tuple(float(getattr(fov, axis)) / int(getattr(matrix, axis))
                                 for axis in axes)
    reconstructor = JointReconstructor(
        data.kspace, data.smaps, data.sampling_idx,
        motion_signal=data.motion_signal, params=data.params,
        kspace_scale=data.kspace_scale, motion_plot_context=data.motion_plot_context,
        calibration_image_prior=getattr(data, 'grics_reference_image', None),
        voxel_spacing_mm=voxel_spacing_mm,
    )
    image, motion = reconstructor.run()
    reconstructor.finish_run()
    synchronize(data.kspace.device)
    elapsed = time.perf_counter() - started
    return image, motion, elapsed


def export_reconstruction(params, result):
    """Write each final tensor once, after all reconstructions have finished."""
    result['log_file'] = (Path(params.logs_folder) / 'reconstruction.log'
                          if params.save_reconstruction_logs else None)
    metadata = {key: value for key, value in result.items()
                if key not in {'image', 'native_image', 'motion', 'image_stage'}}
    paths = write_reconstruction(
        params, result['image'], result['motion'], native_image=result['native_image'],
        image_stage=result['image_stage'], record_paths=True, status='complete', tensor_export='pipeline', **metadata)
    result.update(paths)


def finish_run(data, results, timings, *, return_tensors, **metadata):
    run = data.params._run_outputs
    run.manifest.update(
        timings=timings, save_reconstruction_logs=data.params.save_reconstruction_logs,
        save_reconstruction_tensors=data.params.save_reconstruction_tensors,
        reconstruction_timing='solver construction, optimization and enabled text logging; synchronized on CUDA; excludes preprocessing, postprocessing, transfers and exports',
        **metadata,
    )
    run.flush()
    if not return_tensors:
        for result in results:
            result.pop('image')
            result.pop('native_image')
            result.pop('motion')
    return {'run_folder': Path(data.params.run_folder), 'timings': timings,
            'reconstructions': results}

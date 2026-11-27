"""Shared Siemens navigator reconstruction, contrast handling, and DICOM export."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pipelines.utils.postprocessing import grics_zero_fill_volume_shapes, zero_fill_loaded_reconstruction
from src.preprocessing.DataPreparer import DataPreparer
from src.preprocessing.real_data.RawDataCache import RawDataCache
from src.preprocessing.real_data.real_data_io import convert_siemens_to_ismrmrd
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.output_layout import managed_execution, record_reconstruction, write_reconstruction
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime

REPO_ROOT = Path(__file__).resolve().parents[2]


def combine_mp2rage(first, second):
    """Return Re(conj(INV1)*INV2)/(|INV1|²+|INV2|²), with zero background set to 0."""
    if first.shape != second.shape or first.ndim != 3 or not first.is_complex() or not second.is_complex():
        raise ValueError('MP2RAGE requires matching complex [x, y, z] contrast images.')
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError('MP2RAGE input images must be finite.')
    denominator = first.abs().square() + second.abs().square()
    valid = denominator > 0
    combined = torch.zeros_like(denominator)
    combined[valid] = (first.conj() * second).real[valid] / denominator[valid]
    print(f'[MP2RAGE] {int((~valid).sum())} voxels with zero denominator explicitly set to 0.', flush=True)
    return combined


def _set_common_motion_reference(data, reference=None):
    """Use INV2 readout means/scales for both contrasts, preserving a common zero."""
    loader = data.input_loader
    synchronized = loader.raw_data_preparer.synchronization
    signals = np.asarray(synchronized['acquisition_values'])
    modes = loader.raw_data_preparer.physiological_reader.metadata['navigator_temporal_modes']
    if reference is None:
        mean, scale = signals.mean(axis=0), signals.std(axis=0)
        reference = {'mean': mean, 'scale': scale, 'navigator_modes': modes.copy()}
    else:
        if modes.shape != reference['navigator_modes'].shape or not np.allclose(modes, reference['navigator_modes'], atol=1e-10, rtol=1e-8):
            raise ValueError('The two contrasts must use identical second-set navigator modes.')
        mean, scale = reference['mean'], reference['scale']
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError('Common navigator normalization requires finite, nonconstant signals.')
    loader.motion_data = torch.as_tensor((signals - mean) / scale, device=loader.motion_data.device, dtype=loader.motion_data.dtype)
    print('[MP2RAGE] Using INV2 readout mean and standard deviation for both motion signals.', flush=True)
    return reference


def _validate_acquisition_sets(source_file, single_contrast):
    """Reject incompatible sequences before reconstructing a partial acquisition."""
    import ismrmrd
    from src.utils.ismrmrd_io import ReadOnlyDataset

    with ReadOnlyDataset(source_file) as dataset:
        heads = dataset.read_acquisition_headers()
    names = ('ACQ_IS_NOISE_MEASUREMENT', 'ACQ_IS_NAVIGATION_DATA', 'ACQ_IS_DUMMYSCAN_DATA',
             'ACQ_IS_PHASECORR_DATA', 'ACQ_IS_HPFEEDBACK_DATA', 'ACQ_IS_RTFEEDBACK_DATA')
    excluded = sum(1 << (getattr(ismrmrd, name) - 1) for name in names)
    sets = set(heads['idx']['set'][(heads['flags'] & excluded) == 0].tolist())
    expected = {0} if single_contrast else {0, 1}
    if sets != expected:
        raise ValueError(f'Pipeline requires imaging set counters {sorted(expected)}; found {sorted(sets)}.')


@managed_execution
def run_navigator_pipeline(raw_data_file, *, acquisition_type, output_root, device, reconstruction_config, overrides, dicom_header_dir, dicom_series_number, no_motion_correction):
    """Reconstruct one MPRAGE image or two MP2RAGE contrasts using explicit source layouts."""
    from pydicom.uid import generate_uid
    from src.utils.dicom_export import write_volume_dicoms

    if acquisition_type not in {"MPRAGE", "MP2RAGE"}:
        raise ValueError("acquisition_type must be MPRAGE or MP2RAGE.")
    single_contrast = acquisition_type == "MPRAGE"
    navigator_set = 0 if single_contrast else 1

    raw_data_file = Path(raw_data_file)
    if not raw_data_file.is_file():
        raise FileNotFoundError(raw_data_file)
    if raw_data_file.suffix.lower() != '.dat':
        raise ValueError('Navigator pipeline input must be a Siemens .dat file.')
    if type(dicom_series_number) is not int or dicom_series_number < 1:
        raise ValueError('dicom_series_number must be a positive integer.')
    if type(no_motion_correction) is not bool:
        raise ValueError('no_motion_correction must be a boolean.')
    overrides = dict(overrides or {})
    owned = {'navigator_type', 'navigator_set', 'navigator_imaging_set', 'data_type', 'output_root', 'workflow_label', 'runtime_device'}
    if owned & overrides.keys():
        raise ValueError(f'Use pipeline arguments instead of overriding pipeline-owned settings: {sorted(owned & overrides.keys())}.')
    if single_contrast:
        if overrides.get('navigator_directions', [0]) != [0]:
            raise ValueError('Single-navigator MPRAGE requires navigator_directions=[0].')
        overrides['navigator_directions'] = [0]
    output_root = Path(output_root)
    results, common_motion, shared_maps, run_folder, geometry = {}, None, None, None, None
    conversion_cache = None
    navigator_reference = None
    contrasts = ((0, "mprage"),) if single_contrast else ((1, "inv2"), (0, "inv1"))
    for set_index, name in contrasts:
        params = load_config(data_type='ismrmrd-navigators', reconstruction_config=REPO_ROOT / reconstruction_config,
                             overrides={**overrides, 'navigator_type': 'mosar-mprage' if single_contrast else 'mosar', 'navigator_set': navigator_set, 'navigator_imaging_set': set_index,
                                        'runtime_device': device, 'output_root': str(output_root.parent if run_folder is None else run_folder / 'contrasts'),
                                        'workflow_label': output_root.name if run_folder is None else name})
        if params.reconstruction_dimension != '3D' or params.reconstruction_motion_type not in {'rigid', 'non-rigid'} or params.kspace_sampling_type != 'from-data' or params.simulated_motion_type != 'as-it-is':
            raise ValueError('Navigator pipelines require 3D rigid or non-rigid reconstruction with recorded sampling and recorded motion.')
        runtime = initialize_runtime(params)
        if run_folder is None:
            run_folder = Path(params.run_folder)
        if conversion_cache is None:
            conversion_cache = RawDataCache(params.cache_root, remove_on_release=params.remove_temporary_data_after_run)
            try:
                source_file = convert_siemens_to_ismrmrd(str(raw_data_file), conversion_cache)
                _validate_acquisition_sets(source_file, single_contrast)
            except BaseException:
                conversion_cache.close()
                raise
        data = DataPreparer(params, runtime, filename=source_file, navigator_reference=navigator_reference)
        try:
            data.load_input_data()
            navigator_reference = data.input_loader.raw_data_preparer.physiological_reader.reference
            if data.input_loader.Nz <= 1:
                raise ValueError('Navigator pipeline requires a 3D slab with more than one partition.')
            target_shape, encoded_shape = grics_zero_fill_volume_shapes(data.input_loader)
            if not single_contrast:
                common_motion = _set_common_motion_reference(data, common_motion)
            data.prepare_reconstructor_inputs(coil_sensitivity_maps=shared_maps)
            if shared_maps is None:
                shared_maps = None if single_contrast else data.smaps.detach().cpu()
                geometry = SimpleNamespace(ismrmrd_header=data.ismrmrd_header, _input_slice_geometry=data._input_slice_geometry)
            else:
                if data.ismrmrd_header != geometry.ismrmrd_header or data._input_slice_geometry != geometry._input_slice_geometry:
                    raise ValueError('INV1 and INV2 must share the same acquisition geometry.')
            print(f'[{acquisition_type}] Reconstructing {name.upper()}.', flush=True)
            reconstructor = JointReconstructor.from_prepared_data(data)
            image, motion = reconstructor.run(motion_correction=not no_motion_correction)
            native_image, motion = image.detach().cpu(), motion.detach().cpu()
            image = zero_fill_loaded_reconstruction(native_image, target_shape, encoded_shape)
            paths = write_reconstruction(params, image, motion, native_image=native_image,
                                         image_stage='after_postprocessing', record_paths=True, status='reconstructed', tensor_export='pipeline',
                                         imaging_set=set_index, navigator_set=navigator_set, coil_maps_source='MPRAGE' if single_contrast else 'INV2', motion_normalization_source='MPRAGE' if single_contrast else 'INV2',
                                         motion_correction=not no_motion_correction)
            results[name] = {'image': image, 'motion': motion, 'params': params, **paths}
            del reconstructor, native_image
        finally:
            data.input_loader.raw_data_cache.close()
            del data
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            # Keep the converted measurement leased until both contrasts finish.
            if conversion_cache is not None and (set_index == 0 or sys.exc_info()[0] is not None):
                conversion_cache.close()
    combined = None if single_contrast else combine_mp2rage(results['inv1']['image'], results['inv2']['image'])
    study_uid, frame_uid = generate_uid(), generate_uid()
    series = {}
    images = (("mprage", results["mprage"]["image"]),) if single_contrast else (("inv1", results["inv1"]["image"]), ("inv2", results["inv2"]["image"]), ("combined", combined))
    for index, (name, image) in enumerate(images):
        bounds = (-0.5, 0.5) if name == 'combined' else (0., max(float(image.abs().max()), np.finfo(float).eps))
        files, uids = write_volume_dicoms(image, run_folder / 'exports/dicom' / name, geometry,
                                         series_number=dicom_series_number + index, reference_dicom_path=dicom_header_dir,
                                         series_description=f'{acquisition_type} GRICS {name.upper()}{" NO_MOCO" if no_motion_correction else ""} RESEARCH ONLY',
                                         study_instance_uid=study_uid, frame_of_reference_uid=frame_uid, pixel_value_range=bounds,
                                         sequence_index=0 if single_contrast else None if name == 'combined' else int(name[-1]) - 1)
        series[name] = {'files': [str(path) for path in files], **uids}
    combined_file = None
    if not single_contrast and results['inv2']['params'].save_reconstruction_tensors:
        combined_file = Path(results['inv2']['output_dir']) / 'image_mp2rage_combined.pt'
        torch.save(combined, combined_file)
    manifest = {'source': str(raw_data_file.resolve()), 'acquisition_type': acquisition_type,
                'motion_correction': not no_motion_correction, 'navigator_set': navigator_set,
                'coil_maps_source': 'MPRAGE' if single_contrast else 'INV2', 'dicom_series': series}
    if not single_contrast:
        manifest.update(motion_normalization_source='INV2', motion_mean=common_motion['mean'].tolist(), motion_scale=common_motion['scale'].tolist(),
                        combination='real(conj(INV1)*INV2)/(abs(INV1)^2+abs(INV2)^2); zero denominator -> 0', combined_tensor=str(combined_file) if combined_file else None)
    manifest_file = run_folder / f'exports/{acquisition_type.lower()}.json'
    manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
    for name, result in results.items():
        record_reconstruction(result['params'], status='complete', dicom_files=series[name]['files'], navigator_manifest=str(manifest_file))
    return {'run_folder': run_folder, 'contrasts': results, 'combined_image': combined, 'dicom_series': series, 'manifest_file': manifest_file}


def run_cli(run_pipeline, *, description, output_root, reconstruction_config, argv=None):
    """Parse common Siemens navigator pipeline arguments."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('raw_data_file', type=Path, help='Siemens .dat file.')
    parser.add_argument('--output-root', type=Path, default=output_root)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--no-motion-correction', action='store_true', help='Run iterative image reconstruction with motion fixed to zero at every resolution.')
    parser.add_argument('--reconstruction-config', type=Path, default=Path(reconstruction_config))
    parser.add_argument('--dicom-header-dir', type=Path)
    parser.add_argument('--dicom-series-number', type=int, default=1001, help='First exported DICOM series number.')
    args = parser.parse_args(argv)
    result = run_pipeline(**vars(args))
    print(f"[run] Outputs: {result['run_folder']}")
    return result

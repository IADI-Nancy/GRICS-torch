#!/usr/bin/env python
"""
Slicewise GRICS-torch reconstruction for multislice Siemens breast T2 data.
Example usage:
python pipelines/siemens_breast_T2.py \
  ../data/GRICS-torch/test_XA61_volunteer/0274_T2_s.dat \
  ../data/GRICS-torch/test_XA61_volunteer/0274_T2_s.saec \
    --dicom-header-dir runs/siemens_breast_T2/2017-110_01-0275-V1MR_2026-06-09/DCM_MR/S2/ \
    --dicom-series-number 100

Preprocessed input with raw fallback:
    python pipelines/siemens_breast_T2.py subject.dat subject.saec --preprocessed-file prepared.h5
    python pipelines/siemens_breast_T2.py subject.mrd subject.saec --preprocessed-file prepared.h5
Preprocessed input only:
    python pipelines/siemens_breast_T2.py prepared.h5
"""


from __future__ import annotations

import argparse
import copy
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# Pipeline-level settings. Reconstruction parameters live in the TOML file below.
OUTPUT_ROOT = Path("runs/siemens_breast_T2")
RECONSTRUCTION_CONFIG = "config/reconstruction/nonrigid_2d_breast.toml"
POSTPROCESSING_CONFIG = "config/postprocessing/nonrigid_2d_breast.toml"
RUNTIME_DEVICE = "cpu"
MAX_WORKERS = None
RECONSTRUCT_SLICE_START = 0
RECONSTRUCT_SLICE_STOP = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines._inputs import data_type_from_raw_data_file as classify_input, resolve_input_files
from src.utils.ismrmrd_io import acquisition_header
import torch

from src.preprocessing.DataLoader import DataLoader
from src.runtime.runtime_config import load_config, load_postprocessing_config
from src.runtime.runtime_setup import initialize_runtime
from src.runtime.output_layout import managed_execution, bind_output_paths, public_config
from src.utils.dicom_export import write_reconstruction_dicom
from src.utils.zero_fill import zero_fill_grics_image_to_shape
from pipelines._execution import (
    reconstruction_overrides, timed_reconstruction, synchronize,
    export_reconstruction, finish_run,
)


# With the fork start method, workers inherit this read-only source dataset
# without serializing and copying the complete acquisition for every slice.
LOADED_DATA = None


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load breast T2 raw data once, then reconstruct all slices in parallel."
    )
    parser.add_argument(
        "raw_data_file",
        type=Path, nargs="?",
        help="Siemens .dat, ISMRMRD .h5/.mrd, or preprocessed .h5 file.",
    )
    parser.add_argument("saec_file", type=Path, nargs="?", help="SAEC file required for raw input.")
    parser.add_argument("--preprocessed-file", type=Path, help="Preferred preprocessed HDF5; use raw_data_file and SAEC if missing.")
    parser.add_argument(
        "--dicom-header-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing Siemens DICOMs to use as public metadata "
            "header donors. Files are searched recursively and matched by slice location."
        ),
    )
    parser.add_argument(
        "--dicom-series-number",
        type=int,
        default=1001,
        help=(
            "SeriesNumber assigned to the exported GRICS DICOM series. If donor DICOMs "
            "are provided, this must be different from the Siemens source series number."
        ),
    )
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default=RUNTIME_DEVICE)
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--slice-start', type=int, default=RECONSTRUCT_SLICE_START)
    parser.add_argument('--slice-stop', type=int, default=RECONSTRUCT_SLICE_STOP)
    parser.add_argument('--save-reconstruction-tensors', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--no-dicom', action='store_true', help='Skip DICOM export.')
    parser.add_argument('--save-reconstruction-logs', action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args(argv)



def data_type_from_raw_data_file(raw_data_file: Path, saec_file: Path | None = None) -> str:
    return classify_input(raw_data_file, saec_file, dimension="2D")


def load_all_slices(raw_data_file: Path, saec_file: Path | None = None, *, output_root=OUTPUT_ROOT,
                    device=RUNTIME_DEVICE, reconstruction_config=RECONSTRUCTION_CONFIG,
                    postprocessing_config=POSTPROCESSING_CONFIG, overrides=None,
                    save_reconstruction_logs=None, save_reconstruction_tensors=None) -> DataLoader:
    """Load the acquisition once, without running the per-slice preprocessing."""
    data_type = data_type_from_raw_data_file(raw_data_file, saec_file)
    raw_input = data_type != "preprocessed-real"
    params = load_config(
        data_type=data_type,
        reconstruction_config=REPO_ROOT / reconstruction_config,
        coil_sensitivity_config=REPO_ROOT / "config/coil_sensitivity/odille_spline.toml",
        real_data_config=REPO_ROOT / "config/real_data/saec.toml" if raw_input else None,
        ismrmrd_reader_config=REPO_ROOT / "config/real_data/ismrmrd_reader.toml" if raw_input else None,
        overrides=reconstruction_overrides(
            output_root, device, overrides, save_reconstruction_logs=save_reconstruction_logs,
            save_reconstruction_tensors=save_reconstruction_tensors),
    )
    postprocessing = load_postprocessing_config(REPO_ROOT / postprocessing_config)
    if postprocessing.normalize_image_by_grics_reference and params.coil_sensitivity_method != "odille-spline":
        raise ValueError("Reference-image normalization requires coil_sensitivity_method='odille-spline'.")
    sp_device, t_device = initialize_runtime(params)
    data = DataLoader(
        params=params,
        t_device=t_device,
        sp_device=sp_device,
        filename=(str(raw_data_file), str(saec_file)) if raw_input else str(raw_data_file),
        run_pipeline=False,
    )
    data.postprocessing = postprocessing
    data.load_data()
    return data


def selected_slices(nslices: int, start=0, stop=None) -> list[int]:
    """Resolve and validate the configured half-open slice range."""
    stop = nslices if stop is None else stop
    if start < 0 or stop < start or stop > nslices:
        raise ValueError(f"Invalid slice range [{start}, {stop}) for {nslices} slices.")
    return list(range(start, stop))


def grics_reference_image_for_normalization(data: DataLoader, image: torch.Tensor) -> torch.Tensor:
    reference_image = getattr(data, "grics_reference_image", None)
    if reference_image is None:
        raise ValueError(
            "GRICS reference normalization requires the smoothed reference image "
            "calculated together with the coil sensitivity maps."
        )
    reference_image = reference_image.to(device=image.device)

    if reference_image.ndim == image.ndim and image.ndim >= 3:
        reference_image = reference_image.unsqueeze(0)
    if reference_image.ndim == image.ndim + 1 and reference_image.shape[-1] == 1:
        reference_image = reference_image[..., 0]

    if tuple(reference_image.shape) == tuple(image.shape):
        return reference_image
    if reference_image.shape[0] == 1 and tuple(reference_image.shape[1:]) == tuple(image.shape[1:]):
        return reference_image.expand(image.shape[0], *reference_image.shape[1:])

    raise ValueError(
        "GRICS reference image shape is incompatible with reconstructed image: "
        f"reference={tuple(reference_image.shape)}, image={tuple(image.shape)}."
    )


def normalize_reconstruction_by_grics_reference(
    image: torch.Tensor,
    reference_image: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(image.shape) != tuple(reference_image.shape):
        raise ValueError(
            "GRICS reference normalization requires matching image shapes: "
            f"image={tuple(image.shape)}, reference={tuple(reference_image.shape)}."
        )

    reference_magnitude = torch.abs(reference_image).to(device=image.device, dtype=image.real.dtype)
    threshold = 1.0 * torch.mean(reference_magnitude)
    reference_magnitude = torch.clamp(reference_magnitude, min=threshold)
    return image / reference_magnitude, reference_magnitude


def reconstruct_slice(slice_idx: int, source=None) -> dict:
    """Compute one slice in memory. Exports happen after all workers finish."""
    source = source if source is not None else LOADED_DATA
    if source is None:
        raise RuntimeError("Source data is not initialized.")
    data = copy.copy(source)
    data.params = copy.copy(source.params)
    bind_output_paths(data.params, Path(data.params.run_folder) / 'reconstructions' / f'slice_{slice_idx + 1:03d}')
    started = time.perf_counter()
    data.run_slice_pipeline(slice_idx=slice_idx)
    synchronize(data.kspace.device)
    preprocessing_seconds = time.perf_counter() - started
    image, motion, reconstruction_seconds = timed_reconstruction(data)
    started = time.perf_counter()
    if data.postprocessing.normalize_image_by_grics_reference:
        reference_image = grics_reference_image_for_normalization(data, image)
        image, _ = normalize_reconstruction_by_grics_reference(image, reference_image)
    target_shape, encoded_shape = source.zero_fill_shapes
    image = zero_fill_loaded_reconstruction(image, target_shape, encoded_shape)
    synchronize(image.device)
    postprocessing_seconds = time.perf_counter() - started
    # NumPy transport avoids PyTorch multiprocessing shared-memory lifetime issues.
    # Transfers and IPC are excluded from the per-slice solver timer.
    return {
        'slice_idx': slice_idx, 'slice_number': slice_idx + 1,
        'image': image.detach().cpu().numpy(), 'motion': motion.detach().cpu().numpy(),
        'preprocessing_seconds': preprocessing_seconds,
        'reconstruction_seconds': reconstruction_seconds,
        'postprocessing_seconds': postprocessing_seconds,
        'image_stage': 'after_postprocessing',
        '_configuration': public_config(data.params),
    }


def grics_zero_fill_shapes(raw_data: DataLoader) -> tuple[tuple[int, int], tuple[int, int]]:
    """Read encoded and reconstruction matrix sizes from the ISMRMRD header."""
    header = acquisition_header(raw_data)

    enc = header.encoding[0]
    recon = enc.reconSpace.matrixSize
    encoded = enc.encodedSpace.matrixSize
    target_y = int(recon.y)
    encoded_y = int(encoded.y)
    if encoded_y <= 0:
        raise ValueError(f"ISMRMRD encoded matrix y must be positive, got {encoded_y}.")
    if target_y == 0:
        # Some converted headers omit the reconstruction matrix. Preserve the
        # encoded pixel spacing when deriving the crop from the declared FOVs.
        recon_fov_y = float(enc.reconSpace.fieldOfView_mm.y)
        encoded_fov_y = float(enc.encodedSpace.fieldOfView_mm.y)
        if not all(math.isfinite(value) and value > 0 for value in (recon_fov_y, encoded_fov_y)):
            raise ValueError("A missing reconstruction matrix y requires positive, finite reconstruction and encoded FOV y.")
        target_y = round(encoded_y * recon_fov_y / encoded_fov_y)
        print(
            f"[input] ISMRMRD reconstruction matrix y is zero; "
            f"using {target_y} from the encoded matrix and field-of-view ratio.",
            flush=True,
        )
    elif target_y > encoded_y:
        target_y //= 2

    target_shape = (int(raw_data.Nx), target_y)
    encoded_shape = (int(raw_data.Nx), encoded_y)
    if any(size <= 0 for size in (*target_shape, *encoded_shape)):
        raise ValueError(
            f"GRICS zero-fill sizes must be positive: target_shape={target_shape}, encoded_shape={encoded_shape}."
        )
    if int(raw_data.Ny) > encoded_y:
        raise ValueError(
            f"Loaded phase-encode size {raw_data.Ny} exceeds ISMRMRD encoded matrix y {encoded_y}."
        )
    if any(target > encoded for target, encoded in zip(target_shape, encoded_shape)):
        raise ValueError(
            "GRICS zero-fill sizes are inconsistent: "
            f"target_shape={target_shape} is larger than encoded_shape={encoded_shape}."
        )
    return target_shape, encoded_shape


def zero_fill_loaded_reconstruction(
    image: torch.Tensor,
    target_shape: tuple[int, int],
    encoded_shape: tuple[int, int],
) -> torch.Tensor:
    return zero_fill_grics_image_to_shape(
        image,
        target_shape=target_shape,
        encoded_shape=encoded_shape,
        spatial_dims=(-2, -1),
    )


def _initialize_worker(source):
    global LOADED_DATA
    LOADED_DATA = source
    torch.set_num_threads(1)
    # Inter-op thread count can only be set once in a process; callers may have
    # already used PyTorch before forking. Intra-op limiting is still effective.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def reconstruct_slices_in_parallel(source, slice_indices, max_workers=None):
    if not slice_indices:
        raise ValueError("At least one slice must be selected for reconstruction.")
    if max_workers is not None and (isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1):
        raise ValueError("max_workers must be a positive integer or None.")
    if source.params.runtime_device == 'gpu':
        if max_workers not in (None, 1):
            raise ValueError("GPU slice reconstruction requires max_workers=1.")
        max_workers = 1
    workers = min(max_workers or (os.cpu_count() or 1), len(slice_indices))
    if workers == 1:
        results = [reconstruct_slice(index, source) for index in slice_indices]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                                 initializer=_initialize_worker, initargs=(source,)) as executor:
            futures = [executor.submit(reconstruct_slice, index) for index in slice_indices]
            results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda item: item['slice_idx'])
    for result in results:
        result['image'] = torch.from_numpy(result['image'])
        result['motion'] = torch.from_numpy(result['motion'])
    return results, workers


@managed_execution
def run_pipeline(raw_data_file=None, saec_file=None, *, preprocessed_file=None, output_root=OUTPUT_ROOT,
                 device=RUNTIME_DEVICE, max_workers=MAX_WORKERS,
                 slice_start=RECONSTRUCT_SLICE_START, slice_stop=RECONSTRUCT_SLICE_STOP,
                 reconstruction_config=RECONSTRUCTION_CONFIG,
                 postprocessing_config=POSTPROCESSING_CONFIG, overrides=None,
                 save_reconstruction_logs=None, save_reconstruction_tensors=None,
                 return_tensors=True, export_dicom=False,
                 dicom_header_dir=None, dicom_series_number=1001) -> dict:
    """Reconstruct one acquisition using explicit settings and return its results.

    preprocessed_file is preferred when present; otherwise raw_data_file and SAEC are used.
    reconstructions contains ordered per-slice CPU image/motion tensors, paths,
    and solver times. Images include configured normalization and zero-filling;
    motion tensors stay on the reconstruction grid. Slice range is half-open.
    All final tensor/DICOM writes are deferred until every slice has finished.
    Common save_reconstruction_logs/tensors flags control output independently;
    None uses the TOML/overrides value. export_dicom is independent.
    Text logs are written inside the solver timer; run/configuration metadata
    is always saved.
    """
    started = time.perf_counter()
    raw_data_file, saec_file = resolve_input_files(raw_data_file, saec_file, preprocessed_file)
    data = load_all_slices(raw_data_file, saec_file, output_root=output_root, device=device,
                           reconstruction_config=reconstruction_config,
                           postprocessing_config=postprocessing_config, overrides=overrides,
                           save_reconstruction_logs=save_reconstruction_logs,
                           save_reconstruction_tensors=save_reconstruction_tensors)
    data.params._run_outputs.manifest['inputs'] = {
        'raw_data_file': str(raw_data_file.resolve()), 'saec_file': str(saec_file.resolve()) if saec_file is not None else None}
    indices = selected_slices(int(data.Nz), slice_start, slice_stop)
    data.zero_fill_shapes = grics_zero_fill_shapes(data)
    synchronize(data.kspace.device)
    load_seconds = time.perf_counter() - started
    compute_started = time.perf_counter()
    results, workers = reconstruct_slices_in_parallel(data, indices, max_workers)
    compute_wall_seconds = time.perf_counter() - compute_started

    export_started = time.perf_counter()
    uids = {}
    if export_dicom:
        from pydicom.uid import generate_uid
        uids = {key: generate_uid() for key in
                ('study_instance_uid', 'series_instance_uid', 'frame_of_reference_uid')}
    for result in results:
        params = copy.copy(data.params)
        vars(params).update(result.pop('_configuration'))
        result['dicom_file'] = None
        if export_dicom:
            dicom_path = Path(data.params.run_folder) / 'exports/dicom' / f"slice_{result['slice_number']:03d}.dcm"
            write_reconstruction_dicom(
                result['image'], dicom_path, raw_data=data, slice_index=result['slice_idx'],
                series_description='GRICS reconstruction RESEARCH ONLY',
                images_in_acquisition=len(indices), series_number=dicom_series_number,
                reference_dicom_path=Path(dicom_header_dir) if dicom_header_dir is not None else None,
                **uids)
            result['dicom_file'] = dicom_path
        export_reconstruction(params, result)
    timings = {
        'load_seconds': load_seconds,
        'compute_wall_seconds': compute_wall_seconds,
        'reconstruction_seconds_sum': sum(item['reconstruction_seconds'] for item in results),
        'export_seconds': time.perf_counter() - export_started,
        'pipeline_seconds': time.perf_counter() - started,
    }
    return finish_run(
        data, results, timings, return_tensors=return_tensors,
        selected_slices=indices, max_workers=workers, postprocessing=vars(data.postprocessing),
        export_dicom=export_dicom, dicom_uids=uids,
        dicom_series_number=dicom_series_number if export_dicom else None,
        dicom_header_dir=str(dicom_header_dir) if dicom_header_dir is not None else None,
        compute_wall_timing='slice preprocessing, solver, postprocessing, worker startup and result transfer; excludes exports',
    )


def main(argv=None) -> dict:
    args = parse_args(argv)
    result = run_pipeline(
        args.raw_data_file, args.saec_file, preprocessed_file=args.preprocessed_file, output_root=args.output_root, device=args.device,
        max_workers=args.max_workers, slice_start=args.slice_start, slice_stop=args.slice_stop,
        save_reconstruction_tensors=args.save_reconstruction_tensors,
                          save_reconstruction_logs=args.save_reconstruction_logs, return_tensors=False, export_dicom=not args.no_dicom,
        dicom_header_dir=args.dicom_header_dir, dicom_series_number=args.dicom_series_number,
    )
    print(f"[run] Compute wall time: {result['timings']['compute_wall_seconds']:.2f} s. "
          f"Run: {result['run_folder']}")
    return result


if __name__ == '__main__':
    main()

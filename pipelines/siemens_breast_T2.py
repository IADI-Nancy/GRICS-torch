#!/usr/bin/env python
"""
Slicewise GRICS-torch reconstruction for multislice Siemens breast T2 data.
Example usage:
python pipelines/siemens_breast_T2.py \
  ../data/GRICS-torch/test_XA61_volunteer/0274_T2_s.dat \
  ../data/GRICS-torch/test_XA61_volunteer/0274_T2_s.saec \
    --dicom-header-dir runs/siemens_breast_T2/2017-110_01-0275-V1MR_2026-06-09/DCM_MR/S2/ \
    --dicom-series-number 100
"""


from __future__ import annotations

import argparse
import copy
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
JUPYTER_NOTEBOOK_FLAG = False
# Save diagnostic motion, acquisition-order, and reconstruction figures.
SAVE_DEBUG_PLOTS = False
# Request deterministic PyTorch/cuDNN algorithms independently of the random seed.
USE_DETERMINISTIC_ALGORITHMS = False


# Each process reconstructs one slice, so numerical libraries must not create
# additional thread pools inside every worker.
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
    "SimpleITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
    "KMP_BLOCKTIME": "0",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}
for _key, _value in _THREAD_ENV.items():
    os.environ.setdefault(_key, _value)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.ismrmrd_io import acquisition_header
import torch

from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.runtime_config import load_config, load_postprocessing_config
from src.runtime.runtime_setup import initialize_runtime
from src.runtime.output_layout import managed_execution, bind_output_paths, record_reconstruction
from src.utils.dicom_export import write_reconstruction_dicom
from src.utils.plotting import show_and_save_image
from src.utils.zero_fill import zero_fill_grics_image_to_shape


# With the fork start method, workers inherit this read-only source dataset
# without serializing and copying the complete acquisition for every slice.
LOADED_DATA = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load breast T2 raw data once, then reconstruct all slices in parallel."
    )
    parser.add_argument(
        "raw_data_file",
        type=Path,
        help="Input Siemens .dat file or already converted ISMRMRD .h5/.mrd file.",
    )
    parser.add_argument("saec_file", type=Path, help="Input SAEC physiological .h5 file.")
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
    return parser.parse_args()




def data_type_from_raw_data_file(raw_data_file: Path) -> str:
    suffix = raw_data_file.suffix.lower()
    if suffix == ".dat":
        return "siemens-saec"
    if suffix in {".h5", ".mrd"}:
        return "ismrmrd-saec"
    raise ValueError(
        "raw_data_file must be a Siemens .dat file or an ISMRMRD .h5/.mrd file. "
        f"Got: {raw_data_file}"
    )


def require_existing_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def output_overrides() -> dict:
    return {
        "jupyter_notebook_flag": JUPYTER_NOTEBOOK_FLAG,
        "flip_for_display": True,
        "output_root": str(OUTPUT_ROOT.parent),
        "workflow_label": OUTPUT_ROOT.name,
        "runtime_device": RUNTIME_DEVICE,
        "save_debug_plots": SAVE_DEBUG_PLOTS,
        "use_deterministic_algorithms": USE_DETERMINISTIC_ALGORITHMS,
        "verbose": False,
        "print_to_console": False,
    }


def load_all_slices(raw_data_file: Path, saec_file: Path) -> DataLoader:
    """Load the acquisition once, without running the per-slice preprocessing."""
    require_existing_file(raw_data_file, "raw_data_file")
    require_existing_file(saec_file, "saec_file")
    data_type = data_type_from_raw_data_file(raw_data_file)
    params = load_config(
        data_type=data_type,
        reconstruction_config=RECONSTRUCTION_CONFIG,
        coil_sensitivity_config="config/coil_sensitivity/odille_spline.toml",
        real_data_config=("config/real_data/saec.toml" if data_type.endswith("-saec") else None),
        ismrmrd_reader_config="config/real_data/ismrmrd_reader.toml",
        overrides=output_overrides(),
    )
    postprocessing = load_postprocessing_config(POSTPROCESSING_CONFIG)
    if postprocessing.normalize_image_by_grics_reference and params.coil_sensitivity_method != "odille-spline":
        raise ValueError("Reference-image normalization requires coil_sensitivity_method='odille-spline'.")
    sp_device, t_device = initialize_runtime(params)
    data = DataLoader(
        params=params,
        t_device=t_device,
        sp_device=sp_device,
        filename=(str(raw_data_file), str(saec_file)),
        run_pipeline=False,
    )
    data.postprocessing = postprocessing
    data.load_data()
    return data


def selected_slices(nslices: int) -> list[int]:
    """Resolve and validate the configured half-open slice range."""
    start = RECONSTRUCT_SLICE_START
    stop = RECONSTRUCT_SLICE_STOP if RECONSTRUCT_SLICE_STOP is not None else nslices
    if start < 0 or stop < start or stop > nslices:
        raise ValueError(f"Invalid slice range [{start}, {stop}) for {nslices} slices.")
    return list(range(start, stop))


def set_worker_output_folders(data: DataLoader, slice_output_dir: Path) -> None:
    bind_output_paths(data.params, slice_output_dir)


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
) -> torch.Tensor:
    if tuple(image.shape) != tuple(reference_image.shape):
        raise ValueError(
            "GRICS reference normalization requires matching image shapes: "
            f"image={tuple(image.shape)}, reference={tuple(reference_image.shape)}."
        )

    reference_magnitude = torch.abs(reference_image).to(device=image.device, dtype=image.real.dtype)
    threshold = 1.0 * torch.mean(reference_magnitude)
    below_threshold = reference_magnitude < threshold
    print(
        "[normalize] reference clamp: "
        f"min={float(torch.min(reference_magnitude).item()):.6g}, "
        f"mean={float(torch.mean(reference_magnitude).item()):.6g}, "
        f"max={float(torch.max(reference_magnitude).item()):.6g}, "
        f"threshold={float(threshold.item()):.6g}, "
        f"clamped={100.0 * float(torch.mean(below_threshold.to(torch.float64)).item()):.3f}%",
        flush=True,
    )
    reference_magnitude = torch.clamp(reference_magnitude, min=threshold)
    return image / reference_magnitude, reference_magnitude


def reconstruct_slice(slice_idx: int) -> dict:
    """Preprocess and reconstruct one slice inside a worker process."""
    if LOADED_DATA is None:
        raise RuntimeError("LOADED_DATA is not initialized in the worker process.")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    output_root = Path(LOADED_DATA.params.run_folder)
    slice_output_dir = output_root / "reconstructions" / f"slice_{slice_idx + 1:03d}"
    slice_output_dir.mkdir(parents=True, exist_ok=True)

    # Isolate mutable slice state while retaining the large inherited source tensors.
    data = copy.copy(LOADED_DATA)
    data.params = copy.copy(LOADED_DATA.params)
    set_worker_output_folders(data, slice_output_dir)
    data.run_slice_pipeline(slice_idx=slice_idx)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [run] slice {slice_idx + 1:03d} starting reconstruction", flush=True)

    reconstructor = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=data.params,
        kspace_scale=data.kspace_scale,
        motion_plot_context=data.motion_plot_context,
    )

    t0 = time.time()
    image, alpha = reconstructor.run()
    debug = data.params.save_debug_plots and data.params.save_reconstruction_outputs
    post_dir = Path(data.params.debug_folder) / 'postprocessing'
    if data.postprocessing.normalize_image_by_grics_reference:
        reference_image = grics_reference_image_for_normalization(data, image)
        image, reference_denominator = normalize_reconstruction_by_grics_reference(image, reference_image)
        if debug:
            post_dir.mkdir(parents=True, exist_ok=True)
            for name, tensor in [('reference_smoothed', reference_image),
                                 ('reference_denominator', reference_denominator),
                                 ('image_normalized', image)]:
                torch.save(tensor.detach().cpu(), post_dir / f'{name}.pt')
                show_and_save_image(tensor[0] if tensor.ndim == 3 else tensor,
                                    name, str(post_dir), flip_for_display=data.params.flip_for_display)

    target_shape, encoded_shape = LOADED_DATA.zero_fill_shapes
    image = zero_fill_loaded_reconstruction(image, target_shape, encoded_shape)
    if debug:
        post_dir.mkdir(parents=True, exist_ok=True)
        torch.save(image.detach().cpu(), post_dir / 'image_postprocessed.pt')
        show_and_save_image(image.mean(dim=0), 'image_postprocessed', str(post_dir),
                            flip_for_display=data.params.flip_for_display)
    dicom_path = output_root / 'exports' / 'dicom' / f'slice_{slice_idx + 1:03d}.dcm'
    write_reconstruction_dicom(
        image, dicom_path, raw_data=LOADED_DATA, slice_index=slice_idx,
        series_description='GRICS reconstruction RESEARCH ONLY',
        images_in_acquisition=LOADED_DATA.export_slice_count,
        series_number=LOADED_DATA.export_series_number,
        reference_dicom_path=LOADED_DATA.export_reference,
        **LOADED_DATA.export_uids)
    elapsed_s = time.time() - t0
    record_reconstruction(data.params, status='complete', elapsed_s=elapsed_s,
                          postprocessed_shape=list(image.shape),
                          postprocessing=vars(data.postprocessing),
                          dicom_file=str(dicom_path.relative_to(output_root)),
                          dicom_representation='magnitude of repetition mean, scaled to uint12')
    return {
        'slice_idx': slice_idx, 'slice_number': slice_idx + 1, 'elapsed_s': elapsed_s,
        'output_dir': str(slice_output_dir.relative_to(output_root)),
        'dicom_file': str(dicom_path.relative_to(output_root)),
    }


def grics_zero_fill_shapes(raw_data: DataLoader) -> tuple[tuple[int, int], tuple[int, int]]:
    """Read encoded and reconstruction matrix sizes from the ISMRMRD header."""
    header = acquisition_header(raw_data)

    enc = header.encoding[0]
    recon = enc.reconSpace.matrixSize
    encoded = enc.encodedSpace.matrixSize
    target_y = int(recon.y)
    encoded_y = int(encoded.y)
    if target_y > encoded_y:
        target_y //= 2

    target_shape = (int(raw_data.Nx), target_y)
    encoded_shape = (int(raw_data.Nx), encoded_y)
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


def initialize_source_data(args: argparse.Namespace) -> DataLoader:
    """Load source data and expose it to forked reconstruction workers."""
    global LOADED_DATA
    print(f"[load pid={os.getpid()}] Loading all slices with DataLoader...")
    LOADED_DATA = load_all_slices(args.raw_data_file, args.saec_file)
    LOADED_DATA.params._run_outputs.manifest['inputs'] = {
        'raw_data_file': str(args.raw_data_file.resolve()), 'saec_file': str(args.saec_file.resolve())}
    LOADED_DATA.params._run_outputs.manifest['postprocessing'] = vars(LOADED_DATA.postprocessing)
    LOADED_DATA.params._run_outputs.snapshot(LOADED_DATA.params)
    LOADED_DATA.params._run_outputs.flush()
    print(
        f"[load pid={os.getpid()}] Source data loaded once: "
        f"kspace_shape={tuple(LOADED_DATA._source_kspace.shape)}, "
        f"nslices={int(LOADED_DATA.Nz)}"
    )
    return LOADED_DATA


def reconstruct_slices_in_parallel(slice_indices: list[int]) -> tuple[list[dict], int]:
    """Reconstruct the selected slices in separate forked processes."""
    if not slice_indices:
        raise ValueError("At least one slice must be selected for reconstruction.")

    max_workers = MAX_WORKERS or min(len(slice_indices), os.cpu_count() or 1)
    max_workers = min(max_workers, len(slice_indices))
    print(
        f"[run pid={os.getpid()}] Reconstructing {len(slice_indices)} slices "
        f"with {max_workers} forked workers."
    )

    results = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        futures = [executor.submit(reconstruct_slice, slice_idx) for slice_idx in slice_indices]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[run] slice {result['slice_number']:03d} finished in {result['elapsed_s']:.2f} s")

    results.sort(key=lambda item: item["slice_idx"])
    return results, max_workers


def build_manifest(
    args: argparse.Namespace,
    raw_data: DataLoader,
    slice_indices: list[int],
    results: list[dict],
    max_workers: int,
    zero_filled_shape: tuple[int, int],
    dicom_dir: Path,
    dicom_uids: dict[str, str],
    elapsed_s: float,
) -> dict:
    """Collect run provenance in a JSON-serializable dictionary."""
    return {
        "raw_data_file": str(args.raw_data_file),
        "saec_file": str(args.saec_file),
        "reconstruction_config": RECONSTRUCTION_CONFIG,
        "postprocessing_config": POSTPROCESSING_CONFIG,
        "output_root": ".",
        "dicom_dir": str(dicom_dir.relative_to(Path(raw_data.params.run_folder))),
        "dicom_uids": dicom_uids,
        "dicom_header_dir": (
            str(args.dicom_header_dir) if args.dicom_header_dir is not None else None
        ),
        "dicom_series_number": args.dicom_series_number,
        "zero_filled_shape": zero_filled_shape,
        "nslices": int(raw_data.Nz),
        "selected_slices": slice_indices,
        "max_workers": max_workers,
        "runtime_device": raw_data.params.runtime_device,
        "normalize_image_by_grics_reference": raw_data.postprocessing.normalize_image_by_grics_reference,
        "elapsed_s": elapsed_s,
        "slice_results": results,
    }


@managed_execution
def main() -> None:
    """Run the Siemens breast T2 reconstruction and export pipeline."""
    args = parse_args()
    run_started_at = time.time()

    # 1. Load the complete acquisition once for all forked slice workers.
    raw_data = initialize_source_data(args)
    slice_indices = selected_slices(int(raw_data.Nz))

    from pydicom.uid import generate_uid
    raw_data.zero_fill_shapes = grics_zero_fill_shapes(raw_data)
    raw_data.export_uids = {key: generate_uid() for key in
                           ('study_instance_uid', 'series_instance_uid', 'frame_of_reference_uid')}
    raw_data.export_reference = args.dicom_header_dir
    raw_data.export_series_number = args.dicom_series_number
    raw_data.export_slice_count = len(slice_indices)

    # Workers save reconstruction results, then postprocess and export in memory.
    results, max_workers = reconstruct_slices_in_parallel(slice_indices)

    zero_filled_shape = raw_data.zero_fill_shapes[0]
    output_root = Path(raw_data.params.run_folder)
    dicom_dir = output_root / 'exports' / 'dicom'
    dicom_uids = raw_data.export_uids

    # 4. Save complete run provenance after all outputs have been produced.
    elapsed_s = time.time() - run_started_at
    manifest = build_manifest(
        args,
        raw_data,
        slice_indices,
        results,
        max_workers,
        zero_filled_shape,
        dicom_dir,
        dicom_uids,
        elapsed_s,
    )
    raw_data.params._run_outputs.manifest.update(manifest)
    raw_data.params._run_outputs.flush()
    print(f"[run] Done in {elapsed_s:.2f} s. Manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()

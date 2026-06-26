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
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


OUTPUT_ROOT = Path("runs/siemens_breast_T2")
RECONSTRUCTION_CONFIG = "config/reconstruction/nonrigid_2d_breast.toml"
RUNTIME_DEVICE = "cpu"
MAX_WORKERS = None
RECONSTRUCT_SLICE_START = 0
RECONSTRUCT_SLICE_STOP = None
JUPYTER_NOTEBOOK_FLAG = False
DEBUG_FLAG = False


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

import ismrmrd
import torch

from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from src.utils.dicom_export import write_reconstruction_dicom
from src.utils.zero_fill import zero_fill_grics_image_to_shape


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


def output_overrides(folder: Path) -> dict:
    return {
        "jupyter_notebook_flag": JUPYTER_NOTEBOOK_FLAG,
        "clean_output_folders_before_run": False,
        "runtime_device": RUNTIME_DEVICE,
        "debug_folder": str(folder / "debug") + os.sep,
        "logs_folder": str(folder / "logs") + os.sep,
        "results_folder": str(folder / "results") + os.sep,
        "initial_data_folder": str(folder / "initial_data") + os.sep,
        "debug_flag": DEBUG_FLAG,
        "verbose": False,
        "print_to_console": False,
    }


def load_all_slices(raw_data_file: Path, saec_file: Path) -> DataLoader:
    require_existing_file(raw_data_file, "raw_data_file")
    require_existing_file(saec_file, "saec_file")
    data_type = data_type_from_raw_data_file(raw_data_file)
    params = load_config(
        data_type=data_type,
        reconstruction_config=RECONSTRUCTION_CONFIG,
        overrides=output_overrides(OUTPUT_ROOT / "load"),
    )
    sp_device, t_device = initialize_runtime(params)
    data = DataLoader(
        params=params,
        t_device=t_device,
        sp_device=sp_device,
        filename=(str(raw_data_file), str(saec_file)),
        run_pipeline=False,
    )
    data.load_data()
    return data


def selected_slices(nslices: int) -> list[int]:
    start = RECONSTRUCT_SLICE_START
    stop = RECONSTRUCT_SLICE_STOP if RECONSTRUCT_SLICE_STOP is not None else nslices
    if start < 0 or stop < start or stop > nslices:
        raise ValueError(f"Invalid slice range [{start}, {stop}) for {nslices} slices.")
    return list(range(start, stop))


def set_worker_output_folders(data: DataLoader, slice_output_dir: Path) -> None:
    for key, value in output_overrides(slice_output_dir).items():
        setattr(data.params, key, value)


def reconstruct_slice(slice_idx: int) -> dict:
    if LOADED_DATA is None:
        raise RuntimeError("LOADED_DATA is not initialized in the worker process.")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    slice_output_dir = OUTPUT_ROOT / "reconstructions" / f"slice{slice_idx + 1:03d}"
    slice_output_dir.mkdir(parents=True, exist_ok=True)

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
    elapsed_s = time.time() - t0

    torch.save(image, slice_output_dir / "GricsRecon.pt")
    torch.save(alpha, slice_output_dir / "GricsAlphaMaps.pt")

    return {
        "slice_idx": slice_idx,
        "slice_number": slice_idx + 1,
        "elapsed_s": elapsed_s,
        "output_dir": str(slice_output_dir),
    }


def grics_zero_fill_shapes(raw_data: DataLoader) -> tuple[tuple[int, int], tuple[int, int]]:
    ismrmrd_file = getattr(raw_data, "source_ismrmrd_file", None)
    if not ismrmrd_file:
        raise ValueError("Cannot determine GRICS zero-fill sizes: raw_data.source_ismrmrd_file is not set.")

    dset = ismrmrd.Dataset(str(ismrmrd_file), "dataset", create_if_needed=False)
    try:
        header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    finally:
        dset.close()

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


def zero_fill_reconstructed_images(results: list[dict], raw_data: DataLoader) -> tuple[int, int]:
    target_shape, encoded_shape = grics_zero_fill_shapes(raw_data)
    print(
        f"[zero-fill] Loading GRICS images, padding to encoded matrix {encoded_shape}, "
        f"then cropping to recon matrix {target_shape}"
    )

    for result in results:
        image_path = Path(result["output_dir"]) / "GricsRecon.pt"
        zero_filled_path = Path(result["output_dir"]) / "GricsReconZeroFilled.pt"
        image = torch.load(image_path, map_location="cpu")
        image = zero_fill_loaded_reconstruction(image, target_shape, encoded_shape)
        torch.save(image, zero_filled_path)
        result["zero_filled_recon_file"] = str(zero_filled_path)
        print(f"[zero-fill] slice {result['slice_number']:03d}: {zero_filled_path}")

    return target_shape


def write_reconstruction_dicoms(
    results: list[dict],
    raw_data: DataLoader,
    reference_dicom_path: Path | None,
    series_number: int,
) -> tuple[Path, dict[str, str]]:
    from pydicom.uid import generate_uid

    dicom_dir = OUTPUT_ROOT / "dicoms"
    dicom_dir.mkdir(parents=True, exist_ok=True)
    dicom_uids = {
        "study_instance_uid": generate_uid(),
        "series_instance_uid": generate_uid(),
        "frame_of_reference_uid": generate_uid(),
    }
    print(f"[dicom] Exporting {len(results)} reconstructed slices to {dicom_dir}...")
    print(f"[dicom] SeriesInstanceUID: {dicom_uids['series_instance_uid']}")
    for result in results:
        slice_idx = int(result["slice_idx"])
        image_path = Path(result["zero_filled_recon_file"])
        dicom_path = dicom_dir / f"GricsRecon_slice{slice_idx + 1:03d}.dcm"
        image = torch.load(image_path, map_location="cpu")
        write_reconstruction_dicom(
            image,
            dicom_path,
            raw_data=raw_data,
            slice_index=slice_idx,
            series_description="GRICS reconstruction RESEARCH ONLY",
            images_in_acquisition=len(results),
            series_number=series_number,
            reference_dicom_path=reference_dicom_path,
            **dicom_uids,
        )
        result["dicom_file"] = str(dicom_path)
        print(f"[dicom] slice {result['slice_number']:03d}: {dicom_path}")
    return dicom_dir, dicom_uids


def write_manifest(manifest: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def main() -> None:
    global LOADED_DATA

    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[load pid={os.getpid()}] Loading all slices with DataLoader...")
    LOADED_DATA = load_all_slices(args.raw_data_file, args.saec_file)
    nslices = int(LOADED_DATA.Nz)
    print(
        f"[load pid={os.getpid()}] Source data loaded once: "
        f"kspace_shape={tuple(LOADED_DATA._source_kspace.shape)}, nslices={nslices}"
    )
    slices = selected_slices(nslices)
    max_workers = MAX_WORKERS or min(len(slices), os.cpu_count() or 1)
    max_workers = min(max_workers, len(slices))

    print(f"[run pid={os.getpid()}] Reconstructing {len(slices)} slices with {max_workers} forked workers.")
    results = []
    t0 = time.time()

    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        futures = [executor.submit(reconstruct_slice, slice_idx) for slice_idx in slices]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[run] slice {result['slice_number']:03d} finished in {result['elapsed_s']:.2f} s")

    results.sort(key=lambda item: item["slice_idx"])
    zero_filled_shape = zero_fill_reconstructed_images(results, LOADED_DATA)
    dicom_dir, dicom_uids = write_reconstruction_dicoms(
        results,
        LOADED_DATA,
        reference_dicom_path=args.dicom_header_dir,
        series_number=args.dicom_series_number,
    )
    elapsed_s = time.time() - t0
    write_manifest(
        {
            "raw_data_file": str(args.raw_data_file),
            "saec_file": str(args.saec_file),
            "reconstruction_config": RECONSTRUCTION_CONFIG,
            "output_root": str(OUTPUT_ROOT),
            "dicom_dir": str(dicom_dir),
            "dicom_uids": dicom_uids,
            "dicom_header_dir": str(args.dicom_header_dir) if args.dicom_header_dir is not None else None,
            "dicom_series_number": args.dicom_series_number,
            "zero_filled_shape": zero_filled_shape,
            "nslices": nslices,
            "selected_slices": slices,
            "max_workers": max_workers,
            "runtime_device": RUNTIME_DEVICE,
            "elapsed_s": elapsed_s,
            "slice_results": results,
        }
    )
    print(f"[run] Done in {elapsed_s:.2f} s. Manifest: {OUTPUT_ROOT / 'run_manifest.json'}")


if __name__ == "__main__":
    main()

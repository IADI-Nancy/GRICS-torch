"""Download the Siemens 2D phantom if absent and reconstruct all slices sequentially.

Run from the repository root:
    python tests/validate_siemens_phantom_2d.py --device gpu

The default raw data and physiology are downloaded from Zenodo into demos_data
only when absent. Each invocation converts the input afresh, reads it once,
and saves reconstructed tensors, PNGs, and a DICOM series for all slices.
"""

import argparse
import copy
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import torch
from pydicom.uid import generate_uid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.DataPreparer import DataPreparer
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.output_layout import execution_scope, write_reconstruction
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from src.utils.dicom_export import write_reconstruction_dicom
from src.utils.plotting import show_and_save_image

RAW_DATA_FILE = ROOT / "demos_data/meas_MID00020_FID552369_AX_T2_TSE_HR_1NEX_SAT.dat"
DATA_URL = "https://zenodo.org/records/24314067/files"


def print_success(stage, path=None):
    """Separate completed stages from library progress in terminals and logs."""
    border = "=" * 72
    message = f"\n{border}\n[phantom] SUCCESS: {stage}"
    if path is not None:
        message += f"\nSaved to: {path}"
    print(f"{message}\n{border}\n", flush=True)


def download_phantom_data():
    """Download each missing phantom input from Zenodo into demos_data."""
    RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    for path in (RAW_DATA_FILE, RAW_DATA_FILE.with_suffix(".physio.txt")):
        if path.is_file():
            print(f"[phantom] Found dataset: {path}", flush=True)
            continue
        url = f"{DATA_URL}/{path.name}?download=1"
        print(f"[phantom] Dataset not found at {path}; downloading...", flush=True)
        # Publish only a complete download, so an interruption is retried next time.
        with TemporaryDirectory(prefix="phantom-download-", dir=RAW_DATA_FILE.parent) as directory:
            temporary = Path(directory) / path.name
            urlretrieve(url, temporary)
            temporary.replace(path)
        print_success("DOWNLOAD", path)


def run_phantom(raw_data_file, *, device="gpu", output_root=ROOT / "runs"):
    """Convert and load once, then prepare, reconstruct, and export each slice."""
    raw_data_file = Path(raw_data_file)
    if raw_data_file.resolve() == RAW_DATA_FILE.resolve():
        download_phantom_data()
    physio_file = raw_data_file.with_suffix(".physio.txt")
    for path in (raw_data_file, physio_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    converter = shutil.which("siemens_to_ismrmrd")
    if converter is None:
        raise RuntimeError("siemens_to_ismrmrd executable was not found. Install it to run the phantom check.")
    with TemporaryDirectory(prefix="grics-phantom-") as directory, execution_scope():
        ismrmrd_file = Path(directory) / raw_data_file.with_suffix(".mrd").name
        params = load_config(
            data_type="ismrmrd-physio_text", reconstruction_config=ROOT / "config/reconstruction/nonrigid_2d.toml",
            overrides={"N_motion_states": 1, "workflow_label": "siemens_phantom_2d", "runtime_device": device,
                       "output_root": str(output_root), "cache_root": str(Path(directory) / "cache"), "cache_preprocessed_data": False,
                       "save_debug_plots": False, "save_reconstruction_logs": True, "save_reconstruction_tensors": True,
                       "verbose": False, "print_to_console": False},
        )
        runtime = initialize_runtime(params)
        print("[phantom] Starting fresh Siemens conversion", flush=True)
        subprocess.run([converter, "-f", str(raw_data_file), "-z", "2", "-o", str(ismrmrd_file)], check=True)
        if not ismrmrd_file.is_file() or ismrmrd_file.stat().st_size == 0:
            raise RuntimeError("Siemens conversion produced no data.")
        print_success("CONVERSION", ismrmrd_file)
        data = DataPreparer(params=params, runtime=runtime, filename=(str(ismrmrd_file), str(physio_file)))
        data.load_input_data()
        loader = data.input_loader
        if torch.count_nonzero(loader._input_motion_data):
            raise AssertionError("Loaded physiology must be zero")
        print_success("ISMRMRD READ")
        params._output_unit_count = loader.Nz
        uids = {key: generate_uid() for key in
                ("study_instance_uid", "series_instance_uid", "frame_of_reference_uid")}
        for slice_idx in range(loader.Nz):
            prepared = copy.copy(data)
            prepared.params = copy.copy(params)
            prepared.prepare_reconstructor_inputs(slice_idx=slice_idx)
            if prepared.motion_signal.shape != (1, 1) or torch.count_nonzero(prepared.motion_signal):
                raise AssertionError("Expected a single zero-valued motion state")
            if torch.count_nonzero(prepared.motion_labels):
                raise AssertionError("All readouts must belong to motion state zero")
            reconstructor = JointReconstructor.from_prepared_data(prepared)
            print_success(f"PREPARATION (slice index {slice_idx})")
            image, motion = reconstructor.run()
            expected_shape = (prepared.Nx, prepared.Ny)
            if tuple(image.shape) != expected_shape:
                raise AssertionError(f"Unexpected reconstructed image shape: {tuple(image.shape)} != {expected_shape}")
            if not (torch.isfinite(image).all() and torch.isfinite(motion).all()):
                raise AssertionError("Reconstruction contains non-finite values")
            if not torch.count_nonzero(image):
                raise AssertionError("Reconstruction produced an empty image")
            if torch.count_nonzero(motion):
                raise AssertionError("Zero physiology must produce zero motion")
            print_success(f"RECONSTRUCTION (slice index {slice_idx})")
            dicom_file = write_reconstruction_dicom(
                image, Path(params.run_folder) / "exports/dicom" / f"slice_{slice_idx + 1:03d}.dcm",
                raw_data=prepared, slice_index=slice_idx, images_in_acquisition=loader.Nz, series_number=1001,
                series_description="GRICS reconstruction RESEARCH ONLY", **uids)
            print_success(f"DICOM EXPORT (slice index {slice_idx})", dicom_file)
            paths = write_reconstruction(prepared.params, image, motion, record_paths=True, dicom_file=dicom_file)
            show_and_save_image(image, "image_reconstructed", str(paths["output_dir"]),
                                flip_for_display=params.flip_for_display, jupyter_display=False)
            print_success(f"IMAGE EXPORT (slice index {slice_idx})", paths["output_dir"] / "image_reconstructed.png")
            del reconstructor, prepared, image, motion
        print_success(f"ALL {loader.Nz} SLICES")
        return Path(params.run_folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-file", type=Path, default=RAW_DATA_FILE)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    run_folder = run_phantom(args.raw_data_file, device=args.device, output_root=args.output_root)
    print(f"[phantom] Outputs: {run_folder}")


if __name__ == "__main__":
    main()

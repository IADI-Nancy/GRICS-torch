"""Small file operations shared by recorded-data loading paths."""

import os
import shutil
import subprocess

import h5py

from src.runtime.data_cache import acquire_cached, cache_key, source_identity
from src.runtime.hdf5_cache import read_tree


def convert_siemens_to_ismrmrd(path_to_siemens, params):
    if path_to_siemens is None:
        raise ValueError("Missing Siemens raw data file.")
    path_to_siemens = os.fspath(path_to_siemens)
    if not os.path.exists(path_to_siemens):
        raise FileNotFoundError(f"Siemens raw data file does not exist: {path_to_siemens}")

    converter = shutil.which("siemens_to_ismrmrd")
    if converter is None:
        raise RuntimeError("siemens_to_ismrmrd executable was not found. Install it or supply an existing MRD file.")
    key = cache_key([path_to_siemens], {
        "measurement": 2, "schema": 1, "converter": source_identity(converter),
    })

    def build(output_path):
        cmd = [converter, "-f", path_to_siemens, "-z", "2", "-o", str(output_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Siemens conversion failed: {exc.stdout}\n{exc.stderr}") from exc

    lease = acquire_cached(
        params.cache_root, "converted", key, ".mrd", build,
        remove=params.remove_temporary_data_after_run,
    )
    print(f"[InputDataLoader] Using cached conversion: {lease.path}")
    return str(lease.path)


def read_preprocessed_data(path_to_data, from_data):
    data = {}
    with h5py.File(path_to_data, "r") as handle:
        if from_data:
            for key in ("motion_data", "idx_ky", "idx_kz", "idx_nex"):
                data[key] = handle[key][:]
        data["kspace"] = handle["kspace"][:]
        if "slice_geometry" in handle:
            data["slice_geometry"] = {
                int(key): value for key, value in read_tree(handle["slice_geometry"]).items()
            }
        if "ismrmrd_header" in handle:
            data["ismrmrd_header"] = handle["ismrmrd_header"].asstr()[()]
        if "reference_kspace" in handle:
            data["reference_kspace"] = handle["reference_kspace"][:]
    return data

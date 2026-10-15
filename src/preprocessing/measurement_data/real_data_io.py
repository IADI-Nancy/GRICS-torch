"""Small file operations shared by recorded-data loading paths."""

import json
import os
import shutil
import subprocess

import h5py
import numpy as np


def _write_hdf5_tree(group, values):
    for key, value in values.items():
        name = str(key)
        if isinstance(value, dict):
            node = group.create_group(name)
            node.attrs["kind"] = "dict"
            node.attrs["keys"] = json.dumps(list(value))
            _write_hdf5_tree(node, value)
        elif isinstance(value, (list, tuple)):
            node = group.create_group(name)
            node.attrs["kind"] = "list"
            _write_hdf5_tree(node, dict(enumerate(value)))
        elif value is None:
            node = group.create_group(name)
            node.attrs["kind"] = "none"
        elif isinstance(value, str):
            group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))
        else:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            group.create_dataset(name, data=value)


def _read_hdf5_tree(group):
    result = {}
    for name, node in group.items():
        if isinstance(node, h5py.Group):
            kind = node.attrs["kind"]
            content = _read_hdf5_tree(node)
            if kind == "none":
                value = None
            elif kind == "list":
                value = [content[str(i)] for i in range(len(content))]
            else:
                value = {key: content[str(key)] for key in json.loads(node.attrs["keys"])}
        else:
            value = node.asstr()[()] if h5py.check_string_dtype(node.dtype) else node[()]
            if isinstance(value, np.generic):
                value = value.item()
        result[name] = value
    return result


def write_preprocessed_cache(path, data, metadata):
    """Write a complete prepared acquisition and its preprocessing metadata."""
    with h5py.File(path, "w") as handle:
        _write_hdf5_tree(handle, data)
        cache_metadata = handle.create_group("_cache")
        cache_metadata.attrs["kind"] = "dict"
        cache_metadata.attrs["keys"] = json.dumps(list(metadata))
        _write_hdf5_tree(cache_metadata, metadata)


def read_preprocessed_cache(path):
    """Read a complete prepared acquisition and its preprocessing metadata."""
    with h5py.File(path, "r") as handle:
        data = _read_hdf5_tree(handle)
    return data, data.pop("_cache")


def convert_siemens_to_ismrmrd(path_to_siemens, cache):
    if path_to_siemens is None:
        raise ValueError("Missing Siemens raw data file.")
    path_to_siemens = os.fspath(path_to_siemens)
    if not os.path.exists(path_to_siemens):
        raise FileNotFoundError(f"Siemens raw data file does not exist: {path_to_siemens}")

    converter = shutil.which("siemens_to_ismrmrd")
    if converter is None:
        raise RuntimeError("siemens_to_ismrmrd executable was not found. Install it or supply an existing MRD file.")
    key = cache.key([path_to_siemens], {
        "measurement": 2, "schema": 1, "converter": cache.source_identity(converter),
    })

    def build(output_path):
        cmd = [converter, "-f", path_to_siemens, "-z", "2", "-o", str(output_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Siemens conversion failed: {exc.stdout}\n{exc.stderr}") from exc

    lease = cache.acquire_converted(key, build)
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
                int(key): value for key, value in _read_hdf5_tree(handle["slice_geometry"]).items()
            }
        if "ismrmrd_header" in handle:
            data["ismrmrd_header"] = handle["ismrmrd_header"].asstr()[()]
        if "reference_kspace" in handle:
            data["reference_kspace"] = handle["reference_kspace"][:]
    return data

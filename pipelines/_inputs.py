"""Pipeline input selection and raw/preprocessed data type detection."""
from pathlib import Path
import os
import shutil
import tempfile

import h5py


def resolve_input_files(raw_data_file, saec_file=None, preprocessed_file=None):
    """Prefer an explicit preprocessed file, falling back only when absent."""
    if preprocessed_file is not None:
        preferred = Path(preprocessed_file).expanduser()
        if preferred.exists():
            kind = data_type_from_raw_data_file(preferred)
            if kind != "preprocessed-real":
                raise ValueError("preprocessed_file must contain preprocessed HDF5 data.")
            return preferred, None
        print(f"[input] Preprocessed file missing: {preferred}; using raw data and SAEC.", flush=True)
    if raw_data_file is None:
        raise ValueError("Provide raw_data_file and saec_file when preprocessed_file is unavailable.")
    return (Path(raw_data_file).expanduser(),
            Path(saec_file).expanduser() if saec_file is not None else None)


def require_existing_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")


def data_type_from_raw_data_file(raw_data_file: Path, saec_file: Path | None = None) -> str:
    require_existing_file(raw_data_file, "raw_data_file")
    suffix = raw_data_file.suffix.lower()
    if suffix not in {".dat", ".h5", ".mrd"}:
        raise ValueError("raw_data_file must be a Siemens .dat or HDF5/ISMRMRD .h5/.mrd file.")
    if suffix in {".h5", ".mrd"}:
        with h5py.File(raw_data_file, "r") as source:
            if "kspace" in source:
                if saec_file is not None:
                    raise ValueError("Preprocessed HDF5 already contains physiology; omit saec_file.")
                return "preprocessed-real"
    if saec_file is None:
        raise ValueError("saec_file is required for Siemens/ISMRMRD raw input.")
    require_existing_file(saec_file, "saec_file")
    return "siemens-saec" if suffix == ".dat" else "ismrmrd-saec"


def publish_preprocessed_cache(cache_file, destination):
    """Atomically publish a raw-preprocessing cache as a reusable input file."""
    source = Path(cache_file)
    destination = Path(destination).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Generated preprocessing cache is unavailable: {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing prepared input: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination

"""Shared execution and checks for recorded Siemens phantom validations."""

import copy
from pathlib import Path

import numpy as np
import torch

from src.preprocessing.DataPreparer import DataPreparer
from src.preprocessing.real_data.ISMRMRDReader import _is_noise, _is_non_imaging
from src.preprocessing.real_data.real_data_io import convert_siemens_to_ismrmrd
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.output_layout import execution_scope, write_reconstruction
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from src.utils.ismrmrd_io import ReadOnlyDataset

ROOT = Path(__file__).resolve().parents[2]
PHANTOM_ROOT = ROOT.parent / "data/GRICS-torch/phantom"


def prepare_zero_physiology(ismrmrd_file, physio_file):
    """Write one zero per retained imaging readout in full acquisition order.

    The text format uses SENSOR TIMESTAMP VALUE, with -1 indicating that the
    sample is already synchronized and has no timestamp. Noise and non-imaging
    acquisitions are excluded using the MRI reader's filtering rules.
    """
    dset = ReadOnlyDataset(ismrmrd_file)
    try:
        n_readouts = 0
        for index in range(dset.number_of_acquisitions()):
            acq = dset.read_acquisition(index)
            if not (_is_noise(acq) or _is_non_imaging(acq)):
                n_readouts += 1
    finally:
        dset.close()
    if n_readouts == 0:
        raise AssertionError("The phantom contains no retained imaging readouts")
    physio_file = Path(physio_file)
    if not physio_file.exists():
        with physio_file.open("x", encoding="utf-8") as handle:
            handle.write("0 -1 0\n" * n_readouts)
        print(f"[phantom] Created {physio_file}: {n_readouts} zero samples, no timestamps", flush=True)
    records = np.loadtxt(physio_file, ndmin=2)
    if records.shape != (n_readouts, 3) or not np.all(records == [0, -1, 0]):
        raise AssertionError(f"{physio_file} must contain exactly {n_readouts} rows of '0 -1 0'")
    return n_readouts


def run_phantom(raw_data_file, dimension, *, device="gpu", output_root=ROOT / "runs", slice_idx=None):
    """Run the standard solver schedule with zero physiology and one state."""
    raw_data_file = Path(raw_data_file)
    if not raw_data_file.is_file():
        raise FileNotFoundError(raw_data_file)
    physio_file = raw_data_file.with_suffix(".physio.txt")
    reconstruction_config = "nonrigid_3d_breast.toml" if dimension == "3D" else "nonrigid_2d.toml"
    print(f"[phantom] Reconstruction configuration: {reconstruction_config}; N_motion_states=1", flush=True)
    with execution_scope():
        params = load_config(
            data_type="siemens-physio_text",
            reconstruction_config=ROOT / "config/reconstruction" / reconstruction_config,
            overrides={"N_motion_states": 1, "runtime_device": device,
                       "output_root": str(output_root), "workflow_label": f"siemens_phantom_{dimension.lower()}",
                       "cache_root": str(Path(output_root) / "cache/grics"),
                       "save_debug_plots": False, "save_reconstruction_logs": True,
                       "save_reconstruction_tensors": True, "verbose": False, "print_to_console": False},
        )
        runtime = initialize_runtime(params)
        data = DataPreparer(params=params, runtime=runtime, filename=(str(raw_data_file), str(physio_file)))
        ismrmrd_file = convert_siemens_to_ismrmrd(raw_data_file, data.input_loader.raw_data_cache)
        n_readouts = prepare_zero_physiology(ismrmrd_file, physio_file)
        data.load_input_data()
        loader = data.input_loader
        if loader._input_idx_ky.numel() != n_readouts or torch.count_nonzero(loader._input_motion_data):
            raise AssertionError("Loaded physiology must be zero with one sample per imaging readout")
        if dimension == "3D":
            if loader.Nz <= 1:
                raise AssertionError("The 3D phantom must contain multiple encoded partitions")
            indices = [None]
        else:
            indices = list(range(loader.Nz)) if slice_idx is None else [slice_idx]
        params._output_unit_count = len(indices)
        for index in indices:
            prepared = copy.copy(data)
            prepared.params = copy.copy(params)
            prepared.prepare_reconstructor_inputs(slice_idx=index)
            if prepared.motion_signal.shape != (1, 1) or torch.count_nonzero(prepared.motion_signal):
                raise AssertionError("Expected a single zero-valued motion state")
            if torch.count_nonzero(prepared.motion_labels):
                raise AssertionError("All readouts must belong to motion state zero")
            reconstructor = JointReconstructor.from_prepared_data(prepared)
            image, motion = reconstructor.run()
            expected_shape = (prepared.Nx, prepared.Ny, prepared.Nz) if dimension == "3D" else (prepared.Nx, prepared.Ny)
            if tuple(image.shape) != expected_shape:
                raise AssertionError(f"Unexpected reconstructed image shape: {tuple(image.shape)} != {expected_shape}")
            if not (torch.isfinite(image).all() and torch.isfinite(motion).all()):
                raise AssertionError("Reconstruction contains non-finite values")
            if not torch.count_nonzero(image):
                raise AssertionError("Reconstruction produced an empty image")
            if torch.count_nonzero(motion):
                raise AssertionError("Zero physiology must produce zero motion")
            write_reconstruction(prepared.params, image, motion)
            print(f"[phantom] {dimension} slice={index}: PASSED; image shape={tuple(image.shape)}", flush=True)
            del reconstructor, prepared, image, motion
        return Path(params.run_folder)

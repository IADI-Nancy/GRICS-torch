"""Construct encoding and motion operators from resolution-level data."""

import torch
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator


def compute_dtype_for(params):
    """Operator arithmetic precision: complex64 when compute_precision = "single", else complex128."""
    return torch.complex64 if getattr(params, 'compute_precision', 'double') == 'single' else torch.complex128


def build_motion_operator(Data_res, params):
    Nx, Ny = Data_res["Nx"], Data_res["Ny"]
    alpha = Data_res["MotionModel"]
    compute_dtype = compute_dtype_for(params)
    warp_backend = getattr(params, 'warp_backend', 'sparse')
    if params.reconstruction_motion_type == "rigid":
        motionOperator = MotionOperator(
            Nx, Ny, alpha, params.reconstruction_motion_type, Nz=Data_res.get("Nz", 1), compute_dtype=compute_dtype, warp_backend=warp_backend
        )
    else:
        motion_signal = Data_res["MotionSignal"]
        motionOperator = MotionOperator(
            Nx, Ny, alpha, params.reconstruction_motion_type,
            motion_signal=motion_signal.to(dtype=alpha.dtype), Nz=Data_res.get("Nz", 1), compute_dtype=compute_dtype, warp_backend=warp_backend
        )
    return motionOperator


def build_encoding_operator(Data_res, params):
    E = EncodingOperator(Data_res["SensitivityMaps"], Data_res["Nsamples"], Data_res["SamplingIndices"],
                         params.Nex, Data_res["MotionOperator"])
    return E


def build_motion_perturbation_simulator(Data_res, params):
    J = MotionPerturbationSimulator(Data_res["SensitivityMaps"], Data_res["Nsamples"], Data_res["SamplingIndices"],
                                    params.Nex, Data_res["ReconstructedImage"], Data_res["MotionOperator"],
                                    gradient_boundary=params.motion_gradient_boundary)
    return J

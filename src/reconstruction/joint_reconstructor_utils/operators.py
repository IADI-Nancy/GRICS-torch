"""Construct encoding and motion operators from resolution-level data."""

from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator


def build_motion_operator(Data_res, params):
    Nx, Ny = Data_res["Nx"], Data_res["Ny"]
    alpha = Data_res["MotionModel"]
    if params.reconstruction_motion_type == "rigid":
        motionOperator = MotionOperator(
            Nx, Ny, alpha, params.reconstruction_motion_type, Nz=Data_res.get("Nz", 1)
        )
    else:
        motion_signal = Data_res["MotionSignal"]
        motionOperator = MotionOperator(
            Nx, Ny, alpha, params.reconstruction_motion_type,
            motion_signal=motion_signal.to(dtype=alpha.dtype), Nz=Data_res.get("Nz", 1)
        )
    return motionOperator


def build_encoding_operator(Data_res, params):
    E = EncodingOperator(Data_res["SensitivityMaps"], Data_res["Nsamples"], Data_res["SamplingIndices"],
                         params.Nex, Data_res["MotionOperator"])
    return E


def build_motion_perturbation_simulator(Data_res, params):
    J = MotionPerturbationSimulator(Data_res["SensitivityMaps"], Data_res["Nsamples"], Data_res["SamplingIndices"],
                                    params.Nex, Data_res["ReconstructedImage"], Data_res["MotionOperator"])
    return J

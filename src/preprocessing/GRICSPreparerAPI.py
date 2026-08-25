"""High-level preparation of motion-binned GRICS acquisition metadata."""

import copy

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.preprocessing.MotionBinner import MotionBinner
from src.preprocessing.Sampling import Sampling
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime


@dataclass(frozen=True)
class PreparedGRICSAcquisition:
    """Configuration, motion bins, and named sampling layouts for one acquisition."""

    params: Any
    motion_signal: torch.Tensor
    motion_labels: torch.Tensor
    sampling_indices: dict[str, list]
    chronological_ky: torch.Tensor
    chronological_kz: torch.Tensor | None
    chronological_nex: torch.Tensor


class GRICSPreparerAPI:
    """Load GRICS configuration once and prepare acquisition metadata.

    The constructor loads the reconstruction configuration and initializes the
    runtime. ``prepare_acquisition`` then bins a physiological signal and
    constructs one or more named sampling layouts. Applications can cache the
    result because it depends on acquisition metadata, not learned parameters.
    """

    def __init__(
        self, reconstruction_config, *, data_type="preprocessed-real",
        motion_simulation_model_mode="as-it-is", data_dimension="2D",
        overrides=None, **config_options,
    ):
        self.params = load_config(
            data_type=data_type, reconstruction_config=reconstruction_config,
            motion_simulation_model_mode=motion_simulation_model_mode,
            data_dimension=data_dimension, overrides=overrides, **config_options,
        )
        self.sp_device, self.device = initialize_runtime(self.params)

    def prepare_acquisition(
        self, motion_data, ky_indices, nex_indices, *, Nx, Ny, Nz=1,
        kz_indices=None, sampling_masks: Mapping[str, torch.Tensor] | None = None,
        kspace=None, y_limits=None, seed=None,
    ):
        """Bin motion and build named ``[Nex][Nmotion]`` sampling layouts.

        ``sampling_masks`` maps caller-defined names to Boolean masks over the
        chronological readouts. If omitted, one layout named ``all`` is made.
        This keeps application-specific split policies outside GRICS.
        """
        motion = torch.as_tensor(motion_data, device=self.device)
        if motion.ndim == 1:
            motion = motion.unsqueeze(-1)
        ky = torch.as_tensor(ky_indices, device=self.device, dtype=torch.int64).reshape(-1)
        nex = torch.as_tensor(nex_indices, device=self.device, dtype=torch.int64).reshape(-1)
        kz = None if kz_indices is None else torch.as_tensor(
            kz_indices, device=self.device, dtype=torch.int64,
        ).reshape(-1)
        if ky.numel() == 0:
            raise ValueError("At least one readout is required.")
        if motion.shape[0] != ky.numel() or nex.numel() != ky.numel():
            raise ValueError("motion_data, ky_indices, and nex_indices must describe the same readouts.")
        if kz is not None and kz.numel() != ky.numel():
            raise ValueError("kz_indices must describe the same readouts as ky_indices.")

        params = copy.copy(self.params)
        params.Nex = int(nex.max().item()) + 1
        params.NshotsPerNex = int(Ny)
        params.Nshots = int(Ny) * params.Nex
        masks = {"all": torch.ones(ky.numel(), dtype=torch.bool, device=self.device)}
        if sampling_masks is not None:
            masks = {
                name: torch.as_tensor(mask, device=self.device, dtype=torch.bool).reshape(-1)
                for name, mask in sampling_masks.items()
            }
            if any(mask.numel() != ky.numel() for mask in masks.values()):
                raise ValueError("Every sampling mask must contain one value per readout.")

        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                torch.manual_seed(seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
            _, _, motion_signal, labels, chronological_ky, chronological_kz, chronological_nex = (
                MotionBinner.bin_motion(
                    motion, ky, kz, nex, self.device, params,
                    y_limits=y_limits, return_debug_data=True, kspace=kspace,
                )
            )

        sampling_indices = {}
        for name, selection in masks.items():
            binned_ky = [[
                ky[(nex == excitation) & (labels == state) & selection]
                for state in range(params.N_motion_states)
            ] for excitation in range(params.Nex)]
            binned_kz = None if kz is None else [[
                kz[(nex == excitation) & (labels == state) & selection]
                for state in range(params.N_motion_states)
            ] for excitation in range(params.Nex)]
            sampling_indices[name] = Sampling.build_sampling_per_nex_per_motion(
                binned_ky, self.device, Nx, Ny, Nz=Nz, binned_kz_indices=binned_kz,
            )

        return PreparedGRICSAcquisition(
            params, motion_signal, labels, sampling_indices,
            chronological_ky, chronological_kz, chronological_nex,
        )

"""High-level preparation of motion-binned GRICS acquisition metadata."""

import copy
from os import PathLike
from types import SimpleNamespace

import numpy as np

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.preprocessing.MotionBinner import MotionBinner
from src.preprocessing.Sampling import Sampling
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime


TensorLike = torch.Tensor | np.ndarray | list | tuple


@dataclass(frozen=True)
class PreparedGRICSAcquisition:
    """Configuration, motion bins, and named sampling layouts for one acquisition."""

    params: SimpleNamespace
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
    constructs one or more named sampling layouts. Caller-provided Boolean
    masks support externally defined undersampling or held-out readouts; this
    class translates those selections into GRICS indices but does not generate
    an acceleration pattern. Applications can cache the result because it
    depends on acquisition metadata, not learned parameters.

    Args:
        reconstruction_config: Path to the mandatory reconstruction TOML.
        data_type: External data mode; default ``preprocessed-real``.
        simulated_motion_type: Motion mode; default ``as-it-is``.
        data_dimension: ``2D`` or ``3D``.
        overrides: Optional mapping of configuration overrides.
        **config_options: Additional keyword arguments accepted by ``load_config``.
    """

    def __init__(
        self, reconstruction_config: str | PathLike[str], *,
        data_type: str = "preprocessed-real",
        simulated_motion_type: str = "as-it-is", data_dimension: str = "2D",
        overrides: Mapping[str, Any] | None = None, **config_options: Any,
    ) -> None:
        self.params = load_config(
            data_type=data_type, reconstruction_config=reconstruction_config,
            simulated_motion_type=simulated_motion_type,
            data_dimension=data_dimension, overrides=overrides, **config_options,
        )
        self.sp_device, self.device = initialize_runtime(self.params)

    def prepare_acquisition(
        self, motion_data: TensorLike, ky_indices: TensorLike,
        nex_indices: TensorLike, *, Nx: int, Ny: int, Nz: int = 1,
        kz_indices: TensorLike | None = None,
        sampling_masks: Mapping[str, TensorLike] | None = None,
        kspace: torch.Tensor | None = None,
        y_limits: tuple[float, float] | None = None, seed: int | None = None,
    ) -> PreparedGRICSAcquisition:
        """Bin motion and build named ``[Ne][Nm]`` sampling layouts.

        Args:
            motion_data: Real tensor-like ``[Nr]`` or ``[Nr, Ns]``.
            ky_indices: Integer tensor-like ``[Nr]`` in ``[0, Ny-1]``.
            nex_indices: Integer tensor-like ``[Nr]`` with zero-based values.
            Nx: Positive readout matrix size.
            Ny: Positive phase-encode matrix size.
            Nz: Positive partition count; default ``1`` for 2D.
            kz_indices: Integer tensor-like ``[Nr]`` for 3D, otherwise ``None``.
            sampling_masks: Optional mapping from arbitrary names to Boolean
                tensor-like ``[Nr]`` selections for external undersampling.
            kspace: Optional complex ``[Nc, Ne, Nx, Ny, (Nz)]`` tensor; required
                only when the configured binning mode uses k-space energy.
            y_limits: Optional plotting range ``(minimum, maximum)``.
            seed: Optional integer for reproducible motion binning.

        If ``sampling_masks`` is omitted, one layout named ``all`` is returned.
        The caller creates any acceleration pattern; this method only converts
        readout selections to GRICS sampling indices.
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

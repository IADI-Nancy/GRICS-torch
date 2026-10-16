"""High-level preparation of motion-binned GRICS acquisition metadata."""

import copy
from numbers import Integral
from os import PathLike
from types import SimpleNamespace

import numpy as np

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.preprocessing.MotionSamplingPreparer import MotionSamplingPreparer
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime


TensorLike = torch.Tensor | np.ndarray | list | tuple


def _readout_indices(values: TensorLike, name: str, device: torch.device) -> torch.Tensor:
    indices = torch.as_tensor(values, device=device)
    if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must contain integer indices.")
    return indices.to(torch.int64).reshape(-1)


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
        data_type: Must be ``"preprocessed-real"`` because acquisition metadata
            is supplied directly by the caller.
        overrides: Optional mapping of allowed run-specific overrides.
    """

    def __init__(
        self, reconstruction_config: str | PathLike[str], *,
        data_type: str, overrides: Mapping[str, Any] | None = None,
    ) -> None:
        if data_type != "preprocessed-real":
            raise ValueError("GRICSPreparerAPI accepts only data_type='preprocessed-real'; acquisition metadata must be supplied by the caller.")
        self.params = load_config(
            data_type=data_type, reconstruction_config=reconstruction_config,
            overrides=overrides,
        )
        if self.params.kspace_sampling_type != "from-data" or self.params.simulated_motion_type != "as-it-is":
            raise ValueError("GRICSPreparerAPI bins recorded metadata only; use DataPreparer for simulated sampling or motion.")
        self.sp_device, self.device = initialize_runtime(self.params)

    def prepare_acquisition(
        self, motion_data: TensorLike, ky_indices: TensorLike,
        nex_indices: TensorLike, *, Nx: int, Ny: int, Nz: int,
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
            Nz: Positive partition count.
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
        if motion.ndim != 2:
            raise ValueError("motion_data must have shape [Nreadout] or [Nreadout, Nsensor].")
        if torch.is_complex(motion) or motion.dtype == torch.bool:
            raise ValueError("motion_data must contain real numeric values.")
        if not torch.is_floating_point(motion):
            motion = motion.to(torch.get_default_dtype())
        if not torch.isfinite(motion).all():
            raise ValueError("motion_data must contain finite values.")
        ky = _readout_indices(ky_indices, "ky_indices", self.device)
        nex = _readout_indices(nex_indices, "nex_indices", self.device)
        kz = None if kz_indices is None else _readout_indices(kz_indices, "kz_indices", self.device)
        if any(isinstance(size, bool) or not isinstance(size, Integral) or size < 1 for size in (Nx, Ny, Nz)):
            raise ValueError("Nx, Ny, and Nz must be positive integers.")
        if ky.numel() == 0:
            raise ValueError("At least one readout is required.")
        if motion.shape[0] != ky.numel() or nex.numel() != ky.numel():
            raise ValueError("motion_data, ky_indices, and nex_indices must describe the same readouts.")
        if kz is not None and kz.numel() != ky.numel():
            raise ValueError("kz_indices must describe the same readouts as ky_indices.")
        if torch.any((ky < 0) | (ky >= Ny)):
            raise ValueError("ky_indices must lie in [0, Ny).")
        if torch.any(nex < 0):
            raise ValueError("nex_indices must be nonnegative.")
        if Nz > 1 and kz is None:
            raise ValueError("kz_indices are required when Nz > 1.")
        if kz is not None and torch.any((kz < 0) | (kz >= Nz)):
            raise ValueError("kz_indices must lie in [0, Nz).")

        params = copy.copy(self.params)
        params.Nex = int(nex.max().item()) + 1
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
            prepared = MotionSamplingPreparer.prepare(
                motion, ky, kz, nex, self.device, params, Nx, Ny, Nz=Nz,
                kspace=kspace, sampling_masks=masks, y_limits=y_limits,
            )

        return PreparedGRICSAcquisition(
            params, prepared.motion_signal, prepared.motion_labels, prepared.sampling_indices,
            prepared.chronological_ky, prepared.chronological_kz, prepared.chronological_nex,
        )

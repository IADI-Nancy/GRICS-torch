from dataclasses import dataclass
from typing import Mapping

import torch

from src.preprocessing.MotionBinner import MotionBinner
from src.utils.plotting import save_clustered_motion_plots


@dataclass(frozen=True)
class PreparedMotionSampling:
    binned_ky_indices: list
    binned_kz_indices: list | None
    motion_signal: torch.Tensor | None
    motion_labels: torch.Tensor | None
    chronological_ky: torch.Tensor
    chronological_kz: torch.Tensor | None
    chronological_nex: torch.Tensor
    sampling_indices: dict[str, list]


class MotionSamplingPreparer:
    """Group acquired readouts and build reconstruction sampling layouts."""

    @staticmethod
    def build_sampling_per_nex_per_motion(binned_ky_indices, device, Nx, Ny, Nz=1, binned_kz_indices=None):
        """Return flattened sampling indices grouped as ``[Nex][Nmotion]``."""
        nex_count = len(binned_ky_indices)
        if nex_count == 0:
            raise ValueError("binned_ky_indices cannot be empty.")

        motion_count = len(binned_ky_indices[0])
        if any(len(states) != motion_count for states in binned_ky_indices):
            raise ValueError("Every excitation in binned_ky_indices must have the same number of motion states.")
        if Nz > 1:
            if binned_kz_indices is None:
                raise ValueError("binned_kz_indices is required when Nz > 1.")
            if len(binned_kz_indices) != nex_count or any(len(states) != motion_count for states in binned_kz_indices):
                raise ValueError("binned_kz_indices must match the [Nex][Nmotion] layout of binned_ky_indices.")

        kx = torch.arange(Nx, device=device, dtype=torch.int64)
        sampling = [[None for _ in range(motion_count)] for _ in range(nex_count)]

        for nex in range(nex_count):
            for motion_state in range(motion_count):
                ky = binned_ky_indices[nex][motion_state].to(device=device, dtype=torch.int64).reshape(-1)
                if ky.numel() == 0:
                    sampling[nex][motion_state] = torch.empty(0, dtype=torch.int64, device=device)
                    continue

                if Nz > 1:
                    kz = binned_kz_indices[nex][motion_state].to(device=device, dtype=torch.int64).reshape(-1)
                    if ky.numel() != kz.numel():
                        raise ValueError("Paired ky and kz bins must contain the same number of readouts.")
                    # The reconstruction tensor is flattened in [kx, ky, kz] order.
                    indices = ((ky[:, None] + Ny * kx[None, :]) * Nz + kz[:, None]).reshape(-1)
                else:
                    indices = (ky[:, None] + Ny * kx[None, :]).reshape(-1)

                sampling[nex][motion_state] = indices

        return sampling

    @staticmethod
    def _flatten_indices(values):
        if values is None:
            return None
        if torch.is_tensor(values):
            return values.reshape(-1)
        return torch.cat([value.reshape(-1) for value in values], dim=0)

    @staticmethod
    def _group_by_labels(ky, kz, nex, labels, selection, nex_count, state_count):
        binned_ky = [[
            ky[(nex == repetition) & (labels == state) & selection]
            for state in range(state_count)
        ] for repetition in range(nex_count)]
        binned_kz = None if kz is None else [[
            kz[(nex == repetition) & (labels == state) & selection]
            for state in range(state_count)
        ] for repetition in range(nex_count)]
        return binned_ky, binned_kz

    @staticmethod
    def prepare(motion_curve, ky_idx, kz_idx, nex_idx, device, params, Nx, Ny, Nz=1,
                *, kspace=None, sampling_masks: Mapping[str, torch.Tensor] | None = None,
                y_limits=None, plot_kwargs=None):
        """Return bins, chronological indices, and named flattened layouts.

        With no motion curve, each recorded readout remains its own state within
        its repetition. Named masks require motion labels and are used by the
        external GRICS API.
        """
        ky = MotionSamplingPreparer._flatten_indices(ky_idx)
        kz = MotionSamplingPreparer._flatten_indices(kz_idx)
        nex = MotionSamplingPreparer._flatten_indices(nex_idx)
        if motion_curve is None:
            if sampling_masks is not None:
                raise ValueError("Named sampling masks require a motion curve.")
            binned_ky = [[value.reshape(1) for value in ky[nex == repetition]]
                         for repetition in range(params.Nex)]
            binned_kz = None if kz is None else [
                [value.reshape(1) for value in kz[nex == repetition]]
                for repetition in range(params.Nex)
            ]
            motion_signal = None
            labels = None
        else:
            labels, motion_signal, ky, kz, nex = MotionBinner.assign_states(
                motion_curve, ky, kz, nex, device, params, kspace=kspace,
            )
            all_readouts = torch.ones(ky.numel(), dtype=torch.bool, device=device)
            binned_ky, binned_kz = MotionSamplingPreparer._group_by_labels(
                ky, kz, nex, labels, all_readouts, params.Nex, params.N_motion_states,
            )
            if params.save_debug_plots:
                save_clustered_motion_plots(
                    motion_curve=motion_curve, labels=labels, ky_idx=ky, nex_idx=nex, kz_idx=kz,
                    nbins=params.N_motion_states, output_folder=params.initial_data_folder,
                    resolution_levels=params.ResolutionLevels, data_type=params.data_type,
                    y_limits=y_limits, **(plot_kwargs or {}),
                )

        if sampling_masks is None:
            sampling_masks = {"all": torch.ones(ky.numel(), dtype=torch.bool, device=device)}
        sampling_indices = {}
        for name, mask in sampling_masks.items():
            selection = torch.as_tensor(mask, device=device, dtype=torch.bool).reshape(-1)
            if selection.numel() != ky.numel():
                raise ValueError("Every sampling mask must contain one value per readout.")
            # Every named layout uses the same motion labels; a mask only removes readouts.
            if labels is None or (len(sampling_masks) == 1 and name == "all" and bool(selection.all())):
                selected_ky, selected_kz = binned_ky, binned_kz
            else:
                selected_ky, selected_kz = MotionSamplingPreparer._group_by_labels(
                    ky, kz, nex, labels, selection, params.Nex, params.N_motion_states,
                )
            sampling_indices[name] = MotionSamplingPreparer.build_sampling_per_nex_per_motion(
                selected_ky, device, Nx, Ny, Nz=Nz, binned_kz_indices=selected_kz,
            )

        return PreparedMotionSampling(
            binned_ky, binned_kz, motion_signal, labels, ky, kz, nex, sampling_indices,
        )

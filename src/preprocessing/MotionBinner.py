import torch

from src.utils.plotting import save_clustered_motion_plots


class ConstantPhysiologicalSignalError(ValueError):
    """Raised when physiological samples contain no usable motion variation."""


def _kmeans_torch(x, k, n_iter=20):
    N, _ = x.shape

    # Better init: random unique points
    perm = torch.randperm(N, device=x.device)
    centers = x[perm[:k]].clone()

    for _ in range(n_iter):
        dist = torch.cdist(x, centers)
        labels = dist.argmin(dim=1)

        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(dim=0)
            else:
                # Empty cluster → reinitialize
                idx = torch.randint(0, N, (1,), device=x.device)
                centers[j] = x[idx]

    return labels, centers

def _quantize_motion_features(x, n_bins):
    """Uniformly quantize every physiological channel, as in GRICS++."""
    if n_bins < 2:
        raise ValueError("motion_quantization_bins must be at least 2.")
    lower = x.amin(dim=0)
    upper = x.amax(dim=0)
    scale = upper - lower
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round((x - lower) * (n_bins - 1) / safe_scale).to(torch.int64)
    quantized[:, scale == 0] = 0
    return quantized


def _readout_kspace_energy(kspace, ky_idx, kz_idx, nex_idx):
    """Return sum-of-squares k-space energy for each chronological readout."""
    if kspace is None:
        raise ValueError("kspace must be provided for kspace_energy motion binning.")
    values = torch.as_tensor(kspace)
    if values.ndim == 4:  # [coil, Nex, kx, ky]
        energy = values.abs().square().sum(dim=(0, 2))
        if torch.any(nex_idx < 0) or torch.any(nex_idx >= energy.shape[0]):
            raise ValueError("nex_idx is outside the k-space repetition dimension.")
        if torch.any(ky_idx < 0) or torch.any(ky_idx >= energy.shape[1]):
            raise ValueError("ky_idx is outside the k-space phase dimension.")
        return energy[nex_idx.long(), ky_idx.long()]
    if values.ndim == 5:  # [coil, Nex, kx, ky, kz/slice]
        energy = values.abs().square().sum(dim=(0, 2))
        if kz_idx is None:
            if energy.shape[2] != 1:
                raise ValueError("kz_idx is required for multi-partition k-space.")
            kz_idx = torch.zeros_like(ky_idx)
        if torch.any(nex_idx < 0) or torch.any(nex_idx >= energy.shape[0]):
            raise ValueError("nex_idx is outside the k-space repetition dimension.")
        if torch.any(ky_idx < 0) or torch.any(ky_idx >= energy.shape[1]):
            raise ValueError("ky_idx is outside the k-space phase dimension.")
        if torch.any(kz_idx < 0) or torch.any(kz_idx >= energy.shape[2]):
            raise ValueError("kz_idx is outside the k-space partition dimension.")
        return energy[nex_idx.long(), ky_idx.long(), kz_idx.long()]
    raise ValueError(
        "kspace_energy motion binning expects k-space shaped "
        "[coil, Nex, kx, ky] or [coil, Nex, kx, ky, kz]."
    )


def _kspace_energy_binning(x, k, readout_energy, n_quantization_bins=256):
    """GRICS++ virtual-time quantization and k-space-energy state selection."""
    quantized = _quantize_motion_features(x, n_quantization_bins)
    virtual_states, inverse = torch.unique(quantized, dim=0, return_inverse=True)
    n_virtual = int(virtual_states.shape[0])
    if n_virtual == 1:
        raise ConstantPhysiologicalSignalError(
            "The physiological signal is constant after interpolation and quantization; "
            "motion-corrected reconstruction is skipped."
        )
    if k > n_virtual:
        raise ValueError(
            f"N_motion_states ({k}) exceeds the {n_virtual} quantized virtual times."
        )

    virtual_energy = torch.zeros(n_virtual, dtype=readout_energy.dtype, device=x.device)
    virtual_energy.scatter_add_(0, inverse, readout_energy.to(device=x.device))
    # Stable tie-breaking by virtual-time index makes the result deterministic.
    order = torch.argsort(virtual_energy, descending=True, stable=True)
    selected = order[:k]

    distances = torch.cdist(
        virtual_states.to(dtype=x.dtype),
        virtual_states[selected].to(dtype=x.dtype),
    )
    virtual_labels = distances.argmin(dim=1)
    labels = virtual_labels[inverse]

    # GRICS++ represents a cluster by its selected high-energy virtual time.
    centers = torch.empty((k, x.shape[1]), dtype=x.dtype, device=x.device)
    for state, virtual_idx in enumerate(selected):
        samples = inverse == virtual_idx
        centers[state] = x[samples].mean(dim=0)
    return labels, centers


class MotionBinner:
    @staticmethod
    def _flatten_index_tensor(values, name):
        if torch.is_tensor(values):
            return values.reshape(-1)
        if isinstance(values, list):
            return torch.cat([value.reshape(-1) for value in values], dim=0)
        raise TypeError(f"{name} must be a tensor or list of tensors, got {type(values)!r}.")

    @staticmethod
    def bin_motion(
        motion_curve,
        ky_idx,
        kz_idx,
        nex_idx,
        t_device,
        params,
        tx=None,
        ty=None,
        phi=None,
        tz=None,
        rx=None,
        ry=None,
        rz=None,
        y_limits=None,
        return_debug_data=False,
        kspace=None,
    ):
        motion_curve = motion_curve.to(t_device)

        Nbins = params.N_motion_states
        Nex = params.Nex
        if motion_curve.ndim != 2:
            raise ValueError("motion_curve must have shape [Nreadout, Nsensor].")
        motion_features = motion_curve
        num_motion_samples = int(motion_features.shape[0])

        if num_motion_samples < 1:
            raise ValueError("motion_curve must contain at least one sample.")
        if torch.all(motion_features.amax(dim=0) == motion_features.amin(dim=0)):
            raise ConstantPhysiologicalSignalError(
                "The physiological signal is constant after interpolation; "
                "motion-corrected reconstruction is skipped."
            )
        if Nbins > num_motion_samples:
            raise ValueError(
                f"N_motion_states ({Nbins}) cannot exceed the number of available motion samples "
                f"({num_motion_samples})."
            )

        ky_idx = MotionBinner._flatten_index_tensor(ky_idx, "ky_idx")
        kz_idx = None if kz_idx is None else MotionBinner._flatten_index_tensor(kz_idx, "kz_idx")
        nex_idx = MotionBinner._flatten_index_tensor(nex_idx, "nex_idx")

        binning_mode = str(getattr(params, "motion_binning_mode", "kmeans")).strip().lower()
        if binning_mode == "kmeans":
            labels, centers = _kmeans_torch(motion_features, Nbins)
        elif binning_mode == "kspace_energy":
            readout_energy = _readout_kspace_energy(
                kspace, ky_idx, kz_idx, nex_idx
            ).to(device=t_device, dtype=motion_features.dtype)
            labels, centers = _kspace_energy_binning(
                motion_features,
                Nbins,
                readout_energy,
                n_quantization_bins=int(
                    getattr(params, "motion_quantization_bins", 256)
                ),
            )
        else:
            raise ValueError(
                f"Unsupported motion_binning_mode: {binning_mode!r}. "
                "Supported modes are 'kmeans' and 'kspace_energy'."
            )

        # ---- Allocate output: [Nex][Nbins] ----
        binned_ky_indices = [
            [torch.empty(0, dtype=ky_idx.dtype, device=t_device) for _ in range(Nbins)]
            for _ in range(Nex)
        ]
        binned_kz_indices = None
        if kz_idx is not None:
            binned_kz_indices = [
                [torch.empty(0, dtype=kz_idx.dtype, device=t_device) for _ in range(Nbins)]
                for _ in range(Nex)
            ]

        # ---- Fill bins ----
        for nex in range(Nex):
            nex_mask = nex_idx == nex

            for b in range(Nbins):
                mask = nex_mask & (labels == b)
                binned_ky_indices[nex][b] = ky_idx[mask]
                if binned_kz_indices is not None:
                    binned_kz_indices[nex][b] = kz_idx[mask]

        # ---- Input data plots (always saved) ----
        save_clustered_motion_plots(
            motion_curve=motion_curve,
            labels=labels,
            ky_idx=ky_idx,
            nex_idx=nex_idx,
            kz_idx=kz_idx,
            nbins=Nbins,
            output_folder=params.initial_data_folder,
            resolution_levels=params.ResolutionLevels,
            tx=tx,
            ty=ty,
            phi=phi,
            tz=tz,
            rx=rx,
            ry=ry,
            rz=rz,
            data_type=params.data_type,
            y_limits=y_limits,
        )

        if return_debug_data:
            return (
                binned_ky_indices,
                binned_kz_indices,
                centers,
                labels,
                ky_idx,
                kz_idx,
                nex_idx,
            )
        return binned_ky_indices, binned_kz_indices, centers

    # Compatibility alias for integrations written before bin_motion was public.
    _bin_motion = bin_motion

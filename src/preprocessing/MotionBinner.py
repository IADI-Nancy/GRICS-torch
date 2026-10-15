import torch


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
    """Uniformly quantize each physiological channel into ``n_bins`` levels."""
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
    if values.ndim not in (4, 5):
        raise ValueError("kspace_energy motion binning expects k-space shaped "
                         "[coil, Nex, kx, ky] or [coil, Nex, kx, ky, kz].")
    if values.ndim == 4:  # [coil, Nex, kx, ky]
        values = values.unsqueeze(-1)
        kz_idx = None  # 2D input has no partition index, as in the original path.
    energy = values.abs().square().sum(dim=(0, 2))
    if kz_idx is None:
        if energy.shape[2] != 1:
            raise ValueError("kz_idx is required for multi-partition k-space.")
        kz_idx = torch.zeros_like(ky_idx)
    for indices, size, name, dimension in (
        (nex_idx, energy.shape[0], "nex_idx", "repetition"),
        (ky_idx, energy.shape[1], "ky_idx", "phase"),
        (kz_idx, energy.shape[2], "kz_idx", "partition"),
    ):
        if torch.any(indices < 0) or torch.any(indices >= size):
            raise ValueError(f"{name} is outside the k-space {dimension} dimension.")
    return energy[nex_idx.long(), ky_idx.long(), kz_idx.long()]


def _kspace_energy_binning(x, k, readout_energy, n_quantization_bins=256):
    """Quantize virtual times and select motion states by k-space energy."""
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

    # Represent each cluster by the mean samples at its selected high-energy
    # virtual time, preserving the physiological units of the input features.
    centers = torch.empty((k, x.shape[1]), dtype=x.dtype, device=x.device)
    for state, virtual_idx in enumerate(selected):
        samples = inverse == virtual_idx
        centers[state] = x[samples].mean(dim=0)
    return labels, centers


class MotionBinner:
    @staticmethod
    def assign_states(motion_curve, ky_idx, kz_idx, nex_idx, t_device, params, kspace=None):
        """Return labels and centers for readouts with already-flat index tensors."""

        motion_curve = motion_curve.to(t_device)
        Nbins = params.N_motion_states
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

        binning_mode = str(params.motion_binning_mode).strip().lower()
        if binning_mode == "kmeans":
            labels, centers = _kmeans_torch(motion_features, Nbins)
        elif binning_mode == "kspace_energy":
            readout_energy = _readout_kspace_energy(kspace, ky_idx, kz_idx, nex_idx).to(device=t_device, dtype=motion_features.dtype)
            labels, centers = _kspace_energy_binning(motion_features, Nbins, readout_energy,
                    n_quantization_bins=int(params.motion_quantization_bins))
        else:
            raise ValueError(
                f"Unsupported motion_binning_mode: {binning_mode!r}. "
                "Supported modes are 'kmeans' and 'kspace_energy'."
            )

        return labels, centers

import torch


class Sampling:
    """Build reconstruction sampling layouts from grouped readout indices."""

    @staticmethod
    def build_sampling_per_nex_per_motion(
        binned_ky_indices,
        device,
        Nx,
        Ny,
        Nz=1,
        binned_kz_indices=None,
    ):
        """Return flattened sampling indices grouped as ``[Nex][Nmotion]``."""
        nex_count = len(binned_ky_indices)
        if nex_count == 0:
            raise ValueError("binned_ky_indices cannot be empty.")

        motion_count = len(binned_ky_indices[0])
        if any(len(states) != motion_count for states in binned_ky_indices):
            raise ValueError(
                "Every excitation in binned_ky_indices must have the same "
                "number of motion states."
            )
        if Nz > 1:
            if binned_kz_indices is None:
                raise ValueError("binned_kz_indices is required when Nz > 1.")
            if len(binned_kz_indices) != nex_count or any(
                len(states) != motion_count for states in binned_kz_indices
            ):
                raise ValueError(
                    "binned_kz_indices must match the [Nex][Nmotion] layout "
                    "of binned_ky_indices."
                )

        kx = torch.arange(Nx, device=device, dtype=torch.int64)
        sampling = [[None for _ in range(motion_count)] for _ in range(nex_count)]

        for nex in range(nex_count):
            for motion_state in range(motion_count):
                ky = binned_ky_indices[nex][motion_state].to(
                    device=device, dtype=torch.int64
                ).reshape(-1)
                if ky.numel() == 0:
                    sampling[nex][motion_state] = torch.empty(
                        0, dtype=torch.int64, device=device
                    )
                    continue

                if Nz > 1:
                    kz = binned_kz_indices[nex][motion_state].to(
                        device=device, dtype=torch.int64
                    ).reshape(-1)
                    if ky.numel() != kz.numel():
                        raise ValueError(
                            "Paired ky and kz bins must contain the same number "
                            "of readouts."
                        )
                    indices = (
                        (ky[:, None] + Ny * kx[None, :]) * Nz + kz[:, None]
                    ).reshape(-1)
                else:
                    indices = (ky[:, None] + Ny * kx[None, :]).reshape(-1)

                sampling[nex][motion_state] = indices

        return sampling

import torch

from src.utils.plotting import _visualize_ky_order, _visualize_ky_kz_order

class SamplingSimulator:
    def __init__(self, Ny, params, t_device='cpu'):
        self.params = params
        self.Ny = Ny
        self.t_device = t_device

    @staticmethod
    def _visualize_ky_order(ky_per_shot, Ny, folder, fname="ky_sampling_order.png"):
        _visualize_ky_order(ky_per_shot, Ny, folder, fname)

    @staticmethod
    def _visualize_ky_kz_order(ky_per_block, kz_per_block, Ny, Nz, folder, fname="ky_kz_sampling_order.png"):
        _visualize_ky_kz_order(ky_per_block, kz_per_block, Ny, Nz, folder, fname)

    @staticmethod
    def _flatten_per_nex(values):
        if values is None:
            return None
        return torch.cat(values, dim=0)

    @staticmethod
    def _stack_phase_encode_pairs(
        ky_values,
        kz_values,
    ):
        ky_values = ky_values.to(torch.int64).reshape(-1)
        kz_values = kz_values.to(torch.int64).reshape(-1)
        if ky_values.numel() != kz_values.numel():
            raise ValueError("ky and kz values must have the same number of samples.")
        return torch.stack([ky_values, kz_values], dim=1)

    @staticmethod
    def _build_ordered_ky_values(
        nex,
        nshots,
        Ny,
        device,
        kspace_sampling_type,
    ):
        if kspace_sampling_type in {"linear", "from-data"}:
            return torch.arange(Ny, device=device, dtype=torch.int64)

        if kspace_sampling_type == "interleaved":
            ky_parts = [
                torch.arange(shot, Ny, nshots, device=device, dtype=torch.int64)
                for shot in range(nshots)
            ]
            return torch.cat(ky_parts, dim=0)

        if kspace_sampling_type == "random":
            return torch.randperm(Ny, device=device, dtype=torch.int64)

        raise ValueError(f"Unsupported kspace_sampling_type for 2D sampling: {kspace_sampling_type}")

    @staticmethod
    def _build_ordered_ky_kz_pairs(
        nex,
        shot,
        nshots,
        Ny,
        Nz,
        device,
        kspace_sampling_type,
    ):
        kz_values = torch.arange(Nz, device=device, dtype=torch.int64)

        if kspace_sampling_type in {"linear", "from-data"}:
            start = shot * Ny // nshots
            end = (shot + 1) * Ny // nshots
            ky_values = torch.arange(start, end, device=device, dtype=torch.int64)
            return torch.cartesian_prod(ky_values, kz_values)

        if kspace_sampling_type == "interleaved":
            ky_values = torch.arange(shot, Ny, nshots, device=device, dtype=torch.int64)
            n_ky = int(ky_values.numel())
            stride_y = max(1, min(nshots, n_ky))
            offset_y0 = (nex + shot) % stride_y
            ky_parts = []
            for t in range(stride_y):
                offset = (offset_y0 + t) % stride_y
                ky_parts.append(ky_values[offset::stride_y])
            ky_values = torch.cat(ky_parts, dim=0)

            stride_z = max(1, min(nshots, Nz))
            offset_z0 = (nex + shot) % stride_z
            kz_parts = []
            for t in range(stride_z):
                offset = (offset_z0 + t) % stride_z
                kz_parts.append(
                    torch.arange(offset, Nz, stride_z, device=device, dtype=torch.int64)
                )
            kz_values = torch.cat(kz_parts, dim=0)

            pair_blocks = []
            for i_kz, kz_val in enumerate(kz_values):
                ky_seq = ky_values if (i_kz % 2 == 0) else torch.flip(ky_values, dims=[0])
                kz_seq = torch.full_like(ky_seq, kz_val)
                pair_blocks.append(torch.stack([ky_seq, kz_seq], dim=1))
            return torch.cat(pair_blocks, dim=0)

        raise ValueError(f"Unsupported kspace_sampling_type for 3D sampling: {kspace_sampling_type}")

    def _build_phase_encode_indices_and_nex(self, Nz=1):
        Nshots = self.params.NshotsPerNex
        Nex = self.params.Nex

        # ky_per_shot[nex][shot]
        ky_per_shot = [[] for _ in range(Nex)]
        kz_per_shot = [[] for _ in range(Nex)] if Nz > 1 else None

        ky_idx = []   # motion-state / shot-wise flattened ky indices
        nex_idx = []   # motion-state / shot-wise flattened nex indices
        kz_idx = [] if Nz > 1 else None

        for nex in range(Nex):
            ky_list = []   # chronological chunks (per Nex)
            nex_list = []
            kz_list = [] if Nz > 1 else None

            use_global_random_3d = Nz > 1 and self.params.kspace_sampling_type == "random"

            if use_global_random_3d:
                all_pairs = torch.cartesian_prod(
                    torch.arange(self.Ny, device=self.t_device, dtype=torch.int64),
                    torch.arange(Nz, device=self.t_device, dtype=torch.int64),
                )
                perm = torch.randperm(all_pairs.shape[0], device=self.t_device, dtype=torch.int64)
                all_pairs = all_pairs[perm]
                split_sizes = [
                    (all_pairs.shape[0] // Nshots) + (1 if s < all_pairs.shape[0] % Nshots else 0)
                    for s in range(Nshots)
                ]
                start = 0
            else:
                ky_all = self._build_ordered_ky_values(
                    nex, Nshots, self.Ny, self.t_device, self.params.kspace_sampling_type,
                ).to(torch.int32)
                split_sizes = [
                    (self.Ny // Nshots) + (1 if s < self.Ny % Nshots else 0)
                    for s in range(Nshots)
                ]
                start = 0

            for shot in range(Nshots):
                if Nz > 1:
                    if use_global_random_3d:
                        end = start + split_sizes[shot]
                        phase_encode_pairs = all_pairs[start:end]
                        start = end
                    else:
                        phase_encode_pairs = self._build_ordered_ky_kz_pairs(
                            nex, shot, Nshots, self.Ny, Nz, self.t_device, self.params.kspace_sampling_type,
                        )
                    ky_block = phase_encode_pairs[:, 0].to(torch.int32)
                    kz_block = phase_encode_pairs[:, 1].to(torch.int32)

                    ky_per_shot[nex].append(ky_block)
                    kz_per_shot[nex].append(kz_block)
                    ky_list.append(ky_block)
                    kz_list.append(kz_block)
                    nex_list.append(torch.full_like(ky_block, nex, dtype=torch.int32))
                else:
                    end = start + split_sizes[shot]
                    ky_values = ky_all[start:end]
                    start = end  # advance the pointer

                    ky_per_shot[nex].append(ky_values)
                    ky_list.append(ky_values)
                    nex_list.append(torch.full_like(ky_values, nex, dtype=torch.int32))

            ky_idx.append(torch.cat(ky_list, dim=0))
            nex_idx.append(torch.cat(nex_list, dim=0))
            if Nz > 1:
                kz_idx.append(torch.cat(kz_list, dim=0))
        
            if Nz > 1:
                SamplingSimulator._visualize_ky_kz_order(ky_per_shot[nex], kz_per_shot[nex], Ny=self.Ny,
                                                         Nz=Nz, folder=self.params.initial_data_folder,
                                                         fname=f"ky_kz_order_nex{nex+1}.png")
            else:
                SamplingSimulator._visualize_ky_order(ky_per_shot[nex], Ny=self.Ny,
                                                      folder=self.params.initial_data_folder,
                                                      fname=f"ky_order_nex{nex+1}.png")

        return (
            self._flatten_per_nex(ky_idx),
            self._flatten_per_nex(nex_idx),
            ky_per_shot,
            self._flatten_per_nex(kz_idx),
            kz_per_shot,
        )
    
    @staticmethod
    def _build_sampling_per_nex_per_motion(
        binned_ky_indices,  # [Nex][Nmotion]
        device,
        Nx, Ny,
        Nz=1,
        binned_kz_indices=None,  # [Nex][Nmotion] (optional, for 3D realworld)
    ):
        Nex = len(binned_ky_indices)
        Nmotion = len(binned_ky_indices[0])
        kx = torch.arange(Nx, device=device, dtype=torch.int64)
        Sampling = [[None for _ in range(Nmotion)] for _ in range(Nex)]

        for nex in range(Nex):
            for ms in range(Nmotion):
                ky = binned_ky_indices[nex][ms]

                if ky.numel() == 0:
                    Sampling[nex][ms] = torch.empty(0, dtype=torch.int64, device=device)
                    continue

                if Nz > 1:
                    if binned_kz_indices is None:
                        raise ValueError("binned_kz_indices is required when Nz > 1.")
                    phase_encode_pairs = SamplingSimulator._stack_phase_encode_pairs(ky, binned_kz_indices[nex][ms])
                    ky_pairs = phase_encode_pairs[:, 0]
                    kz_pairs = phase_encode_pairs[:, 1]
                    samp = ((ky_pairs[:, None] + Ny * kx[None, :]) * Nz + kz_pairs[:, None]).reshape(-1)
                else:
                    # Build flattened (kx, ky) sampling indices.
                    ky = ky.to(torch.int64)
                    samp_xy = ky[:, None] + Ny * kx[None, :]
                    samp = samp_xy.reshape(-1)

                Sampling[nex][ms] = samp

        return Sampling

    

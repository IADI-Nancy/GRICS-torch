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
    def _build_ky_kz_pairs(
        ky,
        nex,
        motion_state_idx,
        nmotion,
        Nz,
        device,
        kspace_sampling_type,
        kz_override=None,
    ):
        ky = ky.to(torch.int64).reshape(-1)

        if kz_override is not None:
            kz = kz_override.to(torch.int64).reshape(-1)
            if ky.numel() != kz.numel():
                raise ValueError("ky and kz overrides must have the same number of samples.")
            return ky, kz

        kz_ord = torch.arange(Nz, device=device, dtype=torch.int64)

        if kspace_sampling_type == "random":
            pairs = torch.cartesian_prod(ky, kz_ord)
            perm = torch.randperm(pairs.shape[0], device=device, dtype=torch.int64)
            pairs = pairs[perm]
        elif kspace_sampling_type == "interleaved":
            n_ky = int(ky.numel())
            stride_y = max(1, min(nmotion, n_ky))
            offset_y0 = (nex + motion_state_idx) % stride_y
            ky_parts = []
            for t in range(stride_y):
                offset = (offset_y0 + t) % stride_y
                ky_parts.append(ky[offset::stride_y])
            ky_ord = torch.cat(ky_parts, dim=0)

            stride_z = max(1, min(nmotion, Nz))
            offset_z0 = (nex + motion_state_idx) % stride_z
            kz_parts = []
            for t in range(stride_z):
                offset = (offset_z0 + t) % stride_z
                kz_parts.append(
                    torch.arange(offset, Nz, stride_z, device=device, dtype=torch.int64)
                )
            kz_ord = torch.cat(kz_parts, dim=0)

            pair_blocks = []
            for i_kz, kz_val in enumerate(kz_ord):
                ky_seq = ky_ord if (i_kz % 2 == 0) else torch.flip(ky_ord, dims=[0])
                kz_seq = torch.full_like(ky_seq, kz_val)
                pair_blocks.append(torch.stack([ky_seq, kz_seq], dim=1))
            pairs = torch.cat(pair_blocks, dim=0)
        else:
            pairs = torch.cartesian_prod(ky, kz_ord)

        return pairs[:, 0], pairs[:, 1]

    def _build_ky_and_nex(self, Nz=1):
        Nshots = self.params.NshotsPerNex
        Nex    = self.params.Nex

        # ky_per_shot[nex][shot]
        ky_per_shot = [[] for _ in range(Nex)]
        kz_per_shot = [[] for _ in range(Nex)] if Nz > 1 else None

        ky_idx  = []   # motion-state / shot-wise flattened ky indices
        nex_idx = []   # motion-state / shot-wise flattened nex indices
        kz_idx  = [] if Nz > 1 else None

        for nex in range(Nex):
            ky_list  = []   # chronological chunks (per Nex)
            nex_list = []
            kz_list = [] if Nz > 1 else None

            use_global_random_3d = self.params.kspace_sampling_type == "random" and Nz > 1

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
            elif self.params.kspace_sampling_type == 'random':
                # Independent random ky ordering for each Nex
                ky_all = torch.randperm(self.Ny, device=self.t_device, dtype=torch.int32)
                split_sizes = [
                    (self.Ny // Nshots) + (1 if s < self.Ny % Nshots else 0)
                    for s in range(Nshots)
                ]
                start = 0

            for shot in range(Nshots):
                shot_in_nex = shot

                # ----- ky selection -----
                if use_global_random_3d:
                    end = start + split_sizes[shot]
                    pairs = all_pairs[start:end]
                    start = end
                    ky_block = pairs[:, 0].to(torch.int32)
                    kz_block = pairs[:, 1].to(torch.int32)
                elif self.params.kspace_sampling_type in {'linear', 'from-data'}:
                    start = shot_in_nex * self.Ny // Nshots
                    end   = (shot_in_nex + 1) * self.Ny // Nshots
                    ky = torch.arange(
                        start, end,
                        device=self.t_device,
                        dtype=torch.int32
                    )

                elif self.params.kspace_sampling_type == 'interleaved':
                    ky = torch.arange(
                        shot_in_nex, self.Ny, Nshots,
                        device=self.t_device,
                        dtype=torch.int32
                    )
                else:
                    end = start + split_sizes[shot]
                    ky = ky_all[start:end]
                    start = end  # advance the pointer

                if Nz > 1:
                    if not use_global_random_3d:
                        ky_block, kz_block = self._build_ky_kz_pairs(
                            ky,
                            nex,
                            shot,
                            Nshots,
                            Nz,
                            self.t_device,
                            self.params.kspace_sampling_type,
                        )
                        ky_block = ky_block.to(torch.int32)
                        kz_block = kz_block.to(torch.int32)
                    ky_per_shot[nex].append(ky_block)
                    kz_per_shot[nex].append(kz_block)
                    ky_list.append(ky_block)
                    kz_list.append(kz_block)
                    nex_list.append(torch.full_like(ky_block, nex, dtype=torch.int32))
                else:
                    # ---- shot-wise storage (nested by Nex) ----
                    ky_per_shot[nex].append(ky)

                    # ---- chronological storage ----
                    ky_list.append(ky)
                    nex_list.append(
                        torch.full_like(ky, nex, dtype=torch.int32)
                    )

            ky_idx.append(torch.cat(ky_list, dim=0))
            nex_idx.append(torch.cat(nex_list, dim=0))
            if Nz > 1:
                kz_idx.append(torch.cat(kz_list, dim=0))
        
            if Nz > 1:
                SamplingSimulator._visualize_ky_kz_order(
                    ky_per_shot[nex],
                    kz_per_shot[nex],
                    Ny=self.Ny,
                    Nz=Nz,
                    folder=self.params.initial_data_folder,
                    fname=f"ky_kz_order_nex{nex+1}.png"
                )
            else:
                SamplingSimulator._visualize_ky_order(
                    ky_per_shot[nex],
                    Ny=self.Ny,
                    folder=self.params.initial_data_folder,
                    fname=f"ky_order_nex{nex+1}.png"
                )

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
        kspace_sampling_type="linear",
        binned_kz_indices=None,  # [Nex][Nmotion] (optional, for 3D realworld)
    ):
        Nex = len(binned_ky_indices)
        Nmotion = len(binned_ky_indices[0])

        kx = torch.arange(Nx, device=device, dtype=torch.int64)

        Sampling = [
            [None for _ in range(Nmotion)]
            for _ in range(Nex)
        ]

        for nex in range(Nex):
            for ms in range(Nmotion):
                ky = binned_ky_indices[nex][ms]

                if ky.numel() == 0:
                    Sampling[nex][ms] = torch.empty(
                        0, dtype=torch.int64, device=device
                    )
                    continue

                # Build flattened (kx, ky[, kz]) sampling indices.
                ky = ky.to(torch.int64)
                samp_xy = (
                    ky[:, None]
                    + Ny * kx[None, :]
                )

                if Nz > 1:
                    # When actual (ky, kz) pairs are provided (realworld 3D),
                    # use them directly — no cartesian product needed.
                    kz_override = None if binned_kz_indices is None else binned_kz_indices[nex][ms]
                    ky_pairs, kz_pairs = SamplingSimulator._build_ky_kz_pairs(
                        ky,
                        nex,
                        ms,
                        Nmotion,
                        Nz,
                        device,
                        kspace_sampling_type,
                        kz_override=kz_override,
                    )
                    samp = (
                        (ky_pairs[:, None] + Ny * kx[None, :]) * Nz
                        + kz_pairs[:, None]
                    ).reshape(-1)
                else:
                    samp = samp_xy.reshape(-1)

                Sampling[nex][ms] = samp

        return Sampling

    

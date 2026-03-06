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

    def _build_ky_and_nex(self):
        Nshots = self.params.NshotsPerNex
        Nex    = self.params.Nex

        # ky_per_shot[nex][shot]
        ky_per_shot = [[] for _ in range(Nex)]

        ky_idx  = []   # motion-state / shot-wise flattened ky indices
        nex_idx = []   # motion-state / shot-wise flattened nex indices

        for nex in range(Nex):
            ky_list  = []   # chronological chunks (per Nex)
            nex_list = []

            if self.params.kspace_sampling_type == 'random':
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
                if self.params.kspace_sampling_type in {'linear', 'from-data'}:
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
                elif self.params.kspace_sampling_type == 'random':
                    end = start + split_sizes[shot]
                    ky = ky_all[start:end]
                    start = end  # advance the pointer
                else:
                    raise ValueError("Unknown kspace_sampling_type")

                # ---- shot-wise storage (nested by Nex) ----
                ky_per_shot[nex].append(ky)

                # ---- chronological storage ----
                ky_list.append(ky)
                nex_list.append(
                    torch.full_like(ky, nex, dtype=torch.int32)
                )

            ky_idx.append(torch.cat(ky_list, dim=0))
            nex_idx.append(torch.cat(nex_list, dim=0))
        
            SamplingSimulator._visualize_ky_order(
                ky_per_shot[nex],
                Ny=self.Ny,
                folder=self.params.input_data_folder,
                fname=f"ky_order_nex{nex+1}.png"
            )

        return ky_idx, nex_idx, ky_per_shot
    
    @staticmethod
    def _build_sampling_per_nex_per_motion(
        binned_ky_indices,  # [Nex][Nmotion]
        device,
        Nx, Ny,
        Nz=1,
        kspace_sampling_type="linear",
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
                    # Build (ky, kz) acquisition order, then expand over kx.
                    if kspace_sampling_type == "random":
                        kz_ord = torch.arange(Nz, device=device, dtype=torch.int64)
                        pairs = torch.cartesian_prod(ky, kz_ord)
                        perm = torch.randperm(pairs.shape[0], device=device, dtype=torch.int64)
                        pairs = pairs[perm]
                    elif kspace_sampling_type == "interleaved":
                        n_ky = int(ky.numel())
                        stride_y = max(1, min(Nmotion, n_ky))
                        offset_y0 = (nex + ms) % stride_y
                        ky_parts = []
                        for t in range(stride_y):
                            offset = (offset_y0 + t) % stride_y
                            ky_parts.append(ky[offset::stride_y])
                        ky_ord = torch.cat(ky_parts, dim=0)

                        stride_z = max(1, min(Nmotion, Nz))
                        offset_z0 = (nex + ms) % stride_z
                        kz_parts = []
                        for t in range(stride_z):
                            offset = (offset_z0 + t) % stride_z
                            kz_parts.append(
                                torch.arange(offset, Nz, stride_z, device=device, dtype=torch.int64)
                            )
                        kz_ord = torch.cat(kz_parts, dim=0)

                        # Serpentine traversal to interleave both ky and kz directions.
                        pair_blocks = []
                        for i_kz, kz_val in enumerate(kz_ord):
                            ky_seq = ky_ord if (i_kz % 2 == 0) else torch.flip(ky_ord, dims=[0])
                            kz_seq = torch.full_like(ky_seq, kz_val)
                            pair_blocks.append(torch.stack([ky_seq, kz_seq], dim=1))
                        pairs = torch.cat(pair_blocks, dim=0)
                    elif kspace_sampling_type in {"linear", "from-data"}:
                        kz_ord = torch.arange(Nz, device=device, dtype=torch.int64)
                        pairs = torch.cartesian_prod(ky, kz_ord)
                    else:
                        raise ValueError(
                            f"Unknown kspace_sampling_type: {kspace_sampling_type}"
                        )

                    ky_pairs = pairs[:, 0]
                    kz_pairs = pairs[:, 1]
                    samp = (
                        (ky_pairs[:, None] + Ny * kx[None, :]) * Nz
                        + kz_pairs[:, None]
                    ).reshape(-1)
                else:
                    samp = samp_xy.reshape(-1)

                Sampling[nex][ms] = samp

        return Sampling

    

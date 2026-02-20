import torch

from src.utils.visualize_ky_order import visualize_ky_order

class SamplingSimulator:
    def __init__(self, Ny, params, t_device='cpu'):
        self.params = params
        self.Ny = Ny
        self.t_device = t_device

    @staticmethod
    def visualize_ky_order(ky_per_shot, Ny, folder, fname="ky_sampling_order.png"):
        visualize_ky_order(ky_per_shot, Ny, folder, fname)

    def build_ky_and_nex(self):
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
                if self.params.kspace_sampling_type == 'linear':
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
        
            SamplingSimulator.visualize_ky_order(
                ky_per_shot[nex],
                Ny=self.Ny,
                folder=getattr(self.params, "input_data_folder", self.params.debug_folder),
                fname=f"ky_order_nex{nex+1}.png"
            )

        return ky_idx, nex_idx, ky_per_shot
    
    @staticmethod
    def build_sampling_per_nex_per_motion(
        binned_ky_indices,  # [Nex][Nmotion]
        Nx, Ny,
        device,
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

                # Build flattened (kx, ky) sampling
                samp = (
                    ky[:, None]
                    + Ny * kx[None, :]
                ).reshape(-1)

                Sampling[nex][ms] = samp

        return Sampling

    

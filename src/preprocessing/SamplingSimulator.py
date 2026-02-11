import torch
from Parameters import Parameters
params = Parameters()

class SamplingSimulator:
    def __init__(self, Ny, t_device='cpu'):
        self.params = params
        self.Ny = Ny
        self.t_device = t_device

    def build_ky_and_nex(self):
        Nshots = params.NshotsPerNex
        Nex    = params.Nex

        # ky_per_shot[nex][shot]
        ky_per_shot = [[] for _ in range(Nex)]

        ky_idx  = []   # motion-state / shot-wise flattened ky indices
        nex_idx = []   # motion-state / shot-wise flattened nex indices

        for nex in range(Nex):
            ky_list  = []   # chronological chunks (per Nex)
            nex_list = []

            for shot in range(Nshots):
                shot_in_nex = shot

                # ----- ky selection -----
                if params.kspace_sampling_type == 'linear':
                    start = shot_in_nex * self.Ny // Nshots
                    end   = (shot_in_nex + 1) * self.Ny // Nshots
                    ky = torch.arange(
                        start, end,
                        device=self.t_device,
                        dtype=torch.int32
                    )

                elif params.kspace_sampling_type == 'interleaved':
                    ky = torch.arange(
                        shot_in_nex, self.Ny, Nshots,
                        device=self.t_device,
                        dtype=torch.int32
                    )
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

    
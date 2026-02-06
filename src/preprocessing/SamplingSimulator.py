import torch

class SamplingSimulator:
    def __init__(self, params, Ny, t_device='cpu'):
        self.params = params
        self.Ny = Ny
        self.t_device = t_device

    def build_ky_and_nex(self):
        params = self.params
        Nshots = params.NshotsPerNex
        Nex    = params.Nex

        
        ky_per_shot  = []   # motion-state / shot-wise ky
        ky_idx       = []   # motion-state / shot-wise flattened ky indices
        nex_idx      = []   # motion-state / shot-wise flattened nex indices

        for nex in range(params.Nex):
            ky_list      = []   # chronological chunks
            nex_list     = []
            for shot in range(Nshots):
                # TODO: multi-Nex support later
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

                # shot-wise storage (motion state)
                ky_per_shot.append(ky)

                # chronological storage
                ky_list.append(ky)
                nex_list.append(
                    torch.full_like(ky, nex, dtype=torch.int32)
                )

            ky_idx.append(torch.cat(ky_list, dim=0))
            nex_idx.append(torch.cat(nex_list, dim=0))

        return ky_idx, nex_idx, ky_per_shot
    
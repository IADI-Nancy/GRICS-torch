import torch
import matplotlib.pyplot as plt
import matplotlib
import os

from Parameters import Parameters
params = Parameters()

class SamplingSimulator:
    def __init__(self, Ny, t_device='cpu'):
        self.params = params
        self.Ny = Ny
        self.t_device = t_device

    @staticmethod
    def visualize_ky_order(ky_per_shot, Ny, folder, fname="ky_sampling_order.png"):
        """
        Visualize ky acquisition order as a Ny x Ny square image.
        - Each line colored by acquisition order (viridis colormap)
        - Continuous, no separators
        - Adds a colorbar
        """
        # Create empty image
        img = torch.zeros((Ny, Ny, 3), dtype=torch.float64)  # RGB

        # Concatenate all shot tensors for acquisition order
        all_ky = torch.cat(ky_per_shot)

        # Map ky line -> acquisition order (0..1)
        order_map = torch.zeros(Ny, dtype=torch.float64)
        order_map[all_ky] = torch.linspace(0, 1, len(all_ky))

        # Use viridis colormap
        cmap = plt.get_cmap("viridis")

        # Fill image line by line
        for ky in range(Ny):
            img[ky, :, :] = torch.tensor(cmap(order_map[ky].item())[:3], dtype=torch.float64)

        # Save image
        os.makedirs(folder, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(img.numpy())
        ax.axis("off")
        ax.set_title("Ky Acquisition Order")

        # Add colorbar for acquisition order
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(all_ky)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Acquisition order (first → last)")

        plt.tight_layout()
        plt.savefig(os.path.join(folder, fname))
        plt.close(fig)

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

            if params.kspace_sampling_type == 'random':
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
                elif params.kspace_sampling_type == 'random':
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
        
            if params.debug_flag:
                SamplingSimulator.visualize_ky_order(
                    ky_per_shot[nex],
                    Ny=self.Ny,
                    folder=params.debug_folder,
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

    

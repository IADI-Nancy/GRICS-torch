import numpy as np
import torch
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions
import matplotlib.pyplot as plt
from src.utils.Helpers import build_sampling_from_motion_states


class RigidMotionSimulatorShots:
    def __init__(self, image, smaps, params, sp_device=None, t_device=None):
        self.image = image
        self.smaps = smaps
        self.sp_device = sp_device
        self.t_device = t_device
        self.Nx, self.Ny, self.Nsli, self.Ncha = smaps.shape
        self.create_motion_corrupted_dataset(params)        

    # def create_motion_curves(self, params):
    def get_simulated_sampling(self):
        return self.ky_idx, self.nex_idx, self.TotalKspaceSamples
    
    def get_corrupted_kspace(self):
        return self.kspace
    
    def get_corrupted_image(self):
        return self.image_no_moco

    def get_motion_information(self):
        return self.navigator, self.tx, self.ty, self.phi
    
    def save_debug_plots(self, motion_curve, tx, ty, phi):
        plt.figure()
        plt.plot(motion_curve.cpu().numpy())
        plt.title("Motion Curve")
        plt.savefig("debug_outputs/motion_curve.png")
        plt.close()

        plt.figure()
        plt.plot(tx.cpu().numpy())
        plt.title("tx curve")
        plt.savefig("debug_outputs/tx_curve.png")
        plt.close()

        plt.figure()
        plt.plot(ty.cpu().numpy())
        plt.title("ty curve")
        plt.savefig("debug_outputs/ty_curve.png")
        plt.close()

        plt.figure()
        plt.plot(phi.cpu().numpy())
        plt.title("phi curve")
        plt.savefig("debug_outputs/phi_curve.png")
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(motion_curve.cpu().numpy(), label="PC1 Motion Curve", linewidth=2)
        plt.plot(tx.cpu().numpy(), label="tx", alpha=0.8)
        plt.plot(ty.cpu().numpy(), label="ty", alpha=0.8)
        plt.plot(phi.cpu().numpy(), label="phi", alpha=0.8)

        plt.title("All motion curves (superimposed)")
        plt.xlabel("Acquisition line number")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig("debug_outputs/all_curves.png")
        plt.close()

    def expand_motion_to_ky(self, ky_per_mot_state_idx):
        """
        Expand per-motion-state parameters into per-ky vectors
        in chronological acquisition order.
        """

        # Total number of ky lines
        Ny_total = sum(ky.numel() for ky in ky_per_mot_state_idx)

        assert Ny_total == self.Ny, "Total ky count does not match Ny"

        # Allocate
        self.tx        = torch.empty(self.Ny, device=self.t_device)
        self.ty        = torch.empty(self.Ny, device=self.t_device)
        self.phi       = torch.empty(self.Ny, device=self.t_device)
        self.navigator = torch.empty(self.Ny, device=self.t_device)

        ptr = 0  # write pointer (chronological)

        for m, ky in enumerate(ky_per_mot_state_idx):
            n = ky.numel()

            self.tx[ptr:ptr+n]        = self.tx_mot_state[m]
            self.ty[ptr:ptr+n]        = self.ty_mot_state[m]
            self.phi[ptr:ptr+n]       = self.phi_mot_state[m]
            self.navigator[ptr:ptr+n] = self.navigator_mot_state[m]

            ptr += n

    def create_motion_curves(self, ky_per_mot_state_idx, params):
        Nshots = params.N_mot_states

        alpha = torch.zeros((3, Nshots), device=self.t_device)

        # Translation X: [-max_tx, +max_tx]
        self.tx_mot_state = params.max_tx * (2 * torch.rand(Nshots, device=self.t_device) - 1)
        alpha[0, :] = self.tx_mot_state

        # Translation Y: [-max_ty, +max_ty]
        self.ty_mot_state = params.max_ty * (2 * torch.rand(Nshots, device=self.t_device) - 1)
        alpha[1, :] = self.ty_mot_state

        # Rotation: [-max_phi, +max_phi] degrees → radians
        self.phi_mot_state = params.max_phi * (2 * torch.rand(Nshots, device=self.t_device) - 1) * (torch.pi / 180)
        alpha[2, :] = self.phi_mot_state

        # Centers
        centers = torch.zeros((2, Nshots), device=self.t_device)
        centers[0, :] = self.Nx / 2 + 60.0
        centers[1, :] = self.Ny / 2 + 10.0 * (2 * torch.rand(Nshots, device=self.t_device) - 1)

        # -------------------------------------------------
        # Build navigator = first principal component
        # -------------------------------------------------
        # Stack motion parameters: (Nshots, 3)
        motion_mat = torch.stack([self.tx_mot_state, self.ty_mot_state, self.phi_mot_state], dim=1)  # (Nshots, 3)

        # Center the data
        motion_mat = motion_mat - motion_mat.mean(dim=0, keepdim=True)

        # PCA via SVD
        # motion_mat = U S V^T
        _, _, Vh = torch.linalg.svd(motion_mat, full_matrices=False)

        # First principal component direction
        pc1 = Vh[0]  # (3,)

        # Project motion onto PC1 → navigator
        self.navigator_mot_state = motion_mat @ pc1  # (Nshots,)

        self.expand_motion_to_ky(ky_per_mot_state_idx)
        # save debug plots
        if params.debug_flag:
            self.save_debug_plots(self.navigator, self.tx, self.ty, self.phi)

        return self.navigator, alpha, centers

    def create_motion_corrupted_dataset(self, params):
        self.ky_idx, self.nex_idx, _, ky_per_mot_state_idx = self.build_ky_nex_and_motion_states(params)
        self.sampling_idx, self.nex_offset, self.TotalKspaceSamples = \
            build_sampling_from_motion_states(ky_per_mot_state_idx, self.ky_idx, self.nex_idx, self.Nx, self.Ny, self.t_device)

        self.navigator, alpha, centers = self.create_motion_curves(ky_per_mot_state_idx, params)

        self.MotionOperator = MotionOperator(self.Nx, self.Ny, alpha, centers)

        E = EncodingOperator(
            self.smaps,
            self.TotalKspaceSamples,
            self.sampling_idx,
            self.nex_offset,
            self.MotionOperator
        )
        kspace_corruped = E.forward(self.image)
        self.kspace = kspace_corruped.reshape(params.Nex, self.Nx, self.Ny, self.Nsli, self.Ncha)

        img_cplx = ifftnc(self.kspace[0,:,:,:,:], dims=(0, 1, 2)).to(self.t_device)
        self.image_no_moco = torch.sum(img_cplx * self.smaps.conj(), dim=-1)

    def build_ky_nex_and_motion_states(self, params):
        Nshots = params.N_mot_states
        Nex    = params.Nex

        ky_per_mot_state = []   # list of tensors
        ky_list          = []   # chronological ky
        mot_state_list   = []   # motion state per ky
        nex_list         = []   # nex per ky

        for shot in range(Nshots):
            # TODO: multi-Nex support later
            shot_in_nex = shot
            Nex_idx     = 0

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

            # bookkeeping per motion state
            ky_per_mot_state.append(ky)

            # chronological accumulation
            ky_list.append(ky)
            mot_state_list.append(
                torch.full_like(ky, shot, dtype=torch.int32)
            )
            nex_list.append(
                torch.full_like(ky, Nex_idx, dtype=torch.int32)
            )

        # -------------------------------------------------
        # Concatenate everything chronologically
        # -------------------------------------------------
        ky_idx        = torch.cat(ky_list, dim=0)
        mot_state_idx = torch.cat(mot_state_list, dim=0)
        nex_idx       = torch.cat(nex_list, dim=0)

        return ky_idx, nex_idx, mot_state_idx, ky_per_mot_state



    








        
        
        

    
import numpy as np
import torch
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions
import matplotlib.pyplot as plt
from src.utils.Helpers import build_sampling_from_motion_states


class RigidMotionSimulator:
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
    
    def create_motion_curves(self, params):
        # Time axis: one value per k-space line (Ny)
        t = torch.arange(self.Ny, device=self.t_device, dtype=torch.float32)

        # -------------------------------
        # 1) Generate random motion event times
        # -------------------------------
        # Motion events are random line indices where motion happens
        # Sorted to ensure chronological order
        event_times = torch.sort(
            torch.randint(0, self.Ny, (params.num_motion_events,), device=self.t_device)
        ).values

        # Motion transition sharpness (smaller tau -> faster motion)
        tau = params.motion_tau

        # -------------------------------
        # 2) Generate random amplitudes for each event
        # -------------------------------
        # Each parameter has its own random amplitude per event
        # Range: [-max, max]
        A_tx  = params.max_tx  * (2 * torch.rand(params.num_motion_events, device=self.t_device) - 1)
        A_ty  = params.max_ty  * (2 * torch.rand(params.num_motion_events, device=self.t_device) - 1)
        A_phi = params.max_phi * (2 * torch.rand(params.num_motion_events, device=self.t_device) - 1) * (torch.pi/180)

        # -------------------------------
        # 3) Build independent motion curves
        # -------------------------------
        # Start from zero and add one smooth step per event
        tx  = torch.zeros(self.Ny, device=self.t_device)
        ty  = torch.zeros(self.Ny, device=self.t_device)
        phi = torch.zeros(self.Ny, device=self.t_device)

        for i, ti in enumerate(event_times):
            # Smooth step function: tanh((t-ti)/tau)
            # This creates a rapid but smooth transition at each event time
            tx  += A_tx[i]  * torch.tanh((t - ti) / tau)
            ty  += A_ty[i]  * torch.tanh((t - ti) / tau)
            phi += A_phi[i] * torch.tanh((t - ti) / tau)

        # -------------------------------
        # 4) Compute PCA to obtain a single 1D motion curve (simulation of MRI navigators)
        # -------------------------------
        # Stack the three curves into a matrix (3 x Ny)
        M = torch.stack([tx, ty, phi], dim=0)

        # Mean-center each row (parameter) before PCA
        M_centered = M - M.mean(dim=1, keepdim=True)

        # Perform SVD (PCA)
        # U: 3x3, S: singular values, Vh: NyxNy
        U, S, Vh = torch.linalg.svd(M_centered, full_matrices=False)

        # First principal component direction (in parameter space)
        u1 = U[:, 0]

        # Project the centered data onto the first PC
        # score1 shape: (1 x Ny)
        score1 = torch.matmul(u1.unsqueeze(0), M_centered)

        # Convert to 1D curve
        navigator = score1.squeeze(0)

        # Normalize navigator to [-1, 1] for stability
        navigator = navigator / navigator.abs().max()

        # save debug plots
        if params.debug_flag:
            self.save_debug_plots(navigator, tx, ty, phi, event_times)
        # Return the motion curve, parameter curves, and event times
        return navigator, tx, ty, phi


    def save_debug_plots(self, motion_curve, tx, ty, phi, event_times):
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

         # vertical lines at motion events
        for e in event_times.cpu().numpy():
            plt.axvline(x=e, color="black", linewidth=1)

        plt.title("All motion curves (superimposed)")
        plt.xlabel("Acquisition line number")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig("debug_outputs/all_curves.png")
        plt.close()

   

    def create_motion_corrupted_dataset(self, params):
        # self.Nshots = Nshots = params.NshotsPerNex * params.Nex
        self.ky_idx, self.nex_idx, ky_per_mot_state_idx = self.build_ky_nex_and_motion_states(params)
        self.sampling_idx, self.nex_offset, self.TotalKspaceSamples = \
            build_sampling_from_motion_states(ky_per_mot_state_idx, self.ky_idx, self.nex_idx, self.Nx, self.Ny, self.t_device)

        # generate motion curves and parameters
        self.navigator, self.tx, self.ty, self.phi = self.create_motion_curves(params)

        # build alpha matrix
        alpha = torch.zeros(3, len(self.sampling_idx), device=self.t_device)
        alpha[0, :] = self.tx
        alpha[1, :] = self.ty
        alpha[2, :] = self.phi

        centers = torch.zeros((2, len(self.sampling_idx)), device=self.t_device)
        centers[0, :] = self.Nx / 2 + params.max_center_x * torch.ones(len(self.sampling_idx), device=self.t_device)
        centers[1, :] = self.Ny / 2 + params.max_center_y * torch.randn(len(self.sampling_idx), device=self.t_device)

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
        Nex    = params.Nex  # kept for future multi-Nex support

        ky_chronological = []      # list of scalar ky values
        ky_per_mot_state = []      # list of (1,) tensors
        nex_chronological = []     # optional, for future use

        for shot in range(Nshots):
            shot_in_nex = shot
            Nex_idx     = 0  # TODO: multi-Nex later

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

            # ----- chronological accumulation -----
            for ky_line in ky:
                ky_chronological.append(ky_line)
                ky_per_mot_state.append(ky_line.unsqueeze(0))  # (1,)
                nex_chronological.append(Nex_idx)

        # -------------------------------------------------
        # Final tensors
        # -------------------------------------------------
        ky_idx = torch.stack(ky_chronological)                 # (Ny,)
        nex_idx = torch.tensor(
            nex_chronological,
            device=self.t_device,
            dtype=torch.int32
        )                                                      # (Ny,)

        ky_per_mot_state_idx = ky_per_mot_state                # list of (1,) tensors

        return ky_idx, nex_idx, ky_per_mot_state_idx



    








        
        
        

    
import numpy as np
import torch

from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.preprocessing.SamplingSimulator import SamplingSimulator
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions
import matplotlib.pyplot as plt

from Parameters import Parameters
params = Parameters()

class MotionSimulator:
    def __init__(self, image, smaps, ky_idx, nex_idx, ky_per_motion_state, sp_device=None, t_device=None):
        self.image = image
        self.smaps = smaps
        self.ky_idx = ky_idx
        self.nex_idx = nex_idx
        self.ky_per_motion_state = ky_per_motion_state
        self.sp_device = sp_device
        self.t_device = t_device
        self.Ncha, self.Nx, self.Ny, self.Nsli = smaps.shape  

    def get_simulated_sampling(self):
        return self.ky_idx, self.nex_idx, self.TotalKspaceSamples
    
    def get_corrupted_kspace(self):
        return self.kspace
    
    def get_corrupted_image(self):
        return self.image_no_moco

    def get_motion_information(self):
        return self.navigator, self.tx, self.ty, self.phi
        
    # -------------------------------------------------------
    #------------------- Common functions -------------------
    # -------------------------------------------------------
    
    def save_debug_plots(self, motion_curve, tx, ty, phi, event_times=None):
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

        if event_times is not None:
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

    def apply_motion(self, alpha, centers=None, motion_signal=None, motion_type=None):
        if motion_type is None:
            motion_type = params.motion_type

        self.MotionOperator = MotionOperator(
            self.Nx, self.Ny, alpha, motion_type, centers=centers, motion_signal=motion_signal
        )

        E = EncodingOperator(
            self.smaps,
            self.TotalKspaceSamples,
            self.sampling_idx,
            params.Nex,
            self.MotionOperator
        )
        kspace_corruped = E.forward(self.image)
        self.kspace = kspace_corruped.reshape(self.Ncha, params.Nex, self.Nx, self.Ny, self.Nsli)

        img_cplx = ifftnc(self.kspace, dims=(-3, -2, -1)).to(self.t_device)
        self.image_no_moco = torch.sum(img_cplx * self.smaps.conj().unsqueeze(1).expand(-1, params.Nex, -1, -1, -1), dim=0)

    # -------------------------------------------------------
    #------------------ Simulate zero motion ----------------
    # -------------------------------------------------------
    
    def simulate_no_motion(self):
        """
        Simulate acquisition with NO motion:
        tx = ty = phi = 0 everywhere
        navigator = 0
        """
        # Use shot-wise sampling (same as discrete motion)
        ky_per_mot_state_idx = self.ky_idx.unsqueeze(0)

        self.sampling_idx_per_nex = SamplingSimulator.build_sampling_per_nex_per_motion(ky_per_mot_state_idx, self.Nx, self.Ny, self.t_device)
        
        self.TotalKspaceSamples = self.Nx * self.Ny

        # Expand zero motion to ky (chronological)
        self.tx        = torch.zeros(self.Ny, device=self.t_device)
        self.ty        = torch.zeros(self.Ny, device=self.t_device)
        self.phi       = torch.zeros(self.Ny, device=self.t_device)
        self.navigator = torch.zeros(self.Ny, device=self.t_device)


    # -------------------------------------------------------
    #---------- Realistic motion curves generation ----------
    # -------------------------------------------------------
    
    def create_realistic_motion_curves(self):
        # Time axis: one value per k-space line (Ny)
        t = torch.arange(self.Ny*params.Nex, device=self.t_device, dtype=torch.float64)

        # 1) Generate random motion event times
        event_times = torch.sort(
            torch.randint(0, self.Ny*params.Nex, (params.num_motion_events,), device=self.t_device)
        ).values

        # Motion transition sharpness (smaller tau -> faster motion)
        tau = params.motion_tau

        # 2) Generate random amplitudes for each event
        A_tx  = params.max_tx  * (2 * torch.rand(params.num_motion_events, device=self.t_device) - 1)
        A_ty  = params.max_ty  * (2 * torch.rand(params.num_motion_events, device=self.t_device) - 1)
        A_phi = params.max_phi * (2 * torch.rand(params.num_motion_events, device=self.t_device) - 1) * (torch.pi/180)

        # 3) Build independent motion curves
        tx  = torch.zeros(self.Ny*params.Nex, device=self.t_device)
        ty  = torch.zeros(self.Ny*params.Nex, device=self.t_device)
        phi = torch.zeros(self.Ny*params.Nex, device=self.t_device)

        # Full time axis
        t = torch.arange(self.Ny*params.Nex, device=self.t_device, dtype=torch.float64)

        for i, ti in enumerate(event_times):
            ti = event_times[i].item()
            t_end = min(ti + tau, self.Ny*params.Nex)
            # Normalized time for the transition
            alpha = np.linspace(0.0, 1.0, t_end - ti)
            # Raised cosine ramp (finite, smooth)
            s = 0.5 * (1.0 - np.cos(np.pi * alpha))

            f = torch.zeros(self.Ny*params.Nex, device=self.t_device)
            f[ti:t_end] = torch.from_numpy(s).to(self.t_device)
            if t_end < self.Ny*params.Nex:
                f[t_end:] = 1.0

            # Add motion contribution (same logic as tanh version)
            tx  += A_tx[i]  * f
            ty  += A_ty[i]  * f
            phi += A_phi[i] * f

        # 4) Compute PCA to obtain a single 1D motion curve (simulation of MRI navigators)
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

    def simulate_realistic_rigid_motion(self):
        # Each kspace line is its own motion state
        ky_per_mot_state_idx = [
            ky_line.unsqueeze(-1) for ky_line in self.ky_idx
        ]
        # self.sampling_idx = \
        #     build_sampling_from_motion_states(ky_per_mot_state_idx, self.ky_idx, self.nex_idx, self.Nx, self.Ny, self.t_device)
        self.sampling_idx = SamplingSimulator.build_sampling_per_nex_per_motion(ky_per_mot_state_idx, self.Nx, self.Ny, self.t_device)

        self.TotalKspaceSamples = self.Ny * self.Nx
        # generate motion curves and parameters
        self.navigator, self.tx, self.ty, self.phi = self.create_realistic_motion_curves()

        # Apply simulated motion
        alpha = torch.zeros(3, self.Ny * params.Nex, device=self.t_device)
        alpha[0, :] = self.tx
        alpha[1, :] = self.ty
        alpha[2, :] = self.phi

        centers = torch.zeros((2, self.Ny * params.Nex), device=self.t_device)
        centers[0, :] = self.Nx / 2 + params.max_center_x * torch.ones(self.Ny * params.Nex, device=self.t_device)
        centers[1, :] = self.Ny / 2 + params.max_center_y * torch.randn(self.Ny * params.Nex, device=self.t_device)

        self.apply_motion(alpha, centers)

    # -------------------------------------------------------
    # ---- Generation of discrtete motion states per shot ---
    # -------------------------------------------------------

    def expand_motion_to_ky(self, ky_per_mot_state_idx):
        """
        Expand per-motion-state parameters into per-ky vectors
        in chronological acquisition order.
        """

        # Total number of ky lines
        # Ny_total = sum(ky.numel() for ky in ky_per_mot_state_idx)
        Ny_total = 0
        for ky_list in ky_per_mot_state_idx:
            for ky in ky_list:
                Ny_total += ky.numel()

        # Allocate
        self.tx        = torch.empty(Ny_total, device=self.t_device)
        self.ty        = torch.empty(Ny_total, device=self.t_device)
        self.phi       = torch.empty(Ny_total, device=self.t_device)
        self.navigator = torch.empty(Ny_total, device=self.t_device)

        ptr = 0  # write pointer (chronological)

        m = 0  # motion state index
        for ky_list in ky_per_mot_state_idx:      # loop over Nex
            for ky in ky_list:                    # loop over shots
                n = ky.numel()

                self.tx[ptr:ptr+n]        = self.tx_mot_state[m]
                self.ty[ptr:ptr+n]        = self.ty_mot_state[m]
                self.phi[ptr:ptr+n]       = self.phi_mot_state[m]
                self.navigator[ptr:ptr+n] = self.navigator_mot_state[m]

                ptr += n
                m += 1

    def create_discrete_motion_curves(self, ky_per_mot_state_idx):
        Nshots = params.Nshots

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
        centers[0, :] = self.Nx / 2 + params.max_center_x * torch.ones(Nshots, device=self.t_device)
        centers[1, :] = self.Ny / 2 + params.max_center_y * torch.ones(Nshots, device=self.t_device) # torch.rand(Nshots, device=self.t_device)

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

    def simulate_discrete_rigid_motion(self):
        # Each shot is its own motion state
        ky_per_mot_state_idx = self.ky_per_motion_state

        # self.sampling_idx = \
        #     build_sampling_from_motion_states(ky_per_mot_state_idx, self.ky_idx, self.nex_idx, self.Nx, self.Ny, self.t_device)
        self.sampling_idx = SamplingSimulator.build_sampling_per_nex_per_motion(ky_per_mot_state_idx, self.Nx, self.Ny, self.t_device) # ← for debugging only, ignore output
        
        self.TotalKspaceSamples = self.Ny * self.Nx

        self.navigator, alpha, centers = self.create_discrete_motion_curves(ky_per_mot_state_idx)

        self.apply_motion(alpha, centers)

    def create_discrete_non_rigid_alpha_fields(self):
        # MATLAB-equivalent coordinates and Gaussian maps for motion_type_flag == 2:
        # [X,Y] = meshgrid(1:N,1:N), sigma = N/4
        x = torch.arange(1, self.Nx + 1, device=self.t_device, dtype=torch.float64)
        y = torch.arange(1, self.Ny + 1, device=self.t_device, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        mu_xx = self.Nx / 2.0
        mu_yy = self.Ny / 2.0
        mu_yx = self.Nx / 2.0
        mu_xy = self.Ny / 2.0

        # MATLAB uses sigma = N/4.
        sigma_x = self.Nx / 4.0
        sigma_y = self.Ny / 4.0

        alpha_x = (
            torch.exp(- (X - mu_xx) ** 2 / (2 * sigma_x ** 2))
            * torch.exp(- (Y - mu_xy) ** 2 / (2 * sigma_y ** 2))
        )
        alpha_y = (
            2 * torch.exp(- (X - mu_yx) ** 2 / (2 * sigma_x ** 2))
            * torch.exp(- (Y - mu_yy) ** 2 / (2 * sigma_y ** 2))
        )
        alpha_maps = torch.stack([alpha_x, alpha_y], dim=0)
        return alpha_x, alpha_y, alpha_maps

    def simulate_discrete_non_rigid_motion(self):
        print("Simulating non-rigid motion fields...")

        ky_per_mot_state_idx = self.ky_per_motion_state
        self.sampling_idx = SamplingSimulator.build_sampling_per_nex_per_motion(
            ky_per_mot_state_idx, self.Nx, self.Ny, self.t_device
        )
        self.TotalKspaceSamples = self.Ny * self.Nx

        # Total number of motion states across all Nex acquisitions.
        Nshots = sum(len(shot_list) for shot_list in ky_per_mot_state_idx)

        # MATLAB-equivalent random motion vectors:
        # XTranslationVector = 4 * randn(1,Nshots)
        # YTranslationVector = 2 * randn(1,Nshots)
        # RotationVector     = 3 * randn(1,Nshots) [deg]
        tx_vec = 4.0 * torch.randn(Nshots, device=self.t_device)
        ty_vec = 2.0 * torch.randn(Nshots, device=self.t_device)
        phi_vec = (3.0 * torch.randn(Nshots, device=self.t_device)) * (torch.pi / 180.0)
        # For non-rigid motion_type_flag==2, MATLAB uses S = XTranslationVector.
        S = tx_vec

        alpha_x, alpha_y, alpha_maps = self.create_discrete_non_rigid_alpha_fields()
        self.alpha_maps = alpha_maps

        if params.debug_flag:
            print("Visualizing non-rigid alpha fields (alpha_x, alpha_y)...")
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(alpha_x.detach().cpu().numpy(), cmap="viridis")
            plt.title("alpha_x")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(alpha_y.detach().cpu().numpy(), cmap="viridis")
            plt.title("alpha_y")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(f"{params.debug_folder}nonrigid_alpha_fields.png")
            plt.close()

        self.apply_motion(alpha_maps, centers=None, motion_signal=S, motion_type='non-rigid')

        self.navigator_mot_state = S
        self.tx_mot_state = tx_vec
        self.ty_mot_state = ty_vec
        self.phi_mot_state = phi_vec
        self.expand_motion_to_ky(ky_per_mot_state_idx)



        




    








        
        
        

    

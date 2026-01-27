import numpy as np
import torch
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions
import matplotlib.pyplot as plt


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
    
    def get_motion_operator(self):
        return self.MotionOperator
    
    def get_corrupted_kspace(self):
        return self.kspace
    
    def get_corrupted_image(self):
        return self.image_no_moco
    
    def create_motion_curves(self, params):
        # Set random seed for reproducibility
        torch.manual_seed(params.seed)

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
        A_phi = params.max_rot * (2 * torch.rand(params.num_motion_events, device=self.t_device) - 1) * (torch.pi/180)

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
        curve = score1.squeeze(0)

        # Normalize curve to [-1, 1] for stability
        curve = curve / curve.abs().max()

        # Return the motion curve, parameter curves, and event times
        return curve, tx, ty, phi, event_times


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

    def bin_motion_by_pc1(self, motion_curve, tx, ty, phi, line_idx, num_motion_events):
        Nbins = num_motion_events + 1

        # Ensure all are torch tensors on GPU
        motion_curve = motion_curve.to(self.t_device)
        tx = tx.to(self.t_device)
        ty = ty.to(self.t_device)
        phi = phi.to(self.t_device)
        line_idx = line_idx.to(self.t_device)

        # Bin edges based on PC1
        min_val = motion_curve.min()
        max_val = motion_curve.max()
        bins = torch.linspace(min_val, max_val, Nbins + 1, device=self.t_device)

        # Digitize (bucketize) using PC1
        bin_ids = torch.bucketize(motion_curve, bins) - 1
        bin_ids = torch.clamp(bin_ids, 0, Nbins - 1)

        binned_indices = [None] * Nbins
        bin_centers_tx = torch.zeros(Nbins, device=self.t_device)
        bin_centers_ty = torch.zeros(Nbins, device=self.t_device)
        bin_centers_phi = torch.zeros(Nbins, device=self.t_device)

        for b in range(Nbins):
            mask = (bin_ids == b)

            # Save line indices
            binned_indices[b] = line_idx[mask]

            if mask.any():
                bin_centers_tx[b]  = tx[mask].mean()
                bin_centers_ty[b]  = ty[mask].mean()
                bin_centers_phi[b] = phi[mask].mean()
            else:
                bin_centers_tx[b]  = float('nan')
                bin_centers_ty[b]  = float('nan')
                bin_centers_phi[b] = float('nan')

        return binned_indices, bin_centers_tx, bin_centers_ty, bin_centers_phi

    
    # def generate_line_idx(self, params):
    #     """
    #     Generate line sampling order on GPU as torch tensor.
    #     """
    #     if params.kspace_sampling_type == 'linear':
    #         return torch.arange(self.Ny, device=self.t_device, dtype=torch.int32)

    #     elif params.kspace_sampling_type == 'interleaved':
    #         # 0, 2, 4 ... 1, 3, 5 ...
    #         evens = torch.arange(0, self.Ny, 2, device=self.t_device, dtype=torch.int32)
    #         odds  = torch.arange(1, self.Ny, 2, device=self.t_device, dtype=torch.int32)
    #         return torch.cat([evens, odds])

    #     else:
    #         raise ValueError("Unknown kspace_sampling_type")

    def create_motion_corrupted_dataset(self, params):
        # self.Nshots = Nshots = params.NshotsPerNex * params.Nex
        self.simulate_kspace_sampling(params)

        # idx_ky = self.generate_line_idx(params)

        # generate motion curves and parameters
        motion_curve, tx, ty, phi, event_times = self.create_motion_curves(params)

        # save debug plots
        self.save_debug_plots(motion_curve, tx, ty, phi, event_times)

        # build alpha matrix
        alpha = torch.zeros(5, len(self.sampling_idx), device=self.t_device)
        alpha[0, :] = tx
        alpha[1, :] = ty
        alpha[2, :] = phi

        centers = torch.zeros((2, len(self.sampling_idx)), device=self.t_device)
        centers[0, :] = self.Nx / 2 + 60 * torch.ones(len(self.sampling_idx), device=self.t_device)
        centers[1, :] = self.Ny / 2 + 10 * torch.randn(len(self.sampling_idx), device=self.t_device)

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

        # binned_indices, bin_centers_tx, bin_centers_ty, bin_centers_phi = \
        #     self.bin_motion_by_pc1(motion_curve, tx, ty, phi, self.ky_idx, params.num_motion_events)



    def simulate_kspace_sampling(self, params):
        Nshots = self.Ny

        self.sampling_idx = []   # list of tensors, one per shot
        self.nex_offset      = []   # one scalar per shot
        self.ky_idx          = []   # ky indices per shot
        self.nex_idx         = []   # nex index per ky per shot
        self.TotalKspaceSamples = 0

        kx = torch.arange(self.Nx, device=self.t_device, dtype=torch.int32)

        # -------------------------------------------------
        # Loop over all shots (UNCHANGED logic)
        # -------------------------------------------------
        for shot in range(Nshots):
            shot_in_nex = shot % params.NshotsPerNex
            Nex_idx     = shot // params.NshotsPerNex

            # -------------------------------------------------
            # Compute ky indices for this shot
            # -------------------------------------------------
            if params.kspace_sampling_type == 'linear':
                start = shot_in_nex * self.Ny // params.NshotsPerNex
                end   = (shot_in_nex + 1) * self.Ny // params.NshotsPerNex
                ky = torch.arange(start, end,
                                device=self.t_device,
                                dtype=torch.int32)

            elif params.kspace_sampling_type == 'interleaved':
                ky = torch.arange(shot_in_nex, self.Ny, params.NshotsPerNex,
                                device=self.t_device,
                                dtype=torch.int32)
            else:
                raise ValueError("Unknown kspace_sampling_type")

            # -------------------------------------------------
            # Build flattened (kx, ky) sampling indices
            # idx = ky + Ny * kx
            # -------------------------------------------------
            sampling_idx_nex = (
                ky.unsqueeze(0) +
                self.Ny * kx.unsqueeze(1)
            ).reshape(-1)

            # -------------------------------------------------
            # Store results (per shot)
            # -------------------------------------------------
            self.sampling_idx.append(sampling_idx_nex)
            self.nex_offset.append(Nex_idx)

            self.ky_idx.append(ky)
            self.nex_idx.append(
                torch.full((ky.numel(),), Nex_idx,
                        device=self.t_device,
                        dtype=torch.int32)
            )

            self.TotalKspaceSamples += sampling_idx_nex.numel()








        
        
        

    
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import math

from src.utils.espiritmaps import calc_espirit_maps, from_espirit_dims, to_espirit_dims
from src.utils.Helpers import from_espirit_to_grics_dims, from_grics_to_espirit_dims
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.preprocessing.RawDataReader import DataReader
from src.preprocessing.RigidMotionSimulator import RigidMotionSimulator
from src.preprocessing.RigidMotionSimulatorShots import RigidMotionSimulatorShots
from src.utils.Helpers import build_sampling_from_motion_states, kmeans_torch
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions


class DataLoader:
    def __init__(self, params, sp_device=None, t_device=None):
        self.sp_device = sp_device
        self.t_device = t_device

        if params.data_type == 'fastMRI':
            self.load_fastMRI_data(params.path_to_data, params)
        elif params.data_type == 'shepp-logan':    
            self.generate_shepp_logan(N=128, Ncoils=16, Nz=1, random_phase=True)
        elif params.data_type == 'real-world':
            self.load_nonrigid_data(params.path_to_data)
        self.Ncha, self.Nx, self.Ny, self.Nsli = self.kspace.shape

        # Calculate ESPIRiT maps and non-corrected image
        self.smaps = from_espirit_to_grics_dims(calc_espirit_maps(self.kspace, params.acs, params.kernel_width, sp_device=self.sp_device))
        kspace_perm = from_espirit_dims(self.kspace)
        self.img_cplx = ifftnc(kspace_perm, dims=(-4, -3, -2))
        self.image_ground_truth = torch.sum(self.img_cplx*self.smaps.conj(), dim=-1).to(self.t_device)

        # Simulate motion-corrupted dataset
        simulator = RigidMotionSimulator(self.image_ground_truth, self.smaps, params, sp_device=self.sp_device, t_device=self.t_device)
        self.ky_idx, self.nex_idx, self.TotalKspaceSamples = simulator.get_simulated_sampling()
        self.kspace = simulator.get_corrupted_kspace()
        self.image_no_moco = simulator.get_corrupted_image()
        navigator, tx, ty, phi = simulator.get_motion_information()

        self.binned_indices = self.bin_motion_rigid(navigator, self.ky_idx, params)

        self.sampling_idx, self.nex_offset, self.TotalKspaceSamples = \
            build_sampling_from_motion_states(self.binned_indices, self.ky_idx, self.nex_idx, self.Nx, self.Ny, self.t_device)
        # TODO include multiple Nex support        
        self.nex_offset = torch.zeros(len(self.binned_indices), device=self.t_device)


    def load_fastMRI_data(self, path_to_mri_data, params):
        self.kspace = np.load(path_to_mri_data)['arr_0']
        self.kspace = torch.from_numpy(self.kspace).to(self.t_device)  

    # N coils should be perfect square for coil map generation
    def generate_shepp_logan(self, N=128, Ncoils=4, Nz=1, random_phase=True):
        # --- 1. Create phantom ---
        phantom_np_2d = resize(shepp_logan_phantom(), (N, N))
        phantom_img_2d = torch.tensor(phantom_np_2d, dtype=torch.float32, device=self.t_device)
        phantom_img = phantom_img_2d.unsqueeze(-1).repeat(1, 1, Nz)  # NxNxNz

        # --- 2. Create coil sensitivity maps ---
        X, Y = torch.meshgrid(
            torch.arange(1, N+1, device=self.t_device),
            torch.arange(1, N+1, device=self.t_device),
            indexing='ij'
        )
        sigma = N / 4

        # Generate coil centers on a grid
        grid_size = math.ceil(math.sqrt(Ncoils))  # minimal square grid to hold all coils
        xs = torch.linspace(N/4, 3*N/4, grid_size, device=self.t_device)
        ys = torch.linspace(N/4, 3*N/4, grid_size, device=self.t_device)
        centers = [(x.item(), y.item()) for y in ys for x in xs][:Ncoils]  # pick exactly Ncoils

        sensitivity_maps_2d = torch.zeros((N, N, Ncoils), dtype=torch.cfloat, device=self.t_device)
        for c, (x0, y0) in enumerate(centers):
            profile = torch.exp(-((X-x0)**2 + (Y-y0)**2) / (2*sigma**2))
            phase = torch.exp(1j * 2 * math.pi * torch.rand(1, device=self.t_device)) if random_phase else 1.0
            sensitivity_maps_2d[:, :, c] = profile * phase

        # Repeat along Nz dimension to create full 3D coil maps
        sensitivity_maps = sensitivity_maps_2d.unsqueeze(2).repeat(1, 1, Nz, 1)  # Nx x Ny x Nz x Ncoils

        # --- 3. Generate k-space (vectorized) ---
        # Add coil dimension to phantom: Nx x Ny x Nz x 1
        phantom_exp = phantom_img.unsqueeze(-1)

        # Multiply by coil maps: Nx x Ny x Nz x Ncoils
        coil_imgs = phantom_exp * sensitivity_maps

        # Permute to Ncoils x Nx x Ny x Nz
        coil_imgs = coil_imgs.permute(3, 0, 1, 2)

        # Compute k-space using fftnc (assumes fftnc supports 4D input)
        self.kspace = fftnc(coil_imgs)  # Ncoils x Nx x Ny x Nz          

    def load_nonrigid_data(self, path_to_data):
        data = DataReader.read_kspace_and_motion_data_from_h5(path_to_data)
        self.kspace = data['kspace']
        self.kspace = torch.from_numpy(self.kspace).to(self.t_device)
        # self.Ncha, self.Nx, self.Ny, self.Nsli = self.kspace.shape
        # data['motion_data'] = f['motion_data'][:]
        #     data['prior_image'] = f['prior_image'][:]
        #     data['line_idx'] = f['line_idx'][:]
        #     data['kspace'] = f['kspace'][:]
        #     data['smap'] = f['smap'][:]
        #     data['bin_centers'] = f['bin_centers'][:]
    


    def bin_motion_rigid(self, motion_curve, line_idx, params):
        Nbins = params.num_motion_events + 1

        # Ensure tensors on device
        motion_curve = motion_curve.to(self.t_device)
        line_idx = line_idx.to(self.t_device)

        # K-means clustering
        labels, centers = kmeans_torch(motion_curve.unsqueeze(1), Nbins)
        binned_indices = [None] * Nbins

        for b in range(Nbins):
            mask = labels == b
            binned_indices[b] = line_idx[mask]

        # ---- Plot ----
        if params.debug_flag:
            # Move to CPU for plotting
            motion_cpu = motion_curve.detach().cpu()
            labels_cpu = labels.detach().cpu()
            line_idx_cpu = line_idx.detach().cpu()

            plt.figure(figsize=(10, 4))
            scatter = plt.scatter(
                line_idx_cpu,
                motion_cpu,
                c=labels_cpu,
                s=10,
            )

            plt.xlabel("Line index")
            plt.ylabel("Motion amplitude")
            plt.title("Motion curve with K-means bin assignment")
            plt.colorbar(scatter, label="Motion bin")
            plt.tight_layout()
            plt.savefig("debug_outputs/binned_curve.png")
            plt.close()

        return binned_indices


    # def bin_motion_rigid(self, motion_curve, tx, ty, phi, line_idx, num_motion_events):
    #     # TODO include multiple Nex support
    #     # line_idx = line_idx[0, :]

    #     Nbins = num_motion_events + 1

    #     # Ensure all are torch tensors on GPU
    #     motion_curve = motion_curve.to(self.t_device)
    #     tx = tx.to(self.t_device)
    #     ty = ty.to(self.t_device)
    #     phi = phi.to(self.t_device)
    #     line_idx = line_idx.to(self.t_device)

    #     # Bin edges based on PC1
    #     min_val = motion_curve.min()
    #     max_val = motion_curve.max()
    #     bins = torch.linspace(min_val, max_val, Nbins + 1, device=self.t_device)

    #     # Digitize (bucketize) using PC1
    #     bin_ids = torch.bucketize(motion_curve, bins) - 1
    #     bin_ids = torch.clamp(bin_ids, 0, Nbins - 1)

    #     binned_indices = [None] * Nbins
    #     bin_centers_tx = torch.zeros(Nbins, device=self.t_device)
    #     bin_centers_ty = torch.zeros(Nbins, device=self.t_device)
    #     bin_centers_phi = torch.zeros(Nbins, device=self.t_device)

    #     for b in range(Nbins):
    #         mask = (bin_ids == b)

    #         # Save line indices
    #         binned_indices[b] = line_idx[mask]

    #         if mask.any():
    #             bin_centers_tx[b]  = tx[mask].mean()
    #             bin_centers_ty[b]  = ty[mask].mean()
    #             bin_centers_phi[b] = phi[mask].mean()
    #         else:
    #             bin_centers_tx[b]  = float('nan')
    #             bin_centers_ty[b]  = float('nan')
    #             bin_centers_phi[b] = float('nan')

    #     return binned_indices, bin_centers_tx, bin_centers_ty, bin_centers_phi


        

    





    
    
        
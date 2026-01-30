import numpy as np
import torch
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions

from src.utils.espiritmaps import calc_espirit_maps, from_espirit_dims, to_espirit_dims
from src.utils.Helpers import from_espirit_to_grics_dims, from_grics_to_espirit_dims
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.preprocessing.RawDataReader import DataReader
from src.preprocessing.RigidMotionSimulator import RigidMotionSimulator
from src.preprocessing.RigidMotionSimulatorShots import RigidMotionSimulatorShots
from src.utils.Helpers import build_sampling_from_motion_states, kmeans_torch


class DataLoader:
    def __init__(self, path_to_data, params, sp_device=None, t_device=None):
        self.sp_device = sp_device
        self.t_device = t_device    
        self.load_rigid_data(path_to_data, params)

        # Calculate ESPIRiT maps and non-corrected image
        self.smaps = from_espirit_to_grics_dims(calc_espirit_maps(self.kspace, params.acs, params.kernel_width, sp_device=self.sp_device))
        kspace_perm = from_espirit_dims(self.kspace)
        self.img_cplx = ifftnc(kspace_perm, dims=(-4, -3, -2))
        self.image_ground_truth = torch.sum(self.img_cplx*self.smaps.conj(), dim=-1).to(self.t_device)

        # Simulate motion-corrupted dataset
        simulator = RigidMotionSimulatorShots(self.image_ground_truth, self.smaps, params, sp_device=self.sp_device, t_device=self.t_device)
        self.ky_idx, self.nex_idx, self.TotalKspaceSamples = simulator.get_simulated_sampling()
        self.kspace = simulator.get_corrupted_kspace()
        self.image_no_moco = simulator.get_corrupted_image()
        navigator, tx, ty, phi = simulator.get_motion_information()

        self.binned_indices = self.bin_motion_rigid(navigator, self.ky_idx, params.num_motion_events)

        self.sampling_idx, self.nex_offset, self.TotalKspaceSamples = \
            build_sampling_from_motion_states(self.binned_indices, self.ky_idx, self.nex_idx, self.Nx, self.Ny, self.t_device)
        # TODO include multiple Nex support        
        self.nex_offset = torch.zeros(len(self.binned_indices), device=self.t_device)


    def load_rigid_data(self, path_to_mri_data, params):
        self.kspace = np.load(path_to_mri_data)['arr_0']
        self.kspace = torch.from_numpy(self.kspace).to(self.t_device)
        self.Ncha, self.Nx, self.Ny, self.Nsli = self.kspace.shape    
          

    def load_nonrigid_data(self, path_to_data):
        data = DataReader.read_kspace_and_motion_data_from_h5(path_to_data)
        self.kspace = data['kspace']
        self.kspace = torch.from_numpy(self.kspace).to(self.t_device)
        self.Ncha, self.Nx, self.Ny, self.Nsli = self.kspace.shape
        # data['motion_data'] = f['motion_data'][:]
        #     data['prior_image'] = f['prior_image'][:]
        #     data['line_idx'] = f['line_idx'][:]
        #     data['kspace'] = f['kspace'][:]
        #     data['smap'] = f['smap'][:]
        #     data['bin_centers'] = f['bin_centers'][:]
    
    def bin_motion_rigid(self, motion_curve, line_idx, num_motion_events):
        Nbins = num_motion_events + 1

        # Ensure tensors on device
        motion_curve = motion_curve.to(self.t_device)
        line_idx = line_idx.to(self.t_device)

        # K-means clustering
        labels, centers = kmeans_torch(motion_curve.unsqueeze(1), Nbins)
        binned_indices = [None] * Nbins

        for b in range(Nbins):
            mask = labels == b
            binned_indices[b] = line_idx[mask]

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


        

    





    
    
        
import numpy as np
import torch
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions

from src.utils.espiritmaps import calc_espirit_maps, from_espirit_dims, to_espirit_dims
from src.utils.Helpers import from_espirit_to_grics_dims, from_grics_to_espirit_dims
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.preprocessing.RawDataReader import DataReader
from src.preprocessing.RigidMotionSimulator import RigidMotionSimulator
from src.utils.Helpers import build_sampling_from_motion_states


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
        simulator = RigidMotionSimulator(self.image_ground_truth, self.smaps, params, sp_device=self.sp_device, t_device=self.t_device)
        self.kx_idx, self.nex_idx, self.TotalKspaceSamples = simulator.get_simulated_sampling()
        self.MotionOperator = simulator.get_motion_operator()
        self.kspace = simulator.get_corrupted_kspace()
        self.image_no_moco = simulator.get_corrupted_image()


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


        

    





    
    
        
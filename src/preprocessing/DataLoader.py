import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import math

from src.preprocessing.RawDataReader import RawDataReader
from src.utils.espiritmaps import calc_espirit_maps
from src.preprocessing.MotionSimulator import MotionSimulator
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions
from src.preprocessing.SamplingSimulator import SamplingSimulator
from src.preprocessing.MotionBinner import MotionBinner

from Parameters import Parameters
params = Parameters()


class DataLoader:
    def __init__(self, sp_device=None, t_device=None):
        self.sp_device = sp_device
        self.t_device = t_device
        
        if params.data_type == 'shepp-logan': # Generation of Shepp-Logan phantom with coil sensitivities + sampling simulation   
            self.generate_shepp_logan(N=params.N_SheppLogan, Ncoils=params.Ncoils_SheppLogan, Nz=params.Nz_SheppLogan, random_phase=True)
        elif params.data_type == 'fastMRI': # Only kspace data per coil, but no acquisition order => simulate sampling
            self.load_fastMRI_data(params.path_to_fastMRI_data)
        elif params.data_type == 'real-world': # Real-world data with acquisition order and motion data
            self.load_realworld_data(params.path_to_realworld_data, slice_idx=16)
        elif params.data_type == 'raw-data': # Real-world data with acquisition order and motion data, loaded from raw data files
            self.load_realworld_data_from_ismrm_and_saec(params.ismrmrd_file, params.saec_file, slice_idx=16)
        else:
            raise ValueError("Unknown data_type")

        # Calculate ESPIRiT maps and input image
        self.smaps = calc_espirit_maps(self.kspace, params.acs, params.kernel_width, sp_device=self.sp_device)
        self.img_cplx = ifftnc(self.kspace, dims=(-3, -2, -1))
        smaps_replicated = self.smaps.unsqueeze(1).expand(-1, params.Nex, -1, -1, -1)
        self.image_ground_truth = torch.sum(self.img_cplx*smaps_replicated.conj(), dim=0).to(self.t_device)

        motionSimulator = MotionSimulator(self.image_ground_truth, self.smaps, self.ky_idx, self.nex_idx, self.ky_per_motion, \
                                            sp_device=self.sp_device, t_device=self.t_device)
        
        if params.simulation_type == 'as-it-is':
            if params.data_type in ['real-world', 'raw-data']:
                self.image_no_moco = self.image_ground_truth.clone()
            else:
                raise ValueError("Simulation type 'as-it-is' is only compatible with real-world or raw-data, which already contain motion. Please choose a different simulation type or data type.")
        else:
            if params.simulation_type == 'no-motion':
                motionSimulator.simulate_no_motion()
                self.image_no_moco = self.image_ground_truth.clone()
            else:      
                if params.simulation_type == 'discrete-rigid':
                    motionSimulator.simulate_discrete_rigid_motion()
                elif params.simulation_type == 'rigid':
                    motionSimulator.simulate_realistic_rigid_motion()
                else:
                    raise ValueError("Unknown simulation_type")
                self.kspace = motionSimulator.get_corrupted_kspace()
                self.image_no_moco = motionSimulator.get_corrupted_image()

            motion_curve, _, _, _ = motionSimulator.get_motion_information()
            self.binned_indices = MotionBinner.bin_motion(motion_curve, self.ky_idx, self.nex_idx, self.t_device)
        
        self.sampling_idx = SamplingSimulator.build_sampling_per_nex_per_motion(
            self.binned_indices,  # [Nex][Nmotion]
            self.Nx, self.Ny,
            self.t_device
        )


    def load_fastMRI_data(self, path_to_mri_data):
        self.kspace = np.load(path_to_mri_data)['arr_0']
        self.kspace = torch.from_numpy(self.kspace).to(self.t_device)
        self.kspace = self.kspace.unsqueeze(1).expand(-1, params.Nex, -1, -1, -1)
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape
        samplingSimulator = SamplingSimulator(self.Ny, self.t_device)
        self.ky_idx, self.nex_idx, self.ky_per_motion = samplingSimulator.build_ky_and_nex()

    # Ncoils should be a perfect square (or close) for coil map generation
    def generate_shepp_logan(self, N=128, Ncoils=4, Nz=1, random_phase=True):
        # 1. Create phantom (Nx, Ny, Nz)
        phantom_np_2d = resize(shepp_logan_phantom(), (N, N))
        phantom_2d = torch.tensor(
            phantom_np_2d,
            dtype=torch.float32,
            device=self.t_device
        )

        phantom = phantom_2d.unsqueeze(-1).expand(N, N, Nz)  # (Nx, Ny, Nz)

        # 2. Create coil sensitivity maps (Ncoils, Nx, Ny, Nz)
        X, Y = torch.meshgrid(
            torch.arange(1, N + 1, device=self.t_device),
            torch.arange(1, N + 1, device=self.t_device),
            indexing="ij"
        )

        sigma = N / 4

        # Coil centers on (approximately) square grid
        grid_size = math.ceil(math.sqrt(Ncoils))
        xs = torch.linspace(N / 4, 3 * N / 4, grid_size, device=self.t_device)
        ys = torch.linspace(N / 4, 3 * N / 4, grid_size, device=self.t_device)
        centers = [(x.item(), y.item()) for y in ys for x in xs][:Ncoils]

        # Allocate coil maps directly in (Ncoils, Nx, Ny)
        smaps_2d = torch.empty(
            (Ncoils, N, N),
            dtype=torch.cfloat,
            device=self.t_device
        )

        for c, (x0, y0) in enumerate(centers):
            profile = torch.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
            phase = (
                torch.exp(1j * 2 * math.pi * torch.rand(1, device=self.t_device))
                if random_phase else 1.0
            )
            smaps_2d[c] = profile * phase

        # Expand to Nz: (Ncoils, Nx, Ny, Nz)
        smaps = smaps_2d.unsqueeze(-1).expand(Ncoils, N, N, Nz)

        # 3. Generate coil images directly in (Ncoils, Nx, Ny, Nz)
        phantom = phantom.unsqueeze(0)          # (1, Nx, Ny, Nz)
        coil_imgs = phantom * smaps             # (Ncoils, Nx, Ny, Nz)
        coil_imgs = coil_imgs.unsqueeze(1).expand(-1, self.params.Nex, -1, -1, -1) # add Nex dimension: (Ncoils, Nex, Nx, Ny, Nz)

        coil_imgs = coil_imgs.contiguous()       # important for FFT speed

        # 4. FFT → k-space (Ncoils, Nx, Ny, Nz)
        self.kspace = fftnc(coil_imgs, dims=(-3, -2, -1))
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape

        # 5. Sampling
        samplingSimulator = SamplingSimulator(self.Ny, self.t_device)
        self.ky_idx, self.nex_idx, self.ky_per_motion = samplingSimulator.build_ky_and_nex()

    def load_realworld_data_from_ismrm_and_saec(self, path_to_ismrm, path_to_saec, slice_idx=0):
        data = RawDataReader.read_data_from_rawdata(path_to_ismrm, path_to_saec, h5filename='data/breast_motion_data.h5') #
        self.kspace = torch.from_numpy(data['kspace']).to(self.t_device, dtype=torch.cfloat)[:, :, :, :, [slice_idx]]
        # self.kspace = from_grics_to_espirit_dims(self.kspace)[:, :, :, [slice_idx]]
        self.ky_idx = torch.from_numpy(data['idx_ky'][slice_idx]).to(self.t_device, dtype=torch.int64)
        self.nex_idx = torch.zeros_like(self.ky_idx, device=self.t_device)
        motion_data = data['motion_data'][slice_idx, :]
        motion_data = torch.from_numpy(motion_data).to(self.t_device)
        self.ky_per_motion = self.binned_indices = MotionBinner.bin_motion(motion_data, self.ky_idx, self.nex_idx, self.t_device)
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape
       
        # self.TotalKspaceSamples = np.prod(self.ky_idx.shape) 

    def load_realworld_data(self, path_to_data, slice_idx=0):
        data = RawDataReader.read_kspace_and_motion_data_from_h5(path_to_data)
        self.kspace = torch.from_numpy(data['kspace']).to(self.t_device, dtype=torch.cfloat)[:, :, :, :, [slice_idx]]
        # self.kspace = from_grics_to_espirit_dims(kspace)[:, :, :, [slice_idx]]
        ky_dx = data['line_idx'][slice_idx]
        self.ky_idx = torch.from_numpy(ky_dx).to(self.t_device, dtype=torch.int64)
        self.nex_idx = torch.zeros_like(self.ky_idx, device=self.t_device)
        motion_data = data['motion_data'][slice_idx, :]
        plt.figure()
        plt.plot(motion_data)
        plt.xlabel("Line index")
        plt.title("Motion Curve")
        plt.savefig("debug_outputs/motion_curve.png")
        plt.close()
        motion_data = torch.from_numpy(motion_data).to(self.t_device)
        self.ky_per_motion = self.binned_indices = MotionBinner.bin_motion(motion_data, self.ky_idx, self.nex_idx, self.t_device)
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape
        # self.TotalKspaceSamples = np.prod(self.ky_idx.shape) 






        

    





    
    
        
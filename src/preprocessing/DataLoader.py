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
from src.preprocessing.MotionSimulator import MotionSimulator
from src.utils.Helpers import build_sampling_from_motion_states, kmeans_torch
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions
from src.preprocessing.SamplingSimulator import SamplingSimulator


class DataLoader:
    def __init__(self, params, sp_device=None, t_device=None):
        self.sp_device = sp_device
        self.t_device = t_device
        self.params = params
        
        if params.data_type == 'shepp-logan': # Generation of Shepp-Logan phantom with coil sensitivities + sampling simulation   
            self.generate_shepp_logan(N=params.N_SheppLogan, Ncoils=params.Ncoils_SheppLogan, Nz=params.Nz_SheppLogan, random_phase=True)
        elif params.data_type == 'fastMRI': # Only kspace data per coil, but no acquisition order => simulate sampling
            self.load_fastMRI_data(params.path_to_data, params)
        elif params.data_type == 'real-world': # Real-world data with acquisition order and motion data
            self.load_realworld_data(params.path_to_data, slice_idx=16)
        else:
            raise ValueError("Unknown data_type")

        # Calculate ESPIRiT maps and input image
        self.smaps = calc_espirit_maps(self.kspace, params.acs, params.kernel_width, sp_device=self.sp_device)
        self.img_cplx = ifftnc(self.kspace, dims=(-3, -2, -1))
        self.image_ground_truth = torch.sum(self.img_cplx*self.smaps.conj(), dim=0).to(self.t_device)

        motionSimulator = MotionSimulator(self.image_ground_truth, self.smaps, self.ky_idx, self.nex_idx, self.ky_per_shot, \
                                            params, sp_device=self.sp_device, t_device=self.t_device)
        
        if params.simulation_type == 'as-it-is':
            self.image_no_moco = self.image_ground_truth.clone()
        else:
            if params.simulation_type == 'no-motion':
                motionSimulator.simulate_no_motion()
                self.image_no_moco = self.image_ground_truth.clone()
                params.N_mot_states = 1
            else:      
                if params.simulation_type == 'discrete-rigid':
                    motionSimulator.simulate_discrete_rigid_motion()
                elif params.simulation_type == 'rigid':
                    motionSimulator.simulate_realistic_rigid_motion()
                else:
                    raise ValueError("Unknown simulation_type")
                self.kspace = motionSimulator.get_corrupted_kspace()
                self.image_no_moco = motionSimulator.get_corrupted_image()

            navigator, tx, ty, phi = motionSimulator.get_motion_information()
            self.binned_indices = self.bin_motion_rigid(navigator, self.ky_idx, params)

        self.sampling_idx, self.nex_offset, self.TotalKspaceSamples = \
            build_sampling_from_motion_states(self.binned_indices, self.ky_idx, self.nex_idx, self.Nx, self.Ny, self.t_device)
        # TODO include multiple Nex support
        self.nex_offset = torch.zeros(len(self.binned_indices), device=self.t_device)


    def load_fastMRI_data(self, path_to_mri_data, params):
        self.kspace = np.load(path_to_mri_data)['arr_0']
        self.kspace = torch.from_numpy(self.kspace).to(self.t_device)
        self.Ncha, self.Nx, self.Ny, self.Nsli = self.kspace.shape
        samplingSimulator = SamplingSimulator(params, self.Ny, self.t_device)
        self.ky_idx, self.nex_idx, self.ky_per_shot = samplingSimulator.build_ky_and_nex()

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

        coil_imgs = coil_imgs.contiguous()       # important for FFT speed

        # 4. FFT → k-space (Ncoils, Nx, Ny, Nz)
        self.kspace = fftnc(coil_imgs, dims=(1, 2, 3))
        self.Ncha, self.Nx, self.Ny, self.Nsli = self.kspace.shape

        # 5. Sampling
        samplingSimulator = SamplingSimulator(self.params, self.Ny, self.t_device)
        self.ky_idx, self.nex_idx, self.ky_per_shot = samplingSimulator.build_ky_and_nex()
 

    def load_realworld_data(self, path_to_data, slice_idx=0):
        data = DataReader.read_kspace_and_motion_data_from_h5(path_to_data)
        kspace = torch.from_numpy(data['kspace']).to(self.t_device, dtype=torch.cfloat)
        self.kspace = from_grics_to_espirit_dims(kspace)[:, :, :, [slice_idx]]
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
        self.ky_per_shot = self.binned_indices = self.bin_motion_rigid(motion_data, self.ky_idx, self.params)
        self.Ncha, self.Nx, self.Ny, self.Nsli = self.kspace.shape
        # TODO include multiple Nex support        
        self.nex_offset = torch.zeros(len(self.binned_indices), device=self.t_device)
        self.TotalKspaceSamples = np.prod(self.ky_idx.shape)


        # self.Ncha, self.Nx, self.Ny, self.Nsli = self.kspace.shape
        # data['motion_data'] = f['motion_data'][:]
        #     data['prior_image'] = f['prior_image'][:]
        #     data['line_idx'] = f['line_idx'][:]
        #     data['kspace'] = f['kspace'][:]
        #     data['smap'] = f['smap'][:]
        #     data['bin_centers'] = f['bin_centers'][:]
    


    def bin_motion_rigid(self, motion_curve, line_idx, params):
        Nbins = params.N_mot_states

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


        

    





    
    
        
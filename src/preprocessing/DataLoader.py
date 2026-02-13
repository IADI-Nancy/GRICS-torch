import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import math
import h5py
import sigpy as sp
import sigpy.mri as spmri

from src.preprocessing.RawDataReader import RawDataReader
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
        self.smaps = self.calc_espirit_maps()
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
        coil_imgs = coil_imgs.unsqueeze(1).expand(-1, params.Nex, -1, -1, -1) # add Nex dimension: (Ncoils, Nex, Nx, Ny, Nz)

        coil_imgs = coil_imgs.contiguous()       # important for FFT speed

        # 4. FFT → k-space (Ncoils, Nx, Ny, Nz)
        self.kspace = fftnc(coil_imgs, dims=(-3, -2, -1))
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape

        # 5. Sampling
        samplingSimulator = SamplingSimulator(self.Ny, self.t_device)
        self.ky_idx, self.nex_idx, self.ky_per_motion = samplingSimulator.build_ky_and_nex()

    def load_realworld_data_from_ismrm_and_saec(self, path_to_ismrm, path_to_saec, slice_idx=0):
        reader = RawDataReader(
            ismrmrd_file=path_to_ismrm,
            saec_file=path_to_saec,
            sensor_type="BELT",
            device="cuda"
        )
        data = reader.read_data_from_rawdata()

        self.kspace = torch.from_numpy(data['kspace']).to(self.t_device, dtype=torch.cfloat)[:, :, :, :, [slice_idx]]
        self.ky_idx = torch.from_numpy(data['idx_ky'][slice_idx]).to(self.t_device, dtype=torch.int64)
        self.nex_idx = torch.zeros_like(self.ky_idx, device=self.t_device)
        motion_data = data['motion_data'][slice_idx, :]
        motion_data = torch.from_numpy(motion_data).to(self.t_device)
        self.ky_per_motion = self.binned_indices = MotionBinner.bin_motion(motion_data, self.ky_idx, self.nex_idx, self.t_device)
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape

        if params.debug_flag:
            SamplingSimulator.visualize_ky_order(
                [self.ky_idx.detach().cpu()],
                Ny=self.Ny,
                folder=params.debug_folder,
                fname=f"ky_order_rawdata_slice{slice_idx}.png"
            )
        

    def load_realworld_data(self, path_to_data, slice_idx=0):
        data = {}
        with h5py.File(path_to_data, 'r') as f:
            data['motion_data'] = f['motion_data'][:]
            data['idx_ky'] = f['idx_ky'][:]
            data['idx_kz'] = f['idx_kz'][:]
            data['idx_nex'] = f['idx_nex'][:]
            data['kspace'] = f['kspace'][:]

        self.kspace = torch.from_numpy(data['kspace']).to(self.t_device, dtype=torch.cfloat)[:, :, :, :, [slice_idx]]
        ky_dx = data['idx_ky'][slice_idx]
        self.ky_idx = torch.from_numpy(ky_dx).to(self.t_device, dtype=torch.int64)
        self.nex_idx = torch.zeros_like(self.ky_idx, device=self.t_device) # TODO Add multiple Nex
        motion_data = data['motion_data'][slice_idx, :]
        motion_data = torch.from_numpy(motion_data).to(self.t_device)
        self.ky_per_motion = self.binned_indices = MotionBinner.bin_motion(motion_data, self.ky_idx, self.nex_idx, self.t_device)
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape

        if params.debug_flag:
            SamplingSimulator.visualize_ky_order(
                [self.ky_idx.detach().cpu()],
                Ny=self.Ny,
                folder=params.debug_folder,
                fname=f"ky_order_realworld_slice{slice_idx}.png"
            )

    def calc_espirit_maps(self):
        acs=params.acs
        kernel_width=params.kernel_width
        sp_device=self.sp_device
        kspace = self.kspace
        device = kspace.device             # torch device of input
        
        use_gpu = device.type == "cuda"

        if sp_device is None:
            sp_device = sp.Device(0 if use_gpu else -1)

        nCha, _, nX, nY, nSlices = kspace.shape

        espirit_maps = torch.zeros(
            (nCha, nX, nY, nSlices),
            dtype=torch.complex64,
            device=device
        )

        for i in range(nSlices):

            # ---- GPU path ----
            if use_gpu:
                import cupy as cp
                kspace_cp = cp.asarray(kspace[:, 0, :, :, i].contiguous())
                maps_cp = spmri.app.EspiritCalib(
                    kspace_cp, calib_width=acs,
                    kernel_width=kernel_width,
                    device=sp_device
                ).run()
                maps_cp = maps_cp.astype(cp.complex64, copy=False)
                maps_cp = cp.ascontiguousarray(maps_cp)
                maps_t = sp.to_pytorch(maps_cp)

            # ---- CPU path ----
            else:
                kspace_np = kspace[:, 0, :, :, i].cpu().numpy()
                maps_np = spmri.app.EspiritCalib(
                    kspace_np, calib_width=acs,
                    kernel_width=kernel_width,
                    device=sp_device
                ).run()
                maps_np = maps_np.astype(np.complex64, copy=False)
                maps_t = torch.from_numpy(np.stack([maps_np.real, maps_np.imag], axis=-1))

            espiritual = torch.complex(maps_t[..., 0], maps_t[..., 1])
            espirit_maps[:, :, :, i] = espiritual

            if use_gpu:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                torch.cuda.empty_cache()

        return espirit_maps






        

    





    
    
        

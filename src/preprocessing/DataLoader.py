import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import math
from matplotlib.colors import Normalize

from Parameters import Parameters
from src.preprocessing.RawDataReader import RawDataReader
from src.utils.espiritmaps import calc_espirit_maps
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.preprocessing.MotionSimulator import MotionSimulator
from src.utils.Helpers import kmeans_torch, build_sampling_per_nex_per_motion #build_sampling_from_motion_states
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions
from src.preprocessing.SamplingSimulator import SamplingSimulator

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

        motionSimulator = MotionSimulator(self.image_ground_truth, self.smaps, self.ky_idx, self.nex_idx, self.ky_per_shot, \
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

            navigator, tx, ty, phi = motionSimulator.get_motion_information()
            self.binned_indices = self.bin_motion_rigid(navigator)
        
        self.sampling_idx = build_sampling_per_nex_per_motion(
            self.binned_indices,  # [Nex][Nmotion]
            self.Nx, self.Ny,
            self.t_device
        )
        # TODO include multiple Nex support
        self.nex_offset = torch.zeros(len(self.binned_indices), device=self.t_device)


    def load_fastMRI_data(self, path_to_mri_data):
        self.kspace = np.load(path_to_mri_data)['arr_0']
        self.kspace = torch.from_numpy(self.kspace).to(self.t_device)
        self.kspace = self.kspace.unsqueeze(1).expand(-1, params.Nex, -1, -1, -1) # add Nex dimension
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape
        samplingSimulator = SamplingSimulator(self.Ny, self.t_device)
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
        coil_imgs = coil_imgs.unsqueeze(1).expand(-1, self.params.Nex, -1, -1, -1) # add Nex dimension: (Ncoils, Nex, Nx, Ny, Nz)

        coil_imgs = coil_imgs.contiguous()       # important for FFT speed

        # 4. FFT → k-space (Ncoils, Nx, Ny, Nz)
        self.kspace = fftnc(coil_imgs, dims=(-3, -2, -1))
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape

        # 5. Sampling
        samplingSimulator = SamplingSimulator(self.Ny, self.t_device)
        self.ky_idx, self.nex_idx, self.ky_per_shot = samplingSimulator.build_ky_and_nex()

    def load_realworld_data_from_ismrm_and_saec(self, path_to_ismrm, path_to_saec, slice_idx=0):
        data = RawDataReader.read_data_from_rawdata(path_to_ismrm, path_to_saec, h5filename='data/breast_motion_data.h5') #
        self.kspace = torch.from_numpy(data['kspace']).to(self.t_device, dtype=torch.cfloat)[:, :, :, :, [slice_idx]]
        # self.kspace = from_grics_to_espirit_dims(self.kspace)[:, :, :, [slice_idx]]
        self.ky_idx = torch.from_numpy(data['line_idx'][slice_idx]).to(self.t_device, dtype=torch.int64)
        self.nex_idx = torch.zeros_like(self.ky_idx, device=self.t_device)
        motion_data = data['motion_data'][slice_idx, :]
        motion_data = torch.from_numpy(motion_data).to(self.t_device)
        self.ky_per_shot = self.binned_indices = self.bin_motion_rigid(motion_data, self.ky_idx)
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape
        # TODO include multiple Nex support        
        self.nex_offset = torch.zeros(len(self.binned_indices), device=self.t_device)
        self.TotalKspaceSamples = np.prod(self.ky_idx.shape) 

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
        self.ky_per_shot = self.binned_indices = self.bin_motion_rigid(motion_data, self.ky_idx)
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape
        # TODO include multiple Nex support        
        self.nex_offset = torch.zeros(len(self.binned_indices), device=self.t_device)
        self.TotalKspaceSamples = np.prod(self.ky_idx.shape) 


    def bin_motion_rigid(self, motion_curve):
        ky_idx = self.ky_idx
        nex_idx = self.nex_idx
        motion_curve = motion_curve.to(self.t_device)

        Nbins = params.N_mot_states
        Nex = params.Nex
        norm_color = Normalize(vmin=0, vmax=Nbins - 1)

        # ---- K-means clustering (global, across all Nex) ----
        labels, centers = kmeans_torch(motion_curve.unsqueeze(1), Nbins)

        # ---- Allocate output: [Nex][Nbins] ----
        binned_indices = [
            [torch.empty(0, dtype=ky_idx[0].dtype, device=self.t_device) for _ in range(Nbins)]
            for _ in range(Nex)
        ]
        ky_idx = torch.cat([k.reshape(-1) for k in ky_idx], dim=0)
        nex_idx = torch.cat([nex.reshape(-1) for nex in nex_idx], dim=0)

        # ---- Fill bins ----
        for nex in range(Nex):
            nex_mask = nex_idx == nex

            for b in range(Nbins):
                mask = nex_mask & (labels == b)
                binned_indices[nex][b] = ky_idx[mask]

        # ---- Debug plots ----
        if params.debug_flag:
            markers = ["o", "x", "s", "^", "v", "D", "+", "*", "<", ">"]
            motion_cpu = motion_curve.detach().cpu()
            labels_cpu = labels.detach().cpu()
            nex_cpu = nex_idx.detach().cpu()
            time_idx = torch.arange(len(motion_cpu))
            ky_idx_cpu = ky_idx.detach().cpu()

            # chronological plot
            fig, ax = plt.subplots(figsize=(10, 4))

            for nex in torch.unique(nex_cpu):
                mask = nex_cpu == nex
                sc = ax.scatter(
                    time_idx[mask],
                    motion_cpu[mask],
                    c=labels_cpu[mask],
                    s=12,
                    cmap="tab10",
                    norm=norm_color,                      # ← IMPORTANT
                    marker=markers[int(nex) % len(markers)],
                    label=f"Nex {int(nex)}",
                )

            ax.set_xlabel("Time / acquisition order")
            ax.set_ylabel("Motion amplitude")
            ax.set_title("Chronological motion curve (color = motion state, marker = Nex)")
            ax.legend(title="Nex", loc="best")

            # Proper colorbar (global)
            sm = plt.cm.ScalarMappable(cmap="tab10", norm=norm_color)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Motion bin")
            cbar.set_ticks(range(Nbins))

            fig.tight_layout()
            fig.savefig("debug_outputs/clustered_curve_chronological.png")
            plt.close(fig)



            # ky-sorted plot        
            fig, ax = plt.subplots(figsize=(10, 4))

            for nex in torch.unique(nex_cpu):
                mask = nex_cpu == nex
                sc = ax.scatter(
                    ky_idx_cpu[mask],
                    motion_cpu[mask],
                    c=labels_cpu[mask],
                    s=12,
                    cmap="tab10",
                    norm=norm_color,                      # ← IMPORTANT
                    marker=markers[int(nex) % len(markers)],
                    label=f"Nex {int(nex)}",
                )

            ax.set_xlabel("Line index (ky)")
            ax.set_ylabel("Motion amplitude")
            ax.set_title("Motion curve vs ky (color = motion state, marker = Nex)")
            ax.legend(title="Nex", loc="best")

            sm = plt.cm.ScalarMappable(cmap="tab10", norm=norm_color)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Motion bin")
            cbar.set_ticks(range(Nbins))

            fig.tight_layout()
            fig.savefig("debug_outputs/clustered_curve_sorted_ky.png")
            plt.close(fig)



        return binned_indices



        

    





    
    
        
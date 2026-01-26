import numpy as np
import torch
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions

from src.utils.espiritmaps import calc_espirit_maps, from_espirit_dims, to_espirit_dims
from src.utils.Helpers import from_espirit_to_grics_dims, from_grics_to_espirit_dims
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator


class Data:
    def __init__(self, path_to_kspace, params, sp_device=None, t_device=None):
        self.sp_device = sp_device
        self.t_device = t_device

        self.kspace = np.load(path_to_kspace)['arr_0']
        self.kspace = torch.from_numpy(self.kspace).to(t_device)
        self.Ncha, self.Nx, self.Ny, self.Nsli = self.kspace.shape

        # Calculate ESPIRiT maps and non-corrected image
        self.smaps = from_espirit_to_grics_dims(calc_espirit_maps(self.kspace, params.acs, params.kernel_width, sp_device=sp_device))
        kspace_perm = from_espirit_dims(self.kspace)
        self.img_cplx = ifftnc(kspace_perm, dims=(-4, -3, -2))
        self.image_no_moco = torch.sum(self.img_cplx*self.smaps.conj(), dim=-1).to(t_device)
        

    def create_motion_corrupted_dataset(self, params):
        self.Nshots = Nshots = params.NshotsPerNex * params.Nex
        self.simulate_kspace_sampling(params)

        alpha = torch.zeros((5, Nshots), device=self.t_device)
        alpha[0, :] = 4* torch.randn(Nshots, device=self.t_device) #t_x
        alpha[1, :] = 3 * torch.randn(Nshots, device=self.t_device) #t_y
        alpha[2, :] = 10 * torch.randn(Nshots, device=self.t_device) * (torch.pi / 180) #phi_rot
        centers = torch.zeros((2, Nshots), device=self.t_device)
        centers[0, :] = self.Nx / 2 + 60 * torch.ones(Nshots, device=self.t_device) #center_x
        centers[1, :] = self.Ny / 2 + 10 * torch.randn(Nshots, device=self.t_device) #center_y
        self.MotionOperator = MotionOperator(self.Nx, self.Ny, alpha, centers)

        # self.simulate_rigid_motion_fields(t_x, t_y, phi_rot, rotation_center=[300, 180]) #
        E = EncodingOperator(self.smaps, self.TotalKspaceSamples, self.SamplingIndices, self.KspaceOffset, self.MotionOperator)
        kspace_corruped = E.forward(self.image_no_moco)
        self.kspace = kspace_corruped.reshape(params.Nex, self.Nx, self.Ny, self.Nsli, self.Ncha)
        self.img_cplx = ifftnc(self.kspace[0,:,:,:,:], dims=(0, 1, 2)).to(self.t_device)

        self.image_no_moco = torch.sum(self.img_cplx*self.smaps.conj(), dim=-1)

    
    def simulate_kspace_sampling(self, params):
        Nshots = self.Nshots
        self.KspaceOffset  = []
        self.TotalKspaceSamples = 0
        self.SamplingIndices = []
        # Loop over all shots
        for shot in range(Nshots):
            shot_in_nex = (shot % params.NshotsPerNex)
            Nex_idx = shot // params.NshotsPerNex

            # ----- Build sampling mask (N x N) -----
            KspaceSamplingMask = torch.zeros((self.Nx, self.Ny), dtype=torch.float32, device=self.t_device)

            if params.kspace_sampling_type == 'linear':
                # Linear sampling (contiguous blocks)
                start = shot_in_nex * self.Ny // params.NshotsPerNex
                end   = (shot_in_nex + 1) * self.Ny // params.NshotsPerNex
                KspaceSamplingMask[:, start:end] = 1.0

            elif params.kspace_sampling_type == 'interleaved':
                # Interleaved sampling
                KspaceSamplingMask[:, shot_in_nex::params.NshotsPerNex] = 1.0

            # ----- Build sparse sampling operator -----
            nnz_idx = torch.nonzero(KspaceSamplingMask.flatten(), as_tuple=True)[0]
            Nsamp   = nnz_idx.numel()

            # self.KspaceSamplingOperator.append(SamplingOp)
            self.SamplingIndices.append(nnz_idx)
            self.KspaceOffset.append(Nex_idx* self.Nx* self.Ny)
            self.TotalKspaceSamples += Nsamp





    
    
        
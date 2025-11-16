import numpy as np
import torch
from utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions

from utils.espiritmaps import calc_espirit_maps, from_espirit_dims, to_espirit_dims
from utils.createArtifacts import randomTranslation_3D


class Data:
    def __init__(self, path_to_kspace, params, sp_device=None, t_device=None):
        self.sp_device = sp_device
        self.t_device = t_device

        self.kspace = np.load(path_to_kspace)['arr_0']
        self.kspace = torch.from_numpy(self.kspace).to(t_device)
        self.Ncha, self.Nx, self.Ny, self.Nsli = self.kspace.shape

        # Calculate ESPIRiT maps and non-corrected image
        self.smaps = from_espirit_dims(calc_espirit_maps(self.kspace, params.acs, params.kernel_width, sp_device=sp_device))
        kspace_perm = from_espirit_dims(self.kspace)
        self.img_cplx = ifftnc(kspace_perm, dims=(-4, -3, -2))
        self.image_no_moco = torch.sum(self.img_cplx*self.smaps.conj(), dim=-1).to(t_device)
        

    def simulate_rigid_motion(self, params):
        self.kspace, t_n, masks = randomTranslation_3D(
            self.img_cplx, is_2D=True, sigma=params.max_motion, seed=params.seed
        )
        self.img_cplx = ifftnc(self.kspace, dims=(-4, -3, -2)).to(self.t_device)
        self.smaps = calc_espirit_maps(to_espirit_dims(self.kspace), params.acs, params.kernel_width, sp_device = self.sp_device)
        self.smaps = from_espirit_dims(self.smaps)

        self.image_no_moco = torch.sum(self.img_cplx*self.smaps.conj(), dim=-1)

        return t_n, masks
    

        
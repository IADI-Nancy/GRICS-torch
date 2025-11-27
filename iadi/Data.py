import numpy as np
import torch
from utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions

from utils.espiritmaps import calc_espirit_maps, from_espirit_dims, to_espirit_dims
from iadi.Helpers import from_espirit_to_grics_dims, from_grics_to_espirit_dims
from iadi.EncodingOperator import EncodingOperator
from iadi.MotionOperator import MotionOperator


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

        t_x = 0 * torch.randn(Nshots, device=self.t_device)
        t_y = 0 * torch.randn(Nshots, device=self.t_device)
        s = torch.stack([t_x, t_y], dim=-1)
        phi_rot      = 0 * torch.randn(Nshots, device=self.t_device)
        alpha_x = torch.ones_like(self.image_no_moco, device=self.t_device)
        alpha_y = torch.ones_like(self.image_no_moco, device=self.t_device)
        alpha = torch.stack([alpha_x, alpha_y], dim=-1)

        self.simulate_rigid_motion_fields(t_x, t_y, phi_rot) #, rotation_center=[0, 160]
        E = EncodingOperator(self.smaps, self.TotalKspaceSamples, self.SamplingIndices, self.KspaceOffset, self.MotionOperator)
        kspace_corruped = E.forward(self.image_no_moco)
        self.kspace = kspace_corruped.reshape(params.Nex, self.Nx, self.Ny, self.Nsli, self.Ncha)
        self.img_cplx = ifftnc(self.kspace[0,:,:,:,:], dims=(0, 1, 2)).to(self.t_device)

        self.image_no_moco = torch.sum(self.img_cplx*self.smaps.conj(), dim=-1)
        return s, alpha

    
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

    def simulate_rigid_motion_fields(
        self,
        X_translations,    # shape: (Nshots,)
        Y_translations,    # shape: (Nshots,)
        Rotations,         # degrees, shape: (Nshots,)
        rotation_center=None
    ):
        """
        Creates sparse 2D rigid motion operators for each shot.
        """

        Nshots = X_translations.shape[0]

        # Rotation center (default same as MATLAB)
        if rotation_center is None:
            cx = self.Nx/2 + 1
            cy = self.Ny/2 + 1
        else:
            cx, cy = rotation_center

        cx = torch.tensor(cx, device=self.t_device, dtype=torch.float32)
        cy = torch.tensor(cy, device=self.t_device, dtype=torch.float32)

        # ----------------------------
        # Create pixel grid (X, Y)
        # ----------------------------
        coords_x = torch.arange(1, self.Nx+1, device=self.t_device, dtype=torch.float32)
        coords_y = torch.arange(1, self.Ny+1, device=self.t_device, dtype=torch.float32)

        # Same as MATLAB: X varies along columns, Y along rows
        Y, X = torch.meshgrid(coords_y, coords_x, indexing="xy")   # (Nx, Ny)

        self.MotionOperator = []                 
        self.Ux_list = []
        self.Uy_list = []

        for shot in range(Nshots):
            # ----------------------------
            # Expand translations
            # MATLAB uses inverse displacement: tx = -XTranslation(shot)
            # ----------------------------
            tx = -X_translations[shot].to(self.t_device)
            ty = -Y_translations[shot].to(self.t_device)

            # ----------------------------
            # Rotations (convert to radians, inverse like MATLAB)
            # ----------------------------
            theta = -Rotations[shot].to(self.t_device) * (torch.pi / 180)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)

            # ----------------------------
            # Shift coordinates relative to rotation center
            # ----------------------------
            X0 = X - cx
            Y0 = Y - cy

            # ----------------------------
            # Apply rotation (matrix R)
            # [ cos -sin ] [X0]
            # [ sin  cos ] [Y0]
            # ----------------------------
            X_rot = cos_t * X0 - sin_t * Y0
            Y_rot = sin_t * X0 + cos_t * Y0

            # ----------------------------
            # Apply translation and shift back from center
            # ----------------------------
            X_warped = X_rot + (tx + cx)
            Y_warped = Y_rot + (ty + cy)

            # ----------------------------
            # Final displacement field
            # ----------------------------
            Ux = X_warped - X
            Uy = Y_warped - Y

            self.Ux_list.append(Ux)
            self.Uy_list.append(Uy)

             # Create motion operator (your custom function)
            M = MotionOperator.create_sparse_motion_operator(Ux, Uy)
            self.MotionOperator.append(M)




    
    
        
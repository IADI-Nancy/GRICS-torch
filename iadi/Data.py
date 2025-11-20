import numpy as np
import torch
from utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions

from utils.espiritmaps import calc_espirit_maps, from_espirit_dims, to_espirit_dims
from utils.createArtifacts import randomTranslation_3D
from iadi.Helpers import create_sparse_motion_operator, from_espirit_to_grics_dims, from_grics_to_espirit_dims
from iadi.EncodingOperator import EncodingOperator


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
        # self.smaps = calc_espirit_maps(to_espirit_dims(self.kspace), params.acs, params.kernel_width, sp_device = self.sp_device)
        # self.smaps = from_espirit_dims(self.smaps)
    
        # ----- Copy scalar attributes -----
        # Data["N"] = N
        # Data["Nex"] = Nex
        # Data["Nshots"] = Nshots
        # Data["NshotsPerNex"] = NshotsPerNex

        X_trans = 4 * torch.randn(Nshots, device=self.t_device)
        Y_trans = 2 * torch.randn(Nshots, device=self.t_device)
        Rot      = 3 * torch.randn(Nshots, device=self.t_device)
        self.simulate_rigid_motion_fields(X_trans, Y_trans, Rot, rotation_center=[10, 30])

        E = EncodingOperator(self.smaps, self.KspaceSamplingOperator, self.TotalKspaceSamples, self.KspaceSampleOffset, self.NbKspaceSamplesPerShot, self.t_device)
        kspace_corruped = E.forward(self.image_no_moco, self.MotionOperator)
        self.kspace = kspace_corruped.reshape(params.Nex, self.Nx, self.Ny, self.Nsli, self.Ncha)
        self.img_cplx = ifftnc(self.kspace[0,:,:,:,:], dims=(0, 1, 2)).to(self.t_device)

        self.image_no_moco = torch.sum(self.img_cplx*self.smaps.conj(), dim=-1)

    
    def simulate_kspace_sampling(self, params):
        Nshots = self.Nshots
        self.KspaceSamplingMask      = []
        self.KspaceSamplingOperator  = []
        self.NbKspaceSamplesPerShot  = []
        self.KspaceSampleOffset      = []
        self.TotalKspaceSamples = 0
        # Loop over all shots
        for shot in range(Nshots):

            # MATLAB: shot_in_nex = mod(shot-1, NshotsPerNex) + 1;
            shot_in_nex = (shot % params.NshotsPerNex)

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

            self.KspaceSamplingMask.append(KspaceSamplingMask)

            # ----- Build sparse sampling operator -----
            # MATLAB: nnz_idx = find(KspaceSamplingMask(:))
            nnz_idx = torch.nonzero(KspaceSamplingMask.flatten(), as_tuple=True)[0]
            Nsamp   = nnz_idx.numel()

            # MATLAB: sparse(1:N^2, 1:N^2, KspaceSamplingMask(:), N^2, N^2)
            # PyTorch equivalent: sparse diagonal matrix with mask values
            N2 = self.Nx * self.Ny
            diag_indices = torch.arange(N2, device=self.t_device)

            SamplingOp = torch.sparse_coo_tensor(
                indices=torch.vstack([diag_indices, diag_indices]),
                values=KspaceSamplingMask.flatten(),
                size=(N2, N2),
                device=self.t_device,
                dtype=torch.complex64
            ).coalesce()

            # MATLAB: select only sampled rows → operator of size Nsamp × N²
            # In PyTorch, we filter rows by multiplying with a gather-like mask
            # More efficient: slice via indexing of rows
            # => Extract only rows corresponding to nnz_idx
            SamplingOp = SamplingOp.index_select(0, nnz_idx)

            self.KspaceSamplingOperator.append(SamplingOp)
            self.NbKspaceSamplesPerShot.append(Nsamp)

            # ----- Update shot offsets -----
            self.KspaceSampleOffset.append(self.TotalKspaceSamples)
            self.TotalKspaceSamples += Nsamp

    def simulate_rigid_motion_fields(
        self,
        X_translations,    # shape: (Nshots,)
        Y_translations,    # shape: (Nshots,)
        Rotations,         # degrees, shape: (Nshots,)
        rotation_center=None
    ):
        """
        Replicates the MATLAB code: creates 2D rigid-motion displacement fields
        Ux, Uy for each shot.
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
        print(X.shape, Y.shape)
        #   Note: PyTorch returns shape (Nx, Ny)

        self.MotionOperator = []          # like Data.MotionOperator{shot}
        self.Ux_list = []                 # if you want to store fields
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
            M = create_sparse_motion_operator(Ux, Uy)
            self.MotionOperator.append(M)




    
    
        
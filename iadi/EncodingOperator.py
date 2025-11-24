import torch
import numpy as np
from utils.show_slice import show_slice
import matplotlib.pyplot as plt
from  utils.fftnc import fftnc, ifftnc

class EncodingOperator:
    """
    MRI encoding operator.

    Methods:
    - __init__(smaps, Nsamples, SamplingIndices, KspaceOffset, motionOperator=None)
    - forward(x)   : forward operator (image -> k-space)
    - backward(y)  : adjoint operator (k-space -> image)
    """

    def __init__(self, smaps, Nsamples, SamplingIndices, KspaceOffset, motionOperator=None):
        self.device = smaps.device
        self.smaps = smaps
        self.KspaceOffset = KspaceOffset
        self.Nsamples = Nsamples
        self.SamplingIndices = SamplingIndices
        if motionOperator is None:
            Nshots = len(SamplingIndices)
            Nx, Ny, Nsli, _ = smaps.shape
            Npix = Nx * Ny

            # build a sparse identity (Nx*Ny x Nx*Ny) and reuse for each shot
            idx = torch.arange(Npix, device=self.device)
            indices = torch.stack([idx, idx], dim=0)            # shape (2, Npix)
            vals = torch.ones(Npix, dtype=torch.complex64, device=self.device)
            I_sparse = torch.sparse_coo_tensor(indices, vals, (Npix, Npix),
                                               dtype=torch.complex64,
                                               device=self.device).coalesce()

            self.motionOperator = [I_sparse for _ in range(Nshots)]
        else:
            self.motionOperator = motionOperator
        

    def set_motion_operator(self, motionOperator):
        self.motionOperator = motionOperator

    def forward(self, image):
         # ---- Sizes ----
        Nx, Ny, Nsli, Ncoils = self.smaps.shape
        Nshots         = len(self.SamplingIndices)
        KspaceData = torch.zeros((self.Nsamples, Ncoils), dtype=torch.complex64, device=self.device)

        # ---- Loop over shots ----
        for shot in range(Nshots):
            SamplingIndices = self.SamplingIndices[shot]
            KspaceOffset = self.KspaceOffset[shot]
            MotionOp = self.motionOperator[shot]

            # Apply motion operator -> reshape to image
            WarpedImage = (MotionOp @ image.flatten()).reshape(Nx, Ny)

            # ---- Loop over coils ----
            for coil in range(Ncoils):
                # Coil sensitivity
                WarpedImageSeenByCoil = WarpedImage * self.smaps[:, :, :, coil].squeeze(-1)

                # Fourier encoding: fftshift(fft2(ifftshift)) in 2D
                WarpedImageFT = fftnc(WarpedImageSeenByCoil, dims=(0, 1))

                # Sampling operator
                KspaceData[KspaceOffset + SamplingIndices, coil] = WarpedImageFT.flatten()[SamplingIndices]

        return KspaceData.flatten()

        

    def adjoint(self, KspaceData):
        device = self.device
        Nx, Ny, Nsli, Ncoils = self.smaps.shape
        Nshots = len(self.SamplingIndices)
        KspaceData = KspaceData.reshape(self.Nsamples, Ncoils)
        Image = torch.zeros((Nx, Ny), dtype=torch.complex64, device=device)

        for shot in range(Nshots):
            WarpedImage = torch.zeros((Nx, Ny), dtype=torch.complex64, device=device)
            SamplingIndices = self.SamplingIndices[shot]
            KspaceOffset = self.KspaceOffset[shot]

            for coil in range(Ncoils):
                # Sampling operator
                FullKspaceDataCoil = torch.zeros(Nx*Ny, dtype=KspaceData.dtype, device=self.device)
                FullKspaceDataCoil[SamplingIndices] = KspaceData[KspaceOffset + SamplingIndices, coil]
                FullKspaceDataCoil = FullKspaceDataCoil.reshape(Nx, Ny)
                # Adjoint FFT: fftshift → ifft2 → ifftshift
                image_coil = ifftnc(FullKspaceDataCoil, dims=(0, 1))

                # Adjoint coil sensitivity: multiply by conj(smap)
                smap = self.smaps[:, :, :, coil].squeeze()
                WarpedImage += image_coil * torch.conj(smap)

            # Adjoint motion operator
            MotionOp = self.motionOperator[shot].transpose(0, 1)
            Unwarped = MotionOp @ WarpedImage.reshape(-1)
            Unwarped = Unwarped.reshape(Nx, Ny)

            # Accumulate into full image
            Image += Unwarped

        return Image.flatten()
    
    def normal(self, image):
        return self.adjoint(self.forward(image))
        

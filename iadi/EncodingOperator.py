import torch
import numpy as np
from utils.show_slice import show_slice
import matplotlib.pyplot as plt
from  utils.fftnc import fftnc, ifftnc

class EncodingOperator:
    """
    MRI encoding operator.

    Methods:
    - __init__(sigmas, masks=None, image_shape=None, device=None)
    - forward(x, M)   : forward operator (image -> k-space), M - motion operator
    - backward(y, M)  : adjoint operator (k-space -> image), M - motion operator
    """

    def __init__(self, smaps, Nsamples, SamplingIndices, KspaceOffset, device=None):
        self.smaps = smaps
        self.KspaceOffset = KspaceOffset
        self.Nsamples = Nsamples
        self.SamplingIndices = SamplingIndices
        self.device = device

    def forward(self, image, motionOperator):
         # ---- Sizes ----
        Nx, Ny, Nsli, Ncoils = self.smaps.shape
        Nshots         = len(self.SamplingIndices)
        KspaceData = torch.zeros((self.Nsamples, Ncoils), dtype=torch.complex64, device=self.device)

        # ---- Loop over shots ----
        for shot in range(Nshots):
            SamplingIndices = self.SamplingIndices[shot]
            KspaceOffset = self.KspaceOffset[shot]
            MotionOp = motionOperator[shot]

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

        

    def backward(self, KspaceData, motionOperator):
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
                # 1) Sampling operator
                FullKspaceDataCoil = torch.zeros(Nx*Ny, dtype=KspaceData.dtype, device=self.device)
                FullKspaceDataCoil[SamplingIndices] = KspaceData[KspaceOffset + SamplingIndices, coil]
                FullKspaceDataCoil = FullKspaceDataCoil.reshape(Nx, Ny)
                # 2) Adjoint FFT: fftshift → ifft2 → ifftshift
                image_coil = ifftnc(FullKspaceDataCoil, dims=(0, 1))

                # 3) Adjoint coil sensitivity: multiply by conj(smap)
                smap = self.smaps[:, :, :, coil].squeeze()
                WarpedImage += image_coil * torch.conj(smap)

            # 4) Adjoint motion operator
            MotionOp = motionOperator[shot].transpose(0, 1)
            Unwarped = MotionOp @ WarpedImage.reshape(-1)
            Unwarped = Unwarped.reshape(Nx, Ny)

            # Accumulate into full image
            Image += Unwarped

        return Image.flatten()
    
    def normal(self, image, motionOperator):
        return self.backward(self.forward(image, motionOperator), motionOperator)
        

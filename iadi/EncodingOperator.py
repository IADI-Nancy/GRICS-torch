import torch
import numpy as np
from utils.show_slice import show_slice
import matplotlib.pyplot as plt
from  utils.fftnc import fftnc, ifftnc
debug_folder = "debug_outputs/"

def show_slice_and_save(image, image_name):
    show_slice(image, max_images=1, headline=image_name)
    plt.savefig(debug_folder + image_name + '.png')

class EncodingOperator:
    """
    MRI encoding operator.

    Methods:
    - __init__(sigmas, masks=None, image_shape=None, device=None)
    - E(x)   : forward operator (image -> k-space)
    - Eh(y)  : adjoint operator (k-space -> image)
    """

    def __init__(self, smaps, Nsamples, SamplingIndices, KspaceOffset, device=None):
        self.smaps = smaps
        self.KspaceOffset = KspaceOffset
        self.Nsamples = Nsamples
        self.SamplingIndices = SamplingIndices
        self.device = device

    def forward(self, image, motionOperator):
        """
        Apply forward encoding: multiply image by coil sensitivities, FFT -> k-space,
        and apply mask if provided.

        x: real/complex torch tensor with shape (Nx, Ny, Nz) or (Nx, Ny, Nz, 1)
        returns: k-space tensor with shape (Nx, Ny, Nz, nCha)
        """
         # ---- Sizes ----
        Nx, Ny, Nsli, Ncoils = self.smaps.shape
        Nshots         = len(self.SamplingIndices)
        KspaceData = torch.zeros((self.Nsamples, Ncoils), dtype=torch.complex64, device=self.device)

        # ---- Loop over shots ----
        for shot in range(Nshots):
            SamplingIndices = self.SamplingIndices[shot]
            KspaceOffset = self.KspaceOffset[shot]
            MotionOp = motionOperator[shot]
            # kspaceOffset = 

            # Apply motion operator -> reshape to image
            WarpedImage = (MotionOp @ image.flatten()).reshape(Nx, Ny)
            show_slice_and_save(WarpedImage.unsqueeze(axis=-1), 'warped_image')

            # ---- Loop over coils ----
            for coil in range(Ncoils):
                # Coil sensitivity
                WarpedImageSeenByCoil = WarpedImage * self.smaps[:, :, :, coil].squeeze(-1)
                show_slice_and_save(WarpedImageSeenByCoil.unsqueeze(axis=-1), 'warped_image_by_coil')

                # Fourier encoding: fftshift(fft2(ifftshift)) in 2D
                WarpedImageFT = fftnc(WarpedImageSeenByCoil, dims=(0, 1))

                # Sampling operator
                KspaceData[KspaceOffset + SamplingIndices, coil] = WarpedImageFT.flatten()[SamplingIndices]

        return KspaceData.flatten()

        

    def backward(self, KspaceData, motionOperator):
        """
        Apply adjoint (Hermitian) encoding: mask -> iFFT -> multiply by conj(sens) and sum over coils.

        y: k-space tensor shape (Nx, Ny, Nz, nCha)
        returns: image tensor shape (Nx, Ny, Nz) (complex)
        """
        device = self.device
        Nx, Ny, Nsli, Ncoils = self.smaps.shape
        Nshots = len(self.SamplingIndices)

        # reshape k-space to (Nsamples, Ncoils)
        KspaceData = KspaceData.reshape(self.Nsamples, Ncoils)

        # allocate output image
        Image = torch.zeros((Nx, Ny), dtype=torch.complex64, device=device)

        # ===========================================================
        # Loop over shots
        # ===========================================================
        for shot in range(Nshots):
            # accumulator for coils for this shot
            WarpedImage = torch.zeros((Nx, Ny), dtype=torch.complex64, device=device)

            # extract samples for this shot (slice rows)
            # shot_inds = KspaceSampleOffset + torch.arange(0, Nsamples_in_shot, device=device)
            SamplingIndices = self.SamplingIndices[shot]
            KspaceOffset = self.KspaceOffset[shot]

            # =======================================================
            # Loop over coils
            # =======================================================
            for coil in range(Ncoils):
                # 1) Sampling operator
                FullKspaceDataCoil = torch.zeros(Nx*Ny, dtype=KspaceData.dtype, device=self.device)
                FullKspaceDataCoil[SamplingIndices] = KspaceData[KspaceOffset + SamplingIndices, coil]
                FullKspaceDataCoil = FullKspaceDataCoil.reshape(Nx, Ny)
                # 2) Adjoint FFT: fftshift → ifft2 → ifftshift
                image_coil = ifftnc(FullKspaceDataCoil, dims=(0, 1))
                # Kshift = torch.fft.ifftshift(FullKspaceDataCoil, dim=(0, 1))
                # ImCoil = torch.fft.ifft2(Kshift, norm="forward")
                # ImCoil = torch.fft.fftshift(ImCoil, dim=(0, 1)) * Nx   # MATLAB scaling

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
        

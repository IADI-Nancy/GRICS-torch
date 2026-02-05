import torch
from  src.utils.fftnc import fftnc, ifftnc

class EncodingOperator:
    """
    MRI encoding operator.

    Methods:
    - __init__(smaps, Nsamples, SamplingIndices, KspaceOffset, motionOperator=None)
    - forward(x)   : forward operator (image -> k-space)
    - adjoint(y)  : adjoint operator (k-space -> image)
    """

    def __init__(self, smaps, Nsamples, SamplingIndices, KspaceOffset, motionOperator):
        self.device = smaps.device
        self.smaps = smaps
        self.KspaceOffset = KspaceOffset
        self.Nsamples = Nsamples
        self.SamplingIndices = SamplingIndices
        self.motionOperator = motionOperator

    def forward(self, image):
         # ---- Sizes ----
        Ncoils, Nx, Ny, Nsli = self.smaps.shape
        Nshots         = len(self.SamplingIndices)
        KspaceData = torch.zeros((Ncoils, self.Nsamples), dtype=torch.complex64, device=self.device)

        # ---- Loop over shots ----
        for shot in range(Nshots):
            SamplingIndices = self.SamplingIndices[shot]
            KspaceOffset = self.KspaceOffset[shot]
            MotionOp = self.motionOperator.get_sparse_operator(shot)

            # Apply motion operator -> reshape to image
            WarpedImage = (MotionOp @ image.flatten()).reshape(Nx, Ny)

            # ---- Loop over coils ----
            for coil in range(Ncoils):
                # Coil sensitivity
                WarpedImageSeenByCoil = WarpedImage * self.smaps[coil].squeeze(-1)

                # Fourier encoding: fftshift(fft2(ifftshift)) in 2D
                WarpedImageFT = fftnc(WarpedImageSeenByCoil, dims=(0, 1))

                # Sampling operator
                KspaceData[coil, KspaceOffset + SamplingIndices] = WarpedImageFT.flatten()[SamplingIndices]

        return KspaceData.flatten()

        

    def adjoint(self, KspaceData):
        device = self.device
        Ncoils, Nx, Ny, Nsli = self.smaps.shape
        Nshots = len(self.SamplingIndices)
        KspaceData = KspaceData.reshape(Ncoils, self.Nsamples)
        Image = torch.zeros((Nx, Ny), dtype=torch.complex64, device=device)

        for shot in range(Nshots):
            WarpedImage = torch.zeros((Nx, Ny), dtype=torch.complex64, device=device)
            SamplingIndices = self.SamplingIndices[shot]
            KspaceOffset = self.KspaceOffset[shot]
            if SamplingIndices.numel() == 0:
                continue

            for coil in range(Ncoils):
                # Sampling operator
                FullKspaceDataCoil = torch.zeros(Nx*Ny, dtype=KspaceData.dtype, device=self.device)
                FullKspaceDataCoil[SamplingIndices] = KspaceData[coil, KspaceOffset + SamplingIndices]
                FullKspaceDataCoil = FullKspaceDataCoil.reshape(Nx, Ny)
                # Adjoint FFT: fftshift → ifft2 → ifftshift
                image_coil = ifftnc(FullKspaceDataCoil, dims=(0, 1))

                # Adjoint coil sensitivity: multiply by conj(smap)
                smap = self.smaps[coil].squeeze()
                WarpedImage += image_coil * torch.conj(smap)

            # Adjoint motion operator
            MotionOp = self.motionOperator.get_sparse_operator(shot).coalesce().transpose(0, 1)
            Unwarped = MotionOp @ WarpedImage.reshape(-1)
            Unwarped = Unwarped.reshape(Nx, Ny)

            # Accumulate into full image
            Image += Unwarped

        return Image.flatten()
    
    def normal(self, image):
        return self.adjoint(self.forward(image))
        

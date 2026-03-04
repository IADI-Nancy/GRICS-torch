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

    def __init__(self, smaps, Nsamples, SamplingIndices, Nex, motionOperator):
        self.device = smaps.device
        self.smaps = smaps
        self.Nex = Nex
        self.Nsamples = Nsamples
        self.SamplingIndices = SamplingIndices
        self.motionOperator = motionOperator

    def forward(self, image):
        # ---- Sizes ----
        Ncoils, Nx, Ny, Nz = self.smaps.shape
        N_motion_states = len(self.SamplingIndices[0])  # [Nex][N_motion_states]
        KspaceData = torch.zeros((Ncoils, self.Nex, self.Nsamples), dtype=torch.complex128, device=self.device)

        if Nz > 1:
            image = image.reshape(self.Nex, Nx, Ny, Nz)
            fft_dims = (0, 1, 2)
        else:
            image = image.reshape(self.Nex, Nx, Ny)
            fft_dims = (0, 1)

        # ---- Loop over motion states ----
        for motion_state in range(N_motion_states):
            MotionOp = self.motionOperator._get_sparse_operator(motion_state)

            for nex in range(self.Nex):
                SamplingIndices = self.SamplingIndices[nex][motion_state]
                image_nex = image[nex]
                WarpedImage = (MotionOp @ image_nex.flatten()).reshape(image_nex.shape)

                # ---- Loop over coils ----
                for coil in range(Ncoils):
                    # Coil sensitivity
                    smap = self.smaps[coil] if Nz > 1 else self.smaps[coil].squeeze(-1)
                    WarpedImageSeenByCoil = WarpedImage * smap

                    # Fourier encoding
                    WarpedImageFT = fftnc(WarpedImageSeenByCoil, dims=fft_dims)

                    # Sampling operator
                    KspaceData[coil, nex, SamplingIndices] = WarpedImageFT.flatten()[SamplingIndices]

        return KspaceData.flatten()

        

    def adjoint(self, KspaceData):
        device = self.device
        Ncoils, Nx, Ny, Nz = self.smaps.shape
        N_motion_states = len(self.SamplingIndices[0])  # assuming SamplingIndices is a list of lists with shape [Nex][N_motion_states]
        KspaceData = KspaceData.reshape(Ncoils, self.Nex, self.Nsamples)
        if Nz > 1:
            Image = torch.zeros((self.Nex, Nx, Ny, Nz), dtype=torch.complex128, device=device)
            fft_dims = (0, 1, 2)
            kspace_shape = (Nx, Ny, Nz)
        else:
            Image = torch.zeros((self.Nex, Nx, Ny), dtype=torch.complex128, device=device)
            fft_dims = (0, 1)
            kspace_shape = (Nx, Ny)

        
        for nex in range(self.Nex):

            for motion_state in range(N_motion_states):
                WarpedImage = torch.zeros(kspace_shape, dtype=torch.complex128, device=device)
                SamplingIndices = self.SamplingIndices[nex][motion_state]
                if SamplingIndices.numel() == 0:
                    continue
                
                for coil in range(Ncoils):
                    # Sampling operator
                    KspaceDataCoilNex = torch.zeros(Nx * Ny * Nz, dtype=KspaceData.dtype, device=self.device)
                    KspaceDataCoilNex[SamplingIndices] = KspaceData[coil, nex, SamplingIndices]
                    KspaceDataCoilNex = KspaceDataCoilNex.reshape(kspace_shape)
                    # Adjoint FFT
                    image_coil = ifftnc(KspaceDataCoilNex, dims=fft_dims)

                    # Adjoint coil sensitivity: multiply by conj(smap)
                    smap = self.smaps[coil] if Nz > 1 else self.smaps[coil].squeeze()
                    WarpedImage += image_coil * torch.conj(smap)

                # Adjoint motion operator
                MotionOp = self.motionOperator._get_sparse_operator(motion_state).coalesce().transpose(0, 1)
                Unwarped = MotionOp @ WarpedImage.reshape(-1)
                Unwarped = Unwarped.reshape(kspace_shape)

                # Accumulate into full image
                Image[nex] += Unwarped

        return Image.flatten()
    
    def normal(self, image):
        return self.adjoint(self.forward(image))
        

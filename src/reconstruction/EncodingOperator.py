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
        Ncoils, Nx, Ny, Nsli = self.smaps.shape
        N_mot_states         = len(self.SamplingIndices[0])  # assuming SamplingIndices is a list of lists with shape [Nex][N_mot_states]
        KspaceData = torch.zeros((Ncoils, self.Nex, self.Nsamples), dtype=torch.complex128, device=self.device)
        image = image.reshape(self.Nex, Nx, Ny)

        # ---- Loop over motion states ----
        for motion_state in range(N_mot_states):
            
            MotionOp = self.motionOperator.get_sparse_operator(motion_state)

            for nex in range(self.Nex):
                SamplingIndices = self.SamplingIndices[nex][motion_state]
                image_nex = image[nex]
                # Apply motion operator -> reshape to image
                WarpedImage = (MotionOp @ image_nex.flatten()).reshape(Nx, Ny)

                # ---- Loop over coils ----
                for coil in range(Ncoils):
                    # Coil sensitivity
                    WarpedImageSeenByCoil = WarpedImage * self.smaps[coil].squeeze(-1)

                    # Fourier encoding: fftshift(fft2(ifftshift)) in 2D
                    WarpedImageFT = fftnc(WarpedImageSeenByCoil, dims=(0, 1))

                    # Sampling operator
                    KspaceData[coil, nex, SamplingIndices] = WarpedImageFT.flatten()[SamplingIndices]

        return KspaceData.flatten()

        

    def adjoint(self, KspaceData):
        device = self.device
        Ncoils, Nx, Ny, Nsli = self.smaps.shape
        N_mot_states = len(self.SamplingIndices[0])  # assuming SamplingIndices is a list of lists with shape [Nex][N_mot_states]
        KspaceData = KspaceData.reshape(Ncoils, self.Nex, self.Nsamples)
        Image = torch.zeros((self.Nex, Nx, Ny), dtype=torch.complex128, device=device)

        
        for nex in range(self.Nex):

            for motion_state in range(N_mot_states):
                WarpedImage = torch.zeros((Nx, Ny), dtype=torch.complex128, device=device)
                SamplingIndices = self.SamplingIndices[nex][motion_state]
                if SamplingIndices.numel() == 0:
                    continue
                
                for coil in range(Ncoils):
                    # Sampling operator
                    KspaceDataCoilNex = torch.zeros(Nx*Ny, dtype=KspaceData.dtype, device=self.device)
                    KspaceDataCoilNex[SamplingIndices] = KspaceData[coil, nex, SamplingIndices]
                    KspaceDataCoilNex = KspaceDataCoilNex.reshape(Nx, Ny)
                    # Adjoint FFT: fftshift → ifft2 → ifftshift
                    image_coil = ifftnc(KspaceDataCoilNex, dims=(0, 1))

                    # Adjoint coil sensitivity: multiply by conj(smap)
                    smap = self.smaps[coil].squeeze()
                    WarpedImage += image_coil * torch.conj(smap)

                # Adjoint motion operator
                MotionOp = self.motionOperator.get_sparse_operator(motion_state).coalesce().transpose(0, 1)
                Unwarped = MotionOp @ WarpedImage.reshape(-1)
                Unwarped = Unwarped.reshape(Nx, Ny)

                # Accumulate into full image
                Image[nex] += Unwarped

        return Image.flatten()
    
    def normal(self, image):
        return self.adjoint(self.forward(image))
        

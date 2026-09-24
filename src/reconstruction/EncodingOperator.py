import torch
from src.utils.fftnc import fftnc, ifftnc

class EncodingOperator:
    """
    MRI encoding operator.

    SamplingIndices describes complete readout lines, assigned to disjoint
    motion states within each repetition, as produced by Sampling.

    Methods:
    - __init__(smaps, Nsamples, SamplingIndices, KspaceOffset, motionOperator=None)
    - forward(x)   : forward operator (image -> k-space)
    - adjoint(y)  : adjoint operator (k-space -> image)
    - normal(x)   : normal operator (image -> image)
    """

    def __init__(self, smaps, Nsamples, SamplingIndices, Nex, motionOperator):
        self.device = smaps.device
        self.smaps = smaps
        self.Nex = Nex
        self.Nsamples = Nsamples
        self.SamplingIndices = SamplingIndices
        self.motionOperator = motionOperator
        # Sampling is fixed throughout a CG solve; build the phase masks once for optimized normal operator.
        self._phase_encoding_masks = self._build_phase_encoding_masks()

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
    
    def _build_phase_encoding_masks(self):
        """Extract each state's phase mask from its complete readout lines."""
        Ncoils, Nx, Ny, Nz = self.smaps.shape
        phase_shape = (Ny, Nz) if Nz > 1 else (Ny,)
        masks = []
        for nex in range(self.Nex):
            nex_masks = []
            for motion_state in range(len(self.SamplingIndices[nex])):
                SamplingIndices = self.SamplingIndices[nex][motion_state]
                if SamplingIndices.numel() == 0:
                    nex_masks.append(None)
                    continue
                # Flattened indices are x * (Ny * Nz) + y * Nz + z.
                # The x=0 samples identify every acquired phase line, regardless
                # of index ordering. No full spatial mask is needed.
                phase_indices = SamplingIndices[SamplingIndices < Ny * Nz]
                phase_mask = torch.zeros(Ny * Nz, dtype=torch.bool, device=self.device)
                phase_mask[phase_indices] = True
                nex_masks.append(phase_mask.reshape(phase_shape))
            masks.append(nex_masks)
        return masks

    def normal(self, image):
        """Apply E^H E using only the phase-encoding FFTs (y, or y and z).

        For one coil/state, E = P F_x F_phase S W. Complete readout
        lines make P^H P independent of x, so it commutes with F_x:

            F_x^H (P^H P) F_x = P^H P.

        Thus E^H E = W^H S^H F_phase^H (P^H P) F_phase S W.
        Coil weighting and image warping stay on their original sides of
        the FFTs; neither needs to commute with the sampling mask.
        """
        Ncoils, Nx, Ny, Nz = self.smaps.shape
        kspace_shape = (Nx, Ny, Nz) if Nz > 1 else (Nx, Ny)
        fft_dims = (1, 2) if Nz > 1 else (1,)
        image = image.reshape(self.Nex, *kspace_shape)
        Image = torch.zeros(image.shape, dtype=torch.complex128, device=self.device)

        for nex, nex_masks in enumerate(self._phase_encoding_masks):
            for motion_state, phase_mask in enumerate(nex_masks):
                if phase_mask is None:
                    continue
                MotionOp = self.motionOperator._get_sparse_operator(motion_state)
                image_nex = image[nex]
                WarpedImage = (MotionOp @ image_nex.flatten()).reshape(kspace_shape)
                ImageSum = torch.zeros(kspace_shape, dtype=torch.complex128, device=self.device)

                for coil in range(Ncoils):
                    smap = self.smaps[coil] if Nz > 1 else self.smaps[coil].squeeze(-1)
                    WarpedImageSeenByCoil = WarpedImage * smap
                    WarpedImageFT = fftnc(WarpedImageSeenByCoil, dims=fft_dims)
                    # Broadcast the phase mask across all readout positions.
                    KspaceDataCoilNex = WarpedImageFT * phase_mask
                    image_coil = ifftnc(KspaceDataCoilNex, dims=fft_dims)
                    ImageSum += image_coil * torch.conj(smap)

                # Interpolation weights are real, so transpose is the adjoint,
                # matching the existing general encoding operator.
                MotionOp = MotionOp.coalesce().transpose(0, 1)
                Unwarped = (MotionOp @ ImageSum.flatten()).reshape(kspace_shape)
                Image[nex] += Unwarped

        return Image.flatten()

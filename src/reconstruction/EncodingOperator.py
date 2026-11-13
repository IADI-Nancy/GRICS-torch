import torch
from src.utils.fftnc import fftnc, ifftnc

class EncodingOperator:
    """
    MRI encoding of one shared image into all acquisition repetitions.

    SamplingIndices describes complete readout lines, assigned to disjoint
    motion states within each repetition, as produced by MotionSamplingPreparer.
    Forward encoding uses the same image for every repetition; the adjoint
    sums their contributions without averaging. Nex counts acquisitions,
    while the reconstructed image has only spatial axes.

    Methods:
    - forward(x)   : forward operator (image -> k-space)
    - adjoint(y)  : adjoint operator (k-space -> image)
    - normal(x)   : normal operator (image -> image)

    Implementation notes (performance; results identical to the reference
    implementation up to floating-point summation order):
    - the coil axis is processed as one batched tensor (one FFT call per state
      instead of one per coil);
    - the warped image of a state is computed once and reused for every
      repetition (it does not depend on the repetition);
    - the adjoint warp uses the transposed sparse operator cached inside
      MotionOperator instead of re-transposing the operator at every call;
    - contributions of all repetitions are accumulated before the adjoint
      warp (the warp is linear), so each state costs one adjoint warp.
    """

    def __init__(self, smaps, Nsamples, SamplingIndices, Nex, motionOperator):
        self.device = smaps.device
        self.smaps = smaps
        self.Nex = Nex
        self.Nsamples = Nsamples
        self.SamplingIndices = SamplingIndices
        self.motionOperator = motionOperator
        Ncoils, Nx, Ny, Nz = smaps.shape
        self._is_3d = int(Nz) > 1
        self._shape = (Nx, Ny, Nz) if self._is_3d else (Nx, Ny)
        # Coil maps with the coil axis first and spatial axes matching the image.
        self._smaps_b = smaps if self._is_3d else smaps.squeeze(-1)
        self._smaps_b_conj = self._smaps_b.conj()
        self._fft_dims = (1, 2, 3) if self._is_3d else (1, 2)
        self._phase_fft_dims = (2, 3) if self._is_3d else (2,)
        # Sampling is fixed throughout a CG solve; build the phase masks once for optimized normal operator.
        self._phase_encoding_masks = self._build_phase_encoding_masks()

    def _warp(self, image_flat, motion_state):
        return (self.motionOperator._get_sparse_operator(motion_state) @ image_flat).reshape(self._shape)

    def _unwarp(self, image, motion_state):
        return (self.motionOperator._get_sparse_operator_T(motion_state) @ image.reshape(-1)).reshape(self._shape)

    def forward(self, image):
        Ncoils = self.smaps.shape[0]
        N_motion_states = len(self.SamplingIndices[0])  # [Nex][N_motion_states]
        KspaceData = torch.zeros((Ncoils, self.Nex, self.Nsamples), dtype=torch.complex128, device=self.device)
        image_flat = image.reshape(-1)

        for motion_state in range(N_motion_states):
            WarpedImage = self._warp(image_flat, motion_state)
            # Coil sensitivity, Fourier encoding of all coils at once.
            WarpedImageFT = fftnc(WarpedImage.unsqueeze(0) * self._smaps_b, dims=self._fft_dims).reshape(Ncoils, -1)
            for nex in range(self.Nex):
                SamplingIndices = self.SamplingIndices[nex][motion_state]
                if SamplingIndices.numel() == 0:
                    continue
                # Sampling operator
                KspaceData[:, nex, SamplingIndices] = WarpedImageFT[:, SamplingIndices]

        return KspaceData.flatten()

    def adjoint(self, KspaceData):
        Ncoils = self.smaps.shape[0]
        N_motion_states = len(self.SamplingIndices[0])
        KspaceData = KspaceData.reshape(Ncoils, self.Nex, self.Nsamples)
        n_voxels = self._smaps_b[0].numel()
        Image = torch.zeros(self._shape, dtype=torch.complex128, device=self.device)

        for motion_state in range(N_motion_states):
            WarpedImage = None
            for nex in range(self.Nex):
                SamplingIndices = self.SamplingIndices[nex][motion_state]
                if SamplingIndices.numel() == 0:
                    continue
                # Sampling operator (all coils at once)
                KspaceDataNex = torch.zeros((Ncoils, n_voxels), dtype=KspaceData.dtype, device=self.device)
                KspaceDataNex[:, SamplingIndices] = KspaceData[:, nex, SamplingIndices]
                # Adjoint FFT and adjoint coil sensitivity, summed over coils
                image_coils = ifftnc(KspaceDataNex.reshape(Ncoils, *self._shape), dims=self._fft_dims)
                contribution = (image_coils * self._smaps_b_conj).sum(dim=0)
                WarpedImage = contribution if WarpedImage is None else WarpedImage + contribution
            if WarpedImage is None:
                continue
            # Adjoint motion operator (interpolation weights are real, so the transpose is the adjoint)
            Image += self._unwarp(WarpedImage, motion_state)

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
        N_motion_states = len(self.SamplingIndices[0])
        image_flat = image.reshape(-1)
        Image = torch.zeros(self._shape, dtype=torch.complex128, device=self.device)

        for motion_state in range(N_motion_states):
            WarpedImageFT = None
            ImageSum = None
            for nex in range(self.Nex):
                phase_mask = self._phase_encoding_masks[nex][motion_state]
                if phase_mask is None:
                    continue
                if WarpedImageFT is None:
                    WarpedImage = self._warp(image_flat, motion_state)
                    WarpedImageFT = fftnc(WarpedImage.unsqueeze(0) * self._smaps_b, dims=self._phase_fft_dims)
                # Broadcast the phase mask across all readout positions and coils.
                image_coils = ifftnc(WarpedImageFT * phase_mask, dims=self._phase_fft_dims)
                contribution = (image_coils * self._smaps_b_conj).sum(dim=0)
                ImageSum = contribution if ImageSum is None else ImageSum + contribution
            if ImageSum is None:
                continue
            Image += self._unwarp(ImageSum, motion_state)

        return Image.flatten()

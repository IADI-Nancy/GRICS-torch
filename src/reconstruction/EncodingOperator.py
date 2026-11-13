import os
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
        # Arithmetic precision follows the motion operator (complex64 or complex128); inputs are cast on
        # entry and outputs are returned in complex128 so that the solver is unaffected.
        self.compute_dtype = getattr(motionOperator, 'compute_dtype', torch.complex128)
        # Coil maps with the coil axis first and spatial axes matching the image.
        self._smaps_b = (smaps if self._is_3d else smaps.squeeze(-1)).to(self.compute_dtype)
        self._smaps_b_conj = self._smaps_b.conj()
        self._fft_dims = (1, 2, 3) if self._is_3d else (1, 2)
        self._phase_fft_dims = (2, 3) if self._is_3d else (2,)
        # Sampling is fixed throughout a CG solve; build the phase masks once for optimized normal operator.
        self._phase_encoding_masks = self._build_phase_encoding_masks()
        # State batching (gather backend): all motion states go through one FFT / gather / index_add per call.
        # State batching pays off for launch-bound (small) problems and costs memory traffic on large ones:
        # default = batch when a state has at most GRICS_STATE_BATCH_MAX_VOXELS voxels (2^19), gather backend only.
        _mode = os.environ.get('GRICS_STATE_BATCH', 'auto')
        _small = self._smaps_b[0].numel() <= int(os.environ.get('GRICS_STATE_BATCH_MAX_VOXELS', str(2 ** 19)))
        self._batched = (getattr(motionOperator, 'warp_backend', 'sparse') == 'gather'
                         and (_mode == '1' or (_mode == 'auto' and _small)))
        if self._batched:
            self._stacked_masks = self._stack_phase_masks()

    def _warp(self, image_flat, motion_state):
        return self.motionOperator.warp(image_flat, motion_state).reshape(self._shape)

    def _unwarp(self, image, motion_state):
        return self.motionOperator.warp_T(image.reshape(-1), motion_state).reshape(self._shape)

    def _stack_phase_masks(self):
        """Per repetition: boolean mask [Nm, 1, Ny(, Nz)] (empty states -> all False) for the batched normal operator."""
        stacked = []
        for nex_masks in self._phase_encoding_masks:
            shape = next((m.shape for m in nex_masks if m is not None), None)
            if shape is None:
                stacked.append(None); continue
            masks = [m if m is not None else torch.zeros(shape, dtype=torch.bool, device=self.device) for m in nex_masks]
            # [Nm, 1 (coil), 1 (readout axis), Ny(, Nz)] so that it broadcasts against [Nm, Nc, Nx, Ny(, Nz)]
            stacked.append(torch.stack(masks).unsqueeze(1).unsqueeze(1))
        return stacked

    def _chunks(self):
        Ncoils = self.smaps.shape[0]
        return self.motionOperator.state_chunks(Ncoils * self._smaps_b[0].numel() * self._smaps_b.element_size())

    def forward(self, image):
        if self._batched:
            return self._forward_batched(image)
        Ncoils = self.smaps.shape[0]
        N_motion_states = len(self.SamplingIndices[0])  # [Nex][N_motion_states]
        KspaceData = torch.zeros((Ncoils, self.Nex, self.Nsamples), dtype=self.compute_dtype, device=self.device)
        image_flat = image.reshape(-1).to(self.compute_dtype)

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

        return KspaceData.flatten().to(torch.complex128)

    def _forward_batched(self, image):
        Ncoils = self.smaps.shape[0]
        KspaceData = torch.zeros((Ncoils, self.Nex, self.Nsamples), dtype=self.compute_dtype, device=self.device)
        image_flat = image.reshape(-1).to(self.compute_dtype)
        for s0, s1 in self._chunks():
            W = self.motionOperator.warp_all(image_flat, (s0, s1)).reshape(s1 - s0, 1, *self._shape)
            WFT = fftnc(W * self._smaps_b.unsqueeze(0), dims=tuple(d + 1 for d in self._fft_dims)).reshape(s1 - s0, Ncoils, -1)
            for motion_state in range(s0, s1):
                for nex in range(self.Nex):
                    SamplingIndices = self.SamplingIndices[nex][motion_state]
                    if SamplingIndices.numel() == 0:
                        continue
                    KspaceData[:, nex, SamplingIndices] = WFT[motion_state - s0][:, SamplingIndices]
        return KspaceData.flatten().to(torch.complex128)

    def _adjoint_batched(self, KspaceData):
        Ncoils = self.smaps.shape[0]
        KspaceData = KspaceData.reshape(Ncoils, self.Nex, self.Nsamples).to(self.compute_dtype)
        n_voxels = self._smaps_b[0].numel()
        Image = torch.zeros(n_voxels, dtype=self.compute_dtype, device=self.device)
        for s0, s1 in self._chunks():
            Y = None
            for nex in range(self.Nex):
                Kfull = torch.zeros((s1 - s0, Ncoils, n_voxels), dtype=self.compute_dtype, device=self.device)
                any_lines = False
                for motion_state in range(s0, s1):
                    SamplingIndices = self.SamplingIndices[nex][motion_state]
                    if SamplingIndices.numel() == 0:
                        continue
                    Kfull[motion_state - s0][:, SamplingIndices] = KspaceData[:, nex, SamplingIndices]
                    any_lines = True
                if not any_lines:
                    continue
                img = ifftnc(Kfull.reshape(s1 - s0, Ncoils, *self._shape), dims=tuple(d + 1 for d in self._fft_dims))
                contribution = (img * self._smaps_b_conj.unsqueeze(0)).sum(dim=1).reshape(s1 - s0, -1)
                Y = contribution if Y is None else Y + contribution
            if Y is not None:
                Image += self.motionOperator.warp_T_all(Y, (s0, s1))
        return Image.to(torch.complex128)

    def _normal_batched(self, image):
        Ncoils = self.smaps.shape[0]
        image_flat = image.reshape(-1).to(self.compute_dtype)
        n_voxels = image_flat.numel()
        Image = torch.zeros(n_voxels, dtype=self.compute_dtype, device=self.device)
        phase_dims = tuple(d + 1 for d in self._phase_fft_dims)
        for s0, s1 in self._chunks():
            W = self.motionOperator.warp_all(image_flat, (s0, s1)).reshape(s1 - s0, 1, *self._shape)
            WFT = fftnc(W * self._smaps_b.unsqueeze(0), dims=phase_dims)
            Y = None
            for nex in range(self.Nex):
                masks = self._stacked_masks[nex]
                if masks is None:
                    continue
                img = ifftnc(WFT * masks[s0:s1], dims=phase_dims)
                contribution = (img * self._smaps_b_conj.unsqueeze(0)).sum(dim=1).reshape(s1 - s0, -1)
                Y = contribution if Y is None else Y + contribution
            if Y is not None:
                Image += self.motionOperator.warp_T_all(Y, (s0, s1))
        return Image.to(torch.complex128)

    def adjoint(self, KspaceData):
        if self._batched:
            return self._adjoint_batched(KspaceData)
        Ncoils = self.smaps.shape[0]
        N_motion_states = len(self.SamplingIndices[0])
        KspaceData = KspaceData.reshape(Ncoils, self.Nex, self.Nsamples).to(self.compute_dtype)
        n_voxels = self._smaps_b[0].numel()
        Image = torch.zeros(self._shape, dtype=self.compute_dtype, device=self.device)

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

        return Image.flatten().to(torch.complex128)

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
        if self._batched:
            return self._normal_batched(image)
        N_motion_states = len(self.SamplingIndices[0])
        image_flat = image.reshape(-1).to(self.compute_dtype)
        Image = torch.zeros(self._shape, dtype=self.compute_dtype, device=self.device)

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

        return Image.flatten().to(torch.complex128)

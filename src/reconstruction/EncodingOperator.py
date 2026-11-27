import os
import torch
from src.utils.fftnc import fftnc, ifftnc
from src.reconstruction.coil_batching import CoilBatchPlanner

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
    - the coil axis is processed in automatic GPU-memory-sized FFT batches;
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
        self._smaps_b = smaps if self._is_3d else smaps.squeeze(-1)
        self._coil_batches = CoilBatchPlanner(self._smaps_b, self.compute_dtype, 'encoding')
        self._fft_dims = (1, 2, 3) if self._is_3d else (1, 2)
        self._phase_fft_dims = (2, 3) if self._is_3d else (2,)
        # Sampling is fixed throughout a CG solve; build the phase masks once for optimized normal operator.
        self._phase_encoding_masks = self._build_phase_encoding_masks()
        # normal() assumes complete readout lines; otherwise fall back to adjoint(forward).
        self._complete_lines = self._check_complete_lines()
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
        budget = int(float(os.environ.get('GRICS_STATE_BATCH_BUDGET_GB', '2')) * 2**30)
        return self.motionOperator.state_chunks(Ncoils * self._smaps_b[0].numel() * self._coil_batches.element_size,
                                               budget_bytes=self._coil_batches.state_budget(budget))

    def forward(self, image, *, output_device=None):
        output_device = self.device if output_device is None else torch.device(output_device)
        if self._batched:
            return self._forward_batched(image, output_device)
        Ncoils = self.smaps.shape[0]
        N_motion_states = len(self.SamplingIndices[0])
        KspaceData = torch.zeros((Ncoils, self.Nex, self.Nsamples), dtype=torch.complex128, device=output_device)
        image_flat = image.reshape(-1).to(self.compute_dtype)
        for motion_state in range(N_motion_states):
            WarpedImage = self._warp(image_flat, motion_state)
            for c0, c1 in self._coil_batches.batches():
                WarpedImageFT = fftnc(WarpedImage.unsqueeze(0) * self._coil_batches.coil_maps(c0, c1), dims=self._fft_dims).reshape(c1 - c0, -1)
                for nex in range(self.Nex):
                    indices = self.SamplingIndices[nex][motion_state]
                    if indices.numel():
                        KspaceData[c0:c1, nex, indices.to(output_device)] = WarpedImageFT[:, indices].to(device=output_device, dtype=KspaceData.dtype)
                del WarpedImageFT
        return KspaceData.flatten()

    def _forward_batched(self, image, output_device):
        Ncoils = self.smaps.shape[0]
        KspaceData = torch.zeros((Ncoils, self.Nex, self.Nsamples), dtype=torch.complex128, device=output_device)
        image_flat = image.reshape(-1).to(self.compute_dtype)
        for s0, s1 in self._chunks():
            W = self.motionOperator.warp_all(image_flat, (s0, s1)).reshape(s1 - s0, 1, *self._shape)
            for c0, c1 in self._coil_batches.batches(s1 - s0):
                WFT = fftnc(W * self._coil_batches.coil_maps(c0, c1).unsqueeze(0), dims=tuple(d + 1 for d in self._fft_dims)).reshape(s1 - s0, c1 - c0, -1)
                for motion_state in range(s0, s1):
                    for nex in range(self.Nex):
                        indices = self.SamplingIndices[nex][motion_state]
                        if indices.numel():
                            KspaceData[c0:c1, nex, indices.to(output_device)] = WFT[motion_state - s0][:, indices].to(device=output_device, dtype=KspaceData.dtype)
                del WFT
        return KspaceData.flatten()

    def residual(self, image, measured):
        """Compute measured-minus-predicted with one full buffer, CPU-stored if needed."""
        device = self._coil_batches.kspace_output_device(self.Nex, self.Nsamples)
        residual = self.forward(image, output_device=device).reshape(self.smaps.shape[0], self.Nex, self.Nsamples)
        measured = measured.reshape_as(residual)
        # Subtract coil by coil to avoid a second full prediction or host copy.
        for coil in range(self.smaps.shape[0]):
            torch.sub(measured[coil].to(device), residual[coil], out=residual[coil])
        return residual.flatten()

    def _adjoint_batched(self, KspaceData):
        Ncoils = self.smaps.shape[0]
        KspaceData = KspaceData.reshape(Ncoils, self.Nex, self.Nsamples)
        n_voxels = self._smaps_b[0].numel()
        Image = torch.zeros(n_voxels, dtype=self.compute_dtype, device=self.device)
        for s0, s1 in self._chunks():
            Y = torch.zeros((s1 - s0, n_voxels), dtype=self.compute_dtype, device=self.device)
            for c0, c1 in self._coil_batches.batches(s1 - s0):
                for nex in range(self.Nex):
                    Kfull = torch.zeros((s1 - s0, c1 - c0, n_voxels), dtype=self.compute_dtype, device=self.device)
                    for motion_state in range(s0, s1):
                        indices = self.SamplingIndices[nex][motion_state]
                        if indices.numel():
                            Kfull[motion_state - s0][:, indices] = self._coil_batches.kspace_samples(KspaceData, c0, c1, nex, indices)
                    img = ifftnc(Kfull.reshape(s1 - s0, c1 - c0, *self._shape), dims=tuple(d + 1 for d in self._fft_dims))
                    Y += (img * self._coil_batches.coil_maps(c0, c1, conjugate=True).unsqueeze(0)).sum(dim=1).reshape(s1 - s0, -1)
                    del Kfull, img
            Image += self.motionOperator.warp_T_all(Y, (s0, s1))
        return Image.to(torch.complex128)

    def _normal_batched(self, image):
        image_flat = image.reshape(-1).to(self.compute_dtype)
        Image = torch.zeros(image_flat.numel(), dtype=self.compute_dtype, device=self.device)
        phase_dims = tuple(d + 1 for d in self._phase_fft_dims)
        for s0, s1 in self._chunks():
            W = self.motionOperator.warp_all(image_flat, (s0, s1)).reshape(s1 - s0, 1, *self._shape)
            Y = torch.zeros((s1 - s0, image_flat.numel()), dtype=self.compute_dtype, device=self.device)
            for c0, c1 in self._coil_batches.batches(s1 - s0):
                WFT = fftnc(W * self._coil_batches.coil_maps(c0, c1).unsqueeze(0), dims=phase_dims)
                for nex in range(self.Nex):
                    masks = self._stacked_masks[nex]
                    if masks is None:
                        continue
                    img = ifftnc(WFT * masks[s0:s1], dims=phase_dims)
                    Y += (img * self._coil_batches.coil_maps(c0, c1, conjugate=True).unsqueeze(0)).sum(dim=1).reshape(s1 - s0, -1)
                    del img
                del WFT
            Image += self.motionOperator.warp_T_all(Y, (s0, s1))
        return Image.to(torch.complex128)

    def adjoint(self, KspaceData):
        if self._batched:
            return self._adjoint_batched(KspaceData)
        Ncoils = self.smaps.shape[0]
        N_motion_states = len(self.SamplingIndices[0])
        KspaceData = KspaceData.reshape(Ncoils, self.Nex, self.Nsamples)
        n_voxels = self._smaps_b[0].numel()
        Image = torch.zeros(self._shape, dtype=self.compute_dtype, device=self.device)
        for motion_state in range(N_motion_states):
            WarpedImage = torch.zeros_like(Image)
            for c0, c1 in self._coil_batches.batches():
                for nex in range(self.Nex):
                    indices = self.SamplingIndices[nex][motion_state]
                    if not indices.numel():
                        continue
                    KspaceDataNex = torch.zeros((c1 - c0, n_voxels), dtype=self.compute_dtype, device=self.device)
                    KspaceDataNex[:, indices] = self._coil_batches.kspace_samples(KspaceData, c0, c1, nex, indices)
                    image_coils = ifftnc(KspaceDataNex.reshape(c1 - c0, *self._shape), dims=self._fft_dims)
                    WarpedImage += (image_coils * self._coil_batches.coil_maps(c0, c1, conjugate=True)).sum(dim=0)
                    del KspaceDataNex, image_coils
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

    def _check_complete_lines(self):
        """True when every state's samples are whole readout lines (all kx for each acquired ky[, kz])."""
        Ncoils, Nx, Ny, Nz = self.smaps.shape
        n_phase = Ny * Nz
        for nex in range(self.Nex):
            for motion_state, indices in enumerate(self.SamplingIndices[nex]):
                if indices.numel() == 0:
                    continue
                phase = indices % n_phase
                lines = torch.unique(phase)
                if indices.numel() != Nx * lines.numel() or torch.unique(indices).numel() != indices.numel():
                    print(f"[encoding] repetition {nex}, state {motion_state}: samples are not complete readout lines; "
                          "E^H E will use adjoint(forward), which is slower.", flush=True)
                    return False
        return True

    def normal(self, image):
        """Apply E^H E using only the phase-encoding FFTs (y, or y and z).

        For one coil/state, E = P F_x F_phase S W. Complete readout
        lines make P^H P independent of x, so it commutes with F_x:

            F_x^H (P^H P) F_x = P^H P.

        Thus E^H E = W^H S^H F_phase^H (P^H P) F_phase S W.
        Coil weighting and image warping stay on their original sides of
        the FFTs; neither needs to commute with the sampling mask.
        """
        if not self._complete_lines:
            return self.adjoint(self.forward(image, output_device=self._coil_batches.kspace_output_device(self.Nex, self.Nsamples)))
        if self._batched:
            return self._normal_batched(image)
        N_motion_states = len(self.SamplingIndices[0])
        image_flat = image.reshape(-1).to(self.compute_dtype)
        Image = torch.zeros(self._shape, dtype=self.compute_dtype, device=self.device)
        for motion_state in range(N_motion_states):
            WarpedImage = self._warp(image_flat, motion_state)
            ImageSum = torch.zeros_like(Image)
            for c0, c1 in self._coil_batches.batches():
                WarpedImageFT = fftnc(WarpedImage.unsqueeze(0) * self._coil_batches.coil_maps(c0, c1), dims=self._phase_fft_dims)
                for nex in range(self.Nex):
                    phase_mask = self._phase_encoding_masks[nex][motion_state]
                    if phase_mask is None:
                        continue
                    image_coils = ifftnc(WarpedImageFT * phase_mask, dims=self._phase_fft_dims)
                    ImageSum += (image_coils * self._coil_batches.coil_maps(c0, c1, conjugate=True)).sum(dim=0)
                    del image_coils
                del WarpedImageFT
            Image += self._unwarp(ImageSum, motion_state)
        return Image.flatten().to(torch.complex128)

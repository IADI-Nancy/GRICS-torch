import torch
from  src.utils.fftnc import fftnc, ifftnc

"""
Jacobian of the encoding operator with respect to motion model perturbation.
Performes the simulation of motion perturbation effects in k-space data.
J_ᾱ(E)·δᾱ = δkspace, where δkspace is the residual from the image reconstruction step - is the equation to solve.

J_ᾱ(E)·δᾱ = E(image, MotionOperator(ᾱ+δᾱ)) - E(image, MotionOperator(ᾱ)) =
= (Sampling · F · smaps · MotionOperator(ᾱ+δᾱ) · image) - (Sampling · F · smaps · MotionOperator(ᾱ) · image) =
= Sampling · F · smaps · (MotionOperator(ᾱ+δᾱ) - MotionOperator(ᾱ))· image =
            <optical flow first-order approximation>
= Sampling · F · smaps · sum_{i,j}(∇_i(MotionOperator(ᾱ) · image) · ∂x_i/∂ᾱ_j · δᾱ_j)
where ∂x_i/∂ᾱ_j = J_(i,j) is the Jacobian matrix of (X,Y) grid derivatives with respect to motion parameters
ᾱ_i (translation, rotation, center).

Implementation notes (performance; results identical to the reference implementation up to
floating-point summation order): the image and the motion operator are fixed for the lifetime of
this object (one Gauss-Newton motion solve), so the warped image and its spatial gradients are
computed once per motion state and cached; the coil axis is processed as one batched tensor;
contributions of all repetitions are accumulated before the (linear) adjoint Jacobian.
"""

class MotionPerturbationSimulator:
    def __init__(self, smaps, Nsamples, SamplingIndices, Nex, image, motionOperator,
                 gradient_boundary="one_sided"):
        self.device = smaps.device
        self.SensitivityMaps = smaps
        self.Nex = Nex
        self.Nsamples = Nsamples
        self.SamplingIndices = SamplingIndices
        self.image = image
        if gradient_boundary not in ("zero", "one_sided"):
            raise ValueError('gradient_boundary must be "zero" or "one_sided".')
        self.gradient_boundary = gradient_boundary
        expected_ndim = 3 if int(smaps.shape[-1]) > 1 else 2
        if image.ndim != expected_ndim:
            raise ValueError(
                f"Motion optimization requires a {expected_ndim}D shared image; "
                f"got shape {tuple(image.shape)}.")
        self.motionOperator = motionOperator
        self.Nalpha = motionOperator.alpha.shape[0]  # number of displacement or rigid parameters
        if motionOperator.motion_type == "non-rigid":
            if getattr(motionOperator, "Nz", 1) > 1:
                if motionOperator.alpha.ndim != 5:
                    raise ValueError(
                        "3D non-rigid alpha must have shape [Nalpha, Nx, Ny, Nz, Nsensor]. "
                        f"Got {tuple(motionOperator.alpha.shape)}."
                    )
            elif motionOperator.alpha.ndim != 4:
                raise ValueError(
                    "2D non-rigid alpha must have shape [Nalpha, Nx, Ny, Nsensor]. "
                    f"Got {tuple(motionOperator.alpha.shape)}."
                )
            self.Nphysio = int(motionOperator.alpha.shape[-1])
        else:
            self.Nphysio = 1
        Ncoils, Nx, Ny, Nz = smaps.shape
        self._is_3d = int(Nz) > 1
        self._shape = (Nx, Ny, Nz) if self._is_3d else (Nx, Ny)
        self._smaps_b = smaps if self._is_3d else smaps.squeeze(-1)
        self._smaps_b_conj = self._smaps_b.conj()
        self._fft_dims = (1, 2, 3) if self._is_3d else (1, 2)
        self._state_cache = {}

    def _gradient_2d(self, img):
        # Central differences in the interior, one-sided at boundaries.
        gx = torch.empty_like(img)
        gy = torch.empty_like(img)

        # x-gradient (axis 0)
        gx[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])
        gx[0, :] = img[1, :] - img[0, :]
        gx[-1, :] = img[-1, :] - img[-2, :]

        # y-gradient (axis 1)
        gy[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
        gy[:, 0] = img[:, 1] - img[:, 0]
        gy[:, -1] = img[:, -1] - img[:, -2]

        if self.gradient_boundary == "zero":
            gx[0, :] = gx[-1, :] = 0
            gy[:, 0] = gy[:, -1] = 0
        return gx, gy

    def _gradient_3d(self, img):
        # Central differences in the interior, one-sided at boundaries.
        gx = torch.empty_like(img)
        gy = torch.empty_like(img)
        gz = torch.empty_like(img)

        # x-gradient (axis 0)
        gx[1:-1, :, :] = 0.5 * (img[2:, :, :] - img[:-2, :, :])
        gx[0, :, :] = img[1, :, :] - img[0, :, :]
        gx[-1, :, :] = img[-1, :, :] - img[-2, :, :]

        # y-gradient (axis 1)
        gy[:, 1:-1, :] = 0.5 * (img[:, 2:, :] - img[:, :-2, :])
        gy[:, 0, :] = img[:, 1, :] - img[:, 0, :]
        gy[:, -1, :] = img[:, -1, :] - img[:, -2, :]

        # z-gradient (axis 2)
        gz[:, :, 1:-1] = 0.5 * (img[:, :, 2:] - img[:, :, :-2])
        gz[:, :, 0] = img[:, :, 1] - img[:, :, 0]
        gz[:, :, -1] = img[:, :, -1] - img[:, :, -2]

        if self.gradient_boundary == "zero":
            gx[0, :, :] = gx[-1, :, :] = 0
            gy[:, 0, :] = gy[:, -1, :] = 0
            gz[:, :, 0] = gz[:, :, -1] = 0
        return gx, gy, gz

    def _warped_gradients(self, motion_state):
        """Warped image gradients of one state (image and motion are fixed for this object)."""
        cached = self._state_cache.get(motion_state)
        if cached is None:
            MotionOp = self.motionOperator._get_sparse_operator(motion_state)
            WarpedImage = (MotionOp @ self.image.flatten()).reshape(self._shape)
            cached = self._gradient_3d(WarpedImage) if self._is_3d else self._gradient_2d(WarpedImage)
            self._state_cache[motion_state] = cached
        return cached

    def _motion_signal(self, dtype):
        motion_signal = torch.as_tensor(self.motionOperator.motion_signal, device=self.device, dtype=dtype)
        if motion_signal.ndim != 2:
            raise ValueError(
                "non-rigid motion_signal must have shape [Nstate, Nsensor]. "
                f"Got {tuple(motion_signal.shape)}."
            )
        return motion_signal

    def forward(self, MotionModelPerturbation):
        Ncoils, Nx, Ny, Nz = self.SensitivityMaps.shape
        is_3d = self._is_3d
        N_motion_states = len(self.SamplingIndices[0])  # assuming SamplingIndices is a list of lists with shape [Nex][N_motion_states]

        if self.motionOperator.motion_type == "rigid":
            MotionModelPerturbation = MotionModelPerturbation.reshape(self.Nalpha, N_motion_states)
        else:
            if is_3d:
                MotionModelPerturbation = MotionModelPerturbation.reshape(self.Nalpha, Nx, Ny, Nz, self.Nphysio)
            else:
                MotionModelPerturbation = MotionModelPerturbation.reshape(self.Nalpha, Nx, Ny, self.Nphysio)
            motion_signal = self._motion_signal(MotionModelPerturbation.dtype)
        # output k-space residual
        ResidualKspace = torch.zeros((Ncoils, self.Nex, self.Nsamples),
                                     dtype=torch.complex128,
                                     device=self.device)

        # ---- Loop over motion states ----
        for motion_state in range(N_motion_states):
            if all(self.SamplingIndices[nex][motion_state].numel() == 0 for nex in range(self.Nex)):
                continue
            # 1-2) Warped image and its spatial gradients (cached)
            grads = self._warped_gradients(motion_state)

            # 3) motion model and displacement field perturbations
            if self.motionOperator.motion_type == "rigid":
                if is_3d:
                    dux, duy, duz = self.motionOperator._apply_J(MotionModelPerturbation[:, motion_state], motion_state)
                else:
                    dux, duy = self.motionOperator._apply_J(MotionModelPerturbation[:, motion_state], motion_state)
            else:
                # Non-rigid model is modulated by the state-dependent motion signal.
                if is_3d:
                    weights = motion_signal[motion_state].reshape(1, 1, 1, -1)
                    dux = torch.sum(MotionModelPerturbation[0] * weights, dim=-1)
                    duy = torch.sum(MotionModelPerturbation[1] * weights, dim=-1)
                    duz = torch.sum(MotionModelPerturbation[2] * weights, dim=-1)
                else:
                    weights = motion_signal[motion_state].reshape(1, 1, -1)
                    dux = torch.sum(MotionModelPerturbation[0] * weights, dim=-1)
                    duy = torch.sum(MotionModelPerturbation[1] * weights, dim=-1)

            # 4) optical-flow first-order perturbation
            if is_3d:
                Gx, Gy, Gz = grads
                WarpedImageError = Gx * dux + Gy * duy + Gz * duz
            else:
                Gx, Gy = grads
                WarpedImageError = Gx * dux + Gy * duy

            # 5) Coil sensitivities and FFT of all coils at once
            ImFT = fftnc(WarpedImageError.unsqueeze(0) * self._smaps_b, dims=self._fft_dims).reshape(Ncoils, -1)

            for nex in range(self.Nex):
                SamplingIndices = self.SamplingIndices[nex][motion_state]
                if SamplingIndices.numel() == 0:
                    continue
                # Sampling operator: extract measured samples and store in the global vector
                ResidualKspace[:, nex, SamplingIndices] = ImFT[:, SamplingIndices]

        return ResidualKspace.flatten()

    def adjoint(self, ResidualKspace):
        """
        Adjoint of the motion-perturbation forward operator.
        Input:
            ResidualKspace: flattened k-space residual, shape [Nsamples*Ncoils]
        Output:
            MotionModelAdjoint: shape [2, N_motion_states]
        """

        Ncoils, Nx, Ny, Nz = self.SensitivityMaps.shape
        is_3d = self._is_3d
        N_motion_states = len(self.SamplingIndices[0])  # assuming SamplingIndices is a list of lists with shape [Nex][N_motion_states]
        n_voxels = Nx * Ny * Nz

        ResidualKspace = ResidualKspace.reshape(Ncoils, self.Nex, self.Nsamples)

        if self.motionOperator.motion_type == "rigid":
            MotionModelPerturbation = torch.zeros((self.Nalpha, N_motion_states),
                                            dtype=torch.complex128,
                                            device=self.device)
        else:
            if is_3d:
                MotionModelPerturbation = torch.zeros((self.Nalpha, Nx, Ny, Nz, self.Nphysio), dtype=torch.complex128, device=self.device)
            else:
                MotionModelPerturbation = torch.zeros((self.Nalpha, Nx, Ny, self.Nphysio), dtype=torch.complex128, device=self.device)
            motion_signal = self._motion_signal(MotionModelPerturbation.dtype)

        for motion_state in range(N_motion_states):
            # 3-6) Back-project the residual of every repetition into the image domain,
            # summed over coils and repetitions (the adjoint Jacobian below is linear).
            ResidualImage = None
            for nex in range(self.Nex):
                SamplingIndices = self.SamplingIndices[nex][motion_state]
                if SamplingIndices.numel() == 0:
                    continue
                KspaceDataNex = torch.zeros((Ncoils, n_voxels), dtype=torch.complex128, device=self.device)
                KspaceDataNex[:, SamplingIndices] = ResidualKspace[:, nex, SamplingIndices]
                ImageSeenByCoils = ifftnc(KspaceDataNex.reshape(Ncoils, *self._shape), dims=self._fft_dims)
                contribution = (ImageSeenByCoils * self._smaps_b_conj).sum(dim=0)
                ResidualImage = contribution if ResidualImage is None else ResidualImage + contribution
            if ResidualImage is None:
                continue

            # 1-2) Warped image gradients (cached)
            grads = self._warped_gradients(motion_state)

            # 7) Inner products with gradients -> scalar/field displacement contributions
            if is_3d:
                Gx, Gy, Gz = grads
                du_z = (ResidualImage * Gz.conj())
            else:
                Gx, Gy = grads
            du_x = (ResidualImage * Gx.conj())
            du_y = (ResidualImage * Gy.conj())

            # Apply the adjoint Jacobian
            if self.motionOperator.motion_type == "rigid":
                if is_3d:
                    MotionModelPerturbation[:, motion_state] += self.motionOperator._apply_JH(du_x, du_y, motion_state, du_z=du_z)
                else:
                    MotionModelPerturbation[:, motion_state] += self.motionOperator._apply_JH(du_x, du_y, motion_state)
            else:
                # Adjoint of multiplication by motion_signal is multiplication by its complex conjugate.
                if is_3d:
                    weights = motion_signal[motion_state].conj().reshape(1, 1, 1, -1)
                    MotionModelPerturbation[0] += du_x.unsqueeze(-1) * weights
                    MotionModelPerturbation[1] += du_y.unsqueeze(-1) * weights
                    MotionModelPerturbation[2] += du_z.unsqueeze(-1) * weights
                else:
                    weights = motion_signal[motion_state].conj().reshape(1, 1, -1)
                    MotionModelPerturbation[0] += du_x.unsqueeze(-1) * weights
                    MotionModelPerturbation[1] += du_y.unsqueeze(-1) * weights

        return MotionModelPerturbation.flatten()

    def normal(self, image):
            return self.adjoint(self.forward(image))

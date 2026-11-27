import os
import torch
from src.utils.fftnc import fftnc, ifftnc
from src.reconstruction.coil_batching import CoilBatchPlanner

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
computed once per motion state and cached; coils use automatic GPU-memory-sized FFT batches;
contributions of all repetitions are accumulated before the (linear) adjoint Jacobian.
"""

class MotionPerturbationSimulator:
    def __init__(self, smaps, Nsamples, SamplingIndices, Nex, image, motionOperator,
                 gradient_boundary="one_sided", rotate_gradients=True):
        self.device = smaps.device
        # The warp is a pull-back, warped(r) = rho(T(r)) with T(r) = c + R (r - c) + t, so the
        # chain rule needs (grad rho)(T(r)) = R grad(warped)(r). GRICS-torch 2.0 used
        # grad(warped) directly, which is off by R^T; keep that path for comparisons.
        self.rotate_gradients = bool(rotate_gradients) and motionOperator.motion_type == "rigid"
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
        self.compute_dtype = getattr(motionOperator, 'compute_dtype', torch.complex128)
        self._real_dtype = torch.float32 if self.compute_dtype == torch.complex64 else torch.float64
        self._image_c = image.to(self.compute_dtype)
        self._smaps_b = smaps if self._is_3d else smaps.squeeze(-1)
        self._coil_batches = CoilBatchPlanner(self._smaps_b, self.compute_dtype, 'motion Jacobian')
        self._fft_dims = (1, 2, 3) if self._is_3d else (1, 2)
        self._state_cache = {}
        # State batching pays off for launch-bound (small) problems and costs memory traffic on large ones:
        # default = batch when a state has at most GRICS_STATE_BATCH_MAX_VOXELS voxels (2^19), gather backend only.
        _mode = os.environ.get('GRICS_STATE_BATCH', 'auto')
        _small = self._smaps_b[0].numel() <= int(os.environ.get('GRICS_STATE_BATCH_MAX_VOXELS', str(2 ** 19)))
        self._batched = (getattr(motionOperator, 'warp_backend', 'sparse') == 'gather'
                         and (_mode == '1' or (_mode == 'auto' and _small)))

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
            WarpedImage = self.motionOperator.warp(self._image_c.flatten(), motion_state).reshape(self._shape)
            cached = self._gradient_3d(WarpedImage) if self._is_3d else self._gradient_2d(WarpedImage)
            if self.rotate_gradients:
                R = self.motionOperator.rigid_rotation_matrix(motion_state).to(self._real_dtype)
                cached = tuple(sum(R[i, j] * cached[j] for j in range(len(cached))) for i in range(len(cached)))
            self._state_cache[motion_state] = cached
        return cached

    def _grads_all(self):
        """Warped-image gradients of all states, stacked [Nm, dim, N] (computed once per object)."""
        G = self.__dict__.get("_grads_all_cache")
        if G is None:
            n_states = len(self.SamplingIndices[0])
            G = torch.stack([torch.stack([g.reshape(-1) for g in self._warped_gradients(s)]) for s in range(n_states)])
            self.__dict__["_grads_all_cache"] = G
        return G

    def _chunks(self):
        Ncoils = self.SensitivityMaps.shape[0]
        budget = int(float(os.environ.get('GRICS_STATE_BATCH_BUDGET_GB', '2')) * 2**30)
        return self.motionOperator.state_chunks(Ncoils * self._smaps_b[0].numel() * self._coil_batches.element_size,
                                               budget_bytes=self._coil_batches.state_budget(budget))

    def _displacements_all(self, MotionModelPerturbation, N_motion_states):
        """Displacement perturbations of all states [Nm, dim, N] in the operator precision."""
        dim = 3 if self._is_3d else 2
        if self.motionOperator.motion_type == "rigid":
            rows = []
            for s in range(N_motion_states):
                du = self.motionOperator._apply_J(MotionModelPerturbation[:, s], s)
                rows.append(torch.stack([d.reshape(-1) for d in du]))
            return torch.stack(rows).to(self.compute_dtype)
        signal = self._motion_signal(MotionModelPerturbation.dtype)
        a = MotionModelPerturbation[:dim].reshape(dim, -1, self.Nphysio)          # [dim, N, S]
        return torch.einsum('dns,ms->mdn', a, signal).to(self.compute_dtype)   # [Nm, dim, N]

    def _forward_batched(self, MotionModelPerturbation, output_device):
        Ncoils, Nx, Ny, Nz = self.SensitivityMaps.shape
        N_motion_states = len(self.SamplingIndices[0])
        if self.motionOperator.motion_type == "rigid":
            MotionModelPerturbation = MotionModelPerturbation.reshape(self.Nalpha, N_motion_states)
        else:
            MotionModelPerturbation = MotionModelPerturbation.reshape(self.Nalpha, -1, self.Nphysio)
        du = self._displacements_all(MotionModelPerturbation, N_motion_states)
        G = self._grads_all()
        ResidualKspace = torch.zeros((Ncoils, self.Nex, self.Nsamples), dtype=torch.complex128, device=output_device)
        fft_dims = tuple(d + 1 for d in self._fft_dims)
        for s0, s1 in self._chunks():
            WE = (G[s0:s1] * du[s0:s1]).sum(dim=1).reshape(s1 - s0, 1, *self._shape)
            for c0, c1 in self._coil_batches.batches(s1 - s0):
                ImFT = fftnc(WE * self._coil_batches.coil_maps(c0, c1).unsqueeze(0), dims=fft_dims).reshape(s1 - s0, c1 - c0, -1)
                for motion_state in range(s0, s1):
                    for nex in range(self.Nex):
                        indices = self.SamplingIndices[nex][motion_state]
                        if indices.numel():
                            ResidualKspace[c0:c1, nex, indices.to(output_device)] = ImFT[motion_state - s0][:, indices].to(device=output_device, dtype=ResidualKspace.dtype)
                del ImFT
        return ResidualKspace.flatten()

    def _adjoint_batched(self, ResidualKspace):
        Ncoils, Nx, Ny, Nz = self.SensitivityMaps.shape
        N_motion_states = len(self.SamplingIndices[0])
        n_voxels = Nx * Ny * Nz
        dim = 3 if self._is_3d else 2
        ResidualKspace = ResidualKspace.reshape(Ncoils, self.Nex, self.Nsamples)
        G = self._grads_all()
        fft_dims = tuple(d + 1 for d in self._fft_dims)
        RI = torch.zeros((N_motion_states, n_voxels), dtype=self.compute_dtype, device=self.device)
        for s0, s1 in self._chunks():
            for c0, c1 in self._coil_batches.batches(s1 - s0):
                for nex in range(self.Nex):
                    Kfull = torch.zeros((s1 - s0, c1 - c0, n_voxels), dtype=self.compute_dtype, device=self.device)
                    for motion_state in range(s0, s1):
                        indices = self.SamplingIndices[nex][motion_state]
                        if indices.numel():
                            Kfull[motion_state - s0][:, indices] = self._coil_batches.kspace_samples(ResidualKspace, c0, c1, nex, indices)
                    img = ifftnc(Kfull.reshape(s1 - s0, c1 - c0, *self._shape), dims=fft_dims)
                    RI[s0:s1] += (img * self._coil_batches.coil_maps(c0, c1, conjugate=True).unsqueeze(0)).sum(dim=1).reshape(s1 - s0, -1)
                    del Kfull, img
        du = RI.unsqueeze(1) * G.conj()                                             # [Nm, dim, N]
        if self.motionOperator.motion_type == "rigid":
            out = torch.zeros((self.Nalpha, N_motion_states), dtype=torch.complex128, device=self.device)
            for s in range(N_motion_states):
                fields = [du[s, d].reshape(self._shape) for d in range(dim)]
                if self._is_3d:
                    out[:, s] += self.motionOperator._apply_JH(fields[0], fields[1], s, du_z=fields[2])
                else:
                    out[:, s] += self.motionOperator._apply_JH(fields[0], fields[1], s)
            return out.flatten()
        signal = self._motion_signal(torch.complex128)
        out = torch.einsum('mdn,ms->dns', du.to(torch.complex128), signal.conj())   # [dim, N, S]
        if self._is_3d:
            return out.reshape(self.Nalpha, Nx, Ny, Nz, self.Nphysio).flatten()
        return out.reshape(self.Nalpha, Nx, Ny, self.Nphysio).flatten()

    def _motion_signal(self, dtype):
        motion_signal = torch.as_tensor(self.motionOperator.motion_signal, device=self.device, dtype=dtype)
        if motion_signal.ndim != 2:
            raise ValueError(
                "non-rigid motion_signal must have shape [Nstate, Nsensor]. "
                f"Got {tuple(motion_signal.shape)}."
            )
        return motion_signal

    def forward(self, MotionModelPerturbation, *, output_device=None):
        output_device = self.device if output_device is None else torch.device(output_device)
        if self._batched:
            return self._forward_batched(MotionModelPerturbation, output_device)
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
                                     device=output_device)

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

            # 4) optical-flow first-order perturbation (displacement perturbations cast to the operator precision)
            if is_3d:
                Gx, Gy, Gz = grads
                WarpedImageError = Gx * dux.to(self.compute_dtype) + Gy * duy.to(self.compute_dtype) + Gz * duz.to(self.compute_dtype)
            else:
                Gx, Gy = grads
                WarpedImageError = Gx * dux.to(self.compute_dtype) + Gy * duy.to(self.compute_dtype)

            # 5) Coil sensitivities and FFT, with the same grouping as encoding.
            for c0, c1 in self._coil_batches.batches():
                ImFT = fftnc(WarpedImageError.unsqueeze(0) * self._coil_batches.coil_maps(c0, c1), dims=self._fft_dims).reshape(c1 - c0, -1)
                for nex in range(self.Nex):
                    indices = self.SamplingIndices[nex][motion_state]
                    if indices.numel():
                        ResidualKspace[c0:c1, nex, indices.to(output_device)] = ImFT[:, indices].to(device=output_device, dtype=ResidualKspace.dtype)
                del ImFT
        return ResidualKspace.flatten()

    def adjoint(self, ResidualKspace):
        """
        Adjoint of the motion-perturbation forward operator.
        Input:
            ResidualKspace: flattened k-space residual, shape [Nsamples*Ncoils]
        Output:
            MotionModelAdjoint: shape [2, N_motion_states]
        """

        if self._batched:
            return self._adjoint_batched(ResidualKspace)
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
            ResidualImage = torch.zeros(self._shape, dtype=self.compute_dtype, device=self.device)
            for c0, c1 in self._coil_batches.batches():
                for nex in range(self.Nex):
                    indices = self.SamplingIndices[nex][motion_state]
                    if not indices.numel():
                        continue
                    KspaceDataNex = torch.zeros((c1 - c0, n_voxels), dtype=self.compute_dtype, device=self.device)
                    KspaceDataNex[:, indices] = self._coil_batches.kspace_samples(ResidualKspace, c0, c1, nex, indices)
                    ImageSeenByCoils = ifftnc(KspaceDataNex.reshape(c1 - c0, *self._shape), dims=self._fft_dims)
                    ResidualImage += (ImageSeenByCoils * self._coil_batches.coil_maps(c0, c1, conjugate=True)).sum(dim=0)
                    del KspaceDataNex, ImageSeenByCoils
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
        device = self._coil_batches.kspace_output_device(self.Nex, self.Nsamples)
        return self.adjoint(self.forward(image, output_device=device))

import torch
from  utils.fftnc import fftnc, ifftnc

"""
Jacobian of the encoding operator with respect to motion model perturbation.
Performes the simulation of motion perturbation effects in k-space data.
(∇_u(E)·δu) = δkspace, where δkspace is the residual from the image reconstruction step - is the equation to solve.

(∇_u(E)·δu) = E(image, MotionOperator(u+δu)) - E(image, MotionOperator(u)) = 
= (Sampling · F · smaps · MotionOperator(u+δu) · image) - (Sampling · F · smaps · MotionOperator(u) · image) = 
= Sampling · F · smaps · (MotionOperator(u+δu) - MotionOperator(u))· image = 
            <optical flow first-order approximation>
= Sampling · F · smaps · MotionOperator(u) · (∇image · δu)
"""

class MotionPerturbationSimulator:
    def __init__(self, smaps, Nsamples, SamplingIndices, KspaceOffset, image, motionOperator=None):
        self.device = smaps.device
        self.SensitivityMaps = smaps
        self.KspaceOffset = KspaceOffset
        self.Nsamples = Nsamples
        self.SamplingIndices = SamplingIndices
        self.image = image
        if motionOperator is None:
            Nshots = len(SamplingIndices)
            Nx, Ny, Nsli, _ = smaps.shape
            Npix = Nx * Ny

            # build a sparse identity (Nx*Ny x Nx*Ny) and reuse for each shot
            idx = torch.arange(Npix, device=self.device)
            indices = torch.stack([idx, idx], dim=0)            # shape (2, Npix)
            vals = torch.ones(Npix, dtype=torch.complex64, device=self.device)
            I_sparse = torch.sparse_coo_tensor(indices, vals, (Npix, Npix),
                                               dtype=torch.complex64,
                                               device=self.device).coalesce()

            self.motionOperator = [I_sparse for _ in range(Nshots)]
        else:
            self.motionOperator = motionOperator

    def set_image(self, image):
        self.image = image

    def gradient_2d(self, img):
        gx = torch.zeros_like(img)
        gy = torch.zeros_like(img)

        gx[:-1, :] = img[1:, :] - img[:-1, :]
        gy[:, :-1] = img[:, 1:] - img[:, :-1]

        return gx, gy

    def forward(self, MotionModelPerturbation):
        Nx, Ny, Nsli, Ncoils = self.SensitivityMaps.shape
        Nshots = len(self.SamplingIndices)

        # reshape 2×Nshots perturbation vector
        # MotionModelPerturbation: shape [2*Nshots]
        MotionModelPerturbation = MotionModelPerturbation.reshape(2, Nshots)

        # output k-space residual
        ResidualKspace = torch.zeros((self.Nsamples, Ncoils),
                                     dtype=torch.complex64,
                                     device=self.device)

        # ---- Loop over shots ----
        for shot in range(Nshots):
            SamplingIndices = self.SamplingIndices[shot]    
            KspaceOffset = self.KspaceOffset[shot]
            MotionOp = self.motionOperator[shot]

            # 1) Warp the image using the motion operator
            WarpedImage = (MotionOp @ self.image.flatten()).reshape(Nx, Ny)

            # 2) Spatial gradients ∂I/∂x, ∂I/∂y
            Gx, Gy = self.gradient_2d(WarpedImage)

            # 3) translation perturbation dux, duy (constant matrix)
            dux = MotionModelPerturbation[0, shot]
            duy = MotionModelPerturbation[1, shot]

            dux = dux * torch.ones((Nx, Ny), device=self.device)
            duy = duy * torch.ones((Nx, Ny), device=self.device)

            # 4) optical-flow first-order perturbation
            WarpedImageError = Gx * dux + Gy * duy

            # 5) Loop over coils
            for coil in range(Ncoils):
                # apply coil sensitivities
                WarpedImageSeenByCoil = WarpedImageError * self.SensitivityMaps[:, :, :, coil].squeeze()

                # FFT
                ImFT = fftnc(WarpedImageSeenByCoil, dims=(0, 1))

                # Sampling operator: extract measured samples
                sampled = ImFT.flatten()[SamplingIndices]

                # store in global vector
                ResidualKspace[KspaceOffset + SamplingIndices, coil] = sampled

        return ResidualKspace.flatten()
    

    def adjoint(self, ResidualKspace):
        """
        Adjoint of the motion-perturbation forward operator.
        Input:
            ResidualKspace: flattened k-space residual, shape [Nsamples*Ncoils]
        Output:
            MotionModelAdjoint: shape [2, Nshots]
        """

        Nx, Ny, Nsli, Ncoils = self.SensitivityMaps.shape
        Nshots = len(self.SamplingIndices)

        ResidualKspace = ResidualKspace.reshape(self.Nsamples, Ncoils)

        # output: 2 × Nshots (dux and duy)
        MotionModelPerturbation = torch.zeros((2, Nshots),
                                        dtype=torch.float32,
                                        device=self.device)

        # --- Loop over shots -----------------------------------------------------
        for shot in range(Nshots):
            SamplingIndices = self.SamplingIndices[shot]
            KspaceOffset  = self.KspaceOffset[shot]
            MotionOp     = self.motionOperator[shot]

            # 1) Warp image with shot operator
            WarpedImage = (MotionOp @ self.image.flatten()).reshape(Nx, Ny)

            # 2) Gradients of warped image
            Gx, Gy = self.gradient_2d(WarpedImage)

            # 3) Allocate adjoint image accumulator (summed over coils)
            ResidualImage = torch.zeros((Nx, Ny), dtype=torch.complex64, device=self.device)

            # --- Coil loop --------------------------------------------------------
            for coil in range(Ncoils):

                # 4) Extract coil residual samples and place them back into k-space
                FullKspaceDataCoil = torch.zeros((Nx * Ny,), dtype=torch.complex64, device=self.device)
                FullKspaceDataCoil[SamplingIndices] = ResidualKspace[KspaceOffset + SamplingIndices, coil]

                FullKspaceDataCoil = FullKspaceDataCoil.reshape(Nx, Ny)

                # 5) Inverse FFT to go back to image domain
                ImageSeenByCoil = ifftnc(FullKspaceDataCoil, dims=(0, 1))

                # 6) Apply coil sensitivity adjoint (complex conjugate)
                ResidualImage += ImageSeenByCoil * self.SensitivityMaps[:, :, :, coil].conj().squeeze(-1)

            # 7) Inner products with Gx and Gy → scalar dux, duy contributions
            dux = torch.sum(torch.real(ResidualImage * Gx.conj()))
            duy = torch.sum(torch.real(ResidualImage * Gy.conj()))

            MotionModelPerturbation[0, shot] = dux
            MotionModelPerturbation[1, shot] = duy

        return MotionModelPerturbation.flatten()
    
    def normal(self, image):
            return self.adjoint(self.forward(image))


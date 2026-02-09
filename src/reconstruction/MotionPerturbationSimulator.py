import torch
from  src.utils.fftnc import fftnc, ifftnc
from src.reconstruction.MotionOperator import MotionOperator

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
"""

class MotionPerturbationSimulator:
    def __init__(self, smaps, Nsamples, SamplingIndices, Nex, image, motionOperator):
        self.device = smaps.device
        self.SensitivityMaps = smaps
        self.Nex = Nex
        self.Nsamples = Nsamples
        self.SamplingIndices = SamplingIndices
        self.image = image
        self.motionOperator = motionOperator
        self.Nalpha = motionOperator.alpha.shape[0]  # number of motion parameters (t_x, t_y, phi)
        

    def set_image(self, image):
        self.image = image.reshape(self.Nex, self.SensitivityMaps.shape[1], self.SensitivityMaps.shape[2])

    def gradient_2d(self, img):
        gx = torch.zeros_like(img)
        gy = torch.zeros_like(img)

        gx[:-1, :] = img[1:, :] - img[:-1, :]
        gy[:, :-1] = img[:, 1:] - img[:, :-1]

        return gx, gy

    def forward(self, MotionModelPerturbation):
        Ncoils, Nx, Ny, Nsli = self.SensitivityMaps.shape
        N_mot_states = len(self.SamplingIndices[0])  # assuming SamplingIndices is a list of lists with shape [Nex][N_mot_states]

        # reshape 3×Nshots perturbation vector
        # MotionModelPerturbation: shape [3*Nshots]
        MotionModelPerturbation = MotionModelPerturbation.reshape(self.Nalpha, N_mot_states)
        # output k-space residual
        ResidualKspace = torch.zeros((Ncoils, self.Nex, self.Nsamples),
                                     dtype=torch.complex64,
                                     device=self.device)

        # ---- Loop over shots ----
        for motion_state in range(N_mot_states):
              
            MotionOp = self.motionOperator.get_sparse_operator(motion_state)

            for nex in range(self.Nex):
                SamplingIndices = self.SamplingIndices[nex][motion_state]
                if SamplingIndices.numel() == 0:
                    continue
                image_nex = self.image[nex]
                # 1) Warp the image using the motion operator
                WarpedImage = (MotionOp @ image_nex.flatten()).reshape(Nx, Ny)

                # 2) Spatial gradients ∂I/∂x, ∂I/∂y
                Gx, Gy = self.gradient_2d(WarpedImage)

                # 3) motion model and displacement field perturbations
                dux, duy = self.motionOperator.apply_J(MotionModelPerturbation[:, motion_state], motion_state)

                # 4) optical-flow first-order perturbation
                WarpedImageError = Gx * dux + Gy * duy

                # 5) Loop over coils
                for coil in range(Ncoils):
                    # apply coil sensitivities
                    WarpedImageSeenByCoil = WarpedImageError * self.SensitivityMaps[coil].squeeze()

                    # FFT
                    ImFT = fftnc(WarpedImageSeenByCoil, dims=(0, 1))

                    # Sampling operator: extract measured samples
                    sampled = ImFT.flatten()[SamplingIndices]

                    # store in global vector
                    ResidualKspace[coil, nex, SamplingIndices] = sampled

        return ResidualKspace.flatten()
    

    def adjoint(self, ResidualKspace):
        """
        Adjoint of the motion-perturbation forward operator.
        Input:
            ResidualKspace: flattened k-space residual, shape [Nsamples*Ncoils]
        Output:
            MotionModelAdjoint: shape [2, Nshots]
        """

        Ncoils, Nx, Ny, Nsli = self.SensitivityMaps.shape
        N_mot_states = len(self.SamplingIndices[0])  # assuming SamplingIndices is a list of lists with shape [Nex][N_mot_states]

        ResidualKspace = ResidualKspace.reshape(Ncoils, self.Nex, self.Nsamples)

        # output: 2 × Nshots (dux and duy)
        MotionModelPerturbation = torch.zeros((self.Nalpha, N_mot_states),
                                        dtype=torch.float32,
                                        device=self.device)

        for motion_state in range(N_mot_states):
            
            MotionOp     = self.motionOperator.get_sparse_operator(motion_state)
            
            for nex in range(self.Nex):
                SamplingIndices = self.SamplingIndices[nex][motion_state]
                if SamplingIndices.numel() == 0:
                    continue
                image_nex = self.image[nex]
                # 1) Warp image with shot operator
                WarpedImage = (MotionOp @ image_nex.flatten()).reshape(Nx, Ny)

                # 2) Gradients of warped image
                Gx, Gy = self.gradient_2d(WarpedImage)

                # 3) Allocate adjoint image accumulator (summed over coils)
                ResidualImage = torch.zeros((Nx, Ny), dtype=torch.complex64, device=self.device)

                # --- Coil loop --------------------------------------------------------
                for coil in range(Ncoils):

                    # 4) Extract coil residual samples and place them back into k-space
                    KspaceDataCoilNex = torch.zeros((Nx * Ny,), dtype=torch.complex64, device=self.device)

                    KspaceDataCoilNex[SamplingIndices] = ResidualKspace[coil, nex, SamplingIndices]

                    KspaceDataCoilNex = KspaceDataCoilNex.reshape(Nx, Ny)
                    # 5) Inverse FFT to go back to image domain
                    ImageSeenByCoil = ifftnc(KspaceDataCoilNex, dims=(0, 1))

                    # 6) Apply coil sensitivity adjoint (complex conjugate)
                    ResidualImage += ImageSeenByCoil * self.SensitivityMaps[coil].conj().squeeze(-1)

                # 7) Inner products with Gx and Gy → scalar dux, duy contributions
                du_x = (ResidualImage * Gx.conj())
                du_y = (ResidualImage * Gy.conj())

                # Apply the adjoint Jacobian
                MotionModelPerturbation[:, motion_state] = self.motionOperator.apply_JH(du_x, du_y, motion_state)
            
        return MotionModelPerturbation.flatten()
    
    def normal(self, image):
            return self.adjoint(self.forward(image))


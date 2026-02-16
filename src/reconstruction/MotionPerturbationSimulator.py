import torch
import torch.nn.functional as F
from  src.utils.fftnc import fftnc

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

    def _grid_sample_complex(self, image, ux, uy):
        nx, ny = image.shape
        dtype = image.real.dtype
        device = image.device

        x = torch.arange(nx, device=device, dtype=dtype)
        y = torch.arange(ny, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")

        src_x = xx + ux
        src_y = yy + uy

        if nx > 1:
            y_norm = 2.0 * src_x / (nx - 1) - 1.0
        else:
            y_norm = torch.zeros_like(src_x)
        if ny > 1:
            x_norm = 2.0 * src_y / (ny - 1) - 1.0
        else:
            x_norm = torch.zeros_like(src_y)

        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
        real = image.real.unsqueeze(0).unsqueeze(0)
        imag = image.imag.unsqueeze(0).unsqueeze(0)
        real_w = F.grid_sample(
            real, grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0]
        imag_w = F.grid_sample(
            imag, grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0]
        return torch.complex(real_w, imag_w)

    def _displacement_fields(self, alpha, motion_state, nx, ny, dtype, device):
        if self.motionOperator.motion_type == "rigid":
            centers = self.motionOperator.centers
            if centers is None:
                centers = torch.zeros((2, alpha.shape[1]), dtype=dtype, device=device)
                centers[0, :] = nx / 2.0
                centers[1, :] = ny / 2.0

            tx = alpha[0, motion_state]
            ty = alpha[1, motion_state]
            phi = alpha[2, motion_state]
            cx = centers[0, motion_state]
            cy = centers[1, motion_state]

            x = torch.arange(nx, device=device, dtype=dtype)
            y = torch.arange(ny, device=device, dtype=dtype)
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            xmc = xx - cx
            ymc = yy - cy
            c = torch.cos(phi)
            s = torch.sin(phi)
            xp = cx + c * xmc - s * ymc + tx
            yp = cy + s * xmc + c * ymc + ty
            ux = xp - xx
            uy = yp - yy
            return ux, uy

        motion_signal = torch.as_tensor(
            self.motionOperator.motion_signal, device=device, dtype=dtype
        )
        ux = alpha[0] * motion_signal[motion_state]
        uy = alpha[1] * motion_signal[motion_state]
        return ux, uy

    def _forward_alpha(self, alpha):
        Ncoils, nx, ny, _ = self.SensitivityMaps.shape
        n_states = len(self.SamplingIndices[0])
        image = self.image.reshape(self.Nex, nx, ny)
        kspace = torch.zeros(
            (Ncoils, self.Nex, self.Nsamples), dtype=torch.complex64, device=self.device
        )

        for motion_state in range(n_states):
            ux, uy = self._displacement_fields(
                alpha, motion_state, nx, ny, alpha.dtype, alpha.device
            )
            for nex in range(self.Nex):
                idx = self.SamplingIndices[nex][motion_state]
                if idx.numel() == 0:
                    continue
                warped = self._grid_sample_complex(image[nex], ux, uy)
                for coil in range(Ncoils):
                    coil_img = warped * self.SensitivityMaps[coil].squeeze(-1)
                    k = fftnc(coil_img, dims=(0, 1))
                    kspace[coil, nex, idx] = k.flatten()[idx]
        return kspace.flatten()

    def forward(self, MotionModelPerturbation):
        n_states = len(self.SamplingIndices[0])
        if self.motionOperator.motion_type == "rigid":
            delta = MotionModelPerturbation.reshape(self.Nalpha, n_states)
        else:
            nx, ny = self.SensitivityMaps.shape[1], self.SensitivityMaps.shape[2]
            delta = MotionModelPerturbation.reshape(self.Nalpha, nx, ny)

        alpha0 = self.motionOperator.alpha.detach().to(self.device, dtype=torch.float32)
        delta = delta.to(self.device)

        def directional(d):
            d = d.to(dtype=torch.float32)
            scale = torch.max(torch.abs(d)).item()
            eps = 1e-3 / max(scale, 1e-6)
            with torch.no_grad():
                y_p = self._forward_alpha(alpha0 + eps * d)
                y_m = self._forward_alpha(alpha0 - eps * d)
            return (y_p - y_m) / (2.0 * eps)

        if torch.is_complex(delta):
            return directional(delta.real) + 1j * directional(delta.imag)
        return directional(delta)
    

    def adjoint(self, ResidualKspace):
        alpha0 = self.motionOperator.alpha.detach().to(self.device, dtype=torch.float32)
        with torch.enable_grad():
            _, vjp = torch.autograd.functional.vjp(
                self._forward_alpha,
                alpha0,
                v=ResidualKspace.reshape(-1),
                create_graph=False,
                strict=False,
            )
        return vjp.reshape(-1)
    
    def normal(self, image):
            return self.adjoint(self.forward(image))

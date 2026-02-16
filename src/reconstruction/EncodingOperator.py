import torch
from  src.utils.fftnc import fftnc
import torch.nn.functional as F

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

    def _grid_sample_complex(self, image, ux, uy):
        # image: [Nx, Ny] complex
        # ux, uy: [Nx, Ny] real, inverse-warp displacements along x(row), y(col)
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

        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1,Nx,Ny,2]

        real = image.real.unsqueeze(0).unsqueeze(0)
        imag = image.imag.unsqueeze(0).unsqueeze(0)
        real_w = F.grid_sample(
            real, grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0]
        imag_w = F.grid_sample(
            imag, grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0]
        return torch.complex(real_w, imag_w)

    def _displacement_fields(self, motion_state, nx, ny, dtype, device):
        if self.motionOperator.motion_type == "rigid":
            alpha = self.motionOperator.alpha
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

        signal = torch.as_tensor(
            self.motionOperator.motion_signal, device=device, dtype=dtype
        )
        alpha_x = self.motionOperator.alpha[0]
        alpha_y = self.motionOperator.alpha[1]
        ux = alpha_x * signal[motion_state]
        uy = alpha_y * signal[motion_state]
        return ux, uy

    def _forward_impl(self, image):
        # ---- Sizes ----
        Ncoils, nx, ny, _ = self.smaps.shape
        N_mot_states = len(self.SamplingIndices[0])
        KspaceData = torch.zeros(
            (Ncoils, self.Nex, self.Nsamples), dtype=torch.complex64, device=self.device
        )
        image = image.reshape(self.Nex, nx, ny)

        for motion_state in range(N_mot_states):
            ux, uy = self._displacement_fields(
                motion_state, nx, ny, image.real.dtype, image.device
            )

            for nex in range(self.Nex):
                SamplingIndices = self.SamplingIndices[nex][motion_state]
                if SamplingIndices.numel() == 0:
                    continue
                image_nex = image[nex]
                WarpedImage = self._grid_sample_complex(image_nex, ux, uy)

                for coil in range(Ncoils):
                    WarpedImageSeenByCoil = WarpedImage * self.smaps[coil].squeeze(-1)
                    WarpedImageFT = fftnc(WarpedImageSeenByCoil, dims=(0, 1))
                    KspaceData[coil, nex, SamplingIndices] = WarpedImageFT.flatten()[SamplingIndices]

        return KspaceData.flatten()

    def forward(self, image):
        return self._forward_impl(image)

        

    def adjoint(self, KspaceData):
        with torch.enable_grad():
            Ncoils, nx, ny, _ = self.smaps.shape
            image0 = torch.zeros((self.Nex, nx, ny), dtype=torch.complex64, device=self.device, requires_grad=True)
            y = self._forward_impl(image0.reshape(-1))
            grad = torch.autograd.grad(
                outputs=y,
                inputs=image0,
                grad_outputs=KspaceData.reshape_as(y),
                retain_graph=False,
                create_graph=False,
            )[0]
        return grad.reshape(-1)
    
    def normal(self, image):
        return self.adjoint(self.forward(image))
        

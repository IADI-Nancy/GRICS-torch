import torch

"""
Translational and rotational motion derivatives:
∂u_x/∂t_x = 1
∂u_y/∂t_x = 0

∂u_x/∂t_y = 0
∂u_y/∂t_y = 1

∂u_x/∂θ =-(x-c_x)sinθ-(y-c_y)cosθ
∂u_y/∂θ = (x-c_x)cosθ-(y-c_y)sinθ

∂u_x/∂c_x = 1 - cosθ
∂u_y/∂c_x = sinθ

∂u_x/∂c_y = -sinθ
∂u_y/∂c_y = 1 - cosθ

"""


class MotionOperator:
    def __init__(self, alpha, s, motion_type='translation'):
        self.alpha = alpha
        self.s = s
        self.motion_type = motion_type

    def theta_derivative(self, theta, center):
        """
        Compute dX/dθ and dY/dθ fields for all pixels given theta and center (cx,cy).
        Returns tensors shape (Nx,Ny) of floats (same device).
        """
        Nx, Ny = self.Nx, self.Ny
        device = self.device
        dtype = torch.float32

        # i = row index [0..Nx-1], j = col index [0..Ny-1]
        coords_x = torch.arange(Nx, device=device, dtype=dtype)
        coords_y = torch.arange(Ny, device=device, dtype=dtype)
        Y, X = torch.meshgrid(coords_y, coords_x, indexing='ij')

        cx, cy = center
        xmc = X - cx   # (x - cx): note x as row index
        ymc = Y - cy   # (y - cy): note y as col index

        st = torch.sin(theta)
        ct = torch.cos(theta)

        dX_dtheta = -st * xmc - ct * ymc
        dY_dtheta =  ct * xmc - st * ymc

        return dX_dtheta.to(self.device), dY_dtheta.to(self.device)
    
    def cx_derivative(self, theta):
        """
        Compute dX/dc_x and dY/dc_x fields for all pixels given theta.
        Returns tensors shape (Nx,Ny) of floats (same device).
        """
        Nx, Ny = self.Nx, self.Ny
        device = self.device
        dtype = torch.float32

        # i = row index [0..Nx-1], j = col index [0..Ny-1]
        coords_x = torch.arange(Nx, device=device, dtype=dtype)
        coords_y = torch.arange(Ny, device=device, dtype=dtype)
        Y, X = torch.meshgrid(coords_y, coords_x, indexing='ij')

        st = torch.sin(theta)
        ct = torch.cos(theta)

        dX_dc_x = 1 - ct
        dY_dc_x = st

        return dX_dc_x.to(self.device), dY_dc_x.to(self.device)
    
    def cy_derivative(self, theta):
        """
        Compute dX/dc_y and dY/dc_y fields for all pixels given theta.
        Returns tensors shape (Nx,Ny) of floats (same device).
        """
        Nx, Ny = self.Nx, self.Ny
        device = self.device
        dtype = torch.float32

        # i = row index [0..Nx-1], j = col index [0..Ny-1]
        coords_x = torch.arange(Nx, device=device, dtype=dtype)
        coords_y = torch.arange(Ny, device=device, dtype=dtype)
        Y, X = torch.meshgrid(coords_y, coords_x, indexing='ij')

        st = torch.sin(theta)
        ct = torch.cos(theta)

        dX_dc_y = -st
        dY_dc_y = 1 - ct

        return dX_dc_y.to(self.device), dY_dc_y.to(self.device)

    #  # translations (constant)
    #         dux = d[0, shot] * torch.ones((Nx, Ny), device=self.device)
    #         duy = d[1, shot] * torch.ones((Nx, Ny), device=self.device)

    #         # rotation component
    #         dtheta = d[2, shot]
    #         if dtheta != 0:
    #             # decide center for this shot
    #             if self.centers is not None:
    #                 center = (float(self.centers[shot, 0]), float(self.centers[shot, 1]))
    #             else:
    #                 center = self.center_global
    #             # use current rotation angle if available (otherwise 0)
    #             theta0 = float(self.rotations[shot]) if (self.rotations is not None) else 0.0
    #             # dX_dtheta/dY_dtheta fields evaluated at theta0
    #             dX_dtheta, dY_dtheta = self._rotation_fields(theta0, center)
    #             dux = dux + dX_dtheta * dtheta
    #             duy = duy + dY_dtheta * dtheta

    @staticmethod
    def create_sparse_motion_operator(Ux, Uy):
        """
        PyTorch version of MATLAB create_sparse_motion_operator.
        Produces a sparse linear interpolation matrix M for the displacement fields Ux, Uy.

        Inputs:
            Ux, Uy : displacement fields of shape (Ny, Nx)
                    (inverse displacement, as in MATLAB)

        Output:
            M : torch.sparse_coo_tensor of shape (Ny*Nx, Ny*Nx)
        """

        device = Ux.device
        dtype  = Ux.dtype

        Nx, Ny = Ux.shape

        # -----------------------------
        # Output grid (same as MATLAB's meshgrid(1:Nx, 1:Ny))
        # -----------------------------
        coords_x = torch.arange(1, Nx + 1, device=device, dtype=dtype)
        coords_y = torch.arange(1, Ny + 1, device=device, dtype=dtype)
        Y, X = torch.meshgrid(coords_y, coords_x, indexing="xy")   # shape (Nx, Ny)

        # -----------------------------
        # Coordinates after displacement
        # -----------------------------
        Xi = X + Ux
        Yi = Y + Uy

        # -----------------------------
        # Surrounding integer coordinates
        # -----------------------------
        Xi_i = Xi.floor()
        Yi_i = Yi.floor()

        # Boundaries
        Xi_i = Xi_i.clamp(1, Nx - 1)
        Yi_i = Yi_i.clamp(1, Ny - 1)

        # -----------------------------
        # Flatten sizes
        # -----------------------------
        N3  = Nx * Ny
        N3i = N3

        # -----------------------------
        # Prepare row indices (same as repmat(1:N3i, 4, 1))
        # -----------------------------
        row = torch.arange(1, N3i + 1, device=device, dtype=torch.long)
        rowIndices = row.repeat(4) - 1   # PyTorch 0-based indexing

        # -----------------------------
        # Prepare column indices
        # -----------------------------
        Xi_i_flat = Xi_i.reshape(-1).long()     # (N3,)
        Yi_i_flat = Yi_i.reshape(-1).long()

        # MATLAB uses: index = y + (x-1)*Ny  (1-based)
        # Convert to 0-based PyTorch:
        base = (Yi_i_flat - 1) + (Xi_i_flat - 1) * Ny

        col1 = base
        col2 = (Yi_i_flat - 1) + (Xi_i_flat) * Ny          # x+1
        col3 = (Yi_i_flat)     + (Xi_i_flat - 1) * Ny      # y+1
        col4 = (Yi_i_flat)     + (Xi_i_flat) * Ny          # x+1, y+1

        colIndices = torch.cat([col1, col2, col3, col4], dim=0)

        # -----------------------------
        # Interpolation weights
        # -----------------------------
        Xi_f = Xi.reshape(-1)
        Yi_f = Yi.reshape(-1)

        d_x_xi   = 1 - torch.abs(Xi_i_flat - Xi_f)
        d_y_yi   = 1 - torch.abs(Yi_i_flat - Yi_f)
        d_x_xip1 = 1 - d_x_xi
        d_y_yip1 = 1 - d_y_yi

        val1 = d_x_xi   * d_y_yi
        val2 = d_x_xip1 * d_y_yi
        val3 = d_x_xi   * d_y_yip1
        val4 = d_x_xip1 * d_y_yip1

        values = torch.cat([val1, val2, val3, val4], dim=0)

        # -----------------------------
        # Build sparse matrix
        # -----------------------------
        indices = torch.stack([rowIndices, colIndices], dim=0)
        M = torch.sparse_coo_tensor(indices, values, size=(N3i, N3), device=device, dtype=torch.complex64)

        return M
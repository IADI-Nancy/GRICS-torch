import torch

"""
Translational and rotational motion derivatives:
∂M_x/∂t_x = 1
∂M_y/∂t_x = 0

∂M_x/∂t_y = 0
∂M_y/∂t_y = 1

∂M_x/∂θ = -(x-c_x)sinθ-(y-c_y)cosθ
∂M_y/∂θ = (x-c_x)cosθ-(y-c_y)sinθ

∂M_x/∂c_x = 1 - cosθ
∂M_y/∂c_x = -sinθ

∂M_x/∂c_y = sinθ
∂M_y/∂c_y = 1 - cosθ

"""


class MotionOperator:
    def __init__(self, Nx, Ny, device):
        self.Nx = Nx
        self.Ny = Ny
        self.device = device

        # Precompute meshgrid
        coords_x = torch.arange(Nx, device=device, dtype=torch.float32)
        coords_y = torch.arange(Ny, device=device, dtype=torch.float32)
        self.X, self.Y = torch.meshgrid(coords_x, coords_y, indexing='ij')


    # ---------------------------------------------------------
    # Geometric derivatives
    # ---------------------------------------------------------

    def translation_derivative(self):
        ones = torch.ones((self.Nx, self.Ny), device=self.device)
        zeros = torch.zeros_like(ones)
        return (ones, zeros), (zeros, ones)  # (∂Mx/∂tx, ∂My/∂tx), (∂Mx/∂ty, ∂My/∂ty)


    def theta_derivative(self, theta, center):
        cx, cy = center
        xmc = self.X - cx
        ymc = self.Y - cy

        st = torch.sin(theta)
        ct = torch.cos(theta)

        dMx = -st * xmc - ct * ymc
        dMy =  ct * xmc - st * ymc

        return dMx, dMy


    def cx_derivative(self, theta):
        ones = torch.ones((self.Nx, self.Ny), device=self.device)
        st = torch.sin(theta)
        ct = torch.cos(theta)

        dMx = (1 - ct) * ones
        dMy = -st * ones

        return dMx, dMy


    def cy_derivative(self, theta):
        ones = torch.ones((self.Nx, self.Ny), device=self.device)
        st = torch.sin(theta)
        ct = torch.cos(theta)

        dMx = st * ones
        dMy = (1 - ct) * ones
        return dMx, dMy


    # ---------------------------------------------------------
    # FULL JACOBIAN OPERATOR  J * delta_alpha = du
    # ---------------------------------------------------------

    def apply_J(self, delta_alpha, theta, center):
        """
        delta_alpha : (5,) tensor [dt_x, dt_y, dtheta, dc_x, dc_y]
        returns:
            du_x, du_y  (each Nx x Ny)
        """

        dt_x, dt_y, dtheta, dc_x, dc_y = delta_alpha

        # Derivative fields
        (dMx_dtx, dMy_dtx), (dMx_dty, dMy_dty) = self.translation_derivative()
        dMx_dtheta, dMy_dtheta = self.theta_derivative(theta, center)
        dMx_dcx, dMy_dcx       = self.cx_derivative(theta)
        dMx_dcy, dMy_dcy       = self.cy_derivative(theta)

        # Combine linearly
        du_x = (
            dt_x * dMx_dtx +
            dt_y * dMx_dty +
            dtheta * dMx_dtheta +
            dc_x  * dMx_dcx +
            dc_y  * dMx_dcy
        )

        du_y = (
            dt_x * dMy_dtx +
            dt_y * dMy_dty +
            dtheta * dMy_dtheta +
            dc_x  * dMy_dcx +
            dc_y  * dMy_dcy
        )

        return du_x, du_y


    # ---------------------------------------------------------
    # ADJOINT (TRANSPOSE–CONJUGATE) JACOBIAN   J^H * du
    # ---------------------------------------------------------

    def apply_JH(self, du_x, du_y, theta, center):
        """
        Applies J^H to (du_x, du_y).
        Returns
            delta_alpha : (5,) tensor
        """
        # zeros = torch.zeros((self.Nx, self.Ny), device=self.device)

        (dMx_dtx, dMy_dtx), (dMx_dty, dMy_dty) = self.translation_derivative()
        dMx_dtheta, dMy_dtheta = self.theta_derivative(theta, center)
        dMx_dcx, dMy_dcx       = self.cx_derivative(theta)
        dMx_dcy, dMy_dcy       = self.cy_derivative(theta)

        # Each parameter is a scalar product <v_m, du>
        dt_x  = torch.sum(dMx_dtx * du_x + dMy_dtx * du_y)
        dt_y  = torch.sum(dMx_dty * du_x + dMy_dty * du_y)
        dtheta = torch.sum(dMx_dtheta * du_x + dMy_dtheta * du_y)
        dc_x   = torch.sum(dMx_dcx * du_x + dMy_dcx * du_y)
        dc_y   = torch.sum(dMx_dcy * du_x + dMy_dcy * du_y)

        return torch.stack([dt_x, dt_y, dtheta, dc_x, dc_y])


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
import torch


class MotionOperator:
    def __init__(self, alpha, s, motion_type='translation'):
        self.alpha = alpha
        self.s = s
        self.motion_type = motion_type

    # def get_operator(self, s, alpha):
    #     Nshots = s.shape[0]
    #     if self.motion_type == 'translation':
    #         t_x = s[:, 0]
    #         t_y = s[:, 1]   
    #         self.MotionOperator = []                 
    #         self.Ux_list = []
    #         self.Uy_list = []

    #         for shot in range(Nshots):
    #             # ----------------------------
    #             # Expand translations
    #             # MATLAB uses inverse displacement: tx = -XTranslation(shot)
    #             # ----------------------------
    #             tx = -t_x[shot].to(self.t_device)
    #             ty = -t_y[shot].to(self.t_device)
    #     else:
    #         raise NotImplementedError(f"Motion type {self.motion_type} not implemented.")

    #     M = self.create_sparse_motion_operator(Ux, Uy)
    #     return M

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
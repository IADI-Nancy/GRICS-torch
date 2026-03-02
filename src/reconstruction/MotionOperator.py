import torch

"""
Translational and rotational motion derivatives:
∂X/∂t_x = 1
∂Y/∂t_x = 0

∂X/∂t_y = 0
∂Y/∂t_y = 1

∂X/∂φ = -(x-c_x)sinφ-(y-c_y)cosφ
∂Y/∂φ = (x-c_x)cosφ-(y-c_y)sinφ
"""


class MotionOperator:
    def __init__(self, Nx, Ny, alpha, motion_type, centers=None, motion_signal=None):
        self.Nx = Nx
        self.Ny = Ny
        self.device = alpha.device
        self.alpha = alpha
        self.motion_type = motion_type
        self.centers = centers
        self.motion_signal = motion_signal
        if motion_type == 'rigid':
            self._initialize_rigid_motion_operator()
        elif motion_type == 'non-rigid':
            self._initialize_non_rigid_motion_operator()
        else:
            raise ValueError(f"Unknown motion type: {motion_type}")    

    def get_sparse_operator(self, motion_state):
        return self.sparseMotionOperator[motion_state]
    
    # ---------------------------------------------------------
    # Rigid motion
    # ---------------------------------------------------------
    def _initialize_rigid_motion_operator(self):
        Nx = self.Nx
        Ny = self.Ny
        alpha = self.alpha
        centers = self.centers if hasattr(self, 'centers') else None

        if centers is None:
            centers = torch.zeros((2,alpha.shape[1]), device=self.device)
            centers[0,:] = Nx/2 * torch.ones(alpha.shape[1], device=self.device)
            centers[1,:] = Ny/2 * torch.ones(alpha.shape[1], device=self.device)
        self.centers = centers

        # Precompute meshgrid
        coords_x = torch.arange(Nx, device=self.device, dtype=torch.float64)
        coords_y = torch.arange(Ny, device=self.device, dtype=torch.float64)
        self.X, self.Y = torch.meshgrid(coords_x, coords_y, indexing='ij')

        # Build coordinate grid (absolute coordinates 0..Nx-1, 0..Ny-1)
        xs = torch.arange(Nx, device=self.device)
        ys = torch.arange(Ny, device=self.device)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')

        self.sparseMotionOperator = []

        for motion_state in range(alpha.shape[1]):
            # --- Motion parameters ---
            tx = alpha[0, motion_state]     # translation x
            ty = alpha[1, motion_state]     # translation y
            phi = alpha[2, motion_state]     # rotation (radians)
            cx = centers[0, motion_state]     # center x
            cy = centers[1, motion_state]     # center y

            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            # Shift coordinates to rotation center
            Xc = X - cx
            Yc = Y - cy

            # --- Rigid transform around (cx, cy) ---
            Xp = cx + cos_phi * Xc - sin_phi * Yc + tx
            Yp = cy + sin_phi * Xc + cos_phi * Yc + ty

            # --- Displacement fields ---
            Ux = Xp - X
            Uy = Yp - Y

            # Create sparse operator
            MotionOp = MotionOperator.create_sparse_motion_operator(Ux, Uy)
            self.sparseMotionOperator.append(MotionOp)

    # ------------------------ Geometric derivatives ----------------------------

    def _translation_derivative(self):
        ones = torch.ones((self.Nx, self.Ny), device=self.device)
        zeros = torch.zeros_like(ones)
        return (ones, zeros), (zeros, ones)


    def _phi_derivative(self, motion_state):
        phi = self.alpha[2,motion_state] 
        xmc = self.X - self.centers[0,motion_state]
        ymc = self.Y - self.centers[1,motion_state]

        st = torch.sin(phi)
        ct = torch.cos(phi)

        dX = -st * xmc - ct * ymc
        dY =  ct * xmc - st * ymc

        return dX, dY

    # ----------------- FULL JACOBIAN OPERATOR J * δᾱ_j = sum_j(∂x_i/∂ᾱ_j) δᾱ_j = δu_i -----------------

    def apply_J(self, delta_alpha, motion_state):
        """
        delta_alpha : (3,) tensor [dt_x, dt_y, dphi, dc_x, dc_y]
        returns:
            du_x, du_y  (each Nx x Ny)
        """
        if self.motion_type != 'rigid':
            raise NotImplementedError("apply_J is only implemented for rigid motion.")

        dt_x, dt_y, dphi = delta_alpha

        # Derivative fields
        (dX_dtx, dY_dtx), (dX_dty, dY_dty) = self._translation_derivative()
        dX_dphi, dY_dphi = self._phi_derivative(motion_state)

        # Combine linearly
        du_x = (
            dt_x * dX_dtx +
            dt_y * dX_dty +
            dphi * dX_dphi
        )

        du_y = (
            dt_x * dY_dtx +
            dt_y * dY_dty +
            dphi * dY_dphi
        )

        return du_x, du_y


    # --------- ADJOINT (TRANSPOSE–CONJUGATE) JACOBIAN   JH * δu_i = sum_i(∂x_i/∂ᾱ_j) δu_i = δᾱ_j -----------------

    def apply_JH(self, du_x, du_y, motion_state):
        """
        Applies J^H to (du_x, du_y).
        Returns
            delta_alpha : (3,) tensor
        """
        if self.motion_type != 'rigid':
            raise NotImplementedError("apply_JH is only implemented for rigid motion.")

        (dX_dtx, dY_dtx), (dX_dty, dY_dty) = self._translation_derivative()
        dX_dphi, dY_dphi = self._phi_derivative(motion_state)

        # Each parameter is a scalar product <v_m, du>
        dt_x  = torch.sum(dX_dtx * du_x + dY_dtx * du_y)
        dt_y  = torch.sum(dX_dty * du_x + dY_dty * du_y)
        dphi = torch.sum(dX_dphi * du_x + dY_dphi * du_y)

        return torch.stack([dt_x, dt_y, dphi])
    
    # ---------------------------------------------------------
    # -------------------- Non-rigid motion -------------------
    # ---------------------------------------------------------
    def _initialize_non_rigid_motion_operator(self):
        alpha = self.alpha
        signal = torch.as_tensor(self.motion_signal, device=self.device, dtype=alpha.dtype)

        self.sparseMotionOperator = []
        # Axis convention:
        # alpha[0] -> Ux -> axis 0 displacement (rows)
        # alpha[1] -> Uy -> axis 1 displacement (cols)
        alpha_x = alpha[0]
        alpha_y = alpha[1]
        n_states = signal.numel()

        for motion_state in range(n_states):
            ux = alpha_x * signal[motion_state]
            uy = alpha_y * signal[motion_state]
            motion_op = MotionOperator.create_sparse_motion_operator(ux, uy)
            self.sparseMotionOperator.append(motion_op)




    # ---------------------------------------------------------------------------------
    # -------- Sparse interpolation operator M for applying motion to images ----------
    # ---------------------------------------------------------------------------------

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
        # Interpolation grid is purely geometric; keep it real-valued even if
        # motion coefficients carry a complex diagnostic component.
        Ux = Ux.real if torch.is_complex(Ux) else Ux
        Uy = Uy.real if torch.is_complex(Uy) else Uy
        dtype = Ux.dtype

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

        # Clamp interpolation points to valid non-periodic domain.
        # Use upper bound (N-1) so the +1 neighbor always exists.
        x_hi = max(1, Nx - 1)
        y_hi = max(1, Ny - 1)
        Xi = Xi.clamp(1, x_hi)
        Yi = Yi.clamp(1, y_hi)

        # -----------------------------
        # Surrounding integer coordinates
        # -----------------------------
        Xi_i = Xi.floor()
        Yi_i = Yi.floor()

        # Boundaries
        Xi_i = Xi_i.clamp(1, x_hi)
        Yi_i = Yi_i.clamp(1, y_hi)

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
        M = torch.sparse_coo_tensor(indices, values, size=(N3i, N3), device=device, dtype=torch.complex128)

        return M

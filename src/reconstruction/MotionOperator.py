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
    def __init__(self, Nx, Ny, alpha, motion_type, centers=None, motion_signal=None, Nz=1):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = int(Nz)
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

    def _get_sparse_operator(self, motion_state):
        return self.sparseMotionOperator[motion_state]
    
    # ---------------------------------------------------------
    # Rigid motion
    # ---------------------------------------------------------
    def _initialize_rigid_motion_operator(self):
        if self.Nz > 1:
            self._initialize_rigid_motion_operator_3d()
        else:
            self._initialize_rigid_motion_operator_2d()

    def _initialize_rigid_motion_operator_2d(self):
        Nx = self.Nx
        Ny = self.Ny
        alpha = self.alpha
        centers = self.centers if hasattr(self, "centers") else None

        if centers is None:
            centers = torch.zeros((2, alpha.shape[1]), device=self.device)
            centers[0, :] = Nx / 2 * torch.ones(alpha.shape[1], device=self.device)
            centers[1, :] = Ny / 2 * torch.ones(alpha.shape[1], device=self.device)
        self.centers = centers

        coords_x = torch.arange(Nx, device=self.device, dtype=torch.float64)
        coords_y = torch.arange(Ny, device=self.device, dtype=torch.float64)
        self.X, self.Y = torch.meshgrid(coords_x, coords_y, indexing="ij")

        xs = torch.arange(Nx, device=self.device, dtype=torch.float64)
        ys = torch.arange(Ny, device=self.device, dtype=torch.float64)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")

        self.sparseMotionOperator = []

        for motion_state in range(alpha.shape[1]):
            tx = alpha[0, motion_state]
            ty = alpha[1, motion_state]
            phi = alpha[2, motion_state]
            cx = centers[0, motion_state]
            cy = centers[1, motion_state]

            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            Xc = X - cx
            Yc = Y - cy

            Xp = cx + cos_phi * Xc - sin_phi * Yc + tx
            Yp = cy + sin_phi * Xc + cos_phi * Yc + ty

            Ux = Xp - X
            Uy = Yp - Y

            motion_op = MotionOperator._create_sparse_motion_operator(Ux, Uy)
            self.sparseMotionOperator.append(motion_op)

    @staticmethod
    def _rotation_matrix_3d(rx, ry, rz):
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        rx_m = torch.stack(
            [
                torch.stack([torch.ones_like(rx), torch.zeros_like(rx), torch.zeros_like(rx)]),
                torch.stack([torch.zeros_like(rx), cx, -sx]),
                torch.stack([torch.zeros_like(rx), sx, cx]),
            ]
        )
        ry_m = torch.stack(
            [
                torch.stack([cy, torch.zeros_like(ry), sy]),
                torch.stack([torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry)]),
                torch.stack([-sy, torch.zeros_like(ry), cy]),
            ]
        )
        rz_m = torch.stack(
            [
                torch.stack([cz, -sz, torch.zeros_like(rz)]),
                torch.stack([sz, cz, torch.zeros_like(rz)]),
                torch.stack([torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz)]),
            ]
        )
        return rz_m @ ry_m @ rx_m

    def _initialize_rigid_motion_operator_3d(self):
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        alpha = self.alpha
        centers = self.centers if hasattr(self, "centers") else None

        if centers is None:
            centers = torch.zeros((3, alpha.shape[1]), device=self.device)
            centers[0, :] = Nx / 2 * torch.ones(alpha.shape[1], device=self.device)
            centers[1, :] = Ny / 2 * torch.ones(alpha.shape[1], device=self.device)
            centers[2, :] = Nz / 2 * torch.ones(alpha.shape[1], device=self.device)
        self.centers = centers

        coords_x = torch.arange(Nx, device=self.device, dtype=torch.float64)
        coords_y = torch.arange(Ny, device=self.device, dtype=torch.float64)
        coords_z = torch.arange(Nz, device=self.device, dtype=torch.float64)
        self.X, self.Y, self.Z = torch.meshgrid(coords_x, coords_y, coords_z, indexing="ij")

        X, Y, Z = self.X, self.Y, self.Z
        self.sparseMotionOperator = []

        n_params = int(alpha.shape[0])
        if n_params != 6:
            raise ValueError("3D rigid motion expects alpha with 6 params [tx, ty, tz, rx, ry, rz].")

        for motion_state in range(alpha.shape[1]):
            tx = alpha[0, motion_state]
            ty = alpha[1, motion_state]
            tz = alpha[2, motion_state]
            rx = alpha[3, motion_state]
            ry = alpha[4, motion_state]
            rz = alpha[5, motion_state]

            cx = centers[0, motion_state]
            cy = centers[1, motion_state]
            cz = centers[2, motion_state]

            r = MotionOperator._rotation_matrix_3d(rx, ry, rz).to(dtype=X.dtype, device=self.device)
            xc = X - cx
            yc = Y - cy
            zc = Z - cz

            xp = r[0, 0] * xc + r[0, 1] * yc + r[0, 2] * zc + cx + tx
            yp = r[1, 0] * xc + r[1, 1] * yc + r[1, 2] * zc + cy + ty
            zp = r[2, 0] * xc + r[2, 1] * yc + r[2, 2] * zc + cz + tz

            ux = xp - X
            uy = yp - Y
            uz = zp - Z

            motion_op = MotionOperator._create_sparse_motion_operator_3d(ux, uy, uz)
            self.sparseMotionOperator.append(motion_op)

    # ------------------------ Geometric derivatives ----------------------------

    def _translation_derivative(self):
        if self.Nz > 1:
            ones = torch.ones((self.Nx, self.Ny, self.Nz), device=self.device)
            zeros = torch.zeros_like(ones)
            return (ones, zeros, zeros), (zeros, ones, zeros), (zeros, zeros, ones)
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

    def _rigid_params_3d(self, motion_state):
        n_params = int(self.alpha.shape[0])
        if n_params != 6:
            raise ValueError("3D rigid motion expects alpha with 6 params [tx, ty, tz, rx, ry, rz].")
        tx = self.alpha[0, motion_state]
        ty = self.alpha[1, motion_state]
        tz = self.alpha[2, motion_state]
        rx = self.alpha[3, motion_state]
        ry = self.alpha[4, motion_state]
        rz = self.alpha[5, motion_state]
        return n_params, tx, ty, tz, rx, ry, rz

    @staticmethod
    def _rotation_matrix_derivatives_3d(rx, ry, rz):
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        rx_m = torch.stack(
            [
                torch.stack([torch.ones_like(rx), torch.zeros_like(rx), torch.zeros_like(rx)]),
                torch.stack([torch.zeros_like(rx), cx, -sx]),
                torch.stack([torch.zeros_like(rx), sx, cx]),
            ]
        )
        ry_m = torch.stack(
            [
                torch.stack([cy, torch.zeros_like(ry), sy]),
                torch.stack([torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry)]),
                torch.stack([-sy, torch.zeros_like(ry), cy]),
            ]
        )
        rz_m = torch.stack(
            [
                torch.stack([cz, -sz, torch.zeros_like(rz)]),
                torch.stack([sz, cz, torch.zeros_like(rz)]),
                torch.stack([torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz)]),
            ]
        )

        drx_m = torch.stack(
            [
                torch.stack([torch.zeros_like(rx), torch.zeros_like(rx), torch.zeros_like(rx)]),
                torch.stack([torch.zeros_like(rx), -sx, -cx]),
                torch.stack([torch.zeros_like(rx), cx, -sx]),
            ]
        )
        dry_m = torch.stack(
            [
                torch.stack([-sy, torch.zeros_like(ry), cy]),
                torch.stack([torch.zeros_like(ry), torch.zeros_like(ry), torch.zeros_like(ry)]),
                torch.stack([-cy, torch.zeros_like(ry), -sy]),
            ]
        )
        drz_m = torch.stack(
            [
                torch.stack([-sz, -cz, torch.zeros_like(rz)]),
                torch.stack([cz, -sz, torch.zeros_like(rz)]),
                torch.stack([torch.zeros_like(rz), torch.zeros_like(rz), torch.zeros_like(rz)]),
            ]
        )
        dr_drx = rz_m @ ry_m @ drx_m
        dr_dry = rz_m @ dry_m @ rx_m
        dr_drz = drz_m @ ry_m @ rx_m
        return dr_drx, dr_dry, dr_drz

    def _rigid_3d_rotation_derivatives(self, motion_state):
        _, _, _, _, rx, ry, rz = self._rigid_params_3d(motion_state)
        dr_drx, dr_dry, dr_drz = MotionOperator._rotation_matrix_derivatives_3d(rx, ry, rz)
        dr_drx = dr_drx.to(dtype=self.X.dtype, device=self.device)
        dr_dry = dr_dry.to(dtype=self.X.dtype, device=self.device)
        dr_drz = dr_drz.to(dtype=self.X.dtype, device=self.device)

        cx = self.centers[0, motion_state]
        cy = self.centers[1, motion_state]
        cz = self.centers[2, motion_state]
        xc = self.X - cx
        yc = self.Y - cy
        zc = self.Z - cz

        dX_drx = dr_drx[0, 0] * xc + dr_drx[0, 1] * yc + dr_drx[0, 2] * zc
        dY_drx = dr_drx[1, 0] * xc + dr_drx[1, 1] * yc + dr_drx[1, 2] * zc
        dZ_drx = dr_drx[2, 0] * xc + dr_drx[2, 1] * yc + dr_drx[2, 2] * zc

        dX_dry = dr_dry[0, 0] * xc + dr_dry[0, 1] * yc + dr_dry[0, 2] * zc
        dY_dry = dr_dry[1, 0] * xc + dr_dry[1, 1] * yc + dr_dry[1, 2] * zc
        dZ_dry = dr_dry[2, 0] * xc + dr_dry[2, 1] * yc + dr_dry[2, 2] * zc

        dX_drz = dr_drz[0, 0] * xc + dr_drz[0, 1] * yc + dr_drz[0, 2] * zc
        dY_drz = dr_drz[1, 0] * xc + dr_drz[1, 1] * yc + dr_drz[1, 2] * zc
        dZ_drz = dr_drz[2, 0] * xc + dr_drz[2, 1] * yc + dr_drz[2, 2] * zc

        return (dX_drx, dY_drx, dZ_drx), (dX_dry, dY_dry, dZ_dry), (dX_drz, dY_drz, dZ_drz)

    # ----------------- FULL JACOBIAN OPERATOR J * δᾱ_j = sum_j(∂x_i/∂ᾱ_j) δᾱ_j = δu_i -----------------

    def _apply_J(self, delta_alpha, motion_state):
        """
        delta_alpha : (3,) tensor [dt_x, dt_y, dphi, dc_x, dc_y]
        returns:
            du_x, du_y  (each Nx x Ny)
        """
        if self.Nz > 1:
            _, _, _, _, _, _, _ = self._rigid_params_3d(motion_state)
            dt_x, dt_y, dt_z, dr_x, dr_y, dr_z = delta_alpha

            (dX_dtx, dY_dtx, dZ_dtx), (dX_dty, dY_dty, dZ_dty), (dX_dtz, dY_dtz, dZ_dtz) = self._translation_derivative()
            (dX_drx, dY_drx, dZ_drx), (dX_dry, dY_dry, dZ_dry), (dX_drz, dY_drz, dZ_drz) = self._rigid_3d_rotation_derivatives(motion_state)

            du_x = (
                dt_x * dX_dtx + dt_y * dX_dty + dt_z * dX_dtz +
                dr_x * dX_drx + dr_y * dX_dry + dr_z * dX_drz
            )
            du_y = (
                dt_x * dY_dtx + dt_y * dY_dty + dt_z * dY_dtz +
                dr_x * dY_drx + dr_y * dY_dry + dr_z * dY_drz
            )
            du_z = (
                dt_x * dZ_dtx + dt_y * dZ_dty + dt_z * dZ_dtz +
                dr_x * dZ_drx + dr_y * dZ_dry + dr_z * dZ_drz
            )
            return du_x, du_y, du_z
        else:
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

    def _apply_JH(self, du_x, du_y, motion_state, du_z=None):
        """
        Applies J^H to (du_x, du_y).
        Returns
            delta_alpha : (3,) tensor
        """
        if self.Nz > 1:
            if du_z is None:
                raise ValueError("du_z is required for 3D rigid _apply_JH.")
            _, _, _, _, _, _, _ = self._rigid_params_3d(motion_state)
            (dX_dtx, dY_dtx, dZ_dtx), (dX_dty, dY_dty, dZ_dty), (dX_dtz, dY_dtz, dZ_dtz) = self._translation_derivative()
            (dX_drx, dY_drx, dZ_drx), (dX_dry, dY_dry, dZ_dry), (dX_drz, dY_drz, dZ_drz) = self._rigid_3d_rotation_derivatives(motion_state)

            dt_x = torch.sum(dX_dtx * du_x + dY_dtx * du_y + dZ_dtx * du_z)
            dt_y = torch.sum(dX_dty * du_x + dY_dty * du_y + dZ_dty * du_z)
            dt_z = torch.sum(dX_dtz * du_x + dY_dtz * du_y + dZ_dtz * du_z)
            dr_x = torch.sum(dX_drx * du_x + dY_drx * du_y + dZ_drx * du_z)
            dr_y = torch.sum(dX_dry * du_x + dY_dry * du_y + dZ_dry * du_z)
            dr_z = torch.sum(dX_drz * du_x + dY_drz * du_y + dZ_drz * du_z)
            return torch.stack([dt_x, dt_y, dt_z, dr_x, dr_y, dr_z])

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
        n_states = signal.numel()
        if self.Nz > 1:
            if alpha.ndim != 4 or alpha.shape[0] < 3:
                raise ValueError("3D non-rigid motion expects alpha with shape (3, Nx, Ny, Nz).")
            alpha_x = alpha[0]
            alpha_y = alpha[1]
            alpha_z = alpha[2]
            for motion_state in range(n_states):
                ux = alpha_x * signal[motion_state]
                uy = alpha_y * signal[motion_state]
                uz = alpha_z * signal[motion_state]
                motion_op = MotionOperator._create_sparse_motion_operator_3d(ux, uy, uz)
                self.sparseMotionOperator.append(motion_op)
        else:
            # Axis convention:
            # alpha[0] -> Ux -> axis 0 displacement (rows)
            # alpha[1] -> Uy -> axis 1 displacement (cols)
            alpha_x = alpha[0]
            alpha_y = alpha[1]
            for motion_state in range(n_states):
                ux = alpha_x * signal[motion_state]
                uy = alpha_y * signal[motion_state]
                motion_op = MotionOperator._create_sparse_motion_operator(ux, uy)
                self.sparseMotionOperator.append(motion_op)




    # ---------------------------------------------------------------------------------
    # -------- Sparse interpolation operator M for applying motion to images ----------
    # ---------------------------------------------------------------------------------

    @staticmethod
    def _create_sparse_motion_operator(Ux, Uy):
        """
        PyTorch version of MATLAB _create_sparse_motion_operator.
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

        # Clamp interpolation points to the full valid domain.
        # We clamp the lower neighbor separately so exact samples on the final
        # voxel layer are preserved instead of being collapsed onto N-1.
        x_hi = max(1, Nx - 1)
        y_hi = max(1, Ny - 1)
        Xi = Xi.clamp(1, Nx)
        Yi = Yi.clamp(1, Ny)

        # -----------------------------
        # Surrounding integer coordinates
        # -----------------------------
        Xi_i = Xi.floor().clamp(1, x_hi)
        Yi_i = Yi.floor().clamp(1, y_hi)

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

    @staticmethod
    def _create_sparse_motion_operator_3d(Ux, Uy, Uz):
        """
        3D trilinear sparse interpolation operator for displacement fields Ux, Uy, Uz.

        Inputs:
            Ux, Uy, Uz : displacement fields of shape (Nx, Ny, Nz)
        Output:
            M : sparse matrix of shape (Nx*Ny*Nz, Nx*Ny*Nz)
        """
        device = Ux.device
        Ux = Ux.real if torch.is_complex(Ux) else Ux
        Uy = Uy.real if torch.is_complex(Uy) else Uy
        Uz = Uz.real if torch.is_complex(Uz) else Uz
        dtype = Ux.dtype

        Nx, Ny, Nz = Ux.shape

        x = torch.arange(1, Nx + 1, device=device, dtype=dtype)
        y = torch.arange(1, Ny + 1, device=device, dtype=dtype)
        z = torch.arange(1, Nz + 1, device=device, dtype=dtype)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

        x_hi = max(1, Nx - 1)
        y_hi = max(1, Ny - 1)
        z_hi = max(1, Nz - 1)
        Xi = (X + Ux).clamp(1, Nx)
        Yi = (Y + Uy).clamp(1, Ny)
        Zi = (Z + Uz).clamp(1, Nz)

        Xi_i = Xi.floor().clamp(1, x_hi)
        Yi_i = Yi.floor().clamp(1, y_hi)
        Zi_i = Zi.floor().clamp(1, z_hi)

        n = Nx * Ny * Nz
        row = torch.arange(n, device=device, dtype=torch.long)
        row_idx = row.repeat(8)

        xi = Xi_i.reshape(-1).long()
        yi = Yi_i.reshape(-1).long()
        zi = Zi_i.reshape(-1).long()

        base = ((xi - 1) * Ny + (yi - 1)) * Nz + (zi - 1)
        c000 = base
        c100 = ((xi) * Ny + (yi - 1)) * Nz + (zi - 1)
        c010 = ((xi - 1) * Ny + yi) * Nz + (zi - 1)
        c110 = (xi * Ny + yi) * Nz + (zi - 1)
        c001 = ((xi - 1) * Ny + (yi - 1)) * Nz + zi
        c101 = (xi * Ny + (yi - 1)) * Nz + zi
        c011 = ((xi - 1) * Ny + yi) * Nz + zi
        c111 = (xi * Ny + yi) * Nz + zi
        col_idx = torch.cat([c000, c100, c010, c110, c001, c101, c011, c111], dim=0)

        Xi_f = Xi.reshape(-1)
        Yi_f = Yi.reshape(-1)
        Zi_f = Zi.reshape(-1)
        wx2 = Xi_f - xi.to(dtype)
        wy2 = Yi_f - yi.to(dtype)
        wz2 = Zi_f - zi.to(dtype)
        wx1 = 1 - wx2
        wy1 = 1 - wy2
        wz1 = 1 - wz2

        w000 = wx1 * wy1 * wz1
        w100 = wx2 * wy1 * wz1
        w010 = wx1 * wy2 * wz1
        w110 = wx2 * wy2 * wz1
        w001 = wx1 * wy1 * wz2
        w101 = wx2 * wy1 * wz2
        w011 = wx1 * wy2 * wz2
        w111 = wx2 * wy2 * wz2
        vals = torch.cat([w000, w100, w010, w110, w001, w101, w011, w111], dim=0)

        indices = torch.stack([row_idx, col_idx], dim=0)
        return torch.sparse_coo_tensor(indices, vals, size=(n, n), device=device, dtype=torch.complex128).coalesce()

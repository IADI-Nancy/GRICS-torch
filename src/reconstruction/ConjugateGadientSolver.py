import torch

"""
Source: ChatGPT + Wikipedia
"""
class ConjugateGradientSolver:
    """
    CG and PCG solver for equations of the form:
        A(x) = b
    where A(x) = Eh(E) ('E' is the encoding operator, "h" - Hermitian conjugate).
    """

    def __init__(
        self,
        encoding_operator,
        reg_lambda=0.0,
        regularizer="Tikhonov",
        regularization_shape=None,
        verbose=False,
        early_stopping=True,
        true_residual_interval=10,
        max_stag_steps=3,
        max_more_steps=None,
    ):
        """
        encoding_operator : instance of EncodingOperator
        motion_operator   : list of motion operators (same used inside forward/backward)
        """
        self.E = encoding_operator
        self.device = encoding_operator.device
        self.lambda_ = reg_lambda
        self.regularizer = regularizer
        self.regularization_shape = regularization_shape
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.true_residual_interval = true_residual_interval
        self.max_stag_steps = max_stag_steps
        self.max_more_steps = max_more_steps

    # --------------------------------------------------------------
    # Regularized linear operator: A(x) = Eh(E(x)) + lambda_scaled * x
    # --------------------------------------------------------------
    def A(self, x):
        lam = self.lambda_scaled if hasattr(self, "lambda_scaled") else self.lambda_
        return self.E.normal(x) + lam * self.regularization(x)
    
    def regularization(self, x):
        if self.regularizer == "Tikhonov":
            return x
        elif self.regularizer == "Tikhonov_gradient":
            return self.gradient_op(x)
        elif self.regularizer == "Tikhonov_laplacian":
            return self.laplacian_op(x)
        else:
            raise ValueError("Unknown regularizer")
    
    def gradient_op(self, x):
        if self.regularization_shape is None:
            raise ValueError("regularization_shape must be set for Tikhonov_gradient regularization.")
        field = x.view(*self.regularization_shape)
        # Forward differences with zero-gradient (non-periodic) boundaries.
        dx = torch.zeros_like(field)
        dy = torch.zeros_like(field)
        dx[..., :-1, :] = field[..., 1:, :] - field[..., :-1, :]
        dy[..., :, :-1] = field[..., :, 1:] - field[..., :, :-1]

        # Adjoint divergence of the forward differences.
        dxx = torch.zeros_like(field)
        dyy = torch.zeros_like(field)
        dxx[..., 0, :] = -dx[..., 0, :]
        dxx[..., 1:-1, :] = dx[..., :-2, :] - dx[..., 1:-1, :]
        dxx[..., -1, :] = dx[..., -2, :]
        dyy[..., :, 0] = -dy[..., :, 0]
        dyy[..., :, 1:-1] = dy[..., :, :-2] - dy[..., :, 1:-1]
        dyy[..., :, -1] = dy[..., :, -2]

        return (dxx + dyy).reshape(-1)
    
    def laplacian_op(self, x):
        field = x.view(*self.regularization_shape)
        nx = field.shape[-2]
        ny = field.shape[-1]

        if nx < 2 or ny < 2:
            return torch.zeros_like(field).reshape(-1)

        # MATLAB del2-like boundary treatment: linear extrapolation with
        # one-pixel ghost borders, then 5-point stencil over all pixels.
        pad = torch.zeros(*field.shape[:-2], nx + 2, ny + 2, dtype=field.dtype, device=field.device)
        pad[..., 1:-1, 1:-1] = field

        # Edge ghost values.
        pad[..., 0, 1:-1] = 2 * field[..., 0, :] - field[..., 1, :]
        pad[..., -1, 1:-1] = 2 * field[..., -1, :] - field[..., -2, :]
        pad[..., 1:-1, 0] = 2 * field[..., :, 0] - field[..., :, 1]
        pad[..., 1:-1, -1] = 2 * field[..., :, -1] - field[..., :, -2]

        # Corner ghost values (bilinear completion from adjacent ghosts).
        pad[..., 0, 0] = pad[..., 0, 1] + pad[..., 1, 0] - pad[..., 1, 1]
        pad[..., 0, -1] = pad[..., 0, -2] + pad[..., 1, -1] - pad[..., 1, -2]
        pad[..., -1, 0] = pad[..., -1, 1] + pad[..., -2, 0] - pad[..., -2, 1]
        pad[..., -1, -1] = pad[..., -1, -2] + pad[..., -2, -1] - pad[..., -2, -2]

        lap = (
            pad[..., :-2, 1:-1]
            + pad[..., 2:, 1:-1]
            + pad[..., 1:-1, :-2]
            + pad[..., 1:-1, 2:]
            - 4 * pad[..., 1:-1, 1:-1]
        ) / 4.0

        return (-lap).reshape(-1)


    # --------------------------------------------------------------
    # Preconditioners ----------------------------------------------
    # --------------------------------------------------------------

    def jacobi_preconditioner(self, N):
        """ 
        Build Jacobi preconditioner: M^{-1} = diag(A)^(-1)
        Approximated by applying A to basis vectors implicitly via ones-vector.
        """
        ones_img = torch.ones(N, dtype=torch.complex128, device=self.device)
        d = self.A(ones_img).real   # diagonal approximation
        d = torch.clamp(d, min=1e-6)
        return lambda r: r / d

    # --------------------------------------------------------------
    # Conjugate Gradient Solver (with optional preconditioner)
    # --------------------------------------------------------------
    def cg(self, b, x0=None, max_iter=20, tol=1e-3, M=None):
        """
        Solve A(x) = b using Conjugate Gradient.

        Parameters:
            b        : RHS vector, shape (NxNy,)
            x0       : initial guess
            max_iter : max iterations
            tol      : tolerance
            M        : preconditioner function M(r) ~ M^{-1} r
        """
        with torch.no_grad():
            b = b.to(self.device)
            n = b.numel()

            if x0 is None:
                x = torch.zeros_like(b)
            else:
                x = x0.clone().to(self.device)

            # A(x)
            Ax = self.A(x)

            # Residual
            r = b - Ax
            b_norm = torch.linalg.norm(b) + 1e-12
            tolb = tol * b_norm

            # Preconditioned residual
            if M is None:
                z = r.clone()
            else:
                z = M(r)

            p = z.clone()
            rz_old = torch.dot(torch.conj(r), z).real
            eps = torch.finfo(r.real.dtype).eps
            stag = 0
            moresteps = 0
            max_stag_steps = self.max_stag_steps
            if self.max_more_steps is None:
                max_more_steps = max(1, min(n // 50, 5, max(n - max_iter, 1)))
            else:
                max_more_steps = max(1, int(self.max_more_steps))
            best_x = x.clone()
            best_rel = torch.linalg.norm(r) / b_norm

            for it in range(max_iter):
                Ap = self.A(p)

                denom = torch.dot(torch.conj(p), Ap).real
                if denom.abs() < 1e-15:
                    break

                alpha = rz_old / denom
                if torch.linalg.norm(p) * alpha.abs() < eps * torch.linalg.norm(x):
                    stag += 1
                else:
                    stag = 0

                x = x + alpha * p
                r = r - alpha * Ap
                res_norm = torch.linalg.norm(r)
                refresh_true_residual = (
                    (it + 1) % self.true_residual_interval == 0
                    or res_norm <= tolb
                    or (self.early_stopping and stag >= max_stag_steps)
                    or (self.early_stopping and moresteps > 0)
                )
                if refresh_true_residual:
                    # MATLAB-style stabilization: periodically recompute true residual.
                    r = b - self.A(x)
                    res_norm = torch.linalg.norm(r)
                rel_res = (res_norm / b_norm).item()
                if rel_res < best_rel:
                    best_rel = rel_res
                    best_x = x.clone()
                if self.verbose:
                    print(
                        f"CG Iter {it+1}/{max_iter}, Residual norm: {res_norm.item():.6e}, "
                        f"Rel residual: {rel_res:.6e}"
                    )
                # torch.cuda.synchronize()

                if res_norm <= tolb:
                    break
                if self.early_stopping and refresh_true_residual:
                    if stag >= max_stag_steps and moresteps == 0:
                        stag = 0
                    moresteps += 1
                    if moresteps >= max_more_steps:
                        break

                if M is None:
                    z = r.clone()
                else:
                    z = M(r)

                rz_new = torch.dot(torch.conj(r), z).real
                if rz_old.abs() < 1e-15:
                    break
                beta = rz_new / rz_old
                rz_old = rz_new

                p = z + beta * p

            return best_x
        
    # --------------------------------------------------------------
    # Convenience function: solve with simple CG
    # --------------------------------------------------------------
    def solve_cg(self, b, **kwargs):
        self.lambda_scaled = self.lambda_ * torch.norm(b, p=2)
        return self.cg(b, M=None, **kwargs)

    # --------------------------------------------------------------
    # Convenience function: solve with Jacobi PCG
    # --------------------------------------------------------------
    def solve_pcg_jacobi(self, b, **kwargs):
        self.lambda_scaled = self.lambda_ * torch.norm(b, p=2)
        M = self.jacobi_preconditioner(b.shape[0])
        return self.cg(b, M=M, **kwargs)

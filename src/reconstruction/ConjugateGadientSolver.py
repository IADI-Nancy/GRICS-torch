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

    def __init__(self, encoding_operator, reg_lambda=0.0, regularizer="Tikhonov", verbose=False):
        """
        encoding_operator : instance of EncodingOperator
        motion_operator   : list of motion operators (same used inside forward/backward)
        """
        self.E = encoding_operator
        self.device = encoding_operator.device
        self.lambda_ = reg_lambda
        self.regularizer = regularizer
        self.verbose = verbose

    # --------------------------------------------------------------
    # Regularized linear operator: A(x) = Eh(E(x)) + lambda_scaled * x
    # --------------------------------------------------------------
    def A(self, x):
        return self.E.normal(x) + self.lambda_ * self.regularization(x)
    
    def regularization(self, x):
        if self.regularizer == "Tikhonov":
            return x
        elif self.regularizer == "Tikhonov_gradient":
            return self.gradient_op(x)
        elif self.regularizer == "Tikhonov_laplacian":
            return self.laplacian_op(x)
        else:
            raise ValueError("Unknown regularizer")
    
    # def gradient_op(self, x):
    #     # x is (N) or 2D image flattened
    #     Nx, Ny = self.E.SensitivityMaps.shape[:2]
    #     img = x.view(Nx, Ny)

    #     dx = torch.roll(img, -1, dims=0) - img
    #     dy = torch.roll(img, -1, dims=1) - img

    #     # return L^H L x
    #     # because CG needs LᵀL—not L
    #     dxx = dx - torch.roll(dx, 1, dims=0)
    #     dyy = dy - torch.roll(dy, 1, dims=1)

    #     return (dxx + dyy).reshape(-1)
    
    # def laplacian_op(self, x):
    #     Nx, Ny = self.E.SensitivityMaps.shape[:2]
    #     img = x.view(Nx, Ny)

    #     lap = (
    #         -4*img
    #         + torch.roll(img, 1, 0) + torch.roll(img, -1, 0)
    #         + torch.roll(img, 1, 1) + torch.roll(img, -1, 1)
    #     )
    #     return lap.reshape(-1)

    # --------------------------------------------------------------
    # Preconditioners ----------------------------------------------
    # --------------------------------------------------------------

    def jacobi_preconditioner(self, N):
        """ 
        Build Jacobi preconditioner: M^{-1} = diag(A)^(-1)
        Approximated by applying A to basis vectors implicitly via ones-vector.
        """
        ones_img = torch.ones(N, dtype=torch.complex64, device=self.device)
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

            if x0 is None:
                x = torch.zeros_like(b)
            else:
                x = x0.clone().to(self.device)

            # A(x)
            Ax = self.A(x)

            # Residual
            r = b - Ax

            # Preconditioned residual
            if M is None:
                z = r.clone()
            else:
                z = M(r)

            p = z.clone()
            rz_old = torch.dot(torch.conj(r), z).real

            for it in range(max_iter):
                Ap = self.A(p)

                denom = torch.dot(torch.conj(p), Ap).real
                if denom.abs() < 1e-15:
                    break

                alpha = rz_old / denom

                x = x + alpha * p
                r = r - alpha * Ap

                if self.verbose:
                    res_norm = torch.linalg.norm(r)
                    print(f"CG Iter {it+1}/{max_iter}, Residual norm: {res_norm.item():.6e}")
                # torch.cuda.synchronize()

                if torch.linalg.norm(r) < tol:
                    break

                if M is None:
                    z = r.clone()
                else:
                    z = M(r)

                rz_new = torch.dot(torch.conj(r), z).real
                beta = rz_new / rz_old
                rz_old = rz_new

                p = z + beta * p

            return x
        
    def cg_keep_best(self, b, x0=None, max_iter=20, tol=1e-3, M=None):
        """
        Solve A(x) = b using Conjugate Gradient.
        Returns the x that achieved the smallest residual norm during iterations.
        """
        with torch.no_grad():
            b = b.to(self.device)

            if x0 is None:
                x = torch.zeros_like(b)
            else:
                x = x0.clone().to(self.device)

            # Compute initial residual
            Ax = self.A(x)
            r = b - Ax

            # Precondition
            z = r.clone() if M is None else M(r)
            p = z.clone()
            rz_old = torch.dot(torch.conj(r), z).real

            # === NEW: store best iterate ===
            best_x = x.clone()
            best_res = torch.linalg.norm(r)

            # CG iterations
            for it in range(max_iter):
                Ap = self.A(p)
                denom = torch.dot(torch.conj(p), Ap).real

                if denom.abs() < 1e-15:
                    break

                alpha = rz_old / denom

                x = x + alpha * p
                r = r - alpha * Ap

                res_norm = torch.linalg.norm(r)

                if self.verbose:
                    print(f"CG Iter {it+1}/{max_iter}, Residual norm: {res_norm.item():.6e}")

                # === NEW: update best solution ===
                if res_norm < best_res:
                    best_res = res_norm.clone()
                    best_x = x.clone()

                if res_norm < tol:
                    break

                # Precondition
                z = r.clone() if M is None else M(r)

                rz_new = torch.dot(torch.conj(r), z).real
                beta = rz_new / rz_old
                rz_old = rz_new

                p = z + beta * p

            # === Return best solution, not last one ===
            return best_x

    # --------------------------------------------------------------
    # Convenience function: solve with simple CG
    # --------------------------------------------------------------
    def solve_cg(self, b, **kwargs):
        self.lambda_scaled = self.lambda_ * torch.norm(b, p=2)
        return self.cg(b, M=None, **kwargs)
    
    def solve_cg_keep_best(self, b, **kwargs):
        self.lambda_scaled = self.lambda_ * torch.norm(b, p=2)
        return self.cg_keep_best(b, M=None, **kwargs)

    # --------------------------------------------------------------
    # Convenience function: solve with Jacobi PCG
    # --------------------------------------------------------------
    def solve_pcg_jacobi(self, b, **kwargs):
        self.lambda_scaled = self.lambda_ * torch.norm(b, p=2)
        M = self.jacobi_preconditioner(b.shape[0])
        return self.cg(b, M=M, **kwargs)

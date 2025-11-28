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

    def __init__(self, encoding_operator, reg_lambda=0.0, verbose=False):
        """
        encoding_operator : instance of EncodingOperator
        motion_operator   : list of motion operators (same used inside forward/backward)
        """
        self.E = encoding_operator
        self.device = encoding_operator.device
        self.lambda_ = reg_lambda
        self.verbose = verbose

    # --------------------------------------------------------------
    # Regularized linear operator: A(x) = Eh(E(x)) + lambda_scaled * x
    # --------------------------------------------------------------
    def A(self, x):
        return self.E.normal(x) + self.lambda_scaled * x 

    # --------------------------------------------------------------
    # Preconditioners ----------------------------------------------
    # --------------------------------------------------------------

    def jacobi_preconditioner(self):
        """ 
        Build Jacobi preconditioner: M^{-1} = diag(A)^(-1)
        Approximated by applying A to basis vectors implicitly via ones-vector.
        """
        NxNy = self.E.smaps.shape[0] * self.E.smaps.shape[1]
        ones_img = torch.ones(NxNy, dtype=torch.complex64, device=self.device)
        d = self.A(ones_img).real   # diagonal approximation
        d = torch.clamp(d, min=1e-6)
        return lambda r: r / d

    # --------------------------------------------------------------
    # Conjugate Gradient Solver (with optional preconditioner)
    # --------------------------------------------------------------
    def cg(self, b, x0=None, max_iter=50, tol=1e-6, M=None):
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
        M = self.jacobi_preconditioner()
        return self.cg(b, M=M, **kwargs)

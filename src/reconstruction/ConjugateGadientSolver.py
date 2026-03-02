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
        use_reg_scale_proxy=False,
        reg_scale_num_probes=8,
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
        self.use_reg_scale_proxy = use_reg_scale_proxy
        self.reg_scale_num_probes = reg_scale_num_probes
        self.reg_scale = 1.0
        self.last_info = None

    # --------------------------------------------------------------
    # Regularized linear operator: A(x) = Eh(E(x)) + lambda_eff * R(x)
    # --------------------------------------------------------------
    def A(self, x):
        return self.E.normal(x) + self.effective_lambda() * self.regularization(x)

    def effective_lambda(self):
        return self.lambda_ * self.reg_scale

    def update_regularization_scale(self, reference):
        if (not self.use_reg_scale_proxy) or self.lambda_ == 0.0:
            self.reg_scale = 1.0
            return self.reg_scale

        ref = reference.to(self.device)
        eps = 1e-12
        ratios = []
        n = ref.numel()
        is_complex = torch.is_complex(ref)

        for _ in range(max(1, int(self.reg_scale_num_probes))):
            if is_complex:
                v = torch.randn(n, device=self.device, dtype=torch.float64) + 1j * torch.randn(
                    n, device=self.device, dtype=torch.float64
                )
                v = v.to(ref.dtype)
            else:
                v = torch.randn(n, device=self.device, dtype=ref.dtype)

            v = v / (torch.linalg.norm(v) + eps)
            data_norm = torch.linalg.norm(self.E.normal(v))
            reg_norm = torch.linalg.norm(self.regularization(v))

            if reg_norm > eps and torch.isfinite(data_norm) and torch.isfinite(reg_norm):
                ratios.append((data_norm / reg_norm).real.item())

        if len(ratios) == 0:
            self.reg_scale = 1.0
        else:
            r = torch.tensor(ratios, dtype=torch.float64, device=self.device)
            self.reg_scale = max(1e-12, torch.median(r).item())

        if self.verbose:
            print(
                f"Regularizer proxy scale: {self.reg_scale:.6e}, "
                f"lambda_eff={self.effective_lambda():.6e}"
            )
        return self.reg_scale
    
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
    # Conjugate Gradient Solver
    # --------------------------------------------------------------
    def cg(self, b, x0=None, max_iter=20, tol=1e-3):
        """
        Solve A(x) = b using Conjugate Gradient.

        Parameters:
            b        : RHS vector, shape (NxNy,)
            x0       : initial guess
            max_iter : max iterations
            tol      : tolerance
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

            z = r.clone()
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
            iters_done = 0
            res_norm = torch.linalg.norm(r)
            rel_res = (res_norm / b_norm).item()
            residual_norm_history = [float(res_norm.item())]
            relres_history = [float(rel_res)]
            converged = bool(res_norm <= tolb)
            stop_reason = "initial_tolerance"

            for it in range(max_iter):
                Ap = self.A(p)
                iters_done = it + 1

                denom = torch.dot(torch.conj(p), Ap).real
                if denom.abs() < 1e-15:
                    stop_reason = "breakdown_denom"
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
                residual_norm_history.append(float(res_norm.item()))
                relres_history.append(float(rel_res))
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
                    converged = True
                    stop_reason = "tolerance"
                    break
                if self.early_stopping and refresh_true_residual:
                    if stag >= max_stag_steps and moresteps == 0:
                        stag = 0
                    moresteps += 1
                    if moresteps >= max_more_steps:
                        stop_reason = "early_stopping"
                        break

                z = r.clone()

                rz_new = torch.dot(torch.conj(r), z).real
                if rz_old.abs() < 1e-15:
                    stop_reason = "breakdown_rz"
                    break
                beta = rz_new / rz_old
                rz_old = rz_new

                p = z + beta * p

            if iters_done == 0 and stop_reason == "initial_tolerance":
                stop_reason = "initial_tolerance"
            elif stop_reason == "initial_tolerance":
                stop_reason = "max_iter"

            self.last_info = {
                "flag": 0 if converged else 1,
                "iterations": int(iters_done),
                "relres": float(rel_res),
                "residual_norm": float(res_norm.item()),
                "residual_norm_history": residual_norm_history,
                "relres_history": relres_history,
                "stop_reason": stop_reason,
            }

            return best_x
        
    # --------------------------------------------------------------
    # Convenience function: solve with simple CG
    # --------------------------------------------------------------
    def solve_cg(self, b, **kwargs):
        self.update_regularization_scale(b)
        return self.cg(b, **kwargs)

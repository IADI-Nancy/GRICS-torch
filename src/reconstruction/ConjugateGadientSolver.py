import torch

"""
Source: ChatGPT + Wikipedia
"""
class ConjugateGradientSolver:
    """
    CG and PCG solver for equations of the form:
        _A(x) = b
    where _A(x) = Eh(E) ('E' is the encoding operator, "h" - Hermitian conjugate).
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
    # Regularized linear operator: _A(x) = Eh(E(x)) + lambda_eff * R(x)
    # --------------------------------------------------------------
    def _A(self, x):
        return self.E.normal(x) + self._effective_lambda() * self._regularization(x)

    def _effective_lambda(self):
        return self.lambda_ * self.reg_scale

    def _update_regularization_scale(self, reference):
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
            reg_norm = torch.linalg.norm(self._regularization(v))

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
                f"lambda_eff={self._effective_lambda():.6e}"
            )
        return self.reg_scale
    
    def _regularization(self, x):
        if self.regularizer == "Tikhonov":
            return x
        elif self.regularizer == "Tikhonov_gradient":
            return self._gradient_op(x)
        elif self.regularizer == "Tikhonov_laplacian":
            return self._laplacian_op(x)
        else:
            raise ValueError("Unknown regularizer")
    
    def _gradient_op(self, x):
        if self.regularization_shape is None:
            raise ValueError("regularization_shape must be set for Tikhonov_gradient _regularization.")
        field = x.view(*self.regularization_shape)
        n_spatial = len(self.regularization_shape) - 1  # first dim is component
        result = torch.zeros_like(field)

        # Compute -div(grad(field)) along every spatial dimension.
        for d in range(1, n_spatial + 1):
            # Forward difference along dimension d (zero-gradient boundary).
            df = torch.zeros_like(field)
            slc_src = [slice(None)] * field.ndim
            slc_dst = [slice(None)] * field.ndim
            slc_src[d] = slice(1, None)
            slc_dst[d] = slice(None, -1)
            df[tuple(slc_dst)] = field[tuple(slc_src)] - field[tuple(slc_dst)]

            # Adjoint divergence (transpose of forward difference).
            div = torch.zeros_like(field)
            # First element
            s0 = [slice(None)] * field.ndim; s0[d] = 0
            div[tuple(s0)] = -df[tuple(s0)]
            # Interior elements
            si = [slice(None)] * field.ndim; si[d] = slice(1, -1)
            si_prev = [slice(None)] * field.ndim; si_prev[d] = slice(0, -2)
            div[tuple(si)] = df[tuple(si_prev)] - df[tuple(si)]
            # Last element
            sn = [slice(None)] * field.ndim; sn[d] = -1
            sn_prev = [slice(None)] * field.ndim; sn_prev[d] = -2
            div[tuple(sn)] = df[tuple(sn_prev)]

            result += div

        return result.reshape(-1)
    
    def _laplacian_op(self, x):
        field = x.view(*self.regularization_shape)
        n_spatial = len(self.regularization_shape) - 1  # first dim is component

        # Guard: need at least 2 pixels in every spatial dimension.
        for d in range(1, n_spatial + 1):
            if field.shape[d] < 2:
                return torch.zeros_like(field).reshape(-1)

        # N-dimensional Laplacian with linear-extrapolation ghost boundaries
        # (MATLAB del2 style).  Pad each spatial dim by 1 on each side.
        pad_shape = list(field.shape)
        for d in range(1, n_spatial + 1):
            pad_shape[d] += 2
        pad = torch.zeros(pad_shape, dtype=field.dtype, device=field.device)

        # Copy interior.
        interior = [slice(None)] + [slice(1, -1)] * n_spatial
        pad[tuple(interior)] = field

        # Fill ghost faces, edges, corners by linear extrapolation per axis.
        for d in range(1, n_spatial + 1):
            # Low face ghost:  pad[..., 0, ...] = 2*field[..., 0, ...] - field[..., 1, ...]
            lo_pad = [slice(1, -1)] * (n_spatial + 1); lo_pad[0] = slice(None); lo_pad[d] = 0
            lo_f0  = [slice(None)] * (n_spatial + 1)
            for dd in range(1, n_spatial + 1):
                lo_f0[dd] = slice(1, -1) if dd != d else 0
            lo_f1 = list(lo_f0); lo_f1[d] = 1
            pad[tuple(lo_pad)] = 2 * pad[tuple(lo_f0)] - pad[tuple(lo_f1)]

            # High face ghost:
            hi_pad = [slice(1, -1)] * (n_spatial + 1); hi_pad[0] = slice(None); hi_pad[d] = -1
            hi_f0  = [slice(None)] * (n_spatial + 1)
            for dd in range(1, n_spatial + 1):
                hi_f0[dd] = slice(1, -1) if dd != d else -2
            hi_f1 = list(hi_f0); hi_f1[d] = -3
            pad[tuple(hi_pad)] = 2 * pad[tuple(hi_f0)] - pad[tuple(hi_f1)]

        # Stencil: sum of neighbours minus 2*n_spatial * center, divided by n_spatial.
        lap = -2 * n_spatial * pad[tuple(interior)].clone()
        for d in range(1, n_spatial + 1):
            lo = list(interior); lo[d] = slice(0, -2)
            hi = list(interior); hi[d] = slice(2, None)
            lap += pad[tuple(lo)] + pad[tuple(hi)]
        lap = lap / float(n_spatial)

        return (-lap).reshape(-1)


    # --------------------------------------------------------------
    # Conjugate Gradient Solver
    # --------------------------------------------------------------
    def cg(self, b, x0=None, max_iter=20, tol=1e-3):
        """
        Solve _A(x) = b using Conjugate Gradient.

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

            # _A(x)
            Ax = self._A(x)

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
                Ap = self._A(p)
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
                    r = b - self._A(x)
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
    def _solve_cg(self, b, **kwargs):
        self._update_regularization_scale(b)
        return self.cg(b, **kwargs)

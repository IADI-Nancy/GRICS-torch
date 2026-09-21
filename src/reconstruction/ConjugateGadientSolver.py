from contextlib import nullcontext

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
    def __init__(self, encoding_operator, *, reg_lambda, regularizer, regularization_shape,
        regularization_spatial_dims, verbose, stop_on_stagnation, true_residual_interval,
        stagnation_consecutive_steps, stagnation_countdown_steps, use_reg_scale_proxy, reg_scale_num_probes):
        """
        encoding_operator : instance of EncodingOperator
        motion_operator   : list of motion operators (same used inside forward/backward)
        """
        self.E = encoding_operator
        self.device = encoding_operator.device
        self.lambda_ = reg_lambda
        self.regularizer = regularizer
        self.regularization_shape = regularization_shape
        self.regularization_spatial_dims = regularization_spatial_dims
        if self.regularizer in ("Tikhonov_gradient", "Tikhonov_laplacian"):
            if self.regularization_shape is None:
                raise ValueError(f"regularization_shape must be set for {self.regularizer} regularization.")
            if self.regularization_spatial_dims is None:
                raise ValueError(f"regularization_spatial_dims must be set for {self.regularizer} regularization.")
        self.verbose = verbose
        self.stop_on_stagnation = stop_on_stagnation
        self.true_residual_interval = true_residual_interval
        self.stagnation_consecutive_steps = stagnation_consecutive_steps
        self.stagnation_countdown_steps = stagnation_countdown_steps
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
                    n, device=self.device, dtype=torch.float64)
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
            print(f"Regularizer proxy scale: {self.reg_scale:.6e}, "f"lambda_eff={self._effective_lambda():.6e}")
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
        spatial_dims = self.regularization_spatial_dims
        if spatial_dims is None:
            raise ValueError("regularization_spatial_dims must be set for Tikhonov_gradient regularization.")
        result = torch.zeros_like(field)

        # Compute -div(grad(field)) along selected spatial dimensions.
        for d in spatial_dims:
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
        if self.regularization_shape is None:
            raise ValueError("regularization_shape must be set for Tikhonov_laplacian regularization.")
        field = x.view(*self.regularization_shape)
        spatial_dims = self.regularization_spatial_dims
        if spatial_dims is None:
            raise ValueError("regularization_spatial_dims must be set for Tikhonov_laplacian regularization.")

        # Guard: need at least 2 pixels in every spatial dimension.
        for d in spatial_dims:
            if field.shape[d] < 2:
                return torch.zeros_like(field).reshape(-1)

        result = torch.zeros_like(field)
        for d in spatial_dims:
            prev_idx = [slice(None)] * field.ndim
            curr_idx = [slice(None)] * field.ndim
            next_idx = [slice(None)] * field.ndim
            inner_idx = [slice(None)] * field.ndim
            prev_idx[d] = slice(None, -2)
            curr_idx[d] = slice(1, -1)
            next_idx[d] = slice(2, None)
            inner_idx[d] = slice(1, -1)
            result[tuple(inner_idx)] += (-field[tuple(prev_idx)] + 2 * field[tuple(curr_idx)] - field[tuple(next_idx)])
        return result.reshape(-1)


    # --------------------------------------------------------------
    # Conjugate Gradient Solver
    # --------------------------------------------------------------
    def cg(self, b, x0=None, max_iter=20, tol=1e-3, differentiable=False):
        """
        Solve _A(x) = b using Conjugate Gradient.

        Parameters:
            b        : RHS vector, shape (NxNy,)
            x0       : initial guess
            max_iter : max iterations
            tol      : tolerance
            differentiable : preserve autograd through CG when True. The default
                             False retains the validated inference behavior.
        """
        context = nullcontext() if differentiable else torch.no_grad()
        with context:
            b = b.to(self.device)
            n = b.numel()

            if x0 is None:
                x = torch.zeros_like(b)
            else:
                x = x0.clone().to(self.device)

            # Here A includes regularization: A(x) = E^H E(x) + lambda_eff R(x).
            # For a motion solve, E is the linearized motion operator J.
            Ax = self._A(x)

            # The "true residual" is b - A(x), evaluated directly at the current x.
            # It measures how well x solves this regularized linear system.
            # It is NOT the k-space data mismatch y - E(x), nor an error against
            # a known ground-truth image/motion. "True" means directly recomputed;
            # this calculation still has ordinary floating-point rounding error.
            r = b - Ax
            b_norm = torch.linalg.norm(b) + 1e-12
            # Convergence means ||b - A(x)|| / ||b|| <= tol (with a tiny guard
            # in b_norm for zero RHS). This is a residual tolerance, not a direct
            # bound on image or motion error.
            tolb = tol * b_norm

            z = r.clone()
            p = z.clone()
            rz_old = torch.dot(torch.conj(r), z).real
            eps = torch.finfo(r.real.dtype).eps
            consecutive_tiny_updates = 0       # Consecutive updates too small relative to x.
            stagnation_countdown_elapsed = 0  # 0: no stagnation countdown; >0: countdown is active.
            stagnation_consecutive_steps = self.stagnation_consecutive_steps
            if self.stagnation_countdown_steps is None:
                stagnation_countdown_steps = max(1, min(n // 50, 5, max(n - max_iter, 1)))
            else:
                stagnation_countdown_steps = max(1, int(self.stagnation_countdown_steps))
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
                # The proposed change is delta_x = alpha * p. If its norm is
                # below machine precision times ||x||, adding it may barely
                # change the stored iterate. Count consecutive such steps.
                # This detects numerical stagnation, NOT a slowly decreasing
                # residual or a visually unchanged reconstructed image.
                if torch.linalg.norm(p) * alpha.abs() < eps * torch.linalg.norm(x):
                    consecutive_tiny_updates += 1
                else:
                    consecutive_tiny_updates = 0

                x = x + alpha * p
                # Cheap recursive residual: reuse Ap from this iteration.
                # In exact arithmetic this equals b - A(x), since A is linear.
                # In floating-point arithmetic, accumulated rounding can make
                # this recursively updated r drift away from b - A(x).
                r = r - alpha * Ap
                res_norm = torch.linalg.norm(r)
                # Recompute directly in four situations:
                # 1. Periodically, to limit drift (not a stopping condition).
                # 2. When r suggests convergence, to verify before accepting it.
                # 3. When enough consecutive tiny updates suggest stagnation.
                # 4. On every step of an already active stagnation countdown.
                refresh_true_residual = (
                    (it + 1) % self.true_residual_interval == 0
                    or res_norm <= tolb
                    or (self.stop_on_stagnation and consecutive_tiny_updates >= stagnation_consecutive_steps)
                    or (self.stop_on_stagnation and stagnation_countdown_elapsed > 0)
                )
                if refresh_true_residual:
                    # Costs an extra A evaluation (encoding + adjoint, plus
                    # regularization). Replace the accumulated residual with
                    # a fresh one; this does not reset x or the search direction.
                    r = b - self._A(x)
                    res_norm = torch.linalg.norm(r)
                rel_res = (res_norm / b_norm).item()
                # History uses the direct residual on refresh steps and the
                # recursive residual on other steps.
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

                # Any apparent convergence above has now been checked using
                # the direct residual. This stop remains enabled even when
                # stop_on_stagnation=False (that flag controls stagnation only).
                if res_norm <= tolb:
                    converged = True
                    stop_reason = "tolerance"
                    break
                # Only detected stagnation starts this countdown; routine
                # residual refreshes must not start it. Allow a few more steps
                # using freshly checked residuals before giving up.
                # The triggering iteration counts as step 1, so a budget of 6
                # allows at most 5 subsequent iterations. Once started, this
                # countdown continues even if a later update is no longer tiny.
                # Convergence or max_iter can still end the solve sooner.
                if self.stop_on_stagnation and (consecutive_tiny_updates >= stagnation_consecutive_steps or stagnation_countdown_elapsed > 0):
                    if consecutive_tiny_updates >= stagnation_consecutive_steps and stagnation_countdown_elapsed == 0:
                        consecutive_tiny_updates = 0
                    stagnation_countdown_elapsed += 1
                    if stagnation_countdown_elapsed >= stagnation_countdown_steps:
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

            # Return the iterate with the smallest recorded relative residual,
            # which need not be the last iterate. last_info above describes the
            # final attempted iteration; residual history is not all "true"
            # residuals because direct recomputation happens only on refreshes.
            return best_x
        
    # --------------------------------------------------------------
    # Convenience function: solve with simple CG
    # --------------------------------------------------------------
    def _solve_cg(self, b, **kwargs):
        self._update_regularization_scale(b)
        return self.cg(b, **kwargs)

from contextlib import nullcontext
import math

import torch

class ConjugateGradientSolver:
    """
    CG and PCG solver for equations of the form:
        _A(x) = b
    where _A(x) = Eh(E) ('E' is the encoding operator, "h" - Hermitian conjugate).
    """
    def __init__(self, encoding_operator, *, reg_lambda, regularizer, regularization_shape,
        regularization_spatial_dims, verbose, stop_on_stagnation, true_residual_interval,
        stagnation_consecutive_steps, stagnation_countdown_steps, use_reg_scale_proxy, reg_scale_num_probes,
        regularization_spacing=None, preconditioner=None):
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
        self.regularization_spacing = regularization_spacing
        self.preconditioner = preconditioner
        if self.regularizer == "Tikhonov_gradient":
            if self.regularization_shape is None:
                raise ValueError(f"regularization_shape must be set for {self.regularizer} regularization.")
            if self.regularization_spatial_dims is None:
                raise ValueError(f"regularization_spatial_dims must be set for {self.regularizer} regularization.")
            if self.regularization_spacing is not None:
                if len(self.regularization_spacing) != len(self.regularization_spatial_dims):
                    raise ValueError("regularization_spacing must have one value per spatial dimension.")
                if any(not math.isfinite(float(value)) or float(value) <= 0
                       for value in self.regularization_spacing):
                    raise ValueError("regularization_spacing must contain finite positive values.")
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
        for index, d in enumerate(spatial_dims):
            spacing_squared = (float(self.regularization_spacing[index]) ** 2
                               if self.regularization_spacing is not None else 1.0)
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

            result += div / spacing_squared

        return result.reshape(-1)
    
    # --------------------------------------------------------------
    # Conjugate Gradient Solver
    # --------------------------------------------------------------
    def cg(self, b, x0=None, max_iter=20, tol=1e-3, differentiable=False):
        """
        Solve _A(x) = b using Conjugate Gradient.

        Return the final iterate on verified convergence; otherwise return the
        lowest-residual candidate among the initial guess, residual refreshes,
        and final iterate. Candidate residuals are evaluated directly as b - A(x).

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
            # Same iteration, same stopping rules and same candidate selection as the reference
            # implementation. The only change is how scalars reach the host: all comparisons of one
            # iteration are made on values moved in a single transfer, instead of one synchronisation
            # per comparison, so the GPU queue is drained once per iteration (twice on refresh steps).
            b = b.to(self.device)
            n = b.numel()

            if x0 is None:
                x = torch.zeros_like(b)
            else:
                x = x0.clone().to(self.device)

            # Here A includes regularization: A(x) = E^H E(x) + lambda_eff R(x).
            Ax = self._A(x)
            # The "true residual" is b - A(x), evaluated directly at the current x (see the reference).
            r = b - Ax
            b_norm = torch.linalg.norm(b) + 1e-12
            tolb = tol * b_norm

            z = r.clone() if self.preconditioner is None else self.preconditioner(r)
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
            iters_done = 0
            res_norm = torch.linalg.norm(r)
            res_norm_v, b_norm_v, tolb_v = torch.stack([res_norm, b_norm, tolb]).tolist()
            rel_res = res_norm_v / b_norm_v
            residual_norm_history = [float(res_norm_v)]
            relres_history = [float(rel_res)]
            converged = bool(res_norm_v <= tolb_v)
            stop_reason = "initial_tolerance"
            best_x = x.clone()
            best_res_norm_v = res_norm_v
            best_iteration = 0

            for it in range(0 if converged else max_iter):
                Ap = self._A(p)
                iters_done = it + 1

                denom = torch.dot(torch.conj(p), Ap).real
                alpha = rz_old / denom
                # Tiny-update test (numerical stagnation), evaluated on the device, read below.
                tiny = torch.linalg.norm(p) * alpha.abs() < eps * torch.linalg.norm(x)

                x_prev = x
                x = x + alpha * p
                # Cheap recursive residual: reuse Ap from this iteration.
                r = r - alpha * Ap
                res_norm = torch.linalg.norm(r)
                # One host transfer for every scalar decision of this iteration.
                denom_v, tiny_v, res_norm_v, rz_old_v = torch.stack(
                    [denom, tiny.to(denom.dtype), res_norm, rz_old]).tolist()
                if abs(denom_v) < 1e-15:
                    # The reference breaks before applying this update.
                    x = x_prev
                    stop_reason = "breakdown_denom"
                    break
                if tiny_v != 0.0:
                    consecutive_tiny_updates += 1
                else:
                    consecutive_tiny_updates = 0
                # Recompute the residual directly in the same four situations as the reference.
                refresh_true_residual = (
                    (it + 1) % self.true_residual_interval == 0
                    or res_norm_v <= tolb_v
                    or (self.stop_on_stagnation and consecutive_tiny_updates >= stagnation_consecutive_steps)
                    or (self.stop_on_stagnation and stagnation_countdown_elapsed > 0)
                )
                if refresh_true_residual:
                    r = b - self._A(x)
                    res_norm = torch.linalg.norm(r)
                    res_norm_v = res_norm.item()
                    if res_norm_v < best_res_norm_v:
                        best_x = x.clone()
                        best_res_norm_v = res_norm_v
                        best_iteration = iters_done
                rel_res = res_norm_v / b_norm_v
                residual_norm_history.append(float(res_norm_v))
                relres_history.append(float(rel_res))
                if self.verbose:
                    print(
                        f"CG Iter {it+1}/{max_iter}, Residual norm: {res_norm_v:.6e}, "
                        f"Rel residual: {rel_res:.6e}"
                    )

                if res_norm_v <= tolb_v:
                    converged = True
                    stop_reason = "tolerance"
                    break
                if self.stop_on_stagnation and (consecutive_tiny_updates >= stagnation_consecutive_steps or stagnation_countdown_elapsed > 0):
                    if consecutive_tiny_updates >= stagnation_consecutive_steps and stagnation_countdown_elapsed == 0:
                        consecutive_tiny_updates = 0
                    stagnation_countdown_elapsed += 1
                    if stagnation_countdown_elapsed >= stagnation_countdown_steps:
                        stop_reason = "early_stopping"
                        break

                z = r.clone() if self.preconditioner is None else self.preconditioner(r)

                rz_new = torch.dot(torch.conj(r), z).real
                if abs(rz_old_v) < 1e-15:
                    stop_reason = "breakdown_rz"
                    break
                beta = rz_new / rz_old
                rz_old = rz_new

                p = z + beta * p

            if iters_done == 0 and stop_reason == "initial_tolerance":
                stop_reason = "initial_tolerance"
            elif stop_reason == "initial_tolerance":
                stop_reason = "max_iter"

            # Verify the final candidate even when stopping between refreshes
            # or on breakdown. Selection always uses the unpreconditioned system.
            res_norm_v = torch.linalg.norm(b - self._A(x)).item()
            final_rel_res = float(res_norm_v / b_norm_v)
            converged = bool(res_norm_v <= tolb_v)
            if converged or res_norm_v < best_res_norm_v:
                best_x = x
                best_res_norm_v = res_norm_v
                best_iteration = iters_done

            self.last_info = {
                "flag": 0 if converged else 1,
                "iterations": int(iters_done),
                "selected_iteration": int(best_iteration),
                "relres": float(best_res_norm_v / b_norm_v),
                "residual_norm": float(best_res_norm_v),
                "final_relres": final_rel_res,
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

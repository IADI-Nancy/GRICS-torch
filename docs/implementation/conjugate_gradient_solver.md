# ConjugateGradientSolver implementation details

Source: [ConjugateGradientSolver](../../src/reconstruction/ConjugateGadientSolver.py).

## Linear system and regularization

CG applies $A(x) = E^H E x + w_{\mathrm{eff}} R(x)$ without assembling a matrix. For motion updates, $E$ is the motion Jacobian. The effective weight is $w_{\mathrm{eff}} = w s$, where $w$ is the configured weight and $s$ is the regularization scale: $w$ is `lambda_r` for images or `lambda_m` for motion. Matrix-free CG avoids storing the large normal matrix. Below, $\epsilon=10^{-12}$ is a numerical guard, distinct from the dtype-dependent machine precision used for stagnation detection. This guard is fixed internally; it is not the input-normalization setting `kspace_norm_eps`.

`Tikhonov` uses $R=I$. `Tikhonov_gradient` uses $R=G^H G$: forward differences with zero-gradient boundaries, summed only over the specified spatial axes. Each axis contributes a factor $1/h_i^2$ when voxel spacing is supplied; otherwise spacing is one. Motion components and sensor channels are not regularized against one another by this gradient. Magnitude penalties stabilize weakly constrained solutions; spatial penalties encourage smooth motion fields.

When `cg_use_reg_scale_proxy=true` and the regularization weight is nonzero, the relative operator scale is estimated matrix-free using `cg_reg_scale_num_probes` normalized Gaussian probe vectors, following randomized probing ideas for estimating matrix spectral properties [1]. This balances data and regularization operator magnitudes without matrix assembly. The median reduces sensitivity to atypical probes: $s = \max(\epsilon,\operatorname{median}_v(\|E^H E v\|_2 / \|R v\|_2))$. Probes are normalized with $\|v\|_2+\epsilon$; non-finite norm ratios or regularizer norms at most $\epsilon$ are discarded. If none remain, the implementation uses $s=1$. Disabled probe scaling or a zero weight skips probe generation and uses $s=1$ before any caller override.

[JointReconstructor](joint_reconstructor.md#regularization-normalization) estimates and caches the proxy once per solve type per resolution level. With `regularization_scaling="voxel_aware"`, its analytic scale overrides the proxy; enabling probes still computes the cached estimate, but that estimate is not used in the final weight. Probe estimation is therefore useful for `direct` mode.

## Stopping and returned solution

Convergence requires $\|b-Ax\|_2/(\|b\|_2+\epsilon) \leq \mathrm{tol}$. Here `tol_recon` or `tol_motion` supplies the image or motion tolerance; `max_iter_recon` or `max_iter_motion` caps the corresponding iterations. This measures the regularized linear-system residual, not image error or the k-space mismatch. Recursive residuals are recomputed directly at configured intervals (`cg_true_residual_interval`), on apparent convergence, during stagnation checks, and at exit. These checks limit residual drift and prevent false convergence.

Stagnation means consecutive updates smaller than machine precision times $\|x\|_2$. With `cg_stop_on_stagnation=true`, `cg_stagnation_consecutive_steps` such updates trigger a countdown of `cg_stagnation_countdown_steps`, including the triggering step. A countdown setting of zero or one stops immediately. Iteration limits and numerical breakdown can also end a solve. Verified convergence returns the final iterate; otherwise CG returns the smallest directly evaluated residual among the initial guess, refresh points, and final iterate. `last_info` records the stopping reason and selected iteration; history includes recursive values between refreshes. Stagnation stopping avoids ineffective iterations; selecting the best verified candidate protects against deterioration in later steps.

`motion_preconditioner` enables the optional non-rigid preconditioner. Preconditioning changes the search, but convergence and candidate selection use the unpreconditioned residual. Autograd is disabled by default; `differentiable=True` retains it through the executed iterations. Preconditioning can reduce iteration counts; disabling autograd saves inference memory. `differentiable` is a `cg()` argument, not a TOML setting; `verbose` controls inner-solver diagnostic printing.

## References

[1] Dixon, J. D. (1983). Estimating extremal eigenvalues and condition numbers of matrices. *SIAM Journal on Numerical Analysis*, **20**(4), 812–814. [doi:10.1137/0720053](https://doi.org/10.1137/0720053).

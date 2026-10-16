# JointReconstructor implementation details

Source: [JointReconstructor](../../src/reconstruction/JointReconstructor.py). See the [algorithm](../algorithm.md) for notation and [architecture](../architecture/reconstruction.md) for state ownership.

## Update choices

`ResolutionLevels` defines the coarse-to-fine scales and `GN_iterations_per_level` the iteration budget at each scale. `N_motion_states` sets one state count or a per-level schedule; merging is supported only with `reconstruction_motion_type="non-rigid"`.

The image solve starts from the current image; the motion solve starts from a zero increment. These starting points reuse the image estimate and express motion refinement as a local correction. `lambda_r` controls image magnitude regularization, or distance to `external_image_regularizer` when supplied. `initial_image`, `initial_motion`, and `external_image_regularizer` are constructor arguments, not TOML settings. `use_calibration_weighted_image=true` instead solves for $p$ in $x=Cp$ using `calibration_image_prior` and penalizes $p$; it cannot be combined with an external image prior. `reconstruction_motion_type` selects rigid increment regularization or non-rigid spatial-gradient regularization, both weighted by `lambda_m`. The non-rigid penalty acts on the spatial gradient of the updated total field and contributes the current field's regularization term to the right-hand side. Penalizing the total field controls accumulated spatial roughness across iterations.

`motion_preconditioner=true` enables non-rigid motion preconditioning using local Jacobian-normal blocks and pseudo Gauss–Seidel factors. Factors use complex64 storage; triangular solves run on the CPU, including for GPU reconstruction, so transfers have a cost. The local approximation avoids a full motion matrix; compact factors save memory. See [preconditioner details](motion_preconditioner.md) for factor layout and limits.

Let $\lambda_{\mathrm{eff}}$ and $\mu_{\mathrm{eff}}$ be the scaled image and motion weights. Without calibration weighting or an external image prior, the image system is:

$$
(E^H E+\lambda_{\mathrm{eff}}I)\rho=E^H k.
$$

For non-rigid motion, the current field contributes to the right-hand side because the penalty acts on $\alpha+\delta\alpha$:

$$
(J^H J+\mu_{\mathrm{eff}}G^H G)\delta\alpha=J^H\delta k-\mu_{\mathrm{eff}}G^H G\alpha.
$$

Here $G$ is the spatial regularization gradient. Rigid motion instead uses $G^H G=I$ on the increment and omits the current-field term.

## Regularization normalization

The effective weights are $\lambda_{\mathrm{eff}}=\lambda_r s_{\mathrm{image}}$ and $\mu_{\mathrm{eff}}=\lambda_m s_{\mathrm{motion}}$. `regularization_scaling` selects `direct` or `voxel_aware`; `cg_use_reg_scale_proxy` controls probe estimation. Weights are not interchangeable between modes, image parameterizations, or data scales.

### Direct weights

With `regularization_scaling="direct"` and `cg_use_reg_scale_proxy=false`, both scales equal one: `lambda_r` and `lambda_m` apply directly.

### Probe-based scaling

With `regularization_scaling="direct"` and `cg_use_reg_scale_proxy=true`, `cg_reg_scale_num_probes` random probes estimate relative data/regularizer operator magnitudes. The estimate is cached separately for image and motion at each resolution level, avoiding repeated probe work and holding the scale fixed within that level. See [CG probe scaling](conjugate_gradient_solver.md#linear-system-and-regularization) for the formula and guards.

### Voxel-aware scaling

With `regularization_scaling="voxel_aware"`, scaling explicitly accounts for resolution, physical spacing, and data magnitude. Write $n_\ell$ and $n_f$ for level and full-resolution voxel counts, $q=\sqrt{n_\ell/n_f}$, $a$ for `kspace_scale`, and $h_i=h_{f,i}N_{f,i}/N_{\ell,i}$ for level voxel spacing in mm:

$$
s_{\mathrm{image}}=q a\|b\|_2,\qquad s_{\mathrm{motion}}=q(\min_i h_i)^4\|J^H\delta k\|_2.
$$

Here $b$ is the back-projected measured data before adding a centered prior. Non-rigid gradient regularization also uses $h_i$. Full-resolution spacing is supplied through the constructor argument `voxel_spacing_mm`; `kspace_scale` is also a constructor argument, derived upstream from `normalize_kspace`, `kspace_norm_mode`, and `kspace_norm_eps`.

These scales are recomputed for each solve and override the proxy scale. If `cg_use_reg_scale_proxy=true`, a cached probe estimate is still computed but is not used in the effective weight.

### External image prior

In [`_solve_image()`](../../src/reconstruction/JointReconstructor.py), an external prior $z$ returned by `external_image_regularizer` uses the same image scale $s$ as the zero-centered penalty. With $\lambda_{\mathrm{eff}}=\lambda_r s$, the centered-prior problem is:

$$
\min_\rho\;\|E\rho-k\|_2^2+\lambda_{\mathrm{eff}}\|\rho-z\|_2^2,\qquad (E^H E+\lambda_{\mathrm{eff}}I)\rho=E^H k+\lambda_{\mathrm{eff}}z.
$$

The scale is determined before adding the prior term to the right-hand side: prior magnitude does not enter the probe estimate or voxel-aware data norm. There is no separate normalization of $z$; the caller must supply it in the current image's units. The API argument `regularization_weight` can replace `lambda_r`, but the selected scaling still applies.

### Calibration-weighted prior

With `use_calibration_weighted_image=true`, the prior is instead multiplicative: `calibration_image_prior` defines $C$, and [`CalibrationPriorEncodingOperator`](../../src/reconstruction/joint_reconstructor_utils/calibration_prior.py) replaces $E$ by $EC$:

$$
\rho=Cp,\qquad \min_p\;\|ECp-k\|_2^2+\lambda_{\mathrm{eff}}\|p\|_2^2.
$$

Probe scaling therefore measures $(EC)^H(EC)$ against the identity, while voxel-aware scaling uses $b=(EC)^Hk$ in the voxel-aware image-scale formula. The internally generated $C$ has mean magnitude one, as described in [coil-map normalization](coil_sensitivity_calculator.md#odille-spline-equations); caller-supplied calibration images are validated but not rescaled to unit mean. The penalty controls the image relative to $C$, rather than its raw intensity. This mode cannot be combined with an external centered prior.

## Outer stopping

`gn_early_stopping=true` enables Gauss–Newton early stopping by comparing $\|\delta k\|_2/(\|k\|_2+10^{-12})$ within a resolution level. This residual uses the updated image and motion from before that iteration's motion update. On an increase, the saved image/motion pair that produced the best accepted residual is restored. `update_motion_on_final_iteration=false` omits the final motion update so the output image matches its motion model. These checks are separate from [CG stopping](conjugate_gradient_solver.md#stopping-and-returned-solution). Restoring an accepted pair prevents a worsening iteration from becoming the level output.

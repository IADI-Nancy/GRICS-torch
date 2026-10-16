# MotionOperator implementation details

Source: [MotionOperator](../../src/reconstruction/MotionOperator.py). Coordinate units and axis order are defined in [conventions](../conventions.md).

## Coordinate model

`reconstruction_motion_type` selects the reconstruction model. The constructor receives `motion_type`, `alpha`, `centers`, `motion_signal`, and `Nz` directly; these are API arguments. `N_motion_states` controls the reconstruction state count upstream, while `reconstruction_dimension` must match the input dimension.

For each output location $r$, the warped image samples the input at $r+u(r)$. Rigid coordinates use $r'=c+R(r-c)+t$, with 3D rotation order $R=R_zR_yR_x$. If no rotation center is supplied, each coordinate is half its matrix size. Non-rigid displacement is a sum of spatial coefficient fields weighted by the state's sensor values. This separates spatial motion patterns from their physiological modulation.

## Interpolation and boundaries

Each state stores a sparse COO matrix with bilinear interpolation in 2D (four neighbor contributions) or trilinear interpolation in 3D (eight). For fractional coordinate offsets $f_i$, weights are products of $f_i$ or $1-f_i$ along each axis. The same real weights interpolate both real and imaginary image parts; matrices are stored as complex128 for complex MRI data. Precomputing them reuses the geometry throughout an image CG solve, at a memory cost proportional to voxel count times state count.

Sampling coordinates are clamped to the valid image domain, giving edge-value extension rather than zero padding or wraparound. The lower neighbor is clamped separately so an exact sample on the final voxel layer remains exact. Linear interpolation has a small local stencil, though repeated resampling can smooth fine image detail. Interpolation order, clamping, and sparse storage precision are fixed choices, with no TOML switches.

For a clamped sampling coordinate $\xi(r)$, let $b(r)$ be its lower-neighbor index and $f_i=\xi_i-b_i$. Bilinear/trilinear interpolation can be written in one expression, with $d=2$ or $3$:

$$
(M_m\rho)(r)=\sum_{\beta\in\{0,1\}^d}\rho(b(r)+\beta)\prod_{i=1}^{d}\bigl[\beta_i f_i+(1-\beta_i)(1-f_i)\bigr].
$$

Each term is one corner of the interpolation cell; the weights sum to one. Non-rigid sampling coordinates use the sensor-weighted displacement:

$$
u_m(r)=\sum_j\alpha_j(r)s_{m,j},\qquad \xi(r)=\operatorname{clamp}\bigl(r+u_m(r)\bigr).
$$

Here $\alpha_j$ is a vector coefficient field and $s_{m,j}$ is channel $j$ of `motion_signal`.

## Adjoint

Encoding applies the matrix transpose to back-project a warped image. Because weights are real, this is also the conjugate transpose: it accumulates the same interpolation contributions back onto source voxels. It is not an inverse warp. Using this matched adjoint preserves the linear-system structure needed by CG.

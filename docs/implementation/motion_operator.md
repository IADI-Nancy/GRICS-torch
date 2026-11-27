# MotionOperator implementation details

Source: [MotionOperator](../../src/reconstruction/MotionOperator.py). Coordinate units and axis order are defined in [conventions](../conventions.md).

## Coordinate model

`reconstruction_motion_type` selects the reconstruction model. The constructor receives `motion_type`, `alpha`, `centers`, `motion_signal`, and `Nz` directly; these are API arguments. `N_motion_states` controls the reconstruction state count upstream, while `reconstruction_dimension` must match the input dimension.

For each output location $r$, the warped image samples the input at $r+u(r)$. Rigid coordinates use $r'=c+R(r-c)+t$, with 3D rotation order $R=R_zR_yR_x$. If no rotation center is supplied, each coordinate is half its matrix size. Non-rigid displacement is a sum of spatial coefficient fields weighted by the state's sensor values. This separates spatial motion patterns from their physiological modulation.

## Interpolation and boundaries

Linear warps use bilinear interpolation in 2D (four neighbor contributions) or trilinear interpolation in 3D (eight). `warp_backend` selects sparse COO matrices or gather index/weight tables. For fractional coordinate offsets $f_i$, weights are products of $f_i$ or $1-f_i$ along each axis. The same real weights interpolate both real and imaginary image parts. Sparse values follow `compute_precision`; gather indices are int64 and weights use the corresponding real precision. Precomputing geometry reuses it throughout a CG solve, at a memory cost proportional to voxel count times state count.

Sampling coordinates are clamped to the valid image domain, giving edge-value extension rather than zero padding or wraparound. The lower neighbor is clamped separately so an exact sample on the final voxel layer remains exact. Linear interpolation has a small local stencil, though repeated resampling can smooth fine image detail. For rigid motion, `rigid_interpolation` selects this interpolation (`"linear"`, default) or the [Fourier rigid warps](#fourier-rigid-warps) (`"fourier"`); clamping and sparse storage precision are fixed choices.

For a clamped sampling coordinate $\xi(r)$, let $b(r)$ be its lower-neighbor index and $f_i=\xi_i-b_i$. Bilinear/trilinear interpolation can be written in one expression, with $d=2$ or $3$:

$$
(M_m\rho)(r)=\sum_{\beta\in\{0,1\}^d}\rho(b(r)+\beta)\prod_{i=1}^{d}\bigl[\beta_i f_i+(1-\beta_i)(1-f_i)\bigr].
$$

Each term is one corner of the interpolation cell; the weights sum to one. Non-rigid sampling coordinates use the sensor-weighted displacement:

$$
u_m(r)=\sum_j\alpha_j(r)s_{m,j},\qquad \xi(r)=\mathrm{clamp}\bigl(r+u_m(r)\bigr).
$$

Here $\alpha_j$ is a vector coefficient field and $s_{m,j}$ is channel $j$ of `motion_signal`.

## Gather-table residency

For CUDA gather warps, construction estimates the full table bank plus 512 bytes per voxel of construction workspace against half of currently available GPU memory, including unused PyTorch-reserved blocks after reserving up to 1 GiB (5% on smaller devices). When the estimate exceeds that budget, tables are stored on CPU and a message reports the decision and estimated sizes. Non-rigid geometry is also built on CPU in this mode; rigid geometry is built state by state on GPU before storage on CPU. Host memory must accommodate the complete table bank.

Warp application transfers and caches only one state on GPU when tables reside on CPU. State batching uses one state per chunk in this mode. Resident tables retain GPU execution, and their stacked cache contains only the requested chunk rather than duplicating the entire state bank. Grid size, state count, interpolation, and configured precision are preserved, with possible floating-point differences between CPU and GPU geometry arithmetic. CPU storage trades transfer time for reduced GPU residency; no configuration change is required. A single state's table and warp temporaries still need to fit on the GPU. This policy is separate from [FFT coil batching](encoding_operator.md#automatic-coil-batching).

## Gather warp workspace

Forward and adjoint gather application split the output-voxel axis into chunks targeting 32 MiB of temporary workspace, independently of coil and motion-state batching. The estimate includes gathered complex values, weighted products, flattened index copies, and chunk reductions; full outputs and interpolation tables are outside this budget. Small warps use a single chunk. Large warps log their chunk size once per operator. Both single-state and simultaneous-state paths use this bound, including tables transferred from CPU. Each forward chunk still gathers from the complete source image; adjoint chunks scatter into the complete output, preserving the global interpolation and its adjoint apart from floating-point accumulation order.

## Fourier rigid warps

With `rigid_interpolation = "fourier"` (rigid motion only), a state's warp is applied without interpolation weights. The translation is a phase ramp along each axis, and the rotation in each coordinate plane is a product of three shears [1]:

$$
R(\theta)=\begin{bmatrix}1&-\tan\frac{\theta}{2}\\0&1\end{bmatrix}\begin{bmatrix}1&0\\\sin\theta&1\end{bmatrix}\begin{bmatrix}1&-\tan\frac{\theta}{2}\\0&1\end{bmatrix}.
$$

A shear shifts the image along one axis by an amount proportional to the centred coordinate along the other axis; it is applied as a phase ramp on the one-dimensional FFT along the shifted axis. In 3D the three plane rotations are applied in the order of $R=R_zR_yR_x$: nine shears, plus the translation, 11 passes per state (4 in 2D). Each pass is exact for band-limited images, so the warp adds no interpolation blur, and its adjoint applies the conjugate ramps in reverse order. The image is zero-padded along each axis by the largest displacement of its content during the passes, plus two voxels, rounded up to an FFT size with prime factors 2, 3, 5 and 7, so that sheared content does not wrap around; content that leaves the grid is lost instead of clamped. For even sizes, the Nyquist bin of a ramp is $\cos(\pi a)$ for a shift $a$, the mean of the $\pm1/2$-cycle ramps, so that a real image stays real. Each state keeps only its ramps and applies them with FFTs, so `warp_backend = "gather"`, which stores interpolation tables, cannot be combined with this option. The motion Jacobian keeps the finite-difference gradients of the warped image ([MotionPerturbationSimulator](motion_perturbation_simulator.md)).

This shear decomposition is intended here for small rotations, typically up to 15 degrees. Its $\tan(\theta/2)$ coefficient grows without bound near odd multiples of 180 degrees, causing excessive padding and FFT allocations. The implementation does not split large rotations or reject them before allocating.

## Adjoint

Encoding applies the matrix transpose to back-project a warped image. Because weights are real, this is also the conjugate transpose: it accumulates the same interpolation contributions back onto source voxels. It is not an inverse warp. Using this matched adjoint preserves the linear-system structure needed by CG.

## References

[1] Unser, M., Thévenaz, P., & Yaroslavsky, L. (1995). Convolution-based interpolation for fast, high-quality rotation of images. *IEEE Transactions on Image Processing*, **4**(10), 1371–1381. [doi:10.1109/83.465102](https://doi.org/10.1109/83.465102).

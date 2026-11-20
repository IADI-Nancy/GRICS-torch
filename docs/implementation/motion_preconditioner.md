# PseudoGaussSeidelMotionPreconditioner implementation details

Source: [PseudoGaussSeidelMotionPreconditioner](../../src/reconstruction/joint_reconstructor_utils/motion_preconditioners.py).

This preconditioner is an AI-assisted translation of the GRICS++ implementation by Freddy Odille.

## Approximation and factors

`motion_preconditioner=true` enables this class for `reconstruction_motion_type="non-rigid"` and requires `lambda_m > 0`. `motion_gradient_boundary` selects the image-gradient boundary rule.

For non-rigid motion, products of warped-image gradients and sensor weights form local blocks coupling displacement components and sensors. These reuse the Jacobian's gradient boundary rule. The blocks omit full Fourier sampling and coil encoding, giving a compact approximation rather than the exact motion normal matrix. Regularization-dependent spatial bands complete the pseudo Gauss–Seidel factors; no full motion matrix is assembled. `lambda_scaled`, `voxel_spacing`, and `image_scale` are builder arguments derived from `lambda_m`, `regularization_scaling`, and the scaling controls described in [JointReconstructor](joint_reconstructor.md#regularization-normalization).

Components are ordered phase, partition when present, then readout, with a matching permutation of spatial axes. Band offsets act on this flattened layout and can cross row or component boundaries; they are not clipped like the exact spatial regularizer. This is a preconditioning approximation, while CG still evaluates the original regularized system.

For real physiological samples $s_{m,a}$ and warped-image gradients $g_{m,i}$, the local block at voxel $r$ is:

$$
B_{(i,a),(j,b)}(r)=\sum_m\overline{g_{m,i}(r)}\,g_{m,j}(r)\,s_{m,a}s_{m,b}.
$$

Indices $i,j$ identify displacement components and $a,b$ sensor channels; gradients include the builder's `image_scale`. This retains local component/sensor coupling without storing interactions between every pair of voxels.

## Precision and execution

Factors use complex64 storage to reduce memory and transfer volume. Two Numba-compiled triangular sweeps run on the CPU; CUDA residuals travel to the CPU and results return to their original device and dtype. Preconditioning can reduce CG iterations, but transfer costs can outweigh that saving for small problems. Factor precision and CPU triangular sweeps are fixed choices; `runtime_device` does not move these sweeps to the GPU. A finite positive factor diagonal is required; invalid factors raise an error. The joint solver enables this path only for non-rigid motion with positive motion regularization and a nonzero right-hand side.

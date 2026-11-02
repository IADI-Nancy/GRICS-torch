# MotionPerturbationSimulator implementation details

Source: [MotionPerturbationSimulator](../../src/reconstruction/MotionPerturbationSimulator.py).

## Linearized motion response

The class applies the motion Jacobian using the optical-flow approximation $\delta\rho_m \approx \nabla(M_m\rho)\cdot\delta u_m$, followed by coil weighting, FFT, and sampling. Rigid displacement increments use analytical translation and rotation derivatives; non-rigid increments combine coefficient fields with sensor values. This avoids assembling a full Jacobian or re-encoding a perturbed image for every parameter.

`reconstruction_motion_type` selects the rigid or non-rigid response. `motion_gradient_boundary` supplies the constructor argument `gradient_boundary`. Image gradients use central differences in the interior. `one_sided` uses one-sided differences at the edges, retaining edge information; `zero` suppresses the corresponding boundary gradients. These differences are in voxel coordinates. They approximate image sensitivity to displacement; they are not exact derivatives of the sparse interpolation weights, especially at clamped boundaries. Small Gauss–Newton increments are therefore assumed. `N_motion_states` sets the state count upstream; `max_iter_motion` and `tol_motion` control the CG solve using this operator, not the gradient stencil.

The image-space perturbation is encoded into k-space as:

$$
(J\delta\alpha)_{c,e,m}=P_{e,m}\mathcal F\!\left[S_c\sum_i g_{m,i}\,\delta u_{m,i}\right],\qquad g_{m,i}=D_i(M_m\rho).
$$

Here $D_i$ is the image-gradient stencil; for non-rigid motion, $\delta u_m=\sum_j s_{m,j}\delta\alpha_j$. At an interior voxel with unit voxel-coordinate spacing, the central difference is:

$$
(D_i I)(r)=\frac{I(r+e_i)-I(r-e_i)}{2}.
$$

Here $e_i$ is the unit offset along axis $i$; edge values follow `motion_gradient_boundary`.

## Matched adjoint

The adjoint back-projects residual k-space with inverse FFT and conjugated coil maps, multiplies by conjugated image gradients, and accumulates through the rigid geometric derivatives or conjugated sensor weights. It matches the implemented linear approximation, enabling CG on its normal operator. `normal()` explicitly composes adjoint and forward; it does not use the readout-FFT cancellation implemented by [EncodingOperator](encoding_operator.md).

These image-gradient boundary settings are independent of the zero-gradient boundaries in [motion regularization](conjugate_gradient_solver.md#linear-system-and-regularization).

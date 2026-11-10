# GRICS reconstruction algorithm

GRICS jointly estimates an image and a motion model from acquired k-space and motion-related signals. Reconstruction proceeds from coarse to fine resolution. At each resolution, Gauss–Newton iterations alternate between reconstructing the image with motion fixed and updating motion from the remaining k-space mismatch. Both linear subproblems are solved with conjugate gradients.

<p align="center">
  <img src="figures/GRICS_schematic.svg" alt="GRICS reconstruction: nested resolution and Gauss–Newton loops with image and motion updates">
</p>

## Figure notation

| Symbol | Meaning |
| --- | --- |
| $\rho$ | Reconstructed image shared across motion states. |
| $k$ | Acquired k-space data. |
| $\alpha$ | Motion-model parameters. |
| $E(\alpha)$ | Encoding operator combining motion, coil sensitivities, Fourier transformation, and sampling. |
| $\lambda$ | Image regularization weight. |
| $\delta k = k - E(\alpha)\rho$ | Residual between acquired and predicted k-space. |
| $J(\rho,\alpha) = \frac{\partial [E(\alpha)\rho]}{\partial\alpha}$ | Motion Jacobian mapping a small parameter change to a linearized k-space change. |
| $\mu$ | Motion regularization weight. |
| $\delta\alpha$ | Motion increment, applied as $\alpha \leftarrow \alpha + \delta\alpha$. |
| $\nabla$ | Spatial gradient used to penalize variation in the updated motion field. |

The figure shows non-rigid motion regularization, which encourages a smooth updated field through $\|\nabla(\alpha + \delta\alpha)\|_2^2$. For rigid motion, the solver instead penalizes the increment magnitude $\|\delta\alpha\|_2^2$. Image and motion estimates initialize the next, finer resolution. See [reconstruction architecture](architecture/reconstruction.md) for component responsibilities and state ownership.

## Motion model

The reconstruction assigns every acquired k-space line to a discrete motion state $m$. Each state has a warp $M_m$ that maps the shared reference image $\rho$ into the anatomy observed in that state. At a grid position $r$, the warp samples the reference image at the displaced coordinate $r'_m(r)$:

$$
(M_m\rho)(r)=\rho\bigl(r'_m(r)\bigr),\qquad r'_m(r)=\operatorname{clamp}\bigl(r+u_m(r)\bigr),
$$

where $u_m(r)$ is the state displacement in grid units. $\operatorname{clamp}$ restricts every coordinate to the valid image bounds; a point moved outside the image samples the nearest boundary voxel. The implementation represents $M_m$ as a sparse bilinear (2D) or trilinear (3D) interpolation matrix. Clamping gives edge-value extension when a sampling coordinate leaves the reconstructed grid. The encoding operator applies $M_m$ before coil sensitivity weighting, Fourier transformation, and selection of the lines acquired in state $m$.

### Rigid motion

Rigid motion stores one transform per state. In 2D, the parameter vector is $\alpha_m=[t_x,t_y,\phi]^T$, where translations are in pixels and $\phi$ is in radians. In 3D, it is $\alpha_m=[t_x,t_y,t_z,r_x,r_y,r_z]^T$, with translations in voxels and rotations in radians. The coordinate transform is

$$
r'=c+R(r-c)+t_m,\qquad u_m(r)=r'-r,
$$

where $c$ is the rotation centre and $t_m$ is the translation. The default centre is the midpoint of the reconstruction grid. In 2D,

$$
R(\phi)=\begin{bmatrix}\cos\phi & -\sin\phi\\ \sin\phi & \cos\phi\end{bmatrix}.
$$

In 3D, the implementation composes axis rotations in the order $R=R_z(r_z)R_y(r_y)R_x(r_x)$. The solver estimates direct updates to the state parameters and regularizes their magnitude with $\mu\lVert\delta\alpha\rVert_2^2$.

### Non-rigid motion

Non-rigid motion represents displacement as spatial coefficient fields driven by physiological signals. Let $s_m\in\mathbb{R}^{N_s}$ contain the $N_s$ sensor values for state $m$. For every sensor $j$, $\alpha_j(r)\in\mathbb{R}^d$ is a $d$-component displacement field, where $d=2$ or $d=3$. The state displacement is

$$
u_m(r)=\sum_{j=1}^{N_s}\alpha_j(r)s_{m,j}.
$$

Thus the model shares one spatial motion basis across states while the sensor values set each state's displacement amplitude. In code, the coefficient tensor has shape $[2,N_x,N_y,N_s]$ in 2D or $[3,N_x,N_y,N_z,N_s]$ in 3D; its leading axis stores displacement along the tensor-grid axes. The Gauss–Newton update estimates these fields and penalizes spatial variation of the updated field, $\mu\lVert\nabla(\alpha+\delta\alpha)\rVert_2^2$, so neighboring voxels receive similar displacement updates.

Both models estimate motion in tensor-grid coordinates. Their anatomical direction depends on the acquisition orientation rather than on a fixed patient-coordinate convention. [MotionOperator](implementation/motion_operator.md) describes the interpolation matrices and their matched adjoints; [motion configuration](usage/configuration/motion.md) describes simulated-motion inputs.

## Assumptions and limitations

- The model uses a single reference image and fixed coil sensitivity maps for all motion states. It models changes between states only through geometric warps, so it does not account for changes in signal intensity, contrast, coil sensitivity, B0, or other non-geometric effects.
- The 2D model only accounts for in-plane motion.
- Non-rigid motion is modeled as a linear combination of spatial coefficient fields and the provided physiological signals. Motion that is not captured by these signals or requires a different spatial basis cannot be represented accurately.
- The motion-update step assumes that the residual after image reconstruction reflects errors in the motion model. As a result, noise, aliasing, and other unmodeled effects may affect the motion estimate.
- Motion sensitivity is calculated using finite-difference image gradients and a first-order Taylor (optical-flow) approximation. This works well for small Gauss–Newton updates but is not the exact derivative of the clamped interpolation operator, especially near image boundaries. Since the optimization is non-convex, the results may depend on initialization, regularization, and state assignment.
- The optimized encoding normal operator assumes that complete readout lines are assigned to separate motion states within each repetition. It does not support arbitrary point-wise sampling patterns.

## Implementation details

- [JointReconstructor](implementation/joint_reconstructor.md): alternating updates, resolution levels, and regularization normalization.
- [ConjugateGradientSolver](implementation/conjugate_gradient_solver.md): linear solves, probe scaling, and stopping criteria.
- [EncodingOperator](implementation/encoding_operator.md): forward/adjoint encoding and the optimized normal operator.
- [MotionOperator](implementation/motion_operator.md): rigid and non-rigid coordinates, interpolation, and boundaries.
- [MotionPerturbationSimulator](implementation/motion_perturbation_simulator.md): motion Jacobian approximation and image gradients.
- [PseudoGaussSeidelMotionPreconditioner](implementation/motion_preconditioner.md): local motion-system approximation and triangular solves.

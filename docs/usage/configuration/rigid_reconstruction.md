# Rigid reconstruction

Set these options in `config/reconstruction/rigid_2d.toml` or `rigid_3d.toml`, or pass them through `load_config(overrides=...)`. The interpolation and motion damping options below apply only to rigid reconstruction.

## Interpolation

`rigid_interpolation` controls how the image is sampled after a translation or rotation moves its pixels or voxels between grid positions.

- Choose `"linear"` if some smoothing of fine detail is acceptable and you want to avoid the repeated Fourier transforms needed by `"fourier"`. Each output value is a weighted average of nearby input pixels or voxels. Outside the image, it repeats the nearest boundary value.
- Choose `"fourier"` if preserving fine detail during small translations and rotations is more important than avoiding additional Fourier transforms and temporary image padding. It applies motion through Fourier transforms instead of averaging neighboring pixels or voxels. Outside the image, it uses zeros, and content moved outside the output image is lost. See [Fourier rigid warps](../../implementation/motion_operator.md#fourier-rigid-warps) for implementation details.

This setting controls reconstruction only. Synthetic acquisition generation uses the independent `simulation_interpolation` setting described in [motion simulation](motion.md#motion-simulation). Changing reconstruction interpolation leaves the simulated acquisition unchanged; changing simulation interpolation generates different data.

## Warp backend

`warp_backend` controls how the code stores and applies linear interpolation. The choices produce the same interpolation results, apart from numerical rounding:

- `"sparse"` stores the interpolation weights in a sparse matrix and applies that matrix to the image.
- `"gather"` stores a list of neighboring input pixels or voxels and their weights for each output position. It reads those values directly and computes their weighted average. Choose it to reduce the memory used to store linear interpolation operators, especially for large volumes or many motion states.

With `rigid_interpolation = "linear"`, either backend is valid. With `rigid_interpolation = "fourier"`, set `warp_backend = "sparse"`: Fourier interpolation uses its own FFT implementation, but the configuration requires this value and rejects `"gather"`.

## Motion damping

`rigid_motion_damping` selects the penalty on each rigid motion increment:

- `"diagonal"` (default in both configurations and when omitted): scales the penalty by each parameter's diagonal curvature, accounting for the different units of translations and rotations. The damping weight is `lambda_m` directly; `regularization_scaling`, `cg_use_reg_scale_proxy`, and `cg_reg_scale_num_probes` do not scale this motion penalty. Image regularization still follows those settings.
- `"identity"`: applies the same magnitude penalty to translations in voxels and rotations in radians, as in GRICS-torch 2.0. Its effective weight is `lambda_m` multiplied by the configured regularization scale, so the scaling settings also affect the motion penalty.

`lambda_m` accepts any finite value greater than or equal to zero. Increasing it strengthens damping of the motion increment; zero removes the damping penalty. Keep `"diagonal"` for unit-normalized damping, and select `"identity"` when reproducing the earlier penalty. A `lambda_m` tuned for one mode is not interchangeable with the other. See [rigid damping](../../implementation/joint_reconstructor.md#rigid-damping) for the equations.

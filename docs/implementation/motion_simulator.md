# MotionSimulator implementation details

Source: [MotionSimulator](../../src/preprocessing/MotionSimulator.py). Configuration and signal definitions are in the [motion guide](../usage/motion.md).

## Synthetic acquisition

`simulated_motion_type` selects rigid/non-rigid and per-readout/per-shot motion; `motion_simulation_config_dimension` must match the input dimension. This is independent of `reconstruction_motion_type`. `Nex` sets the repetition count.

Corruption uses the shared motion and encoding operators: warp the image, apply coil sensitivities, Fourier transform, and retain the samples assigned to each state. Coil maps remain fixed. Per-readout simulation follows acquisition order; per-shot simulation holds a state over a shot. Consecutive identical rigid states are compressed for encoding without changing their transforms. Sharing operators keeps simulation consistent with reconstruction; compression reduces repeated operator work.

Non-rigid simulation multiplies fixed spatial displacement maps by a temporal coefficient: Gaussian per shot or sinusoidal per readout. The sinusoid has random phase and frequency, is normalized to unit maximum absolute amplitude, and expresses frequency per acquisition repeat so changing partition count does not change the intended respiratory rate. Separating spatial maps from temporal coefficients makes displacement shape and amplitude easy to control.

For non-rigid motion, a spatial vector field $a(r)$ is modulated by one scalar per state:

$$
u_m(r)=a(r)s_m,\qquad s_m\sim\mathcal N(0,\sigma_s^2)\quad\text{(per shot)}.
$$

Here $\sigma_s$ is `nonrigid_discrete_s_scale`. In per-readout mode, $a$ includes `nonrigid_motion_amplitude`, while the normalized respiratory curve is:

$$
s_m=\frac{\sin(2\pi f m/L+\phi)}{\max(\max_n|\sin(2\pi f n/L+\phi)|,\epsilon)},\qquad \epsilon=10^{-12}.
$$

The frequency $f$ is drawn between `nonrigid_resp_cycles_min` and `nonrigid_resp_cycles_max`, $\phi$ is a random phase, and $L$ is readouts per repetition. Corrupted samples for coil $c$, repetition $e$, and state $m$ follow:

$$
k_{c,e,m}=P_{e,m}\mathcal F\bigl(S_c M_m\rho\bigr).
$$

Here $P_{e,m}$ selects acquired samples, $S_c$ is the coil map, and $M_m$ is the motion interpolation operator.

## Simulation controls

| Parameters | Effect |
| --- | --- |
| `rigid_motion_amplitude_scale` | Global multiplier for rigid motion amplitudes. |
| `max_tx`, `max_ty`, `max_phi` | 2D translation and rotation limits. |
| `max_tx_3d`, `max_ty_3d`, `max_tz_3d`, `max_rx_3d`, `max_ry_3d`, `max_rz_3d` | 3D translation and rotation limits. |
| `FoVxy_mm`, `FoVz_mm` | Convert physical translation and center offsets to voxel coordinates. |
| `max_center_x`, `max_center_y`, `max_center_x_3d`, `max_center_y_3d`, `max_center_z_3d` | Fixed rotation-center offsets. |
| `num_motion_events`, `motion_tau` | Rigid per-readout event count and transition duration. |
| `nonrigid_discrete_s_scale` | Standard deviation of per-shot Gaussian coefficients. |
| `nonrigid_motion_amplitude`, `nonrigid_resp_cycles_min`, `nonrigid_resp_cycles_max` | Per-readout displacement amplitude and respiratory frequency range. |
| `nonrigid_diaphragm_level`, `nonrigid_diaphragm_sharpness`, `nonrigid_inferior_gain`, `nonrigid_top_decay` | Inferior activation and superior decay of the spatial field. |
| `nonrigid_lateral_sigma_lr`, `nonrigid_lateral_sigma_ap`, `nonrigid_lr_fraction`, `nonrigid_ap_fraction`, `nonrigid_anterior_bias` | Lateral envelope, relative displacement components, and anterior bias. |

## Diagnostic image and reconstruction states

The uncorrected diagnostic image averages k-space over repetitions, applies an inverse FFT, and combines coils with conjugated sensitivity maps. It is distinct from motion-corrected reconstruction. Simulated acquisition states can be finer than the reconstruction bins; [MotionBinner](motion_binner.md) groups the resulting readout signals separately. This allows testing reconstruction with fewer states (`N_motion_states`) than were used to generate the corruption. `seed_enabled` and `seed` control random draws; `use_deterministic_algorithms` separately requests deterministic kernels. `save_debug_plots` enables previews, and `flip_for_display` controls their orientation (also orienting generated non-rigid maps). `runtime_device` selects the preferred simulation device; CPU retry on CUDA memory exhaustion belongs to preparation, as described in the [motion guide](../usage/motion.md#motion-simulation).

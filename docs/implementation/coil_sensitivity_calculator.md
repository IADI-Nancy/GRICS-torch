# CoilSensitivityCalculator implementation details

Source: [CoilSensitivityCalculator](../../src/preprocessing/CoilSensitivityCalculator.py).

## Calibration choices

`coil_sensitivity_config` selects the configuration file containing `coil_sensitivity_method` (`espirit` or `odille-spline`). `reference_kspace` is an optional `calculate()` argument, not a TOML setting.

| Method | Implementation and benefit |
| --- | --- |
| `espirit` | SigPy ESPIRiT uses repetition-averaged reference k-space when supplied, otherwise imaging k-space. A separate reference permits calibration independent of imaging sampling. Calibration is volumetric for 3D and two-dimensional for singleton-z inputs. |
| `odille-spline` | Uses repetition-averaged imaging k-space, even when a separate reference is supplied. Coil magnitudes and phases are smoothed separately, allowing independent control of amplitude and phase variation. |

`espirit_calibration_width` sets the calibration region, `espirit_kernel_width` the kernel size, and `espirit_max_iter` the power-method iteration budget.

## Odille-spline method

These equations describe this implementation's `odille-spline` path. Let $k_{c,e}$ denote imaging k-space for coil $c$ and repetition $e$, with $N_e$ repetitions. Centered unitary inverse FFT gives coil images, their magnitudes, and the root-sum-of-squares reference:

$$
I_c=\mathcal F_{xyz}^{-1}\!\left(\frac{1}{N_e}\sum_e k_{c,e}\right),\qquad A_c=|I_c|,\qquad B=\sqrt{\sum_c A_c^2}.
$$

For 2D inputs the z axis is singleton. The complex phase proxy is damped in weak-signal regions using a fixed global threshold; it is the complex proxy, rather than wrapped phase angles, that is smoothed:

$$
\tau=0.05\max_{c,r}A_c(r),\qquad P_c(r)=\frac{I_c(r)}{\max(A_c(r),\tau)}.
$$

Define the per-plane smoothing filter $H_\eta$ below, where $\mathcal F_{xy}$ is the centered unitary 2D FFT. For centered integer frequency indices $p,q$ and plane size $N_x,N_y$, the exact smoothing symbol includes the unitary-FFT size factors:

$$
H_\eta(y)=\mathcal F_{xy}^{-1}\!\left(\frac{\mathcal F_{xy}(y)}{1+\eta W}\right),\qquad W_{p,q}=16\left[\frac{\sin^2(\pi p/N_x)}{\sqrt{N_x}}+\frac{\sin^2(\pi q/N_y)}{\sqrt{N_y}}\right].
$$

In code, $W$ is built as 16 times the sum of the real 1D FFTs of centered kernels with coefficients $(-1/4,1/2,-1/4)$. If either plane dimension is below three, the code sets $W=0$. Filtering is periodic in x/y and independent across z; $H_0$ is the identity. This direct Fourier division avoids an iterative smoothing solve.

With $\eta_A$=`spline_magnitude_smoothing` and $\eta_P$=`spline_phase_smoothing`, the filtered quantities are:

$$
\widetilde A_c=|H_{\eta_A}(A_c)|,\qquad \widetilde B=|H_{\eta_A}(B)|,\qquad \widetilde P_c=H_{\eta_P}(P_c).
$$

Separate weights control magnitude and phase smoothness. The map denominator replaces only exact reference zeros, using $\epsilon_C$=`coil_sensitivity_eps`; positive low-signal reference values are preserved:

$$
D(r)=\begin{cases}\widetilde B(r),&\widetilde B(r)>0,\\ \epsilon_C,&\widetilde B(r)=0,\end{cases}\qquad S_c^0(r)=\frac{\widetilde A_c(r)e^{i\arg\widetilde P_c(r)}}{D(r)}.
$$

Finally, maps have global maximum magnitude one, while the calibration image $C$ used by `use_calibration_weighted_image` has mean magnitude one. Here $N$ is the full spatial voxel count, including z:

$$
S_c(r)=\frac{S_c^0(r)}{\max_{d,r'}|S_d^0(r')|},\qquad C(r)=\frac{N\widetilde B(r)}{\sum_{r'}\widetilde B(r')}.
$$

Non-positive reference sum or map maximum raises an error. These conventions fix global scale; they do not impose voxel-wise sum-of-squares unity.

## Device and precision choices

ESPIRiT calibrates in complex64 and returns complex128 maps. The CUDA path uses CuPy and DLPack to avoid a host copy. `runtime_device` selects the preferred runtime device; an explicit `sp_device` argument selects the SigPy device. On the CUDA path, CUDA memory exhaustion prints a message and retries calibration on CPU. Other exceptions propagate. Spline filtering uses mixed precision internally and also returns complex128 maps.

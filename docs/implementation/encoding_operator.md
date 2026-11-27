# EncodingOperator implementation details

Source: [EncodingOperator](../../src/reconstruction/EncodingOperator.py).

## Forward and adjoint

`Nex` is the acquisition repetition count resolved from the input or synthetic data configuration. `smaps`, `SamplingIndices`, `Nsamples`, and `motionOperator` are constructor arguments supplied by preparation and reconstruction, not independent TOML controls.

A state encodes the shared image as $E_m=P_m F S M_m$: motion interpolation, fixed coil sensitivity weighting, centered unitary FFT, then sample selection. The adjoint zero-fills the selected samples, applies the inverse FFT and conjugated coil maps, and back-projects with the transpose motion matrix. Contributions from all repetitions are summed without averaging; this is the adjoint of encoding the same image into every repetition.

Unitary FFT normalization makes the inverse FFT its adjoint without additional matrix-size factors. Complex128 output buffers retain precision through repeated forward/adjoint operations. FFT normalization and buffer precision are fixed implementation choices.

Writing $M_m$ for motion interpolation, $S_c$ for coil multiplication, and $P_{e,m}$ for sample selection makes the adjoint sum explicit:

$$
E_{c,e,m}=P_{e,m}\mathcal F S_c M_m,\qquad E^H y=\sum_{c,e,m}M_m^H S_c^H\mathcal F^H P_{e,m}^H y_{c,e,m}.
$$

## Automatic coil batching

`EncodingOperator` and `MotionPerturbationSimulator` each own a shared-policy `CoilBatchPlanner`. After mandatory data/output allocations, CUDA operations query free device memory and add unused memory reserved by PyTorch. The planner reserves up to 1 GiB (5% of device capacity on smaller devices), then budgets half the remainder for temporaries. It estimates eight complex spatial buffers per coil and simultaneous motion state, covering FFT inputs/outputs, shifts, coil weighting, and workspace. All coils share an FFT call when that estimate fits; otherwise every forward, adjoint, and normal path processes coil groups and sums their contributions. Existing state batching uses the same workspace budget.

The planner checks memory again as operations run, allowing smaller groups when motion caches grow. A batch cap can decrease within an operator's lifetime and resets with each new operator; this avoids oscillating allocation sizes during CG. The selected size is logged initially and whenever it decreases. CPU operators process all coils together without querying CUDA. No configuration change is required. Public output vectors remain complex128; input residuals are converted to operator precision only for the selected coil group, avoiding a second full-coil conversion buffer. Coil maps also retain their original shared storage: `CoilBatchPlanner.coil_maps()` converts only the active coil slice to compute precision and optionally conjugates that slice. Neither image encoding nor the motion Jacobian retains an additional full-coil precision copy. This is logged when source and compute precision differ on CUDA; the memory estimate uses compute precision regardless of the source map dtype.

The policy controls FFT intermediates, not total reconstruction memory. Input data, output vectors, coil maps, motion tables, gradients, and preconditioners must still fit. The workspace estimate is conservative but is not an exact cuFFT memory bound, and another process can allocate memory after a check. If even one coil exceeds the estimated budget, the operation raises an explicit memory error. Grouping preserves the image grid, coils, motion states, and precision settings; only floating-point accumulation order can differ.

## K-space buffer storage

`forward(image)` returns complex128 k-space on the operator device by default; `output_device` explicitly permits CPU output storage while FFTs and coil arithmetic remain on the operator device. Adjoint methods accept CPU or GPU k-space and transfer only selected coil/sample slices to compute precision. This also applies to the motion Jacobian.

For iteration residuals and composed normal products, the planner compares the full complex128 output size with half its temporary GPU budget, leaving the other half for FFT work. If the buffer exceeds that share, it logs CPU storage and coil-slice transfers. `EncodingOperator.residual()` reuses the prediction buffer in place for measured-minus-predicted and copies measured data one coil at a time, avoiding a second full buffer. The iteration result retains only the residual, not a redundant prediction. Motion CG and incomplete-line image normal products apply the same output policy; the real-motion wrapper delegates to the Jacobian normal method so it cannot bypass that policy. Image and motion updates remain on GPU, while a CPU residual norm is evaluated on CPU. Host storage adds transfer cost and requires sufficient RAM; it does not change configured arithmetic precision or the normal equations.

## Faster normal operator

Sampling must comprise complete readout lines assigned to disjoint states within each repetition. Their masks depend only on phase coordinates, so the readout FFT and its adjoint cancel in $E^H E$. `normal()` therefore transforms only y in 2D, or y/z in 3D, and caches a phase mask per repetition/state. This reduces FFT work and mask storage in CG while retaining coil weighting and motion in their original order. The optimization depends on complete readout lines; it does not describe arbitrary point-wise sampling. The phase-only optimization is always used by `normal()`; it has no enable flag.

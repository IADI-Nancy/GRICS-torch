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

## Faster normal operator

Sampling must comprise complete readout lines assigned to disjoint states within each repetition. Their masks depend only on phase coordinates, so the readout FFT and its adjoint cancel in $E^H E$. `normal()` therefore transforms only y in 2D, or y/z in 3D, and caches a phase mask per repetition/state. This reduces FFT work and mask storage in CG while retaining coil weighting and motion in their original order. The optimization depends on complete readout lines; it does not describe arbitrary point-wise sampling. The phase-only optimization is always used by `normal()`; it has no enable flag.

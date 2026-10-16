# MotionBinner implementation details

Source: [MotionBinner](../../src/preprocessing/MotionBinner.py). Input normalization and state-count settings are described in the [motion guide](../usage/motion.md#motion-binning-and-reconstruction-states).

## State selection

`motion_binning_mode` selects `kmeans` or `kspace_energy`; `N_motion_states` supplies the requested count. In a per-resolution schedule, preparation uses the full count and reconstruction handles later merging.

Both modes use one feature vector per chronological readout and Euclidean distances across sensor channels. The binner does not standardize the channels; relative channel scales therefore affect assignments.

| Mode | Technical choice |
| --- | --- |
| `kmeans` | Initialize from random distinct sample indices, then perform 20 assignment/mean-update steps. Reinitialize empty clusters from a random sample. Returned centers are cluster means. |
| `kspace_energy` | Uniformly quantize each channel over its observed range, sum readout energy at each quantized feature vector, and select the highest-energy vectors as states. Assign other vectors to their nearest selected vector. |

K-means groups similar physiological states with a fixed iteration budget. Energy selection prioritizes signal-rich states; quantization limits sensitivity to small signal fluctuations. `motion_quantization_bins` sets the quantization levels per channel and applies only to `motion_binning_mode="kspace_energy"`. The 20 k-means updates are fixed internally.

## K-space energy binning

With `motion_binning_mode="kspace_energy"`, state selection reflects the acquired MRI signal as well as physiological variation. High-energy readouts often contain central k-space information that strongly influences image contrast. Prioritizing their motion states aims to give the reconstruction better-supported representatives; energy is a practical heuristic, not a guarantee of balanced sampling or optimal conditioning.

Each sensor channel is uniformly quantized over its observed range into `motion_quantization_bins` levels. Readouts with the same quantized vector form a virtual state $v$. For readout $r$, energy sums squared magnitudes over coils $c$ and readout positions $k_x$; virtual-state energy sums those readout energies:

$$
\mathcal E_r=\sum_c\sum_{k_x}|k_{c,r}(k_x)|^2,\qquad E_v=\sum_{r:q_r=v}\mathcal E_r.
$$

Here $k_{c,r}$ is the acquired line and $q_r$ its quantized sensor vector. Retain the $K$ highest-energy virtual states, with $K$=`N_motion_states`, then assign every readout to the nearest retained vector:

$$
\ell_r=\underset{m\in\{1,\ldots,K\}}{\arg\min}\;\|q_r-v_m\|_2.
$$

Energy is summed rather than averaged, so both signal strength and readout count affect selection. Representatives are the mean original sensor values at each selected virtual state, preserving signal units; they are not means over the entire assigned bin. Stable energy ordering and first-match distance ties make selection deterministic for fixed inputs.

## Invalid signals and reproducibility

Constant input signals raise `ConstantPhysiologicalSignalError`; energy mode also rejects signals that become constant after quantization. Requested state counts cannot exceed available samples, or quantized vectors in energy mode. There is no automatic switch between binning methods. K-means depends on random initialization; energy selection is deterministic for fixed inputs and settings. `seed_enabled` and `seed` control random initialization through the runtime; `use_deterministic_algorithms` separately requests deterministic kernels. Explicit rejection of unusable signals avoids defining arbitrary motion states.

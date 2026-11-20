# Sampling

Configured with:
- `kspace_sampling_type`
- `NshotsPerNex`
- `Nex`

Implemented in `src/preprocessing/SamplingSimulator.py`.

When synthetic sampling is generated and `save_debug_plots=true`, per-repetition debug plots are written to the reconstruction’s `preprocessing/` folder with hardcoded names:
- 2D sampling: `ky_order_nex{nex}.png`
- 3D sampling: `ky_kz_order_nex{nex}.png`

Here, `nex` in the filename is one-based.

For each repetition, acquired readouts are divided into `NshotsPerNex` chronological shot blocks. `acceleration_factor` selects regularly spaced ky lines, and `calibration_lines` retains a central calibration band when acceleration is greater than one. The ordering below is applied to the retained readouts.

## `linear`

In 2D, ky increases monotonically and is split into blocks of approximately equal size. In 3D, each shot covers a contiguous ky band and all kz partitions.

## `interleaved`

The ky ordering is built from groups `ky = s, s + NshotsPerNex, s + 2*NshotsPerNex, ...`. In 2D, retained lines are then split into approximately equal shot blocks. In 3D, each shot uses its interleaved ky group, with repetition-dependent ky/kz ordering and alternating ky traversal direction between partition blocks.

## `random`

Each repetition independently shuffles ky lines in 2D, or all retained `(ky, kz)` pairs in 3D, then splits them into approximately equal shot blocks.

# GRICS-torch reference vs GRICS-torch-batched on NVIDIA RTX PRO 6000 Blackwell Server Edition

## Operator calls, ms per call (speed-up); identity = max relative difference of outputs vs reference

**2d_rigid_128**

| operator | reference | batched, double/sparse | batched, single/gather |
|---|---|---|---|
| E_forward | 5.6 | 0.9 (6.2x) | 0.3 (21.0x) |
| E_adjoint | 8.7 | 1.1 (7.9x) | 0.3 (30.0x) |
| E_normal | 10.3 | 1.4 (7.2x) | 0.2 (56.8x) |
| J_forward | 8.4 | 1.1 (8.0x) | 0.8 (10.2x) |
| J_adjoint | 9.9 | 1.4 (6.9x) | 1.0 (10.0x) |
| J_normal | 18.1 | 2.5 (7.2x) | 1.8 (10.0x) |

Identity vs reference: batched, double/sparse: 3.6e-16, batched, single/gather: 2.4e-07

**2d_nonrigid_128**

| operator | reference | batched, double/sparse | batched, single/gather |
|---|---|---|---|
| E_forward | 5.6 | 0.9 (6.3x) | 0.3 (21.5x) |
| E_adjoint | 8.6 | 1.1 (8.0x) | 0.3 (30.2x) |
| E_normal | 10.4 | 1.4 (7.3x) | 0.2 (58.8x) |
| J_forward | 7.1 | 0.8 (8.4x) | 0.3 (20.9x) |
| J_adjoint | 8.7 | 1.1 (7.7x) | 0.4 (22.4x) |
| J_normal | 15.8 | 2.0 (7.9x) | 0.7 (21.2x) |

Identity vs reference: batched, double/sparse: 2.0e-16, batched, single/gather: 2.7e-07

**3d_rigid_128x128x64**

| operator | reference | batched, double/sparse | batched, single/gather |
|---|---|---|---|
| E_forward | 10.5 | 10.2 (1.0x) | 4.0 (2.7x) |
| E_adjoint | 66.3 | 11.4 (5.8x) | 4.0 (16.5x) |
| E_normal | 71.6 | 16.3 (4.4x) | 6.0 (12.0x) |
| J_forward | 19.1 | 11.1 (1.7x) | 5.9 (3.3x) |
| J_adjoint | 20.8 | 12.6 (1.6x) | 6.9 (3.0x) |
| J_normal | 39.9 | 23.8 (1.7x) | 12.7 (3.1x) |

Identity vs reference: batched, double/sparse: 0.0e+00, batched, single/gather: 2.9e-07

## Full reconstructions: seconds for run() (speed-up, relative image difference vs reference)

| case | reference | batched, double/sparse | batched, single/gather |
|---|---|---|---|
| 2d_rigid | 3.5 s | 1.1 s (3.3x, diff 0e+00) | 0.8 s (4.1x, diff 1e-06) |
| 2d_nonrigid | 4.1 s | 2.1 s (2.0x, diff 0e+00) | 1.3 s (3.2x, diff 3e-07) |
| 3d_rigid | 8.8 s | 2.7 s (3.3x, diff 0e+00) | 2.3 s (3.9x, diff 4e-07) |
| 3d_nonrigid | 6.6 s | 2.7 s (2.5x, diff 0e+00) | 2.2 s (3.1x, diff 2e-07) |
| real | 12.4 s | 3.0 s (4.1x, diff 3e-15) | 2.6 s (4.7x, diff 6e-07) |

## Roofline model (bandwidth-bound time = modelled bytes / measured copy bandwidth 1459 GB/s)

| case | operator | bound c128 (ms) | reference | batched double/sparse | batched single/gather (bound c64) |
|---|---|---|---|---|---|
| 2D 128x128, 8 coils, 8 states, rigid | E_normal | 0.1 | 10.3 ms (1 %) | 1.5 ms (10 %) | 0.2 ms (38 %) |
| 2D 128x128, 8 coils, 8 states, rigid | J_forward | 0.1 | 8.1 ms (1 %) | 1.1 ms (6 %) | 0.9 ms (4 %) |
| 2D 128x128, 8 coils, 8 states, rigid | J_adjoint | 0.1 | 9.9 ms (1 %) | 1.5 ms (4 %) | 1.0 ms (3 %) |
| 2D 320x532, 24 coils, 4 states, non-rigid (Zenodo slice size) | E_normal | 1.9 | 17.2 ms (11 %) | 5.8 ms (33 %) | 2.8 ms (35 %) |
| 2D 320x532, 24 coils, 4 states, non-rigid (Zenodo slice size) | J_forward | 0.9 | 12.0 ms (8 %) | 4.4 ms (21 %) | 2.1 ms (22 %) |
| 2D 320x532, 24 coils, 4 states, non-rigid (Zenodo slice size) | J_adjoint | 0.9 | 12.8 ms (7 %) | 4.7 ms (20 %) | 2.9 ms (16 %) |
| 3D 128x128x64, 4 coils, 8 states, rigid | E_normal | 7.0 | 70.4 ms (10 %) | 16.0 ms (44 %) | 5.7 ms (67 %) |
| 3D 128x128x64, 4 coils, 8 states, rigid | J_forward | 2.2 | 18.3 ms (12 %) | 10.7 ms (21 %) | 5.6 ms (20 %) |
| 3D 128x128x64, 4 coils, 8 states, rigid | J_adjoint | 2.2 | 20.1 ms (11 %) | 12.4 ms (18 %) | 6.5 ms (17 %) |
| 3D 192x192x56, 24 coils, 16 states, non-rigid (paper 3D size class) | E_normal | 100.0 | 508.8 ms (20 %) | 392.0 ms (26 %) | 179.5 ms (29 %) |
| 3D 192x192x56, 24 coils, 16 states, non-rigid (paper 3D size class) | J_forward | 44.9 | 203.8 ms (22 %) | 246.8 ms (18 %) | 115.2 ms (19 %) |
| 3D 192x192x56, 24 coils, 16 states, non-rigid (paper 3D size class) | J_adjoint | 44.9 | 216.7 ms (21 %) | 287.6 ms (16 %) | 135.1 ms (17 %) |

## Zenodo breast slice with the paper's nonrigid_2d.toml, seconds for run() over 3 repeats

| run | shape | coils | states | precision/backend | run() s | peak GiB |
|---|---|---|---|---|---|---|
| batched_double | 320x532x1 | 24 | 4 | double/sparse | 3.0, 2.5, 2.4 | 1.1 |
| batched_single | 320x532x1 | 24 | 4 | single/gather | 2.5, 1.9, 1.8 | 1.4 |
| reference | 320x532x1 | 24 | 4 | double/sparse | 12.2, 11.7, 11.8 | 0.7 |

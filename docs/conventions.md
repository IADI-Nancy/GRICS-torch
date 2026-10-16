# Data and coordinate conventions

These conventions form the shared contract between data loading, preparation,
and reconstruction. File-level input formats are described in the
[input data guide](usage/input_formats.md).

## Symbols

| Symbol | Meaning |
| --- | --- |
| `Nc` | Number of receiver coils |
| `Nex` | Number of acquisition repetitions |
| `Nx`, `Ny`, `Nz` | Readout and spatial-grid dimensions |
| `Nm` | Number of motion states |
| `Ns` | Number of motion-sensor channels |
| `Nalpha` | Number of motion components |

## Tensor layouts

| Data | Shape |
| --- | --- |
| K-space | `[Nc, Nex, Nx, Ny, Nz]` |
| Coil sensitivity maps | `[Nc, Nx, Ny, Nz]` |
| Motion signal | `[Nm, Ns]` |
| Rigid motion model | `[Nalpha, Nm]` |
| Non-rigid motion model | `[Nalpha, Nx, Ny, (Nz), Ns]` |
| Reconstructed image | `[Nx, Ny, (Nz)]` |

Two-dimensional k-space and coil maps retain `Nz = 1`. The reconstructed image
omits that singleton dimension.

## Sampling indices

Sampling indices are zero-based flattened spatial-grid indices, grouped as
`SamplingIndices[Nex][Nm]`. Each group contains the samples acquired during one
repetition and assigned to one motion state.

The flattening order is:

- 2D: `index = kx * Ny + ky`
- 3D: `index = (kx * Ny + ky) * Nz + kz`

Thus `kz` varies fastest in 3D, followed by `ky`, then `kx`. Chronological
sampling arrays contain one `ky`, optional `kz`, and repetition index per
readout; preparation expands each readout across all `kx` positions.

## Motion coordinates and units

Motion components follow tensor-grid axes, not fixed patient-anatomical axes.
Their anatomical interpretation depends on the acquisition orientation.

| Model | Component order | Internal units |
| --- | --- | --- |
| 2D rigid | `[tx, ty, phi]` | Translations in pixels; rotation in radians |
| 3D rigid | `[tx, ty, tz, rx, ry, rz]` | Translations in voxels; rotations in radians |
| 2D non-rigid | `[dx, dy]` | Displacement basis in pixels |
| 3D non-rigid | `[dx, dy, dz]` | Displacement basis in voxels |

Simulation configuration expresses translations and rotation centers in
millimetres and rotations in degrees. Data-preparation helpers convert them to
grid units and radians before reconstruction.

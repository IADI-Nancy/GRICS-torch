"""GRICS++ flat-diagonal pseudo Gauss-Seidel factors for breast 3D motion.

The reference's working axes are phase, partition, readout (Torch y, z, x).
Its flat offsets intentionally cross row/plane/component boundaries. The
sequential triangular substitution runs in compiled CPU code, including for
CUDA reconstructions; only the residual and result move per PCG iteration.
"""
from __future__ import annotations

from itertools import chain

import numpy as np
import torch
from numba import njit


@njit(cache=True)
def _solve_factors(diagonals, offsets, rhs):
    n = rhs.size
    lower = np.empty_like(rhs)
    out = np.empty_like(rhs)
    for k in range(n):
        value = rhs[k]
        for j in range(1, offsets.size):
            previous = k - offsets[j]
            if previous >= 0:
                value -= lower[previous] * diagonals[j, previous]
        lower[k] = value / diagonals[0, k]
    for k in range(n - 1, -1, -1):
        value = lower[k]
        diagonal = diagonals[0, k]
        for j in range(1, offsets.size):
            following = k + offsets[j]
            if following < n:
                value -= out[following] * np.conj(diagonals[j, k]) / diagonal
        out[k] = value / (np.conj(diagonal) / diagonal)
    return out


class GricsCppPseudoGaussSeidelPreconditioner:
    """C1 and C2 from MotionModelPerturbationSimulator::createPreconditioner.

    Input local normal matrix uses [component*sensor, component*sensor,x,y,z].
    Components and voxels are permuted to the reference breast working order.
    Factors have complex64 precision, matching the C++ factor storage.
    """

    def __init__(self, local_normal, *, lambda_scaled, voxel_spacing):
        if local_normal.ndim != 5 or local_normal.shape[0] != local_normal.shape[1]:
            raise ValueError('local_normal must have shape [C,C,Nx,Ny,Nz].')
        count, _, nx, ny, nz = local_normal.shape
        if count % 3 or count == 0 or min(nx, ny, nz) < 1:
            raise ValueError('Expected three components per sensor and positive spatial sizes.')
        if len(voxel_spacing) != 3 or any(not np.isfinite(v) or v <= 0 for v in voxel_spacing):
            raise ValueError('voxel_spacing must contain three finite positive values.')
        if not np.isfinite(lambda_scaled) or lambda_scaled < 0:
            raise ValueError('lambda_scaled must be finite and non-negative.')
        self.count, self.nx, self.ny, self.nz = count, nx, ny, nz
        self.nvoxels = nx * ny * nz
        sensors = count // 3
        order = [d*sensors+s for d in (1, 2, 0) for s in range(sensors)]
        local = local_normal.detach().to(device='cpu', dtype=torch.complex64)
        local = local[order][:, order].permute(0, 1, 2, 4, 3).reshape(count, count, -1).numpy()
        n = count * self.nvoxels
        factors = np.zeros((count + 3, n), dtype=np.complex64)
        for row in range(count):
            for col in range(row + 1):
                factors[row-col, col*self.nvoxels:(col+1)*self.nvoxels] = local[row, col]
        spacing = [voxel_spacing[d] for d in (1, 2, 0)]
        factors[0] += sum(2 * lambda_scaled * h*h for h in spacing)
        for d, h in enumerate(spacing):
            factors[count+d] = -lambda_scaled * h*h
        if not np.isfinite(factors).all() or np.any(factors[0].real <= 0):
            raise ValueError('GRICS++ motion preconditioner requires a finite positive diagonal.')
        self.diagonals = factors
        self.offsets = np.array([c*self.nvoxels for c in range(count)] + [1, ny, ny*nz], dtype=np.int64)

    @classmethod
    def from_warped_gradients(cls, gradients, motion_signal: torch.Tensor,
                              *, lambda_scaled: float,
                              voxel_spacing: tuple[float, float, float]):
        """Build C++'s local ``RᴴR`` approximation from every virtual time."""
        gradients = iter(gradients)
        first = next(gradients, None)
        if first is None:
            raise ValueError('At least one warped-image gradient is required.')
        dimensions = len(first)
        nx, ny, nz = first.shape[1:]
        signal = torch.as_tensor(motion_signal, device=first.device)
        if signal.ndim != 2:
            raise ValueError('motion_signal must have shape [Nstate, Nsensor].')
        sensors = int(signal.shape[1])
        count = dimensions * sensors
        local = torch.zeros((count, count, nx, ny, nz), dtype=torch.complex64,
                            device=first.device)
        for state, gradient in enumerate(chain((first,), gradients)):
            if state >= signal.shape[0]:
                raise ValueError("More gradients than motion states.")
            if tuple(gradient.shape) != (dimensions, nx, ny, nz):
                raise ValueError('All warped gradients must share [dimension, x, y, z].')
            grad = gradient.to(torch.complex64)
            weights = signal[state].to(torch.complex64)
            for row_dim in range(dimensions):
                for row_sensor in range(sensors):
                    row = row_dim * sensors + row_sensor
                    for col_dim in range(dimensions):
                        for col_sensor in range(sensors):
                            col = col_dim * sensors + col_sensor
                            local[row, col] += (
                                grad[row_dim].conj() * grad[col_dim]
                                * weights[row_sensor] * weights[col_sensor]
                            )
        if state + 1 != signal.shape[0]:
            raise ValueError("Fewer gradients than motion states.")
        return cls(local, lambda_scaled=lambda_scaled, voxel_spacing=voxel_spacing)

    def _to_cpp_layout(self, residual):
        if residual.numel() != self.count * self.nvoxels:
            raise ValueError('Unexpected number of motion parameters.')
        field = residual.reshape(3, self.nx, self.ny, self.nz, self.count // 3)
        return field[[1, 2, 0]].permute(0, 4, 1, 3, 2).contiguous().reshape(-1)

    def _from_cpp_layout(self, values):
        field = values.reshape(3, self.count // 3, self.nx, self.nz, self.ny)
        return field.permute(0, 2, 4, 3, 1)[[2, 0, 1]].reshape(-1)

    def __call__(self, residual):
        values = self._to_cpp_layout(residual).detach().cpu().numpy()
        result = _solve_factors(self.diagonals, self.offsets, values)
        return self._from_cpp_layout(torch.from_numpy(result).to(residual.device))

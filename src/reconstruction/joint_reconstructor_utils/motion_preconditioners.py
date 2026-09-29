"""GRICS++ pseudo Gauss-Seidel motion factors for 2D and 3D non-rigid fields.

The C++ working component order is phase, [partition,] readout: Torch
(y, [z,] x). Its flat offsets cross row and component boundaries by design.
Triangular solves run in compiled CPU code, including for CUDA reconstructions.
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


class PseudoGaussSeidelMotionPreconditioner:
    """C1 and C2 from C++ MotionModelPerturbationSimulator::createPreconditioner.

    ``local_normal`` has shape [component*sensor, component*sensor, x, y[, z]].
    Factors use complex64 storage, matching the C++ implementation.
    """

    def __init__(self, local_normal, *, lambda_scaled, voxel_spacing):
        dimensions = local_normal.ndim - 2
        if (dimensions not in (2, 3) or local_normal.shape[0] != local_normal.shape[1]):
            raise ValueError('local_normal must have shape [C,C,Nx,Ny[,Nz]].')
        count = local_normal.shape[0]
        spatial_shape = tuple(local_normal.shape[2:])
        if count == 0 or count % dimensions or min(spatial_shape) < 1:
            raise ValueError('Expected one motion component per spatial dimension and positive spatial sizes.')
        if (len(voxel_spacing) != dimensions
                or any(not np.isfinite(value) or value <= 0 for value in voxel_spacing)):
            raise ValueError('voxel_spacing must contain one finite positive value per spatial dimension.')
        if not np.isfinite(lambda_scaled) or lambda_scaled < 0:
            raise ValueError('lambda_scaled must be finite and non-negative.')

        self.dimensions = dimensions
        self.count = count
        self.spatial_shape = spatial_shape
        self.nvoxels = int(np.prod(spatial_shape))
        sensors = count // dimensions
        # C++ stores phase first, partition second when present, readout last.
        self.component_order = tuple(range(1, dimensions)) + (0,)
        self.inverse_component_order = tuple(self.component_order.index(d)
                                              for d in range(dimensions))
        component_indices = [d * sensors + s for d in self.component_order
                             for s in range(sensors)]
        local = local_normal.detach().to(device='cpu', dtype=torch.complex64)
        local = local[component_indices][:, component_indices]
        if dimensions == 3:
            local = local.permute(0, 1, 2, 4, 3)
        local = local.reshape(count, count, -1).numpy()

        n = count * self.nvoxels
        factors = np.zeros((count + dimensions, n), dtype=np.complex64)
        for row in range(count):
            for col in range(row + 1):
                factors[row - col, col * self.nvoxels:(col + 1) * self.nvoxels] = local[row, col]
        spacing = [voxel_spacing[d] for d in self.component_order]
        factors[0] += sum(2 * lambda_scaled * h * h for h in spacing)
        for d, h in enumerate(spacing):
            factors[count + d] = -lambda_scaled * h * h
        if not np.isfinite(factors).all() or np.any(factors[0].real <= 0):
            raise ValueError('Pseudo Gauss-Seidel motion preconditioner requires a finite positive diagonal.')
        self.diagonals = factors
        spatial_offsets = ([1, spatial_shape[1]] if dimensions == 2
                           else [1, spatial_shape[1], spatial_shape[1] * spatial_shape[2]])
        self.offsets = np.array([c * self.nvoxels for c in range(count)]
                                + spatial_offsets, dtype=np.int64)

    @classmethod
    def from_warped_gradients(cls, gradients, motion_signal: torch.Tensor,
                              *, lambda_scaled: float, voxel_spacing: tuple[float, ...]):
        """Build C++'s local RᴴR approximation from every virtual time."""
        gradients = iter(gradients)
        first = next(gradients, None)
        if first is None:
            raise ValueError('At least one warped-image gradient is required.')
        dimensions = first.shape[0]
        spatial_shape = tuple(first.shape[1:])
        if dimensions not in (2, 3) or len(spatial_shape) != dimensions:
            raise ValueError('Gradients must have shape [D, Nx, Ny[, Nz]] with D=2 or 3.')
        signal = torch.as_tensor(motion_signal, device=first.device)
        if signal.ndim != 2:
            raise ValueError('motion_signal must have shape [Nstate, Nsensor].')
        sensors = int(signal.shape[1])
        count = dimensions * sensors
        local = torch.zeros((count, count, *spatial_shape), dtype=torch.complex64,
                            device=first.device)
        for state, gradient in enumerate(chain((first,), gradients)):
            if state >= signal.shape[0]:
                raise ValueError('More gradients than motion states.')
            if tuple(gradient.shape) != (dimensions, *spatial_shape):
                raise ValueError('All warped gradients must share the same shape.')
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
            raise ValueError('Fewer gradients than motion states.')
        return cls(local, lambda_scaled=lambda_scaled, voxel_spacing=voxel_spacing)

    @classmethod
    def from_motion_simulator(cls, jacobian, *, lambda_scaled, voxel_spacing, image_scale):
        """Build factors using the same warped gradients as the motion Jacobian."""
        _, nx, ny, nz = jacobian.SensitivityMaps.shape
        dimensions = 3 if nz > 1 else 2
        if (jacobian.motionOperator.motion_type != 'non-rigid'
                or jacobian.Nalpha != dimensions):
            raise ValueError('GRICS++ factors require 2D or 3D non-rigid motion.')
        spatial_shape = (nx, ny, nz) if dimensions == 3 else (nx, ny)

        def warped_gradients():
            image = jacobian.image.flatten() * image_scale
            for state in range(len(jacobian.SamplingIndices[0])):
                warp = jacobian.motionOperator._get_sparse_operator(state)
                warped = (warp @ image).reshape(spatial_shape)
                gradient = (jacobian._gradient_3d(warped) if dimensions == 3
                            else jacobian._gradient_2d(warped))
                yield torch.stack(gradient)

        return cls.from_warped_gradients(
            warped_gradients(), jacobian.motionOperator.motion_signal,
            lambda_scaled=lambda_scaled, voxel_spacing=voxel_spacing,
        )

    def _to_cpp_layout(self, residual):
        if residual.numel() != self.count * self.nvoxels:
            raise ValueError('Unexpected number of motion parameters.')
        sensors = self.count // self.dimensions
        field = residual.reshape(self.dimensions, *self.spatial_shape, sensors)
        field = field[list(self.component_order)]
        if self.dimensions == 3:
            return field.permute(0, 4, 1, 3, 2).contiguous().reshape(-1)
        return field.permute(0, 3, 1, 2).contiguous().reshape(-1)

    def _from_cpp_layout(self, values):
        sensors = self.count // self.dimensions
        if self.dimensions == 3:
            nx, ny, nz = self.spatial_shape
            field = values.reshape(3, sensors, nx, nz, ny).permute(0, 2, 4, 3, 1)
        else:
            nx, ny = self.spatial_shape
            field = values.reshape(2, sensors, nx, ny).permute(0, 2, 3, 1)
        return field[list(self.inverse_component_order)].reshape(-1)

    def __call__(self, residual):
        values = self._to_cpp_layout(residual).detach().cpu().numpy()
        result = _solve_factors(self.diagonals, self.offsets, values)
        return self._from_cpp_layout(
            torch.from_numpy(result).to(device=residual.device, dtype=residual.dtype))

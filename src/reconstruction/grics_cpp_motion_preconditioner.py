"""GRICS++ pseudo Gauss-Seidel preconditioner for non-rigid 3D motion PCG."""

from __future__ import annotations

import torch


_WAVEFRONTS: dict[tuple[str, int, int, int], tuple[torch.Tensor, ...]] = {}


def _wavefronts(nx: int, ny: int, nz: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Indices in the C++ x-fastest layout, grouped by x+y+z."""
    key = (str(device), nx, ny, nz)
    cached = _WAVEFRONTS.get(key)
    if cached is not None:
        return cached
    indices = torch.arange(nx * ny * nz, device=device)
    x = indices.remainder(nx)
    y = indices.div(nx, rounding_mode='floor').remainder(ny)
    z = indices.div(nx * ny, rounding_mode='floor')
    order = torch.argsort(x + y + z, stable=True)
    counts = torch.bincount((x + y + z)[order], minlength=nx + ny + nz - 2)
    cached = tuple(torch.split(order, counts.tolist()))
    _WAVEFRONTS[key] = cached
    return cached


class GricsCppPseudoGaussSeidelPreconditioner:
    """Two triangular factors created by GRICS++ ``createPreconditioner``.

    ``local_normal`` is the voxelwise data term in C++ component/sensor order
    ``[component * sensor, component * sensor, x, y, z]``.  The factors use
    C++'s x-fastest spatial layout internally, while callers keep the Torch
    motion-vector convention ``[component, x, y, z, sensor]``.
    """

    def __init__(self, local_normal: torch.Tensor, *, lambda_scaled: float,
                 voxel_spacing: tuple[float, float, float]):
        if local_normal.ndim != 5 or local_normal.shape[0] != local_normal.shape[1]:
            raise ValueError('local_normal must have shape [C, C, Nx, Ny, Nz].')
        count, _, nx, ny, nz = local_normal.shape
        if min(nx, ny, nz) < 1:
            raise ValueError('Preconditioner dimensions must be positive.')
        if len(voxel_spacing) != 3 or any(float(v) <= 0 for v in voxel_spacing):
            raise ValueError('voxel_spacing must contain three positive values.')
        self.count = int(count)
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.nvoxels = self.nx * self.ny * self.nz
        self.device = local_normal.device
        # GRICS++ runs this factor in complex float even when reconstruction
        # arithmetic is complex double.  Keeping its factor precision limits
        # memory while retaining the reference's numerical representation.
        self.factor_dtype = torch.complex64
        local = local_normal.to(self.factor_dtype)
        self.diagonal = torch.stack([
            local[c, c].permute(2, 1, 0).reshape(-1).clone()
            for c in range(self.count)
        ])
        self.lower: dict[tuple[int, int], torch.Tensor] = {}
        for row in range(self.count):
            for col in range(row):
                self.lower[row, col] = local[row, col].permute(2, 1, 0).reshape(-1).clone()

        lam = float(lambda_scaled)
        spacing = tuple(float(value) for value in voxel_spacing)
        self.off = tuple(torch.tensor(-lam * value * value, dtype=self.factor_dtype,
                                      device=self.device) for value in spacing)
        diagonal_add = sum(2.0 * lam * value * value for value in spacing)
        self.diagonal += diagonal_add
        scale = self.diagonal.abs().amax().real
        if not torch.isfinite(self.diagonal).all() or scale <= 0:
            raise ValueError('GRICS++ motion preconditioner has invalid diagonal entries.')
        floor = torch.clamp(scale * 1e-8, min=1e-12)
        self.diagonal = torch.where(self.diagonal.abs() >= floor, self.diagonal,
                                    torch.full_like(self.diagonal, floor))
        self.wavefronts = _wavefronts(self.nx, self.ny, self.nz, self.device)

    @classmethod
    def from_warped_gradients(cls, gradients: list[torch.Tensor], motion_signal: torch.Tensor,
                              *, lambda_scaled: float,
                              voxel_spacing: tuple[float, float, float]):
        """Build C++'s local ``RᴴR`` approximation from every virtual time."""
        if not gradients:
            raise ValueError('At least one warped-image gradient is required.')
        dimensions = len(gradients[0])
        nx, ny, nz = gradients[0].shape[1:]
        signal = torch.as_tensor(motion_signal, device=gradients[0].device)
        if signal.ndim != 2 or signal.shape[0] != len(gradients):
            raise ValueError('motion_signal must have shape [Nstate, Nsensor].')
        sensors = int(signal.shape[1])
        count = dimensions * sensors
        local = torch.zeros((count, count, nx, ny, nz), dtype=torch.complex64,
                            device=gradients[0].device)
        for state, gradient in enumerate(gradients):
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
        return cls(local, lambda_scaled=lambda_scaled, voxel_spacing=voxel_spacing)

    def _to_cpp_layout(self, residual: torch.Tensor) -> torch.Tensor:
        expected = self.count * self.nvoxels
        if residual.numel() != expected:
            raise ValueError(f'Expected {expected} motion parameters, got {residual.numel()}.')
        # Torch: [dimension, x, y, z, sensor]; C++: [dimension, sensor, x-fastest voxel].
        sensors = self.count // 3
        field = residual.reshape(3, self.nx, self.ny, self.nz, sensors)
        return field.permute(0, 4, 3, 2, 1).reshape(self.count, self.nvoxels)

    def _from_cpp_layout(self, values: torch.Tensor) -> torch.Tensor:
        sensors = self.count // 3
        return values.reshape(3, sensors, self.nz, self.ny, self.nx).permute(0, 4, 3, 2, 1).reshape(-1)

    def _lower_solve(self, right_hand_side: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(right_hand_side)
        nx, ny, nxy = self.nx, self.ny, self.nx * self.ny
        for component in range(self.count):
            rhs = right_hand_side[component].clone()
            for previous in range(component):
                rhs -= self.lower[component, previous].to(rhs.dtype) * out[previous]
            result = out[component]
            diagonal = self.diagonal[component].to(rhs.dtype)
            for indices in self.wavefronts:
                value = rhs[indices]
                x = indices.remainder(nx)
                y = indices.div(nx, rounding_mode='floor').remainder(ny)
                z = indices.div(nxy, rounding_mode='floor')
                mask = x > 0
                if mask.any():
                    value[mask] -= self.off[0].to(rhs.dtype) * result[indices[mask] - 1]
                mask = y > 0
                if mask.any():
                    value[mask] -= self.off[1].to(rhs.dtype) * result[indices[mask] - nx]
                mask = z > 0
                if mask.any():
                    value[mask] -= self.off[2].to(rhs.dtype) * result[indices[mask] - nxy]
                result[indices] = value / diagonal[indices]
        return out

    def _upper_solve(self, right_hand_side: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(right_hand_side)
        nx, ny, nxy = self.nx, self.ny, self.nx * self.ny
        for component in range(self.count - 1, -1, -1):
            rhs = right_hand_side[component].clone()
            diagonal = self.diagonal[component].to(rhs.dtype)
            main = diagonal.conj() / diagonal
            for following in range(component + 1, self.count):
                coefficient = self.lower[following, component].to(rhs.dtype).conj() / diagonal
                rhs -= coefficient * out[following]
            result = out[component]
            for indices in reversed(self.wavefronts):
                value = rhs[indices]
                x = indices.remainder(nx)
                y = indices.div(nx, rounding_mode='floor').remainder(ny)
                z = indices.div(nxy, rounding_mode='floor')
                coefficient = self.off[0].to(rhs.dtype).conj() / diagonal[indices]
                mask = x < nx - 1
                if mask.any():
                    value[mask] -= coefficient[mask] * result[indices[mask] + 1]
                coefficient = self.off[1].to(rhs.dtype).conj() / diagonal[indices]
                mask = y < ny - 1
                if mask.any():
                    value[mask] -= coefficient[mask] * result[indices[mask] + nx]
                coefficient = self.off[2].to(rhs.dtype).conj() / diagonal[indices]
                mask = z < self.nz - 1
                if mask.any():
                    value[mask] -= coefficient[mask] * result[indices[mask] + nxy]
                result[indices] = value / main[indices]
        return out

    def __call__(self, residual: torch.Tensor) -> torch.Tensor:
        values = self._to_cpp_layout(residual)
        return self._from_cpp_layout(self._upper_solve(self._lower_solve(values)))

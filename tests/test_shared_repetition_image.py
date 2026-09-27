import unittest

import torch

from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator


def _complex_randn(shape):
    return torch.complex(
        torch.randn(shape, dtype=torch.float64),
        torch.randn(shape, dtype=torch.float64),
    )


def _relative_adjoint_error(operator, x, y):
    lhs = torch.vdot(operator.forward(x), y)
    rhs = torch.vdot(x, operator.adjoint(y))
    return (torch.abs(lhs - rhs) / (torch.abs(lhs) + torch.abs(rhs) + 1e-12)).item()


class _IdentityNonrigidMotion:
    def __init__(self, nvoxels, spatial_shape, nstates):
        indices = torch.arange(nvoxels)
        values = torch.ones(nvoxels, dtype=torch.complex128)
        self._identity = torch.sparse_coo_tensor(
            torch.stack((indices, indices)), values, (nvoxels, nvoxels)
        ).coalesce()
        self.motion_type = "non-rigid"
        self.Nz = spatial_shape[-1]
        self.alpha = torch.zeros((3, *spatial_shape, 1), dtype=torch.float64)
        self.motion_signal = torch.linspace(-1.0, 1.0, nstates, dtype=torch.float64)[:, None]

    def _get_sparse_operator(self, state):
        return self._identity


class SharedRepetitionImageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.ncoils, self.nex = 2, 2
        self.nx, self.ny, self.nz = 3, 4, 2
        self.nvoxels = self.nx * self.ny * self.nz
        self.smaps = _complex_randn((self.ncoils, self.nx, self.ny, self.nz))
        phase0 = torch.tensor([
            (x * self.ny + y) * self.nz + z
            for x in range(self.nx) for y in (0, 2) for z in range(self.nz)
        ])
        phase1 = torch.tensor([
            (x * self.ny + y) * self.nz + z
            for x in range(self.nx) for y in (1, 3) for z in range(self.nz)
        ])
        self.sampling = [[phase0, phase1] for _ in range(self.nex)]
        self.motion = _IdentityNonrigidMotion(
            self.nvoxels, (self.nx, self.ny, self.nz), len(self.sampling[0])
        )

    def test_encoding_maps_two_repetitions_to_one_image_and_is_adjoint(self):
        operator = EncodingOperator(
            self.smaps, self.nvoxels, self.sampling, self.nex, self.motion, Nimages=1
        )
        image = _complex_randn((self.nvoxels,))
        data = _complex_randn((self.ncoils * self.nex * self.nvoxels,))
        self.assertLess(_relative_adjoint_error(operator, image, data), 1e-12)

        predicted = operator.forward(image).reshape(
            self.ncoils, self.nex, self.nvoxels
        )
        torch.testing.assert_close(predicted[:, 0], predicted[:, 1])
        torch.testing.assert_close(operator.normal(image), operator.adjoint(operator.forward(image)))

    def test_motion_jacobian_uses_shared_image_for_both_repetitions(self):
        image = _complex_randn((1, self.nx, self.ny, self.nz))
        operator = MotionPerturbationSimulator(
            self.smaps, self.nvoxels, self.sampling, self.nex,
            image, self.motion, Nimages=1,
        )
        perturbation = _complex_randn((3 * self.nvoxels,))
        data = _complex_randn((self.ncoils * self.nex * self.nvoxels,))
        self.assertLess(_relative_adjoint_error(operator, perturbation, data), 1e-12)

    def test_invalid_intermediate_image_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nimages"):
            EncodingOperator(
                self.smaps, self.nvoxels, self.sampling, 3, self.motion, Nimages=2
            )


if __name__ == "__main__":
    unittest.main()

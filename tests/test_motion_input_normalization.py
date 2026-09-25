"""Regression checks for acquisition-wide respiratory scaling."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.preprocessing.DataLoader import DataLoader
from src.preprocessing.motion_input_normalization import acquisition_zscore_motion_input


class AcquisitionMotionNormalizationTests(unittest.TestCase):
    def test_2d_uses_all_slices_and_preserves_per_sensor_structure(self):
        trace = torch.tensor([
            [[1.0, 20.0], [2.0, 30.0], [3.0, 40.0]],
            [[4.0, 50.0], [5.0, 60.0], [6.0, 70.0]],
        ], dtype=torch.float64)
        normalized, mean, std = acquisition_zscore_motion_input(trace, data_dimension='2D')
        torch.testing.assert_close(mean, trace.mean(dim=(0, 1)))
        torch.testing.assert_close(std, trace.std(dim=(0, 1), correction=0))
        torch.testing.assert_close(normalized.mean(dim=(0, 1)), torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
        torch.testing.assert_close(normalized.std(dim=(0, 1), correction=0), torch.ones(2, dtype=torch.float64))
        self.assertNotAlmostEqual(normalized[0, :, 0].std(correction=0).item(), 1.0)

    def test_2d_sensorless_input(self):
        trace = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        normalized, _, _ = acquisition_zscore_motion_input(trace, data_dimension='2D')
        self.assertAlmostEqual(normalized.mean().item(), 0.0)
        self.assertAlmostEqual(normalized.std(correction=0).item(), 1.0)

    def test_3d_uses_all_readouts_per_sensor(self):
        trace = torch.tensor([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]], dtype=torch.float64)
        normalized, _, _ = acquisition_zscore_motion_input(trace, data_dimension='3D')
        torch.testing.assert_close(normalized.mean(dim=0), torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
        torch.testing.assert_close(normalized.std(dim=0, correction=0), torch.ones(2, dtype=torch.float64))

    def test_real_data_loader_normalizes_before_slice_selection(self):
        loader = DataLoader.__new__(DataLoader)
        loader.params = SimpleNamespace(data_dimension='2D', kspace_sampling_type='from-data',
                                        motion_signal_normalization='acquisition_zscore', Nex=1)
        loader.t_device = torch.device('cpu')
        motion = np.arange(8, dtype=np.float64).reshape(2, 4, 1)
        arrays = dict(kspace=np.zeros((1, 1, 2, 2, 2), dtype=np.complex128), motion_data=motion,
                      idx_ky=np.zeros((2, 4), dtype=np.int64),
                      idx_kz=np.zeros((2, 4), dtype=np.int64),
                      idx_nex=np.zeros((2, 4), dtype=np.int64))
        loader._ingest_realworld_arrays(arrays)
        torch.testing.assert_close(loader._source_motion_data.mean(), torch.zeros((), dtype=torch.float64), atol=1e-14, rtol=0)
        torch.testing.assert_close(loader._source_motion_data.std(correction=0), torch.ones((), dtype=torch.float64))
        self.assertAlmostEqual(motion.mean(), 3.5)  # Input cache is not modified.

    def test_constant_sensor_fails(self):
        trace = torch.tensor([[[1.0, 2.0], [2.0, 2.0]]])
        with self.assertRaisesRegex(ValueError, 'constant'):
            acquisition_zscore_motion_input(trace, data_dimension='2D')


if __name__ == '__main__':
    unittest.main()

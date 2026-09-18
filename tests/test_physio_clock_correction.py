import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from pathlib import Path

import numpy as np

from src.preprocessing.RawDataPreparer import RawDataPreparer
from src.runtime.runtime_config import load_config


class ClockCorrectionTests(unittest.TestCase):
    def synchronize(self, times, values, target, shift=0.0):
        return RawDataPreparer._synchronize_to_sequence_end(
            [times], [values], target, source_sequence_end=0.0,
            bounds='autoregression', physio_clock_drift_seconds=shift)[:, 0]

    def test_both_directions_and_exact_one_second_limit(self):
        times = np.linspace(-10, 0, 501)
        values = 3 + 2 * times
        for shift in (-1.0, -0.25, 0.0, 0.25, 1.0):
            with self.subTest(shift=shift):
                actual = self.synchronize(times, values, times, shift)
                np.testing.assert_allclose(actual, 3 + 2 * (times - shift), atol=1e-8)

    def test_sinusoidal_autoregression_with_intercept(self):
        times = np.linspace(-10, 0, 1001)
        signal = lambda t: 4 + np.sin(2 * np.pi * 0.3 * t)
        for shift in (-0.8, 0.8):
            np.testing.assert_allclose(self.synchronize(times, signal(times), times, shift),
                                       signal(times - shift), atol=1e-7)

    def test_over_one_second_rejected_at_either_edge(self):
        times = np.linspace(-10, 0, 501)
        for shift in (-1.001, 1.001):
            with self.assertRaisesRegex(ValueError, 'not reasonable to extrapolate so far'):
                self.synchronize(times, times, times, shift)

    def test_existing_missing_coverage_counts_toward_limit(self):
        with self.assertRaisesRegex(ValueError, 'maximum is 1 second'):
            self.synchronize(np.linspace(-8, 0, 401), np.ones(401), [-10, 0])

    def test_irregular_constant_data(self):
        np.testing.assert_allclose(self.synchronize(
            np.array([-3, -2.8, -2.3, -1.7, -1, 0]), np.full(6, 7.0),
            [-3.5, -2, 0.5]), 7)

    def test_already_synchronized_data_rejects_shift(self):
        times = np.linspace(-5, 0, 251)
        for kind in ('physio_text', 'physio_array'):
            for shift in (-0.5, 0.0, 0.5):
                with self.subTest(kind=kind, shift=shift):
                    preparer = RawDataPreparer.__new__(RawDataPreparer)
                    preparer.physio_clock_drift_seconds = shift
                    preparer.physiological_format = kind
                    preparer.reader = SimpleNamespace(read_data=lambda: {
                        'time_seconds': torch.tensor(times),
                        'slice_indices': torch.zeros(len(times), dtype=torch.int64)})
                    preparer._physiological_channels = lambda: (
                        [np.full(len(times), -1)], [2 * times + 3], None, 'raise')
                    # Stop after synchronization, before unrelated k-space processing.
                    preparer.physiological_reader = SimpleNamespace(
                        already_synchronized=True,
                        prepare_motion=Mock(side_effect=RuntimeError('stop')))
                    if shift != 0:
                        with self.assertRaisesRegex(ValueError, 'must be zero for already-synchronized'):
                            preparer._prepare_data()
                        preparer.physiological_reader.prepare_motion.assert_not_called()
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'stop'):
                            preparer._prepare_data()
                        np.testing.assert_array_equal(
                            preparer.synchronization['acquisition_values'][:, 0], 2 * times + 3)
                        np.testing.assert_array_equal(
                            preparer.synchronization['physiological_time_seconds'][0], times)

    def test_config_defaults_overrides_and_validation(self):
        root = Path(__file__).resolve().parents[1] / 'config'
        for source in ('ismrmrd', 'siemens'):
            for kind in ('polaris', 'physio_text', 'physio_array'):
                kwargs = dict(data_type=f'{source}-{kind}',
                    reconstruction_config=root / 'reconstruction/nonrigid_2d.toml',
                    coil_sensitivity_config=root / 'coil_sensitivity/espirit.toml',
                    ismrmrd_reader_config=root / 'real_data/ismrmrd_reader.toml')
                if kind == 'polaris':
                    kwargs['polaris_config'] = root / 'real_data/polaris.toml'
                self.assertEqual(load_config(**kwargs).physio_clock_drift_seconds, 0.0)
                for shift in (-0.5, 0.5):
                    self.assertEqual(load_config(**kwargs, overrides={
                        'physio_clock_drift_seconds': shift}).physio_clock_drift_seconds, shift)
                for invalid in (True, '0.5', float('nan'), float('inf')):
                    with self.assertRaises(ValueError):
                        load_config(**kwargs, overrides={'physio_clock_drift_seconds': invalid})


if __name__ == '__main__':
    unittest.main()

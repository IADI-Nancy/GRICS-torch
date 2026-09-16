"""End alignment regression tests for external tracking synchronization."""
import sys
import unittest
import tempfile
from unittest.mock import patch
import h5py
import torch
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.physiological_data.PolarisInfraredTrackerReader import PolarisInfraredTrackerReader
from src.preprocessing.RawDataPreparer import RawDataPreparer


class TrackingSynchronizationTest(unittest.TestCase):
    def test_end_alignment_interpolation_and_slice_endpoint(self):
        tracking = NS(time_seconds=np.array([0., 1., 2.]),
                      tool_positions=np.array([[0., 0., 0.], [2., 4., 6.], [4., 8., 12.]]))
        # A selected slice ends before the full sequence; do not re-zero it.
        out = RawDataPreparer._synchronize_to_sequence_end([tracking.time_seconds] * 3, list(tracking.tool_positions.T), [-1.5, -0.5])
        np.testing.assert_allclose(out, [[1., 2., 3.], [3., 6., 9.]])
        with self.assertRaisesRegex(ValueError, 'extrapolation'):
            RawDataPreparer._synchronize_to_sequence_end([tracking.time_seconds] * 3, list(tracking.tool_positions.T), [-3., 0.])
        tracking.tool_positions[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'invalid physiological'):
            RawDataPreparer._synchronize_to_sequence_end([tracking.time_seconds] * 3, list(tracking.tool_positions.T), [-1.])


    def test_saec_and_polaris_agree(self):
        values = np.array([[0., 2., 4.], [0., 4., 8.]])
        target = torch.tensor([-1.5, -0.5], dtype=torch.float64)
        saec = RawDataPreparer._synchronize_to_sequence_end([np.array([-2., -1., 0.])] * 2, list(values), target.numpy(), source_sequence_end=0.0, bounds='edge')
        polaris = RawDataPreparer._synchronize_to_sequence_end([np.array([10., 11., 12.])] * 2,
                                               list(values), target.numpy())
        np.testing.assert_allclose(saec, polaris)

    def test_saec_independent_channels_keep_trigger_reference(self):
        # Recordings continue after the trigger: do not align their final samples to zero.
        times = [np.array([-2., 0., 2.]), np.array([-3., -1., 1., 3.])]
        values = [np.array([0., 2., 4.]), np.array([0., 4., 8., 12.])]
        out = RawDataPreparer._synchronize_to_sequence_end(times, values, [-1., 0.], source_sequence_end=0.0, bounds='edge')
        np.testing.assert_allclose(out, [[1., 4.], [2., 6.]])
        edge = RawDataPreparer._synchronize_to_sequence_end(times, values, [-4., 4.], source_sequence_end=0.0, bounds='edge')
        np.testing.assert_allclose(edge, [[0., 0.], [4., 12.]])

    def test_preparation_preserves_reader_indices_and_full_sequence_clock(self):
        from src.preprocessing.ISMRMRDReader import ISMRMRDReader
        from src.preprocessing.RawDataPreparer import RawDataPreparer
        reader = ISMRMRDReader('unused')
        reader.reference_kspace = None
        reader._raw_uses_kz_as_volume_axis = False
        # Two complementary ky lines occupy one existing NEX; slice 1 ends early.
        extracted = (
            torch.ones((1, 1, 8, 2, 2), dtype=torch.complex128),
            torch.tensor([-2., -1.5, -1., 0.]),
            torch.tensor([1, 1, 0, 0]), torch.tensor([0, 1, 0, 1]),
            torch.zeros(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64),
            'repetition', torch.tensor([0]), {0: {}, 1: {}},
        )
        tracking = NS(time_seconds=np.array([0., 1., 2.]),
                      tool_positions=np.array([[0., 0., 0.], [2., 4., 6.], [4., 8., 12.]]))
        with tempfile.TemporaryDirectory() as tmp, \
             patch('src.preprocessing.RawDataPreparer.ISMRMRDReader', return_value=reader), \
             patch.object(reader, '_extract_mri_data', return_value=extracted) as read, \
             patch.object(PolarisInfraredTrackerReader, 'read', return_value=tracking), \
             patch.object(PolarisInfraredTrackerReader, '_lowpass', side_effect=lambda t, p: p):
            output = Path(tmp) / 'slice.h5'
            preparer = RawDataPreparer('unused', 'tracking.tsv', physiological_format='PolarisInfraredTracker', sensor_type=None, device='cpu', print_raw_calibration_lines=False, polaris_channel_mode='all')
            preparer.read_data(output_h5_file=output, slice_idx=1)
            sync = preparer.synchronization
            read.assert_called_once_with()
            with h5py.File(output) as f:
                self.assertEqual(f['kspace'].shape, (1, 1, 4, 2, 1))
                np.testing.assert_array_equal(f['idx_ky'][:], [[0, 1]])
                np.testing.assert_array_equal(f['idx_nex'][:], [[0, 0]])
            np.testing.assert_allclose(sync['acquisition_values'][1], [1., 2., 3.])

    def test_saec_preparer_uses_common_synchronizer(self):
        from src.preprocessing.RawDataPreparer import RawDataPreparer
        from src.preprocessing.physiological_data.SAECReader import SAECReader
        raw = {
            'kspace': torch.ones((1, 1, 8, 2, 1), dtype=torch.complex128),
            'time_seconds': torch.tensor([-1., 0.]),
            'slice_indices': torch.tensor([0, 0]),
            'idx_ky': torch.tensor([0, 1]), 'idx_kz': torch.tensor([0, 0]),
            'idx_nex': torch.tensor([0, 0]), 'nex_source': 'repetition',
            'nex_values': torch.tensor([0]), 'slice_geometry': {0: {}},
            'reference_kspace': None, 'uses_kz_as_volume_axis': False,
        }
        preparer = RawDataPreparer('unused', 'physiology.h5', physiological_format='SAEC', sensor_type='BELT', device='cpu', print_raw_calibration_lines=False, polaris_channel_mode='all')
        with patch.object(preparer.reader, 'read_data', return_value=raw), \
             patch.object(SAECReader, '_read_and_process_data',
                          return_value=(np.array([-2., 0., 2.]), np.array([0., 2., 4.]))), \
             patch.object(RawDataPreparer, '_synchronize_to_sequence_end',
                          wraps=RawDataPreparer._synchronize_to_sequence_end) as sync:
            data = preparer.read_data()
        sync.assert_called_once()
        np.testing.assert_allclose(data['motion_data'], [[[1.], [2.]]])
        np.testing.assert_array_equal(data['idx_ky'], [[0, 1]])

    def test_largest_amplitude_uses_range_not_offset_and_exports_one_channel(self):
        raw = {
            'kspace': torch.ones((1, 1, 8, 3, 1), dtype=torch.complex128),
            'time_seconds': torch.tensor([-2., -1., 0.]),
            'slice_indices': torch.zeros(3, dtype=torch.int64),
            'idx_ky': torch.arange(3), 'idx_kz': torch.zeros(3, dtype=torch.int64),
            'idx_nex': torch.zeros(3, dtype=torch.int64), 'nex_source': 'repetition',
            'nex_values': torch.tensor([0]), 'slice_geometry': {0: {}},
            'reference_kspace': None, 'uses_kz_as_volume_axis': False,
        }
        # Huge X offset and a Z excursion before the MRI window must not win.
        tracking = NS(time_seconds=np.array([0., 1., 2., 3.]),
                      tool_positions=np.array([[1000., 0., 100.], [1000., 0., 0.],
                                               [1001., 10., 2.], [1002., 0., 4.]]))
        preparer = RawDataPreparer('unused', 'tracking.tsv',
                                  physiological_format='PolarisInfraredTracker', sensor_type=None, device='cpu',
                                  print_raw_calibration_lines=False, polaris_channel_mode='largest-amplitude')
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(preparer.reader, 'read_data', return_value=raw), \
             patch.object(PolarisInfraredTrackerReader, 'read', return_value=tracking), \
             patch.object(PolarisInfraredTrackerReader, '_lowpass', side_effect=lambda t, p: p):
            output = Path(tmp) / 'slice.h5'
            data = preparer.read_data(output_h5_file=output, slice_idx=0)
            self.assertEqual(data['motion_data'].shape, (1, 3, 1))
            self.assertEqual(preparer.selected_polaris_channels, ['Ty'])
            np.testing.assert_allclose(preparer.polaris_peak_to_peak, [2., 10., 4.])
            expected = np.array([0., 10., 0.])
            expected = (expected - expected.mean()) / expected.std()
            np.testing.assert_allclose(data['motion_data'][0, :, 0], expected)
            self.assertEqual(preparer.synchronization['acquisition_values'].shape, (3, 3))
            with h5py.File(output) as f:
                self.assertEqual(list(f.attrs['polaris_channels']), ['Ty'])

    def test_polaris_lowpass_removes_fast_noise_and_preserves_respiration(self):
        times = np.arange(0., 1000., 0.1)
        noise = np.sin(2 * np.pi * 3.0 * times)
        respiration = np.sin(2 * np.pi * 0.3 * times)
        positions = np.column_stack([1000 + noise, respiration, 2 * respiration])
        tracking = NS(time_seconds=times, tool_positions=positions)
        preparer = RawDataPreparer('unused', 'tracking.tsv',
                                  physiological_format='PolarisInfraredTracker', sensor_type=None, device='cpu',
                                  print_raw_calibration_lines=False, polaris_channel_mode='all')
        with patch.object(PolarisInfraredTrackerReader, 'read', return_value=tracking):
            _, channels, _, _ = preparer._physiological_channels()
        middle = slice(2000, -2000)
        self.assertLess(np.std(channels[0][middle]), 0.05)
        self.assertAlmostEqual(np.mean(channels[0][middle]), 1000., places=3)
        self.assertGreater(np.std(channels[1][middle]) / np.std(respiration[middle]), 0.9)
        np.testing.assert_allclose(channels[2], 2 * channels[1])
        np.testing.assert_array_equal(tracking.tool_positions, positions)

    def test_polaris_lowpass_rejects_short_recordings_and_gaps(self):
        with self.assertRaisesRegex(ValueError, 'at least 7'):
            PolarisInfraredTrackerReader._lowpass(np.arange(3.), np.zeros((3, 3)))
        positions = np.zeros((10, 3))
        positions[4, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'resolve gaps first'):
            PolarisInfraredTrackerReader._lowpass(np.arange(10.), positions)

    def test_invalid_polaris_channel_mode(self):
        with self.assertRaisesRegex(ValueError, 'polaris_channel_mode'):
            PolarisInfraredTrackerReader(channel_mode='invalid')


if __name__ == '__main__':
    unittest.main()

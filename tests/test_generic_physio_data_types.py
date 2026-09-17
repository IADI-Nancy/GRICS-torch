"""Generic physiology parsing, routing, synchronization, and cache regressions."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.preprocessing.DataLoader import DataLoader
from src.preprocessing.RawDataPreparer import RawDataPreparer
from src.preprocessing.physiological_data.PreprocessedPhysioReader import PreprocessedPhysioReader
from src.runtime.runtime_config import load_config
from src.runtime.data_cache import release_leases


class GenericPhysioTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(release_leases)
        self.root = Path(self.tmp.name)

    def arrays(self, times=None, values=None):
        times = np.array([[[10.], [11.], [12.]]]) if times is None else np.asarray(times)
        values = np.array([[[0., 10.], [2., 20.], [4., 30.]]]) if values is None else np.asarray(values)
        paths = (self.root / 'timestamps.npy', self.root / 'values.npy')
        for path, data in zip(paths, (times, values)):
            np.save(path, data)
        return paths

    def text(self, body):
        path = self.root / 'physio.txt'
        path.write_text(body)
        return path

    def raw(self, volume=False):
        return dict(kspace=torch.ones((1, 1, 8, 2, 2), dtype=torch.complex128),
                    time_seconds=torch.tensor([-2., -1.5, -1., 0.]),
                    slice_indices=torch.tensor([1, 0, 1, 0]),
                    idx_ky=torch.tensor([0, 0, 1, 1]), idx_kz=torch.tensor([1, 0, 1, 0]),
                    idx_nex=torch.zeros(4, dtype=torch.int64), nex_source='repetition',
                    nex_values=torch.tensor([0]), slice_geometry={0: {}, 1: {}},
                    ismrmrd_header='<header/>', reference_kspace=None,
                    uses_kz_as_volume_axis=volume)

    def preparer(self, files, fmt='physio_array'):
        scan = self.root / 'scan.mrd'
        scan.touch(exist_ok=True)
        return RawDataPreparer(scan, files, physiological_format=fmt, sensor_type=None,
                               device='cpu', print_raw_calibration_lines=False,
                               polaris_channel_mode=None)

    def config(self, kind, dim='2D'):
        return load_config(data_type=kind,
                           reconstruction_config=f'config/reconstruction/nonrigid_{dim.lower()}.toml',
                           coil_sensitivity_config='config/coil_sensitivity/odille_spline.toml',
                           ismrmrd_reader_config='config/real_data/ismrmrd_reader.toml')

    def test_all_modes_route_and_enforce_slice_rules(self):
        for source in ('ismrmrd', 'siemens'):
            for fmt in ('physio_text', 'physio_array'):
                for dim in ('2D', '3D'):
                    kind = f'{source}-{fmt}'
                    with self.subTest(kind=kind, dim=dim):
                        mri_key = 'ismrmrd_file' if source == 'ismrmrd' else 'siemens_raw_file'
                        physio = {'physio_file': 'physio.txt'} if fmt == 'physio_text' else {
                            'physio_timestamps_file': 'times.npy', 'physio_values_file': 'values.npy'}
                        filenames = {mri_key: 'scan', **physio}
                        params = self.config(kind, dim)
                        self.assertEqual(params.kspace_sampling_type, 'from-data')
                        self.assertEqual(params.simulated_motion_type, 'as-it-is')
                        for filename in (filenames, tuple(filenames.values())):
                            loader = DataLoader(params, filename=filename, run_pipeline=False)
                            with patch.object(loader, '_convert_siemens_to_ismrmrd', return_value='converted') as convert, \
                                 patch('src.preprocessing.DataLoader.RawDataPreparer') as preparer, \
                                 patch.object(loader, '_ingest_realworld_arrays'):
                                loader.load_data()
                            self.assertEqual(convert.call_count, int(source == 'siemens'))
                            self.assertEqual(preparer.call_args.kwargs['physiological_format'], fmt)
                            self.assertEqual(preparer.call_args.kwargs['physiological_file'],
                                             'physio.txt' if fmt == 'physio_text' else ('times.npy', 'values.npy'))
                        if dim == '3D':
                            with self.assertRaisesRegex(ValueError, "data_dimension='3D'"):
                                DataLoader(params, filename=filenames, slice_idx=0, run_pipeline=False)
                        for filename in ({mri_key: 'scan'}, ('scan', None), {**filenames, 'typo': 'bad'}):
                            with self.assertRaises(ValueError):
                                DataLoader(params, filename=filename, run_pipeline=False)

    def test_text_and_arrays_agree_and_preserve_channel_order(self):
        array_reader, text_reader = PreprocessedPhysioReader('physio_array'), PreprocessedPhysioReader('physio_text')
        text = self.text('SENSOR TIMESTAMP VALUE1 VALUE2\n0 10 0 10\n'
                         '0 11 2 20\n0 12 4 30\n')
        at, av, _, _ = array_reader.read_channels(self.arrays())
        tt, tv, _, _ = text_reader.read_channels(text)
        np.testing.assert_array_equal(at, tt)
        np.testing.assert_array_equal(av, tv)
        self.assertEqual(text_reader.metadata['physio_channels'], [[0, 0], [0, 1]])
        preparer = self.preparer(text, 'physio_text')
        with patch.object(preparer.reader, 'read_data', return_value=self.raw()):
            result = preparer.read_data(slice_idx=0)
        np.testing.assert_allclose(result['motion_data'], [[[1., 15.], [4., 30.]]])
        np.testing.assert_array_equal(result['idx_ky'], [[0, 1]])

    def test_text_independent_sampling_rates(self):
        path = self.text('2 10 0\n2 12 4\n0 20 10\n0 21 20\n0 22 30\n')
        p = self.preparer(path, 'physio_text')
        with patch.object(p.reader, 'read_data', return_value=self.raw(True)):
            result = p.read_data()
        np.testing.assert_allclose(result['motion_data'], [[10, 0], [15, 1], [20, 2], [30, 4]])

    def test_synchronized_channels_bypass_interpolation_before_slice_selection(self):
        for fmt in ('physio_array', 'physio_text'):
            files = self.arrays(np.full((1, 4, 1), -1), np.arange(8).reshape(1, 4, 2))
            if fmt == 'physio_text':
                files = self.text(''.join(f'0 -1 {2 * i} {2 * i + 1}\n' for i in range(4)))
            for volume in (False, True):
                p = self.preparer(files, fmt)
                with patch.object(p.reader, 'read_data', return_value=self.raw(volume)), \
                     patch.object(p, '_synchronize_to_sequence_end', side_effect=AssertionError('must bypass')):
                    result = p.read_data(slice_idx=None if volume else 1)
                expected = np.arange(8).reshape(4, 2) if volume else [[[0, 1], [4, 5]]]
                np.testing.assert_array_equal(result['motion_data'], expected)
                np.testing.assert_array_equal(p.synchronization['physiological_time_seconds'][0], [-2, -1.5, -1, 0])
        p = self.preparer(self.arrays(np.full((1, 3, 1), -1)))
        with patch.object(p.reader, 'read_data', return_value=self.raw()), self.assertRaisesRegex(ValueError, 'one value per'):
            p.read_data()

    def test_validation(self):
        reader = PreprocessedPhysioReader('physio_array')
        for times, values in ((np.zeros((1, 3)), np.zeros((1, 3, 1))),
                              (np.zeros((1, 3, 1)), np.zeros((1, 2, 1))),
                              (np.array([[[0], [2], [1]]]), np.zeros((1, 3, 1))),
                              (np.array([[[0], [1], [1]]]), np.zeros((1, 3, 1))),
                              (np.full((1, 3, 1), np.nan), np.zeros((1, 3, 1))),
                              (np.arange(3).reshape(1, 3, 1), np.full((1, 3, 1), 1j))):
            with self.subTest(times=times), self.assertRaises(ValueError):
                reader.read_channels(self.arrays(times, values))
        for text in ('', '0 1', '-1 0 1', '0.5 0 1', '0 nan 2',
                     '0 -1 2\n1 0 2\n1 1 3', '0 0 2\n0 0 3',
                     '0 0 1\n0 1 1 2'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                PreprocessedPhysioReader('physio_text').read_channels(self.text(text))
        p = self.preparer(self.arrays(np.array([[[11.], [11.5], [12.]]])))
        with patch.object(p.reader, 'read_data', return_value=self.raw()), self.assertRaisesRegex(ValueError, 'extrapolation'):
            p.read_data()

    def test_cache_uses_both_array_files_and_restores_metadata(self):
        paths = self.arrays()
        cache = dict(cache_h5=True, cache_root=self.root / 'cache', remove_temporary_data_after_run=False)
        p = self.preparer(paths)
        with patch.object(p.reader, 'read_data', return_value=self.raw()):
            original = p.read_data(**cache)
        # Do not touch the MRI file between readers: its identity is part of the key.
        q = RawDataPreparer(p.reader.ismrmrd_file, paths, physiological_format='physio_array',
                            sensor_type=None, device='cpu', print_raw_calibration_lines=False, polaris_channel_mode=None)
        with patch.object(q, '_prepare_data', side_effect=AssertionError('cache miss')):
            cached = q.read_data(**cache)
        np.testing.assert_array_equal(original['motion_data'], cached['motion_data'])
        self.assertEqual(q.physiological_reader.metadata['physio_channels'], [[0, 0], [0, 1]])
        for path in paths:
            np.save(path, np.load(path) + 1)
            with patch.object(q.reader, 'read_data', return_value=self.raw()) as read:
                changed = q.read_data(**cache)
            read.assert_called_once()
            self.assertNotEqual(cached['realworld_h5_path'], changed['realworld_h5_path'])
            cached = changed


if __name__ == '__main__':
    unittest.main()

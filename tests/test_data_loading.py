"""Regression checks for image conversion, sensor selection, and header retention."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import ismrmrd
import numpy as np

from src.preprocessing.DataLoader import DataLoader
from src.preprocessing.ISMRMRDReader import ISMRMRDReader
from src.preprocessing.RawDataPreparer import RawDataPreparer
from src.preprocessing.physiological_data.SAECReader import SAECReader


class DataLoadingChecks(unittest.TestCase):
    def test_complex_image_uses_magnitude_without_mutating_input(self):
        image = np.array([[1j, 2j], [3j, 4j]])
        np.testing.assert_allclose(DataLoader._normalize_real_image(image), [[0, 1/3], [2/3, 1]])
        np.testing.assert_array_equal(image, [[1j, 2j], [3j, 4j]])
        real = np.array([[1., 2.], [3., 4.]])
        DataLoader._normalize_real_image(real)
        np.testing.assert_array_equal(real, [[1, 2], [3, 4]])

    def test_image_channel_layouts(self):
        image = np.arange(6.).reshape(2, 3)
        expected = image / 5
        for value in (image, image[..., None], np.repeat(image[..., None], 3, axis=-1),
                      np.repeat(image[..., None], 4, axis=-1)):
            np.testing.assert_allclose(DataLoader._normalize_real_image(value), expected)
        self.assertEqual(DataLoader._normalize_real_image(np.ones((1, 3))).shape, (1, 3))
        with self.assertRaises(ValueError):
            DataLoader._normalize_real_image(np.ones((2, 3, 2)))

    def test_single_marmot_rejects_unusable_tracks(self):
        times = [np.arange(12.)]
        for scores in (np.zeros(3), np.full(3, np.nan)):
            with patch.object(SAECReader, '_get_filtered_marmot_data',
                              return_value=(np.zeros((12, 3)), scores)):
                with self.assertRaisesRegex(ValueError, 'No valid MARMOT'):
                    SAECReader._get_filtered_resp_data(times, [np.zeros((12, 3))], '1MARMOT')

    def test_single_marmot_selects_valid_sensor_and_track(self):
        times = [np.arange(12.), np.arange(12.) + 1]
        good = np.column_stack([np.full(12, np.nan), np.arange(12.), np.zeros(12)])
        with patch.object(SAECReader, '_get_filtered_marmot_data', side_effect=[
            (np.zeros((12, 3)), np.zeros(3)), (good, np.array([100., 3., 0.]))]):
            t, signal = SAECReader._get_filtered_resp_data(times, [good, good], '1MARMOT')
        np.testing.assert_array_equal(t, times[1])
        np.testing.assert_allclose(signal, np.arange(12.) / np.std(np.arange(12.)))

    def test_full_header_survives_reader_export_and_reload(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <encoding>
    <encodedSpace>
      <matrixSize><x>8</x><y>2</y><z>1</z></matrixSize>
      <fieldOfView_mm><x>200</x><y>200</y><z>5</z></fieldOfView_mm>
    </encodedSpace>
    <reconSpace>
      <matrixSize><x>4</x><y>2</y><z>1</z></matrixSize>
      <fieldOfView_mm><x>100</x><y>200</y><z>5</z></fieldOfView_mm>
    </reconSpace>
    <encodingLimits>
      <kspace_encoding_step_1><minimum>0</minimum><maximum>1</maximum><center>1</center></kspace_encoding_step_1>
    </encodingLimits>
    <trajectory>cartesian</trajectory>
  </encoding>
  <userParameters><userParameterString><name>note</name><value>Full header: µ</value></userParameterString></userParameters>
</ismrmrdHeader>"""
        acq = ismrmrd.Acquisition()
        acq.resize(8, 1)
        acq.data[:] = 1
        with patch('ismrmrd.Dataset') as dataset:
            dataset.return_value.read_xml_header.return_value = xml.encode('utf-8')
            dataset.return_value.number_of_acquisitions.return_value = 1
            dataset.return_value.read_acquisition.return_value = acq
            raw = ISMRMRDReader('unused').read_data()
        self.assertEqual(raw['ismrmrd_header'], xml)
        preparer = RawDataPreparer('unused', 'unused', physiological_format='SAEC',
            sensor_type='BELT', device='cpu', print_raw_calibration_lines=False, polaris_channel_mode=None)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prepared.h5'
            with patch.object(preparer.reader, 'read_data', return_value=raw), patch.object(
                preparer, '_physiological_channels',
                return_value=([np.array([-1., 1.])], [np.array([0., 1.])], 0., 'edge')):
                result = preparer.read_data(output_h5_file=path, slice_idx=0)
            self.assertEqual(result['ismrmrd_header'], xml)
            with h5py.File(path, 'r') as f:
                self.assertEqual(f['ismrmrd_header'].asstr()[()], xml)
            loader = DataLoader.__new__(DataLoader)
            loader.params = SimpleNamespace(kspace_sampling_type='from-data', save_debug_plots=False)
            with patch.object(loader, '_ingest_realworld_arrays') as ingest:
                loader._load_realworld_data(path)
            self.assertEqual(ingest.call_args.args[0]['ismrmrd_header'], xml)
            # Older exports without a header remain readable.
            with h5py.File(path, 'a') as f:
                del f['ismrmrd_header']
            with patch.object(loader, '_ingest_realworld_arrays') as ingest:
                loader._load_realworld_data(path)
            self.assertNotIn('ismrmrd_header', ingest.call_args.args[0])


if __name__ == '__main__':
    unittest.main()

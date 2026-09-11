"""Configuration and raw-reader routing for SAEC and Polaris sources."""
import unittest
from unittest.mock import patch

from src.runtime.runtime_config import load_config
from src.preprocessing.DataLoader import DataLoader


class PolarisDataTypesTest(unittest.TestCase):
    def config(self, kind, dimension='2D'):
        return load_config(
            data_type=kind,
            reconstruction_config=f'config/reconstruction/nonrigid_{dimension.lower()}.toml',
            simulated_motion_type='as-it-is',
        )

    def test_raw_types_select_reader_and_convert_only_siemens(self):
        for kind in ('ismrmrd-polaris', 'siemens-polaris', 'ismrmrd-saec', 'siemens-saec'):
            for dimension in ('2D', '3D'):
                with self.subTest(kind=kind, dimension=dimension):
                    params = self.config(kind, dimension)
                    self.assertEqual(params.kspace_sampling_type, 'from-data')
                    self.assertTrue(params.flip_for_display)
                    params.debug_flag = False
                    is_polaris = kind.endswith('-polaris')
                    if is_polaris:
                        params.rawdata_sensor_type = None
                    mri_key = 'siemens_file' if kind.startswith('siemens') else 'ismrmrd_file'
                    physiology_key = 'polaris_file' if is_polaris else 'saec_file'
                    loader = DataLoader(params, filename={mri_key: 'scan', physiology_key: 'physiology'},
                                        slice_idx=1 if dimension == '2D' else None, run_pipeline=False)
                    with patch.object(loader, '_convert_siemens_to_ismrmrd', return_value='converted.mrd') as convert, \
                         patch('src.preprocessing.DataLoader.RawDataPreparer') as preparer, \
                         patch.object(loader, '_ingest_realworld_arrays') as ingest:
                        loader._load_source_data()
                    self.assertEqual(convert.call_count, int(kind.startswith('siemens')))
                    kwargs = preparer.call_args.kwargs
                    self.assertEqual(kwargs['physiological_format'], 'PolarisInfraredTracker' if is_polaris else 'SAEC')
                    self.assertEqual(kwargs['physiological_file'], 'physiology')
                    self.assertEqual(kwargs['ismrmrd_file'], 'converted.mrd' if kind.startswith('siemens') else 'scan')
                    ingest.assert_called_once_with(preparer.return_value.read_data.return_value,
                                                   slice_idx=1 if dimension == '2D' else None)

    def test_polaris_filename_validation_and_slice_rules(self):
        for kind in ('ismrmrd-polaris', 'siemens-polaris'):
            with self.subTest(kind=kind):
                params = self.config(kind)
                DataLoader(params, filename=('scan', 'tracking.tsv'), run_pipeline=False)
                for filename in ('scan', ('scan',), ('scan', None), {'polaris_file': 'tracking.tsv'}):
                    with self.assertRaisesRegex(ValueError, 'both files'):
                        DataLoader(params, filename=filename, run_pipeline=False)
                with self.assertRaisesRegex(ValueError, "data_dimension='3D'"):
                    DataLoader(self.config(kind, '3D'), filename=('scan', 'tracking.tsv'),
                               slice_idx=0, run_pipeline=False)


if __name__ == '__main__':
    unittest.main()

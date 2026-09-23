"""CPU integration coverage for the single-volume Siemens breast pipeline."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from pipelines import siemens_breast_3d_lowres as pipeline
from src.preprocessing.CoilSensitivityCalculator import CoilSensitivityCalculator


class SiemensBreast3DTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.h5 = self.root / 'subject.h5'
        n = 8
        rng = np.random.default_rng(1)
        with h5py.File(self.h5, 'w') as f:
            f['kspace'] = rng.normal(size=(1, 1, n, n, n)).astype(np.complex128)
            f['idx_ky'] = np.repeat(np.arange(n), n)
            f['idx_kz'] = np.tile(np.arange(n), n)
            f['idx_nex'] = np.zeros(n * n, dtype=np.int64)
            f['motion_data'] = np.sin(np.arange(n * n) / 5).reshape(-1, 1)

    def test_input_validation(self):
        self.assertEqual(pipeline.data_type_from_raw_data_file(self.h5, None), 'preprocessed-real')
        with self.assertRaisesRegex(FileNotFoundError, 'not a file'):
            pipeline.data_type_from_raw_data_file(self.root, None)
        raw = self.root / 'subject.dat'
        raw.touch()
        with self.assertRaisesRegex(ValueError, 'saec_file is required'):
            pipeline.data_type_from_raw_data_file(raw, None)
        saec = self.root / 'subject.saec'
        saec.touch()
        self.assertEqual(pipeline.data_type_from_raw_data_file(raw, saec), 'siemens-saec')
        mrd = self.root / 'subject.mrd'
        with h5py.File(mrd, 'w') as f:
            f.create_group('dataset')
        self.assertEqual(pipeline.data_type_from_raw_data_file(mrd, saec), 'ismrmrd-saec')
        with self.assertRaisesRegex(ValueError, 'omit saec_file'):
            pipeline.data_type_from_raw_data_file(self.h5, saec)

    def test_rejects_2d_input(self):
        with h5py.File(self.h5, 'a') as f:
            del f['kspace']
            f['kspace'] = np.ones((1, 1, 8, 8, 1), dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, 'Expected 3D'):
            pipeline.data_type_from_raw_data_file(self.h5, None)

    def test_cpu_volume_pipeline(self):
        torch.set_num_threads(1)
        spline = CoilSensitivityCalculator.calculate_iadi_spline
        overrides = dict(ResolutionLevels=[0.5, 1.0], GN_iterations_per_level=[2, 2],
                         max_iter_recon=2, max_iter_motion=2)
        original_timed = pipeline.timed_reconstruction
        def check_preprocessing_metadata(data):
            metadata = json.loads((Path(data.params.reconstruction_folder) / '.metadata.json').read_text())
            self.assertEqual(metadata['status'], 'preprocessing')
            return original_timed(data)
        with patch.object(
            CoilSensitivityCalculator, 'calculate_iadi_spline', autospec=True, side_effect=spline,
        ) as calculate_spline, patch.object(
            pipeline, 'timed_reconstruction', side_effect=check_preprocessing_metadata,
        ), patch(
            'src.reconstruction.joint_reconstructor_utils.logging.show_and_save_image',
            side_effect=AssertionError('Solver must not render plots'),
        ):
            result = pipeline.run_pipeline(str(self.h5), output_root=self.root / 'runs/volume',
                                           device='cpu', overrides=overrides)
            second = pipeline.run_pipeline(self.h5, output_root=self.root / 'runs/volume',
                                           device='cpu', overrides=overrides, save_reconstruction_tensors=False)
            unlogged = pipeline.run_pipeline(
                self.h5, output_root=self.root / 'runs/volume', device='cpu',
                overrides={**overrides, 'save_reconstruction_logs': False},
                save_reconstruction_tensors=True,
            )
        self.assertEqual(calculate_spline.call_count, 3)
        self.assertIsNone(unlogged['reconstructions'][0]['log_file'])
        self.assertFalse(list(unlogged['run_folder'].rglob('*.log')))
        self.assertEqual(len(list(unlogged['run_folder'].rglob('*.pt'))), 2)
        self.assertNotEqual(result['run_folder'], second['run_folder'])
        output = result['reconstructions'][0]['output_dir']
        image = torch.load(output / 'image_reconstructed.pt', weights_only=True)
        alpha = torch.load(output / 'motion_parameters.pt', weights_only=True)
        self.assertEqual(tuple(image.shape), (1, 8, 8, 8))
        self.assertTrue(torch.isfinite(image).all())
        self.assertTrue(torch.isfinite(alpha).all())
        torch.testing.assert_close(result['reconstructions'][0]['image'], image)
        torch.testing.assert_close(second['reconstructions'][0]['image'], image)
        self.assertFalse(list(second['run_folder'].rglob('*.pt')))
        self.assertFalse(list(result['run_folder'].rglob('*.png')))
        self.assertEqual(len(list(result['run_folder'].rglob('*.log'))), 1)
        for run in (result, second):
            log = run['reconstructions'][0]['log_file'].read_text()
            self.assertIn('Model optimization step:', log)
            self.assertIn('Total time of reconstruction run:', log)
            resolved = json.loads((run['run_folder'] / 'config_resolved.json').read_text())
            expected = resolved['save_reconstruction_tensors']
            self.assertIn(f'  save_reconstruction_tensors = {expected}\n', log)
            metadata = json.loads((run['reconstructions'][0]['output_dir'].parent / '.metadata.json').read_text())
            self.assertEqual(metadata['configuration']['save_reconstruction_tensors'], expected)
            self.assertEqual(metadata['tensor_export'], 'pipeline' if expected else 'disabled')
            self.assertIn('inputs', metadata)
            if expected:
                self.assertIn('Tensor export: deferred to caller', log)
        self.assertEqual(len(list(result['run_folder'].rglob('*.pt'))), 2)
        self.assertGreater(result['timings']['reconstruction_seconds'], 0)
        manifest = json.loads((result['run_folder'] / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['inputs']['raw_data_file'], str(self.h5))
        config = json.loads((result['run_folder'] / 'config_resolved.json').read_text())
        self.assertEqual(config['coil_sensitivity_method'], 'odille-spline')


if __name__ == '__main__':
    unittest.main()

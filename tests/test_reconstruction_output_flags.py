"""Independent output controls shared by every reconstruction entry point."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.runtime.runtime_config import load_config
from src.runtime.output_layout import bind_output_paths
from src.reconstruction.joint_reconstructor_utils.logging import JointReconstructionLogger

ROOT = Path(__file__).resolve().parents[1]


def config(**overrides):
    return load_config(
        data_type='preprocessed-real',
        reconstruction_config=ROOT / 'config/reconstruction/nonrigid_3d_breast.toml',
        coil_sensitivity_config=ROOT / 'config/coil_sensitivity/odille_spline.toml',
        overrides=overrides,
    )


class OutputFlagTests(unittest.TestCase):
    def test_independent_logs_tensors_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            for logs, tensors, plots in itertools.product((False, True), repeat=3):
                with self.subTest(logs=logs, tensors=tensors, plots=plots):
                    params = config(save_reconstruction_logs=logs,
                                    save_reconstruction_tensors=tensors, save_debug_plots=plots)
                    folder = Path(tmp) / f'{logs}-{tensors}-{plots}'
                    bind_output_paths(params, folder)
                    with patch('src.reconstruction.joint_reconstructor_utils.logging.show_and_save_image') as preview, patch(
                        'src.reconstruction.joint_reconstructor_utils.logging.save_final_nonrigid_alpha_maps'
                    ), patch('src.reconstruction.joint_reconstructor_utils.logging._save_run_residual_plots'):
                        logger = JointReconstructionLogger(params, [1])
                        logger.start_run()
                        logger.append('test log entry')
                        logger.run_finished()
                        logger.save_final_outputs(torch.ones(1, 4, 4, 4), torch.zeros(3, 4, 4, 4, 1))
                    self.assertEqual((folder / 'reconstruction.log').exists(), logs)
                    self.assertEqual(len(list(folder.rglob('*.pt'))), 2 if tensors else 0)
                    self.assertEqual(preview.called, plots)
                    metadata = json.loads((folder / '.metadata.json').read_text())
                    self.assertEqual(metadata['status'], 'reconstructed')
                    self.assertEqual(metadata['image_shape'], [1, 4, 4, 4])
                    self.assertEqual(metadata['configuration']['save_reconstruction_tensors'], tensors)

    def test_deferred_export_preserves_config_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            params = config(save_reconstruction_logs=False, save_reconstruction_tensors=True,
                            save_debug_plots=False)
            folder = Path(tmp)
            bind_output_paths(params, folder)
            logger = JointReconstructionLogger(params, [1])
            logger.save_final_outputs(torch.ones(1, 4, 4, 4), torch.zeros(3, 4, 4, 4, 1),
                                      defer_tensor_export=True)
            self.assertTrue(params.save_reconstruction_tensors)
            self.assertFalse(list(folder.rglob('*.pt')))
            metadata = json.loads((folder / '.metadata.json').read_text())
            self.assertEqual(metadata['tensor_export'], 'deferred')
            self.assertTrue(metadata['configuration']['save_reconstruction_tensors'])

    def test_flags_are_strict_booleans(self):
        for key in ('save_reconstruction_logs', 'save_reconstruction_tensors'):
            for invalid in ('false', 0, None):
                with self.subTest(key=key, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, f'{key} must be a boolean'):
                        config(**{key: invalid})

    def test_all_reconstruction_configs_use_general_flags(self):
        for path in (ROOT / 'config/reconstruction').glob('*.toml'):
            with self.subTest(path=path):
                params = load_config(
                    data_type='preprocessed-real', reconstruction_config=path,
                    coil_sensitivity_config=ROOT / 'config/coil_sensitivity/odille_spline.toml',
                )
                self.assertTrue(params.save_reconstruction_logs)
                self.assertTrue(params.save_reconstruction_tensors)
                self.assertFalse(hasattr(params, 'save_reconstruction_outputs'))


if __name__ == '__main__':
    unittest.main()

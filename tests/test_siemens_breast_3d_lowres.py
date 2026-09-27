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
from src.preprocessing.MotionBinner import MotionBinner
from src.preprocessing.Sampling import Sampling
from src.reconstruction.joint_reconstructor_utils.resampling.downsample import downsample_data


class SiemensBreast3DTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.h5 = self.root / 'subject.h5'
        n = 8
        rng = np.random.default_rng(1)
        with h5py.File(self.h5, 'w') as f:
            # Two complete acquisitions of the same volume.  The 3D breast
            # profile must reconstruct one shared image from these readouts.
            f['kspace'] = rng.normal(size=(1, 2, n, n, n)).astype(np.complex128)
            f['idx_ky'] = np.tile(np.repeat(np.arange(n), n), 2)
            f['idx_kz'] = np.tile(np.tile(np.arange(n), n), 2)
            f['idx_nex'] = np.repeat(np.arange(2), n * n)
            f['motion_data'] = np.sin(np.arange(2 * n * n) / 5).reshape(-1, 1)

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

    def test_load_grics_cpp_sensitivity_maps_preserves_cpp_axis_order(self):
        nx, ny, nz, ncoils = 2, 3, 4, 2
        # GRICS++ uses column-major [y, x, z, coil] storage.
        cpp = np.empty((ny, nx, nz, ncoils), dtype=np.complex64, order='F')
        for coil in range(ncoils):
            for z in range(nz):
                for x in range(nx):
                    for y in range(ny):
                        cpp[y, x, z, coil] = coil * 1000 + z * 100 + x * 10 + y + 1j
        path = self.root / 'SensitivityMaps.dat'
        cpp.ravel(order='F').tofile(path)
        maps = pipeline.load_grics_cpp_sensitivity_maps(
            path, ncoils=ncoils, spatial_shape=(nx, ny, nz), device=torch.device('cpu'),
        )
        self.assertEqual(tuple(maps.shape), (ncoils, nx, ny, nz))
        for coil in range(ncoils):
            for z in range(nz):
                for x in range(nx):
                    for y in range(ny):
                        self.assertEqual(maps[coil, x, y, z].item(), cpp[y, x, z, coil])

    def test_load_grics_cpp_complex_double_sensitivity_maps(self):
        cpp = np.arange(24, dtype=np.float64).reshape((2, 2, 2, 3), order='F') * (1 + 1j)
        path = self.root / 'SensitivityMaps-double.dat'
        cpp.ravel(order='F').tofile(path)
        maps = pipeline.load_grics_cpp_sensitivity_maps(
            path, ncoils=3, spatial_shape=(2, 2, 2), device=torch.device('cpu'),
        )
        self.assertEqual(maps.dtype, torch.complex128)
        self.assertEqual(maps[2, 1, 0, 1].item(), cpp[0, 1, 1, 2])

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
        self.assertEqual(config['repetition_image_model'], 'shared')


    def test_coarse_kspace_energy_reselects_from_quantized_virtual_states(self):
        # Full-resolution selection keeps virtual times 0 and 3. After the
        # central crop, only 1 and 2 remain and 2 has more energy. GRICS++
        # must select original virtual time 2, not the merged-state center 3.
        motion = torch.arange(4, dtype=torch.float64).unsqueeze(1)
        ky = torch.arange(4, dtype=torch.int64)
        nex = torch.zeros(4, dtype=torch.int64)
        energy = torch.tensor([100.0, 1.0, 80.0, 90.0], dtype=torch.float64)
        kspace = torch.sqrt(energy / 2).reshape(1, 1, 1, 4, 1).expand(
            1, 1, 2, 4, 1,
        ).clone().to(torch.complex128)
        params = type('Params', (), {
            'N_motion_states': 2, 'Nex': 1,
            'motion_binning_mode': 'kspace_energy',
            'motion_quantization_bins': 256, 'save_debug_plots': False,
        })()
        result = MotionBinner.bin_motion(
            motion, ky, None, nex, torch.device('cpu'), params,
            return_debug_data=True, return_virtual_state_data=True, kspace=kspace,
        )
        _, _, selected_signal, selected_labels, _, _, _, virtual_labels, virtual_signal = result

        def sampling(labels, states):
            return Sampling.build_sampling_per_nex_per_motion(
                [[ky[labels == state] for state in range(states)]],
                torch.device('cpu'), Nx=2, Ny=4, Nz=1,
            )

        data_full = {
            'Nx': 2, 'Ny': 4, 'Nz': 1, 'Nimages': 1,
            'SensitivityMaps': torch.ones((1, 2, 4, 1), dtype=torch.complex128),
            'KspaceData': kspace,
            'SamplingIndices': sampling(selected_labels, 2),
        }
        selected_result = downsample_data(
            data_full, 0.5, 1, selected_signal, params, torch.device('cpu'),
        )
        virtual_result = downsample_data(
            data_full, 0.5, 1, selected_signal, params, torch.device('cpu'),
            source_sampling_indices=sampling(virtual_labels, len(virtual_signal)),
            source_motion_signal=virtual_signal,
        )
        torch.testing.assert_close(selected_result['MotionSignal'][:, 0], torch.tensor([3.0], dtype=torch.float64))
        torch.testing.assert_close(virtual_result['MotionSignal'][:, 0], torch.tensor([2.0], dtype=torch.float64))


if __name__ == '__main__':
    unittest.main()

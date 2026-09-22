"""Callable pipeline, deferred output, and solver timing integration tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pipelines import _execution, siemens_breast_T2 as pipeline
from src.preprocessing.DataLoader import DataLoader


class SiemensT2APITests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw, self.saec = self.root / 'subject.mrd', self.root / 'subject.saec'
        self.raw.touch()
        self.saec.touch()
        n = 8
        rng = np.random.default_rng(2)
        arrays = {
            'kspace': rng.normal(size=(1, 1, n, n, 2)).astype(np.complex128),
            'motion_data': np.tile(np.sin(np.arange(n)), (2, 1)),
            'idx_ky': np.tile(np.arange(n), (2, 1)),
            'idx_kz': np.zeros((2, n), dtype=np.int64),
            'idx_nex': np.zeros((2, n), dtype=np.int64),
        }
        # Replace only raw-file decoding; preprocessing and both solvers are real.
        def read_fixture(loader, *args, **kwargs):
            loader._ingest_realworld_arrays(arrays)
        self.addCleanup(patch.stopall)
        patch.object(DataLoader, '_load_realworld_data_from_ismrm_and_physiology', read_fixture).start()
        matrix = SimpleNamespace(x=n, y=n, z=1)
        header = SimpleNamespace(encoding=[SimpleNamespace(
            reconSpace=SimpleNamespace(matrixSize=matrix),
            encodedSpace=SimpleNamespace(matrixSize=matrix))])
        patch.object(pipeline, 'acquisition_header', return_value=header).start()
        patch('src.reconstruction.joint_reconstructor_utils.logging.show_and_save_image',
              side_effect=AssertionError('Solver must not plot')).start()
        self.options = dict(
            output_root=self.root / 'runs/t2', device='cpu',
            overrides=dict(ResolutionLevels=[0.5, 1.0], GN_iterations_per_level=[2, 2],
                           N_motion_states=4, N_motion_states_per_level='full',
                           max_iter_recon=2, max_iter_motion=2),
        )

    def test_sequential_parallel_and_repeated_calls(self):
        sequential = pipeline.run_pipeline(str(self.raw), str(self.saec), max_workers=1, **self.options)
        parallel = pipeline.run_pipeline(self.raw, self.saec, max_workers=2, **self.options)
        self.assertIsNone(pipeline.LOADED_DATA)
        for result in (sequential, parallel):
            self.assertEqual([item['slice_idx'] for item in result['reconstructions']], [0, 1])
            self.assertEqual(len(list(result['run_folder'].rglob('*.pt'))), 4)
            self.assertFalse(list(result['run_folder'].rglob('*.png')))
            self.assertEqual(len(list(result['run_folder'].rglob('*.log'))), 2)
            for item in result['reconstructions']:
                log = item['log_file'].read_text()
                self.assertIn('Fixed point iter', log)
                self.assertIn('Total time of reconstruction run:', log)
                self.assertIn('  save_reconstruction_tensors = True\n', log)
                self.assertIn('Tensor export: deferred to caller', log)
            manifest = json.loads((result['run_folder'] / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            for item in result['reconstructions']:
                self.assertEqual(tuple(item['image'].shape), (1, 8, 8))
                self.assertTrue(torch.isfinite(item['image']).all())
                self.assertGreater(item['reconstruction_seconds'], 0)
                torch.testing.assert_close(item['image'], torch.load(item['image_file'], weights_only=True))
        for left, right in zip(sequential['reconstructions'], parallel['reconstructions']):
            torch.testing.assert_close(left['image'], right['image'])
            torch.testing.assert_close(left['motion'], right['motion'])
        selected = pipeline.run_pipeline(self.raw, self.saec, max_workers=1, slice_start=1,
                                         save_reconstruction_tensors=False, return_tensors=False, **self.options)
        self.assertEqual(len(selected['reconstructions']), 1)
        self.assertNotIn('image', selected['reconstructions'][0])
        self.assertFalse(list(selected['run_folder'].rglob('*.pt')))

    def test_exports_follow_all_reconstructions(self):
        original = pipeline.timed_reconstruction
        completed = []
        def timed(data):
            self.assertFalse(list(Path(data.params.run_folder).rglob('*.pt')))
            self.assertFalse(list(Path(data.params.run_folder).rglob('*.dcm')))
            result = original(data)
            completed.append(data.slice_idx)
            return result
        def export(*args, **kwargs):
            self.assertEqual(completed, [0, 1])
        with patch.object(pipeline, 'timed_reconstruction', side_effect=timed), patch.object(
            pipeline, 'write_reconstruction_dicom', side_effect=export,
        ) as dicom:
            result = pipeline.run_pipeline(self.raw, self.saec, max_workers=1, export_dicom=True,
                                           dicom_series_number=123, **self.options)
        self.assertEqual(dicom.call_count, 2)
        self.assertEqual(dicom.call_args.kwargs['series_number'], 123)
        self.assertIsNotNone(result['reconstructions'][0]['dicom_file'])

    def test_failure_finalizes_manifest_and_next_call_works(self):
        with patch.object(pipeline, 'timed_reconstruction', side_effect=RuntimeError('solver failed')):
            with self.assertRaisesRegex(RuntimeError, 'solver failed'):
                pipeline.run_pipeline(self.raw, self.saec, max_workers=1, **self.options)
        manifests = list((self.root / 'runs/t2').glob('*/manifest.json'))
        self.assertEqual(len(manifests), 1)
        self.assertEqual(json.loads(manifests[0].read_text())['status'], 'failed')
        result = pipeline.run_pipeline(self.raw, self.saec, max_workers=1, slice_stop=1, **self.options)
        self.assertEqual(len(result['reconstructions']), 1)
        self.assertIsNone(pipeline.LOADED_DATA)


class TimingTests(unittest.TestCase):
    def test_cuda_synchronizes_on_both_sides_of_solver(self):
        events = []
        data = SimpleNamespace(kspace=SimpleNamespace(device=torch.device('cuda:0')),
                               smaps=None, sampling_idx=None, motion_signal=None,
                               params=SimpleNamespace(save_reconstruction_tensors=True), kspace_scale=1, motion_plot_context=None)
        def run(**kwargs):
            self.assertEqual(kwargs, {'defer_tensor_export': True})
            events.append('run')
            return 1, 2
        solver = SimpleNamespace(run=run)
        def construct(*args, **kwargs):
            self.assertIs(kwargs['params'], data.params)
            self.assertTrue(kwargs['params'].save_reconstruction_tensors)
            events.append('construct')
            return solver
        def clock():
            events.append('clock')
            return float(len(events))
        with patch.object(_execution.torch.cuda, 'synchronize', side_effect=lambda *_: events.append('sync')), patch.object(
            _execution, 'JointReconstructor', side_effect=construct,
        ), patch.object(_execution.time, 'perf_counter', side_effect=clock):
            image, motion, elapsed = _execution.timed_reconstruction(data)
        self.assertEqual(events, ['sync', 'clock', 'construct', 'run', 'sync', 'clock'])
        self.assertEqual((image, motion), (1, 2))
        self.assertGreater(elapsed, 0)


if __name__ == '__main__':
    unittest.main()

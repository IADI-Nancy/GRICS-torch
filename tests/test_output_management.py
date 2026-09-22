"""Run with: MPLBACKEND=Agg python -m unittest discover -s tests -p 'test_output_management.py'."""
import json
import multiprocessing as mp
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from src.runtime.data_cache import acquire_cached, cache_key, clear_cache, release_leases
from src.runtime.output_layout import RunOutputs, bind_output_paths, execution_scope
from src.preprocessing.RawDataPreparer import RawDataPreparer
from src.reconstruction.joint_reconstructor_utils.logging import JointReconstructionLogger


def hold_cached(root, connection):
    lease = acquire_cached(root, 'converted', 'shared', '.mrd',
                           lambda path: path.write_bytes(b'original'), remove=False)
    connection.send(str(lease.path))
    connection.recv()
    lease.close()
    connection.close()


def build_cached(root, connection, build_started, finish_build):
    def build(path):
        with (Path(root) / 'build_count').open('a') as count:
            count.write('built\n')
        build_started.set()
        if not finish_build.wait(20):
            raise RuntimeError('Test builder timed out')
        path.write_bytes(b'complete')
    lease = acquire_cached(root, 'converted', 'simultaneous', '.mrd', build, remove=False)
    connection.send(str(lease.path))
    connection.recv()
    lease.close()


class OutputManagementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix='grics-output-test-')
        self.root = Path(self.temporary.name)

    def tearDown(self):
        release_leases()
        self.temporary.cleanup()

    def test_cache_reuse_and_active_file_protection_across_processes(self):
        context = mp.get_context('fork')
        parent, child = context.Pipe()
        process = context.Process(target=hold_cached, args=(self.root, child))
        process.start()
        self.assertTrue(parent.poll(20))
        path = Path(parent.recv())
        try:
            lease = acquire_cached(self.root, 'converted', 'shared', '.mrd',
                                   lambda _: self.fail('Cache hit must not rebuild'), remove=True)
            removed, busy = clear_cache(self.root)
            self.assertEqual(removed, [])
            self.assertEqual(busy, [str(path)])
            lease.close()
            self.assertTrue(path.exists(), 'Another process still uses this file')
        finally:
            parent.send('close')
            process.join(20)
            if process.is_alive():
                process.terminate()
                process.join()
        self.assertEqual(process.exitcode, 0)
        self.assertFalse(path.exists(), 'Deferred cleanup must run when final reader exits')

    def test_simultaneous_misses_share_one_build_without_waiting_for_run_end(self):
        context = mp.get_context('fork')
        started, finish = context.Event(), context.Event()
        pipes = [context.Pipe() for _ in range(2)]
        processes = [context.Process(target=build_cached, args=(self.root, child, started, finish))
                     for parent, child in pipes]
        try:
            processes[0].start()
            self.assertTrue(started.wait(20))
            processes[1].start()
            finish.set()
            paths = []
            # Both readers must acquire the result while the other still holds its lease.
            for parent, child in pipes:
                self.assertTrue(parent.poll(20))
                paths.append(parent.recv())
            self.assertEqual(paths[0], paths[1])
            self.assertEqual((self.root / 'build_count').read_text(), 'built\n')
        finally:
            finish.set()
            for parent, child in pipes:
                parent.send('close')
            for process in processes:
                if process.pid is not None:
                    process.join(20)
                    if process.is_alive():
                        process.terminate()
                        process.join()
        self.assertTrue(all(process.exitcode == 0 for process in processes))

    def test_clear_runs_filters_dry_run_and_active_protection(self):
        from src.utils.clear_runs import clear_runs
        def create(label):
            params = SimpleNamespace(output_root=str(self.root), workflow_label=label,
                                     data_dimension='2D', ResolutionLevels=[1.0])
            return RunOutputs(params)
        active = create('selected')
        time.sleep(1.05)
        completed = create('selected')
        completed.close()
        other = create('other')
        other.close()
        unknown = self.root / 'selected' / 'unrecognized'
        unknown.mkdir()
        (unknown / 'keep.txt').write_text('keep')
        try:
            matches, skipped = clear_runs(self.root, workflow_label='selected', dry_run=True)
            self.assertEqual(matches, [str(completed.root)])
            self.assertTrue(completed.root.exists())
            self.assertTrue(any(path == str(active.root) and reason == 'active run' for path, reason in skipped))
            matches, _ = clear_runs(self.root, older_than_days=1)
            self.assertEqual(matches, [])
            matches, _ = clear_runs(self.root, workflow_label='selected')
            self.assertEqual(matches, [str(completed.root)])
            self.assertFalse(completed.root.exists())
            self.assertTrue(active.root.exists())
            self.assertTrue(other.root.exists())
            self.assertTrue((unknown / 'keep.txt').exists())
        finally:
            active.close()
        matches, _ = clear_runs(self.root)
        self.assertEqual(set(matches), {str(active.root), str(other.root)})

    def test_dicom_export_uses_memory_header_and_no_deprecated_calls(self):
        import warnings
        import pydicom
        from src.utils.dicom_export import write_reconstruction_dicom
        from pipelines.siemens_breast_T2 import grics_zero_fill_shapes
        xml = self.header_xml()
        raw = SimpleNamespace(ismrmrd_header=xml, source_ismrmrd_file='/nonexistent/source.mrd', Nx=8,
            _source_slice_geometry={0: {'position': [0., 0., 0.], 'read_dir': [1., 0., 0.],
                                      'phase_dir': [0., 1., 0.], 'slice_dir': [0., 0., 1.]}})
        path = self.root / 'export.dcm'
        with patch('h5py.File', side_effect=AssertionError('Must not reopen MRD')), warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            self.assertEqual(grics_zero_fill_shapes(raw), ((8, 8), (8, 8)))
            write_reconstruction_dicom(torch.arange(64).reshape(1, 8, 8).to(torch.complex128),
                                       path, raw_data=raw, slice_index=0)
        dataset = pydicom.dcmread(path)
        self.assertEqual(dataset.file_meta.TransferSyntaxUID, pydicom.uid.ExplicitVRLittleEndian)
        self.assertEqual(dataset.pixel_array.shape, (8, 8))
        self.assertEqual(int(dataset.pixel_array.max()), 4095)

    @staticmethod
    def header_xml():
        return """<?xml version="1.0" encoding="UTF-8"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <experimentalConditions><H1resonanceFrequency_Hz>123000000</H1resonanceFrequency_Hz></experimentalConditions>
  <encoding>
    <encodedSpace><matrixSize><x>8</x><y>8</y><z>1</z></matrixSize>
      <fieldOfView_mm><x>80</x><y>80</y><z>5</z></fieldOfView_mm></encodedSpace>
    <reconSpace><matrixSize><x>8</x><y>8</y><z>1</z></matrixSize>
      <fieldOfView_mm><x>80</x><y>80</y><z>5</z></fieldOfView_mm></reconSpace>
    <encodingLimits/><trajectory>cartesian</trajectory>
  </encoding>
</ismrmrdHeader>"""

    def test_mrd_fallback_and_dataset_open_read_only(self):
        from src.utils.ismrmrd_io import acquisition_header, ReadOnlyDataset
        path = self.root / 'header.mrd'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('dataset/xml', data=[self.header_xml()], dtype=h5py.string_dtype())
        with h5py.File(path, 'r'):
            with patch('h5py.File', wraps=h5py.File) as opened:
                header = acquisition_header(ismrmrd_file=path)
                self.assertEqual(opened.call_args.args[1], 'r')
                self.assertEqual(header.encoding[0].reconSpace.matrixSize.x, 8)
                reader = ReadOnlyDataset(path)
                try:
                    self.assertEqual(reader._file.mode, 'r')
                    self.assertIn(b'ismrmrdHeader', reader.read_xml_header())
                finally:
                    reader.close()

    def test_failed_builder_never_publishes_partial_artifact(self):
        def fail(path):
            path.write_bytes(b'partial')
            raise ValueError('conversion failed')
        with self.assertRaises(ValueError):
            acquire_cached(self.root, 'converted', 'failed', '.mrd', fail)
        self.assertEqual(list(self.root.rglob('*.mrd')), [])
        self.assertEqual(list(self.root.rglob('*.tmp')), [])
        lease = acquire_cached(self.root, 'converted', 'failed', '.mrd',
                               lambda path: path.write_bytes(b'complete'), remove=False)
        lease.close()
        removed, busy = clear_cache(self.root)
        self.assertEqual(len(removed), 1)
        self.assertEqual(busy, [])

    def test_source_and_settings_invalidate_cache_keys(self):
        source = self.root / 'input.dat'
        source.write_bytes(b'input')
        before = cache_key([source], {'mode': 1})
        self.assertNotEqual(before, cache_key([source], {'mode': 2}))
        source.write_bytes(b'changed source')
        self.assertNotEqual(before, cache_key([source], {'mode': 1}))

    def test_failed_run_manifest_and_cleanup(self):
        params = SimpleNamespace(output_root=str(self.root / 'runs'), workflow_label='test',
                                 data_dimension='2D', ResolutionLevels=[1.0])
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with execution_scope():
                outputs = RunOutputs(params)
                lease = acquire_cached(self.root / 'cache', 'converted', 'key', '.mrd',
                                       lambda path: path.write_bytes(b'mrd'))
                raise RuntimeError('failure')
        manifest = json.loads((outputs.root / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['error'], 'failure')
        self.assertFalse(lease.path.exists())

    def test_hdf5_cache_preserves_metadata_and_is_reused_before_slice_selection(self):
        mrd = self.root / 'input.mrd'
        physiology = self.root / 'physiology.h5'
        mrd.write_bytes(b'source')
        physiology.write_bytes(b'signal')
        calls = []
        def preparer():
            instance = RawDataPreparer.__new__(RawDataPreparer)
            instance.reader = SimpleNamespace(ismrmrd_file=str(mrd))
            instance.physiological_file = str(physiology)
            instance.physiological_format = 'SAEC'
            instance.sensor_type = 'BELT'
            instance.polaris_channel_mode = None
            instance.physio_clock_drift_seconds = 0.0
            instance.physiological_reader = SimpleNamespace(metadata={})
            def prepare():
                calls.append(1)
                instance.synchronization = {'times': [np.arange(3.), np.arange(4.)]}
                instance.physiological_reader.metadata = {'channels': ['Tx', 'Ty']}
                instance._uses_kz_as_volume_axis = False
                return {'kspace': np.ones((1, 1, 4, 4, 2), dtype=np.complex128),
                        'motion_data': np.ones((2, 3, 1)), 'idx_ky': np.zeros((2, 3), dtype=int),
                        'idx_kz': np.zeros((2, 3), dtype=int), 'idx_nex': np.zeros((2, 3), dtype=int),
                        'nex_values': np.array([0]), 'nex_source': 'repetition',
                        'slice_geometry': {0: {'position': [1., 2., 3.]}, 1: {'position': [4., 5., 6.]}},
                        'ismrmrd_header': '<header/>'}
            instance._prepare_data = prepare
            return instance
        first = preparer()
        full = first.read_data(cache_h5=True, cache_root=self.root / 'cache',
                               remove_temporary_data_after_run=False)
        second = preparer()
        selected = second.read_data(cache_h5=True, cache_root=self.root / 'cache',
                                    remove_temporary_data_after_run=False, slice_idx=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(selected['kspace'].shape[-1], 1)
        self.assertEqual(selected['slice_geometry'][0]['position'], [4., 5., 6.])
        self.assertEqual(second.physiological_reader.metadata['channels'], ['Tx', 'Ty'])
        np.testing.assert_equal(second.synchronization['times'][1], np.arange(4.))
        self.assertEqual(selected['realworld_h5_path'], full['realworld_h5_path'])
        with h5py.File(full['realworld_h5_path']) as handle:
            self.assertEqual(handle['kspace'].shape[-1], 2)

    def test_pipeline_exports_postprocessed_image_without_debug_dependency(self):
        from pipelines import siemens_breast_T2 as pipeline
        from src.runtime.runtime_config import load_config
        original = torch.ones((1, 8, 8), dtype=torch.complex128) * 4
        motion = torch.zeros((2, 8, 8, 1))
        for save in (False, True):
            with self.subTest(save=save):
                params = load_config(data_type='preprocessed-real',
                    reconstruction_config='config/reconstruction/nonrigid_2d_breast.toml',
                    coil_sensitivity_config='config/coil_sensitivity/odille_spline.toml',
                    overrides={'output_root': str(self.root), 'workflow_label': f'export_{save}',
                               'runtime_device': 'cpu', 'save_debug_plots': False,
                               'save_reconstruction_tensors': save})
                raw = SimpleNamespace(params=params, Nz=1,
                    postprocessing=SimpleNamespace(normalize_image_by_grics_reference=True),
                    grics_reference_image=torch.ones((8, 8, 1)) * 2,
                    kspace=torch.zeros((1, 1, 8, 8, 1)),
                    run_slice_pipeline=lambda slice_idx: None)
                def load(*args, **kwargs):
                    RunOutputs(params)
                    return raw
                captured = []
                def export(image, path, **kwargs):
                    captured.append(image.clone())
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'dicom')
                with patch.object(pipeline, 'load_all_slices', side_effect=load), \
                     patch.object(pipeline, 'grics_zero_fill_shapes', return_value=((12, 12), (12, 12))), \
                     patch.object(pipeline, 'timed_reconstruction',
                                  side_effect=lambda data: (original.clone(), motion.clone(), 0.01)), \
                     patch.object(pipeline, 'write_reconstruction_dicom', side_effect=export):
                    result = pipeline.run_pipeline('input.mrd', 'motion.saec', max_workers=1,
                        save_reconstruction_tensors=save, export_dicom=True)
                item = result['reconstructions'][0]
                normalized, _ = pipeline.normalize_reconstruction_by_grics_reference(
                    original, pipeline.grics_reference_image_for_normalization(raw, original))
                expected = pipeline.zero_fill_loaded_reconstruction(normalized, (12, 12), (12, 12))
                torch.testing.assert_close(item['image'], expected)
                torch.testing.assert_close(captured[0], expected)
                self.assertEqual(tuple(captured[0].shape), (1, 12, 12))
                saved_path = item['output_dir'] / 'image_reconstructed.pt'
                self.assertEqual(saved_path.exists(), save)
                if save:
                    torch.testing.assert_close(torch.load(saved_path, weights_only=True), expected)
                self.assertFalse(list(result['run_folder'].rglob('image_postprocessed.pt')))
                self.assertTrue(item['dicom_file'].is_file())

    def test_small_cpu_reconstruction_layout_and_debug_controls(self):
        from src.runtime.runtime_config import load_config
        from src.runtime.runtime_setup import initialize_runtime
        from src.preprocessing.DataLoader import DataLoader
        from src.reconstruction.JointReconstructor import JointReconstructor
        torch.set_num_threads(1)
        for debug in (False, True):
            with self.subTest(debug=debug), execution_scope():
                params = load_config(data_type='shepp-logan',
                    reconstruction_config='config/reconstruction/nonrigid_2d.toml',
                    coil_sensitivity_config='config/coil_sensitivity/odille_spline.toml',
                    shepp_logan_config='config/synthetic_data/shepp_logan_2d.toml',
                    sampling_config='config/sampling_simulation/interleaved.toml',
                    motion_simulation_config='config/motion_simulation/nonrigid_2d.toml',
                    overrides={'output_root':str(self.root), 'workflow_label':'smoke', 'runtime_device':'cpu',
                        'N_SheppLogan':16, 'calibration_lines':8, 'NshotsPerNex':2,
                        'ResolutionLevels':[0.5,1.0], 'GN_iterations_per_level':[1,1],
                        'max_iter_recon':2, 'max_iter_motion':2, 'N_motion_states':2,
                        'check_simulated_motion_consistency':False, 'Nex':2,
                        'save_debug_plots':debug, 'print_to_console':False, 'verbose':False})
                sp_device, device = initialize_runtime(params)
                data = DataLoader(params=params, sp_device=sp_device, t_device=device)
                recon = JointReconstructor(data.kspace, data.smaps, data.sampling_idx,
                    motion_signal=data.motion_signal, params=params, kspace_scale=data.kspace_scale,
                    motion_plot_context=data.motion_plot_context)
                image, motion = recon.run()
                folder = Path(params.results_folder)
                torch.testing.assert_close(torch.load(folder / 'image_reconstructed.pt', weights_only=True), image.cpu())
                self.assertEqual((folder / 'image_reconstructed_nex_002.png').exists(), debug)
                self.assertEqual(Path(params.debug_folder).exists(), debug)
                self.assertEqual(Path(params.initial_data_folder).exists(), debug)
                if debug:
                    self.assertTrue((Path(params.debug_folder) / 'level_01' / 'image_reconstructed.png').exists())
                    self.assertTrue((Path(params.debug_folder) / 'level_02' / 'image_reconstructed.png').exists())
                    self.assertTrue((Path(params.debug_folder) / 'residuals' / 'recon_residual.png').exists())
            manifest = json.loads((Path(params.run_folder) / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['reconstructions']['slice_001']['image_shape'], list(image.shape))


if __name__ == '__main__':
    unittest.main()

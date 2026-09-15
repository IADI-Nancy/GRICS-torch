import contextlib
import io
import itertools
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import ismrmrd
import numpy as np
import torch
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from src.preprocessing.RawDataReader import RawDataReader
from src.preprocessing.RawDataPreparer import RawDataPreparer
from src.preprocessing.DataLoader import DataLoader

FLAGS = ('save_debug_plots', 'check_simulated_motion_consistency',
         'use_deterministic_algorithms', 'print_raw_calibration_lines')

class DebugFlagChecks(unittest.TestCase):
    def test_configuration_combinations_and_migration(self):
        with TemporaryDirectory() as folder:
            paths = {key: folder + '/' + key for key in
                     ('debug_folder', 'logs_folder', 'results_folder', 'initial_data_folder')}
            def config(overrides):
                return load_config(data_type='siemens-polaris',
                    reconstruction_config='config/reconstruction/nonrigid_2d.toml',
                    simulated_motion_type='as-it-is', data_dimension='2D',
                    overrides={**paths, **overrides})
            default = config({})
            self.assertEqual(tuple(getattr(default, key) for key in FLAGS), (True, True, True, False))
            for values in itertools.product((False, True), repeat=4):
                with self.subTest(values=values):
                    params = config(dict(zip(FLAGS, values)))
                    self.assertEqual(tuple(getattr(params, key) for key in FLAGS), values)
            with self.assertRaisesRegex(ValueError, 'debug_flag has been replaced'):
                config({'debug_flag': False})
            for key in FLAGS:
                with self.subTest(invalid=key), self.assertRaisesRegex(ValueError, key + ' must be a boolean'):
                    config({key: 'false'})

    def test_runtime_toggle_in_same_process(self):
        old = (torch.are_deterministic_algorithms_enabled(),
               torch.is_deterministic_algorithms_warn_only_enabled(),
               torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        params = NS(clean_output_folders_before_run=False, runtime_device='cpu', seed=None)
        try:
            with patch('src.runtime.runtime_setup._install_runtime_safety_guards'):
                for enabled in (True, False, True, False):
                    params.use_deterministic_algorithms = enabled
                    initialize_runtime(params)
                    self.assertEqual(torch.are_deterministic_algorithms_enabled(), enabled)
                    self.assertEqual(torch.backends.cudnn.deterministic, enabled)
        finally:
            torch.use_deterministic_algorithms(old[0], warn_only=old[1])
            torch.backends.cudnn.deterministic = old[2]
            torch.backends.cudnn.benchmark = old[3]

    def test_raw_logging_preserves_data_and_duplicate_checks(self):
        acquisitions = []
        for ky, flag in enumerate((ismrmrd.ACQ_IS_PARALLEL_CALIBRATION,
                                   ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING, None)):
            acq = ismrmrd.Acquisition()
            acq.resize(4, 1)
            acq.idx.kspace_encode_step_1 = ky
            acq.acquisition_time_stamp = ky * 10
            acq.data[:] = np.arange(4) + ky + 1j
            if flag is not None:
                acq.setFlag(flag)
            acquisitions.append(acq)
        header = NS(encoding=[NS(encodingLimits=NS(slice=None, repetition=None,
            kspace_encoding_step_1=NS(maximum=3), kspace_encoding_step_2=None))],
            measurementInformation=NS(patientPosition=None))
        def extract(enabled, duplicate=False):
            ds = MagicMock()
            rows = acquisitions + ([acquisitions[0]] if duplicate else [])
            ds.number_of_acquisitions.return_value = len(rows)
            ds.read_acquisition.side_effect = rows
            reader = RawDataReader('unused.h5', print_raw_calibration_lines=enabled)
            output = io.StringIO()
            with patch('src.preprocessing.RawDataReader.ismrmrd.Dataset', return_value=ds), \
                 patch('src.preprocessing.RawDataReader.ismrmrd.xsd.CreateFromDocument', return_value=header), \
                 contextlib.redirect_stdout(output):
                result = reader._extract_mri_data()
            ds.close.assert_called_once()
            return result, reader.reference_kspace, output.getvalue()
        quiet, quiet_ref, quiet_log = extract(False)
        loud, loud_ref, loud_log = extract(True)
        self.assertEqual(quiet_log, '')
        self.assertEqual(loud_log.count('parallel calibration line:'), 2)
        self.assertIn('acquisition=0, ky=0, z=0, repetition=0', loud_log)
        for a, b in zip(quiet, loud):
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b)
            else:
                self.assertEqual(a, b)
        torch.testing.assert_close(quiet_ref, loud_ref)
        for enabled in (False, True):
            with self.assertRaisesRegex(ValueError, 'Duplicate parallel-calibration'):
                extract(enabled, duplicate=True)
        self.assertFalse(RawDataReader('unused').print_raw_calibration_lines)
        for enabled in (False, True):
            preparer = RawDataPreparer('unused', 'unused', print_raw_calibration_lines=enabled)
            self.assertEqual(preparer.reader.print_raw_calibration_lines, enabled)

    def test_consistency_check_respects_plot_switch(self):
        loader = DataLoader.__new__(DataLoader)
        loader.params = NS(simulated_motion_type='non-rigid-realistic', Nex=1,
            cg_early_stopping=True, cg_max_stag_steps=3, cg_max_more_steps=3,
            cg_use_reg_scale_proxy=False, cg_reg_scale_num_probes=1,
            debug_folder='unused', flip_for_display=False)
        loader._has_simulated_motion = lambda: True
        loader.t_device = 'cpu'
        loader.Nx = loader.Ny = loader.Nz = loader.Ncha = 1
        loader.motion_signal = torch.ones(1)
        loader.kspace = loader.smaps = loader.image_ground_truth = torch.ones(1, 1, 1)
        loader.sampling_idx = torch.zeros(1, dtype=torch.int64)
        simulator = NS(alpha_maps=torch.zeros(1))
        module = 'src.preprocessing.DataLoader.'
        for enabled in (False, True):
            loader.params.save_debug_plots = enabled
            with patch(module + 'MotionOperator'), patch(module + 'EncodingOperator') as encoding, \
                 patch(module + 'ConjugateGradientSolver') as solver, patch(module + 'show_and_save_image') as plot:
                encoding.return_value.adjoint.return_value = torch.ones(1)
                solver.return_value._solve_cg.return_value = torch.ones(1)
                loader._debug_check_true_motion_image_reconstruction(simulator)
                solver.return_value._solve_cg.assert_called_once()
                self.assertEqual(plot.call_count, int(enabled))

if __name__ == '__main__':
    unittest.main(verbosity=2)

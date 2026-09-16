"""Regression checks for file-owned, explicit runtime configuration."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from src.runtime.runtime_config import load_config, load_postprocessing_config, _load_toml_flat
from src.runtime.runtime_setup import initialize_runtime
from src.preprocessing.DataLoader import DataLoader
from src.preprocessing.SamplingSimulator import SamplingSimulator
from src.preprocessing.CoilSensitivityCalculator import CoilSensitivityCalculator

BASE = dict(data_type='siemens-polaris', reconstruction_config='config/reconstruction/nonrigid_2d.toml',
            coil_sensitivity_config='config/coil_sensitivity/odille_spline.toml',
            ismrmrd_reader_config='config/real_data/ismrmrd_reader.toml',
            polaris_config='config/real_data/polaris.toml')
SYNTH = dict(data_type='shepp-logan', reconstruction_config='config/reconstruction/nonrigid_2d.toml',
             coil_sensitivity_config='config/coil_sensitivity/odille_spline.toml',
             shepp_logan_config='config/synthetic_data/shepp_logan_2d.toml',
             sampling_config='config/sampling_simulation/linear.toml',
             motion_simulation_config='config/motion_simulation/nonrigid_2d.toml')

class StrictConfigurationChecks(unittest.TestCase):
    def test_shipped_configs(self):
        for recon in Path('config/reconstruction').glob('*.toml'):
            dim = '3d' if '3d' in recon.stem else '2d'
            for motion in ('rigid', 'nonrigid'):
                for sampling in ('linear', 'interleaved', 'random'):
                    with self.subTest(recon=recon, motion=motion, sampling=sampling):
                        load_config(**{**SYNTH, 'reconstruction_config': recon,
                            'shepp_logan_config': f'config/synthetic_data/shepp_logan_{dim}.toml',
                            'motion_simulation_config': f'config/motion_simulation/{motion}_{dim}.toml',
                            'sampling_config': f'config/sampling_simulation/{sampling}.toml'})

    def test_mode_specific_settings_reject_inactive_values(self):
        with self.assertRaisesRegex(ValueError, 'FoVxy_mm'):
            load_config(**SYNTH, overrides={'FoVxy_mm': 220.0})
        with self.assertRaisesRegex(ValueError, 'motion_quantization_bins'):
            load_config(**SYNTH, overrides={'motion_quantization_bins': 16})
        breast = {**SYNTH, 'reconstruction_config':'config/reconstruction/nonrigid_2d_breast.toml'}
        with self.assertRaisesRegex(ValueError, 'cg_reg_scale_num_probes'):
            load_config(**breast, overrides={'cg_reg_scale_num_probes': 8})

    def test_invalid_values(self):
        cases = [
            {'save_debug_plot':False}, {'unknown':1}, {'debug_flag':False}, {'acs':8}, {'kernel_width':4},
            {'normalize_by_reference':True}, {'normalize_image_by_grics_reference':True},
            {'nonrigid_lateral_sigma':99}, {'N_motion_states':2.9}, {'N_motion_states':True},
            {'N_motion_states':0}, {'lambda_r':-1}, {'lambda_m':float('nan')},
            {'tol_motion':-1}, {'tol_recon':float('inf')}, {'max_iter_recon':0},
            {'max_iter_motion':True}, {'cg_max_stag_steps':-1}, {'cg_reg_scale_num_probes':0},
            {'ResolutionLevels':[-1,0]}, {'ResolutionLevels':[0.5,0.25,1]}, {'ResolutionLevels':[]},
            {'GN_iterations_per_level':[1]}, {'GN_iterations_per_level':[-1,4,4]},
            {'GN_iterations_per_level':3}, {'N_motion_states_per_level':[0,4,4]},
            {'N_motion_states_per_level':[1.1,2,4]}, {'kspace_norm_mode':'typo'},
            {'kspace_norm_eps':0}, {'coil_sensitivity_eps':float('nan')}, {'normalize_kspace':'false'},
            {'verbose':'false'}, {'seed':True}, {'seed_enabled':'false'},
            {'data_dimension':'2'}, {'data_dimension':'2d'}, {'data_dimension':'3D'},
            {'Ncoils_input':4}, {'espirit_kernel_width':100},
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                load_config(**BASE, overrides=values)
        for values in ({'N_SheppLogan':0}, {'Nz_SheppLogan':0}, {'SheppLoganFillFraction':-1},
                       {'nonrigid_motion_amplitude':float('nan')}, {'max_tx':99},
                       {'acceleration_factor':2.5}, {'calibration_lines':129}, {'NshotsPerNex':999},
                       {'N_motion_states': 'per-shot'}):
            with self.subTest(synthetic=values), self.assertRaises(ValueError):
                load_config(**SYNTH, overrides=values)

    def test_real_data_settings_are_loaded_only_for_saec(self):
        saec = {**BASE, 'data_type':'siemens-saec', 'real_data_config':'config/real_data/saec.toml', 'polaris_config':None}
        self.assertEqual(load_config(**saec).rawdata_sensor_type, '1MARMOT')
        with self.assertRaises(ValueError):
            load_config(**{**BASE, 'data_type':'siemens-saec'})
        with self.assertRaises(ValueError):
            load_config(**{**BASE, 'real_data_config':'config/real_data/saec.toml','ismrmrd_reader_config':'config/real_data/ismrmrd_reader.toml','polaris_config':None})
        with self.assertRaises(ValueError):
            load_config(**SYNTH, overrides={'rawdata_sensor_type':'1MARMOT'})

    def test_file_ownership_and_missing_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.toml'
            for text,kind in [('[runtime]\nseed=1\n[other]\nseed=2\n','general'),
                              ('[reconstruction]\nseed=1\n','reconstruction'),
                              ('[postprocessing]\nnormalize_image_by_grics_reference=true\n','reconstruction')]:
                path.write_text(text)
                with self.assertRaises(ValueError): _load_toml_flat(path,kind)
            path.write_text('[reconstruction]\nreconstruction_dimension="2D"\nreconstruction_motion_type="non-rigid"\n')
            with self.assertRaisesRegex(ValueError,'Missing reconstruction'):
                load_config(**{**BASE,'reconstruction_config':path})
            with self.assertRaises(ValueError):
                load_config(**BASE,shepp_logan_config='config/synthetic_data/shepp_logan_2d.toml')

    def test_motion_common_files_are_strict(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'common.toml').write_text('[motion]\nnonrigid_diaphragm_level = 0.2\n')
            (root / 'selected.toml').write_text('[motion]\ninclude = "common.toml"\nsimulated_motion_type = "non-rigid-realistic"\n')
            self.assertEqual(
                _load_toml_flat(root / 'selected.toml', 'motion')['nonrigid_diaphragm_level'],
                0.2,
            )
            (root / 'duplicate.toml').write_text('[motion]\ninclude = "common.toml"\nnonrigid_diaphragm_level = 0.3\n')
            with self.assertRaisesRegex(ValueError, 'duplicates an included'):
                _load_toml_flat(root / 'duplicate.toml', 'motion')
            (root / 'first.toml').write_text('[motion]\ninclude = "second.toml"\n')
            (root / 'second.toml').write_text('[motion]\ninclude = "first.toml"\n')
            with self.assertRaisesRegex(ValueError, 'include cycle'):
                _load_toml_flat(root / 'first.toml', 'motion')
            (root / 'escape.toml').write_text('[motion]\ninclude = "../common.toml"\n')
            with self.assertRaisesRegex(ValueError, 'below its directory'):
                _load_toml_flat(root / 'escape.toml', 'motion')

    def test_precedence_and_pure_load(self):
        import src.runtime.runtime_config as rc
        real_load = rc._load_toml_flat
        def edited_general(path,kind):
            cfg = real_load(path,kind)
            if kind=='general':cfg.update(verbose=False,print_to_console=False)
            return cfg
        dtype = torch.get_default_dtype()
        with patch.object(rc,'_load_toml_flat',side_effect=edited_general):
            p=load_config(**BASE)
            self.assertFalse(p.verbose); self.assertFalse(p.print_to_console)
            p=load_config(**BASE,overrides={'verbose':True,'jupyter_notebook_flag':True})
            self.assertTrue(p.verbose); self.assertFalse(p.print_to_console)
        self.assertEqual(torch.get_default_dtype(),dtype)
        with tempfile.TemporaryDirectory() as folder:
            p=load_config(**BASE,overrides={'debug_folder':folder+'/does-not-exist'})
            self.assertFalse(Path(p.debug_folder).exists())

    def test_notebook_logging_adjustments_are_announced(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            p=load_config(**BASE,overrides={'jupyter_notebook_flag':True})
        self.assertFalse(p.verbose)
        self.assertFalse(p.print_to_console)
        self.assertIn('verbose changed from True to False', output.getvalue())
        self.assertIn('print_to_console changed from True to False', output.getvalue())
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            p=load_config(**BASE,overrides={'jupyter_notebook_flag':True,'verbose':True,'print_to_console':True})
        self.assertTrue(p.verbose)
        self.assertTrue(p.print_to_console)
        self.assertEqual(output.getvalue(),'')

    def test_per_shot_count_replacement_is_announced(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            p=load_config(**{**SYNTH, 'motion_simulation_config':'config/motion_simulation/nonrigid_per_shot_2d.toml'}, overrides={'N_motion_states':3})
        self.assertEqual(p.N_motion_states,8)
        self.assertIn('N_motion_states changed from 3 to 8',output.getvalue())
        p=load_config(**{**BASE, 'motion_simulation_config':'config/motion_simulation/nonrigid_per_shot_2d.toml',
                      'sampling_config':'config/sampling_simulation/random.toml'}, overrides={'N_motion_states':2, 'calibration_lines':4})
        loader=DataLoader(p,t_device='cpu',filename=('scan.mrd','trace.tsv'),run_pipeline=False)
        ky=np.array([[3,1,0,2]])
        data={'kspace':np.ones((1,1,8,4,1),dtype=np.complex128),
              'idx_ky':ky,'idx_kz':np.zeros_like(ky),'idx_nex':np.zeros_like(ky),
              'motion_data':np.array([[0.,1.,2.,3.]])}
        output=io.StringIO()
        with contextlib.redirect_stdout(output):loader._ingest_realworld_arrays(data,slice_idx=0)
        self.assertEqual(p.N_motion_states,4)

    def test_sampling_and_motion_compatibility(self):
        a=load_config(**BASE); b=load_config(**BASE)
        self.assertEqual(vars(a),vars(b)); self.assertFalse(hasattr(a,'Nex'))
        for args in ({**BASE,'sampling_config':'config/sampling_simulation/random.toml'},):
            with self.assertRaises(ValueError):load_config(**args)
        with self.assertRaises(TypeError): load_config(**BASE, Nex=1)
        with self.assertRaises(TypeError): load_config(**SYNTH, kspace_sampling_type='from-data')
        p=load_config(**{**SYNTH, 'motion_simulation_config':'config/motion_simulation/nonrigid_per_shot_2d.toml'}, overrides={'N_motion_states':3})
        self.assertEqual(p.N_motion_states,p.Nshots)
        self.assertEqual(p.N_motion_states, 8)
        with self.assertRaises(ValueError):SamplingSimulator(8,a)._build_phase_encode_indices_and_nex()

    def test_real_resampling_ignores_original_metadata(self):
        cfg={**BASE,'sampling_config':'config/sampling_simulation/random.toml',
             'motion_simulation_config':'config/motion_simulation/nonrigid_2d.toml'}
        values={'save_debug_plots':False,'calibration_lines':4}
        data=np.arange(1*1*8*8*2).reshape(1,1,8,8,2).astype(np.complex128)
        p=load_config(**cfg,overrides=values)
        loader=DataLoader(p,t_device='cpu',filename='unused.mrd',run_pipeline=False)
        loader._ingest_realworld_arrays({'kspace':data,'motion_data':'invalid','idx_ky':'invalid'},slice_idx=0)
        torch.testing.assert_close(loader.kspace,torch.from_numpy(data[...,:1]))
        self.assertEqual(sorted(loader.ky_idx.tolist()),list(range(8)))
        self.assertEqual(len(loader.ky_per_motion_state[0]),4)
        self.assertIsNone(loader._source_motion_data)
        self.assertIsNone(loader._motion_curve_for_binning)
        loader._select_loaded_slice(1)
        torch.testing.assert_close(loader.kspace,torch.from_numpy(data[...,1:2]))
        bad=np.zeros((1,2,8,8,1),dtype=np.complex128)
        with self.assertRaisesRegex(ValueError,'Nex'):
            loader._ingest_realworld_arrays({'kspace':bad},slice_idx=0)
        cfg['reconstruction_config']='config/reconstruction/nonrigid_3d.toml'
        cfg['motion_simulation_config']='config/motion_simulation/nonrigid_3d.toml'
        p=load_config(**cfg,overrides=values)
        loader=DataLoader(p,t_device='cpu',filename='unused.mrd',run_pipeline=False)
        loader._ingest_realworld_arrays({'kspace':data})
        self.assertEqual(loader.ky_idx.numel(),16)
        pairs=set(zip(loader.ky_idx.tolist(),loader.kz_idx.tolist()))
        self.assertEqual(pairs,{(y,z) for y in range(8) for z in range(2)})
        torch.testing.assert_close(loader.kspace,torch.from_numpy(data))

    def test_kspace_only_hdf_and_ismrmrd_reader_routing(self):
        cfg={**BASE, 'data_type':'preprocessed-real', 'ismrmrd_reader_config':None, 'polaris_config':None,
             'sampling_config':'config/sampling_simulation/random.toml',
             'motion_simulation_config':'config/motion_simulation/nonrigid_2d.toml'}
        p=load_config(**cfg,overrides={'save_debug_plots':False,'calibration_lines':4})
        kspace=np.ones((1,1,8,8,1),dtype=np.complex128)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'kspace.h5'
            with h5py.File(path,'w') as handle:handle['kspace']=kspace
            loader=DataLoader(p,t_device='cpu',filename=str(path),slice_idx=0,run_pipeline=False)
            loader.load_data()
            torch.testing.assert_close(loader.kspace,torch.from_numpy(kspace))
        p=load_config(**{**cfg,'data_type':'ismrmrd-polaris','ismrmrd_reader_config':'config/real_data/ismrmrd_reader.toml','polaris_config':'config/real_data/polaris.toml'},overrides={'save_debug_plots':False,'calibration_lines':4})
        loader=DataLoader(p,t_device='cpu',filename='scan.mrd',slice_idx=0,run_pipeline=False)
        with patch('src.preprocessing.DataLoader.ISMRMRDReader') as reader, patch('src.preprocessing.DataLoader.RawDataPreparer') as preparer:
            reader.return_value.read_data.return_value={'kspace':torch.ones(1,1,16,8,1,dtype=torch.complex128),
                'reference_kspace':None,'slice_geometry':{}}
            reader.return_value._remove_oversampling.return_value=torch.from_numpy(kspace)
            loader.load_data()
            preparer.assert_not_called()
            reader.return_value.read_data.assert_called_once()
            torch.testing.assert_close(loader.kspace,torch.from_numpy(kspace))

    def test_from_data_preserves_recorded_order(self):
        p=load_config(**BASE)
        loader=DataLoader(p,t_device='cpu',filename=('scan.mrd','trace.tsv'),run_pipeline=False)
        ky=np.array([[3,1,0,2]])
        values={'kspace':np.ones((1,1,8,4,1),dtype=np.complex128),
                'idx_ky':ky,'idx_kz':np.zeros_like(ky),'idx_nex':np.zeros_like(ky),
                'motion_data':np.array([[0.,1.,2.,3.]])}
        loader._ingest_realworld_arrays(values,slice_idx=0)
        self.assertEqual(loader.ky_idx.tolist(),[3,1,0,2])
        self.assertEqual(loader.params.Nex,1)
        self.assertEqual(loader._motion_curve_for_binning.flatten().tolist(),[0.,1.,2.,3.])

    def test_known_data_bounds(self):
        for overrides in ({'N_motion_states':1000}, {'N_SheppLogan':16,'calibration_lines':17},
                          {'espirit_calibration_width':256}):
            with self.subTest(overrides=overrides),self.assertRaises(ValueError):
                load_config(**{**SYNTH, 'coil_sensitivity_config':'config/coil_sensitivity/espirit.toml'} if 'espirit_calibration_width' in overrides else SYNTH,overrides=overrides)
        args={**SYNTH,'reconstruction_config':'config/reconstruction/nonrigid_3d.toml',
              'shepp_logan_config':'config/synthetic_data/shepp_logan_3d.toml',
              'motion_simulation_config':'config/motion_simulation/nonrigid_3d.toml'}
        with self.assertRaisesRegex(ValueError,'empty 3D shot'):
            load_config(**args,overrides={'NshotsPerNex':129})

    def test_filename_aliases(self):
        p=load_config(**{**BASE,'data_type':'siemens-saec','real_data_config':'config/real_data/saec.toml','ismrmrd_reader_config':'config/real_data/ismrmrd_reader.toml','polaris_config':None})
        for key in ('siemens_file','dat_file'):
            with self.assertRaises(ValueError):
                DataLoader(p,filename={key:'scan.dat','saec_file':'motion'},run_pipeline=False)
        loader=DataLoader(p,filename={'siemens_raw_file':'scan.dat','saec_file':'motion'},run_pipeline=False)
        self.assertEqual(loader.siemens_filenames,('scan.dat','motion'))

    def test_postprocessing_separate(self):
        p=load_postprocessing_config('config/postprocessing/nonrigid_2d_breast.toml')
        self.assertFalse(p.normalize_image_by_grics_reference)
        p=load_postprocessing_config('config/postprocessing/nonrigid_2d_breast.toml',overrides={'normalize_image_by_grics_reference':True})
        self.assertTrue(p.normalize_image_by_grics_reference)
        with self.assertRaises(ValueError):load_postprocessing_config('config/reconstruction/nonrigid_2d_breast.toml')
        with self.assertRaises(ValueError):load_postprocessing_config('config/postprocessing/nonrigid_2d_breast.toml',overrides={'normalize_image_by_grics_reference':'false'})

    def test_gpu_fallback_visible_and_toggle(self):
        with tempfile.TemporaryDirectory() as folder:
            paths={k:folder+'/'+k for k in ('debug_folder','logs_folder','results_folder','initial_data_folder')}
            p=load_config(**BASE,overrides={**paths,'clean_output_folders_before_run':False,'seed_enabled':False})
            out=io.StringIO()
            old_dtype=torch.get_default_dtype()
            old=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark)
            try:
                with patch('src.runtime.runtime_setup._install_runtime_safety_guards'),patch('torch.cuda.is_available',return_value=False),contextlib.redirect_stdout(out):
                    initialize_runtime(p)
                    self.assertEqual(p.runtime_device,'cpu')
                    self.assertTrue(torch.are_deterministic_algorithms_enabled())
                    p.use_deterministic_algorithms=False
                    initialize_runtime(p)
                    self.assertFalse(torch.are_deterministic_algorithms_enabled())
                self.assertIn('Falling back to CPU',out.getvalue())
            finally:
                torch.set_default_dtype(old_dtype)
                torch.use_deterministic_algorithms(old[0],warn_only=old[1])
                torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark=old[2:]

    def test_csm_config_is_method_specific(self):
        p = load_config(**{**SYNTH, 'coil_sensitivity_config':'config/coil_sensitivity/espirit.toml'},
                        overrides={'N_SheppLogan':16, 'calibration_lines':8, 'espirit_calibration_width':8, 'espirit_kernel_width':4})
        self.assertEqual(p.coil_sensitivity_method, 'espirit')
        with self.assertRaises(ValueError):
            load_config(**BASE, overrides={'espirit_kernel_width':4})
        with self.assertRaises(ValueError):
            load_config(**BASE, overrides={'coil_sensitivity_method':'espirit'})
        with self.assertRaises(TypeError):
            load_config(**BASE, coil_sensitivity_method='espirit')

if __name__=='__main__':unittest.main(verbosity=2)

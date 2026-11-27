"""Check native-grid map sharing and release of old resolution data."""
import importlib.util
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reconstruction.JointReconstructor import JointReconstructor
from src.reconstruction.joint_reconstructor_utils.resampling.downsample import downsample_data
from src.reconstruction.joint_reconstructor_utils.resampling.resize import resize_img_xy


def validate_maps(device):
    for nz in (1, 4):
        shape = (8, 6, nz)
        maps = torch.randn(3, *shape, dtype=torch.complex128, device=device)
        kspace = torch.randn(3, 1, *shape, dtype=torch.complex128, device=device)
        indices = [[torch.arange(8 * 6 * nz, device=device)]]
        full = dict(Nx=8, Ny=6, Nz=nz, SensitivityMaps=maps, KspaceData=kspace, SamplingIndices=indices)
        signal = torch.ones(1, 1, device=device)
        params = SimpleNamespace(Nex=1, motion_binning_mode='uniform')
        before = maps.clone()
        with patch('src.reconstruction.joint_reconstructor_utils.resampling.downsample.resize_img_xy', side_effect=AssertionError('Native-grid maps must not be interpolated')):
            result = downsample_data(full, 1., 1, signal, params, device)
        assert result['SensitivityMaps'] is maps
        torch.testing.assert_close(result['KspaceData'], kspace.reshape(3, 1, -1))
        coarse = downsample_data(full, .5, 1, signal, params, device)
        target = (4, 3, 2) if nz > 1 else (4, 3)
        torch.testing.assert_close(coarse['SensitivityMaps'], resize_img_xy(maps, target))
        assert coarse['SensitivityMaps'].data_ptr() != maps.data_ptr()
        torch.testing.assert_close(maps, before)
    print(f'PASS: {device}: native maps share storage, coarse maps retain interpolation, inputs unchanged.')


def validate_transition():
    spec = importlib.util.spec_from_file_location('mosar_memory_fixture', ROOT / 'validation_studies/navigator_motion_draft/validate_mosar_pipeline.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    prepare = JointReconstructor._prepare_resolution_level
    run_pipeline = fixture.run_pipeline
    references, checked = {}, []
    def prepare_checked(self, index, factor):
        if index:
            assert references['maps']() is None, 'Coarse maps still retained while preparing next level'
            assert references['kspace']() is None, 'Coarse k-space still retained while preparing next level'
            checked.append(index)
        result = prepare(self, index, factor)
        if factor == 1.:
            assert result['SensitivityMaps'] is self.Data_full['SensitivityMaps']
        else:
            references['maps'] = weakref.ref(result['SensitivityMaps'])
            references['kspace'] = weakref.ref(result['KspaceData'])
        return result
    def pipeline_levels(*args, **kwargs):
        overrides = dict(kwargs.get('overrides') or {})
        overrides.update(ResolutionLevels=[.5, 1.], GN_iterations_per_level=[1, 1])
        kwargs['overrides'] = overrides
        return run_pipeline(*args, **kwargs)
    with patch.object(JointReconstructor, '_prepare_resolution_level', prepare_checked), patch.object(fixture, 'run_pipeline', pipeline_levels):
        fixture.run_validation()
    assert checked == [1, 1], checked
    print('PASS: both MOSAR contrasts cross coarse-to-native levels; previous map/k-space objects released before next allocation; reconstruction and DICOM checks pass.')


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.manual_seed(23)
    validate_maps('cpu')
    if torch.cuda.is_available():
        validate_maps('cuda')
    else:
        print('CUDA checks skipped explicitly: GPU unavailable.')
    validate_transition()

"""Validate CPU-stored k-space with GPU encoding, Jacobian, and GN solves."""
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.reconstruction.JointReconstructor import JointReconstructor
from src.reconstruction.coil_batching import CoilBatchPlanner
from src.reconstruction.joint_reconstructor_utils.operators import RealNormalOperator
from src.runtime.runtime_config import load_config

spec = importlib.util.spec_from_file_location('coil_cases', ROOT / 'validation_studies/navigator_motion_draft/validate_coil_batching.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def validate_operators():
    cases = 0
    for nz in (1, 4):
        for dtype in (torch.complex64, torch.complex128):
            for batching in ('0', '1'):
                for model in ('rigid', 'non-rigid'):
                    with patch.dict(os.environ, {'GRICS_STATE_BATCH': batching}):
                        maps, image, alpha, motion, samples = fixture.inputs(nz, dtype, 'gather', model, 'cuda')
                        nv = image.numel()
                        # Incomplete lines exercise E's composed normal path too.
                        samples[0][0] = samples[0][0][1:]
                        measured = torch.randn(maps.shape[0]*2*nv, device='cuda', dtype=torch.complex128)
                        delta = torch.randn(alpha.shape, device='cuda', dtype=torch.complex128)
                        e = EncodingOperator(maps, nv, samples, 2, motion)
                        j = MotionPerturbationSimulator(maps, nv, samples, 2, image, motion)
                        tol = 1e-4 if dtype == torch.complex64 else 1e-10
                        expected_residual = measured-e.forward(image)
                        expected_normal = j.normal(delta)
                        expected_image_normal = e.normal(image)
                        for operator, vector in ((e, image), (j, delta)):
                            expected = operator.forward(vector)
                            actual = operator.forward(vector, output_device='cpu')
                            torch.testing.assert_close(actual, expected.cpu(), atol=tol, rtol=tol)
                            torch.testing.assert_close(operator.adjoint(actual), operator.adjoint(expected), atol=tol, rtol=tol)
                        with patch.object(CoilBatchPlanner, 'kspace_output_device', return_value=torch.device('cpu')):
                            original = e.forward
                            buffers = []
                            def capture(*args, **kwargs):
                                out = original(*args, **kwargs)
                                buffers.append(out)
                                return out
                            with patch.object(e, 'forward', side_effect=capture):
                                residual = e.residual(image, measured)
                            assert residual.device.type == 'cpu'
                            assert residual.data_ptr() == buffers[0].data_ptr(), 'Prediction buffer must become the residual in place'
                            torch.testing.assert_close(residual, expected_residual.cpu(), atol=tol, rtol=tol)
                            torch.testing.assert_close(j.normal(delta), expected_normal, atol=tol, rtol=tol)
                            torch.testing.assert_close(RealNormalOperator(j).normal(delta.real), j.normal(delta.real).real, atol=tol, rtol=tol)
                            torch.testing.assert_close(e.normal(image), expected_image_normal, atol=tol, rtol=tol)
                        cases += 1
    bank = e.smaps.shape[0]*e.Nex*e.Nsamples*16
    with patch.object(e._coil_batches, '_memory_budget', return_value=4*bank):
        assert e._coil_batches.kspace_output_device(e.Nex,e.Nsamples).type == 'cuda'
    with patch.object(e._coil_batches, '_memory_budget', return_value=bank):
        assert e._coil_batches.kspace_output_device(e.Nex,e.Nsamples).type == 'cpu'
    print(f'PASS: {cases} CUDA cases: CPU/GPU k-space outputs and adjoints, residual buffer reuse, E/J normal paths and real wrapper, both precisions, dimensions, motion models, and state batching.')


def validate_gn():
    maps, image, alpha, motion, samples = fixture.inputs(4, torch.complex64, 'gather', 'non-rigid', 'cuda')
    measured = torch.randn(maps.shape[0], 2, *image.shape, dtype=torch.complex128, device='cuda')
    params = load_config(data_type='ismrmrd-navigators', reconstruction_config=ROOT/'config/reconstruction/nonrigid_3d.toml',
                         overrides={'ResolutionLevels':[1.], 'GN_iterations_per_level':[1], 'N_motion_states':4,
                                    'compute_precision':'single', 'warp_backend':'gather', 'verbose':False})
    params.Nex = 2
    params.cg_use_reg_scale_proxy = False
    recon = JointReconstructor(measured, maps, samples, motion.motion_signal, params)
    base = dict(Nx=8, Ny=6, Nz=4, SensitivityMaps=maps, KspaceData=measured.reshape(maps.shape[0],2,-1), Nsamples=image.numel(),
                SamplingIndices=samples, MotionSignal=motion.motion_signal)
    results = []
    original_zeros = torch.zeros
    for host in (False, True):
        data = dict(base, ReconstructedImage=image.clone(), MotionModel=alpha.clone())
        device = torch.device('cpu' if host else 'cuda')
        def checked_zeros(*args, **kwargs):
            shape = args[0] if args else kwargs.get('size')
            if host and isinstance(shape, (tuple, list)) and tuple(shape) == (maps.shape[0],2,image.numel()):
                assert torch.device(kwargs.get('device','cpu')).type == 'cpu', 'Full prediction/Jacobian k-space must not be allocated on CUDA'
            return original_zeros(*args, **kwargs)
        with patch.object(CoilBatchPlanner, 'kspace_output_device', return_value=device), patch('torch.zeros', side_effect=checked_zeros):
            result = recon.gauss_newton_iteration(data, image_cg_iterations=2, motion_cg_iterations=2)
        assert result.image.device.type == result.motion.device.type == 'cuda'
        assert result.residual.device.type == device.type
        results.append(result)
    torch.testing.assert_close(results[0].image, results[1].image, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(results[0].motion, results[1].motion, atol=1e-5, rtol=1e-5)
    print('PASS: actual GPU GN image and motion solves match with CPU-stored residual/Jacobian buffers; parameter updates stay on CUDA.')


def validate_peak():
    maps, image, _, motion, samples = fixture.inputs(32, torch.complex64, 'gather', 'non-rigid', 'cuda', shape=(128,96), coils=32)
    with patch.dict(os.environ, {'GRICS_STATE_BATCH':'0'}):
        operator = EncodingOperator(maps,image.numel(),samples,2,motion)
    unit = image.numel()*8*CoilBatchPlanner.workspace_factor
    report, reference = {}, None
    for device in ('cuda','cpu'):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with patch.object(operator._coil_batches,'_memory_budget',return_value=4*unit):
            output = operator.forward(image,output_device=device)
        torch.cuda.synchronize()
        report[device] = (torch.cuda.max_memory_allocated()-baseline)/2**20
        if reference is None:
            reference = output.cpu()
        else:
            torch.testing.assert_close(output,reference,atol=1e-5,rtol=1e-5)
        del output
    bank = 32*2*image.numel()*16/2**20
    assert report['cpu'] < report['cuda'] - .75*bank
    print(f'KSPACE_GPU_PEAK_MiB {report}; full buffer {bank:.1f} MiB; four-coil groups, actual GPU allocations, with inputs resident.')


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(53)
    if torch.cuda.is_available():
        validate_operators()
        validate_gn()
        validate_peak()
    else:
        print('CUDA checks skipped explicitly: GPU unavailable.')

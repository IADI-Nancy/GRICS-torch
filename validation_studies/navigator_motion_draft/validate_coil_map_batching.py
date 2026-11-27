"""Check bounded coil-map conversion and CG under simulated workspace pressure."""
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reconstruction.coil_batching import CoilBatchPlanner
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.joint_reconstructor_utils.calibration_prior import CalibrationPriorEncodingOperator

spec = importlib.util.spec_from_file_location('coil_inputs', ROOT / 'validation_studies/navigator_motion_draft/validate_coil_batching.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def solver(operator):
    return ConjugateGradientSolver(operator, reg_lambda=.01, regularizer='Tikhonov', regularization_shape=None,
                                  regularization_spatial_dims=None, verbose=False, stop_on_stagnation=False,
                                  true_residual_interval=1, stagnation_consecutive_steps=6, stagnation_countdown_steps=0,
                                  use_reg_scale_proxy=False, reg_scale_num_probes=None)


def validate_precision(device):
    for nz in (1, 4):
        for state_batching in ('0', '1'):
            with patch.dict(os.environ, {'GRICS_STATE_BATCH': state_batching}):
                maps, image, alpha, motion, samples = fixtures.inputs(nz, torch.complex64, 'gather', 'non-rigid', device)
                converted = maps.to(torch.complex64)
                residual = torch.randn(maps.shape[0]*2*image.numel(), device=device, dtype=torch.complex128)
                delta = torch.randn(alpha.shape, device=device, dtype=torch.complex128)
                for cls, args, ref_args, vector in ((EncodingOperator, (maps, image.numel(), samples, 2), (converted, image.numel(), samples, 2), image),
                                                   (MotionPerturbationSimulator, (maps, image.numel(), samples, 2, image), (converted, image.numel(), samples, 2, image), delta)):
                    actual, reference = cls(*args, motion), cls(*ref_args, motion)
                    assert actual._smaps_b.data_ptr() == maps.data_ptr()
                    assert actual._smaps_b.dtype == torch.complex128
                    assert actual._coil_batches.element_size == 8
                    unit = image.numel()*8*CoilBatchPlanner.workspace_factor
                    with patch.object(actual._coil_batches, '_memory_budget', return_value=2*unit), patch.object(reference._coil_batches, '_memory_budget', return_value=1000*unit):
                        torch.testing.assert_close(actual.forward(vector), reference.forward(vector), atol=5e-4, rtol=5e-4)
                        torch.testing.assert_close(actual.adjoint(residual), reference.adjoint(residual), atol=5e-4, rtol=5e-4)
                        torch.testing.assert_close(actual.normal(vector), reference.normal(vector), atol=1e-3, rtol=1e-3)
                    sliced = actual._coil_batches.coil_maps(1, 3, conjugate=True)
                    torch.testing.assert_close(sliced, converted[1:3].squeeze(-1).conj() if nz == 1 else converted[1:3].conj())
                    assert sliced.numel() == 2*image.numel()
    print(f'PASS: {device}: bounded map conversion shares source storage and matches eager precision conversion in E/J forward, adjoint, and normal; both state paths and dimensions.')


def validate_cg_pressure():
    maps = torch.randn(64, 64, 48, 16, dtype=torch.complex128, device='cuda')
    image = torch.randn(64, 48, 16, dtype=torch.complex128, device='cuda')
    alpha = torch.zeros(3, 64, 48, 16, 1, device='cuda', dtype=torch.float64)
    motion = MotionOperator(64, 48, alpha, 'non-rigid', Nz=16, motion_signal=torch.ones(1, 1, device='cuda'), compute_dtype=torch.complex64, warp_backend='gather')
    samples = [[torch.arange(image.numel(), device='cuda')]]
    with patch.dict(os.environ, {'GRICS_STATE_BATCH': '0'}):
        before = torch.cuda.memory_allocated()
        operator = EncodingOperator(maps, image.numel(), samples, 1, motion)
        constructor_bytes = torch.cuda.memory_allocated()-before
        bank_bytes = maps.numel()*8
        assert constructor_bytes < bank_bytes/4, 'Constructor must not retain an all-coil precision copy'
        weighted = CalibrationPriorEncodingOperator(operator, .5 + torch.rand_like(image.real))
        rhs = weighted.normal(image.flatten()) + .01*image.flatten()
        expected = solver(weighted).cg(rhs, max_iter=4, tol=1e-12).cpu()
        baseline = torch.cuda.memory_allocated()
        budget = int(bank_bytes*.9)
        per_coil = image.numel()*8*CoilBatchPlanner.workspace_factor
        def available():
            return min(int(per_coil*1.1), max(0, budget-(torch.cuda.memory_allocated()-baseline)))
        # Recreate the old full-bank residency: the same live budget then fails.
        eager = maps.to(torch.complex64)
        with patch.object(operator._coil_batches, '_memory_budget', side_effect=available):
            try:
                weighted.normal(image.flatten())
            except MemoryError as error:
                assert 'one FFT coil' in str(error)
            else:
                raise AssertionError('Eager conversion must reproduce workspace exhaustion')
        del eager
        with patch.object(operator._coil_batches, '_memory_budget', side_effect=available):
            actual = solver(weighted).cg(rhs, max_iter=4, tol=1e-12).cpu()
        assert operator._coil_batches._batch_size == 1
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        print(f'PASS: GPU calibration-weighted image CG finishes under simulated memory pressure that fails with an eager full map bank; {bank_bytes/2**20:.2f} MiB conversion bank avoided, {constructor_bytes/2**20:.3f} MiB constructor residency; planner headroom unchanged.')


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(47)
    validate_precision('cpu')
    if torch.cuda.is_available():
        validate_precision('cuda')
        validate_cg_pressure()
    else:
        print('CUDA checks skipped explicitly: GPU unavailable.')

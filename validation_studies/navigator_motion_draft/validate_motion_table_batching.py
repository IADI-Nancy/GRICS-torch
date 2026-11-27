"""Compare GPU-resident and CPU-stored gather tables, including state chunks."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator

spec = importlib.util.spec_from_file_location('coil_fixtures', ROOT / 'validation_studies/navigator_motion_draft/validate_coil_batching.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def limited_budget():
    return patch.multiple(torch.cuda, mem_get_info=lambda device: (0, 40 * 2**30), memory_reserved=lambda device: 0, memory_allocated=lambda device: 0)


def validate():
    cases = 0
    for nz in (1, 4):
        for dtype in (torch.complex64, torch.complex128):
            for model in ('rigid', 'non-rigid'):
                smaps, image, alpha, resident, samples = fixtures.inputs(nz, dtype, 'gather', model, 'cuda')
                with limited_budget():
                    streamed = MotionOperator(8, 6, alpha, model, Nz=nz, motion_signal=resident.motion_signal, centers=resident.centers, compute_dtype=dtype, warp_backend='gather')
                assert resident._table_device.type == 'cuda'
                assert streamed._table_device.type == 'cpu'
                assert all(idx.device.type == w.device.type == 'cpu' for idx, w in streamed.gatherMotionOperator)
                assert streamed.state_chunks(image.numel(), budget_bytes=2**40) == [(i, i+1) for i in range(4)]
                tol = 2e-5 if dtype == torch.complex64 else 1e-11
                vector = image.flatten().to(dtype)
                residual = torch.randn(4, vector.numel(), dtype=dtype, device='cuda')
                for states in (None, (1, 3)):
                    first, last = (0, 4) if states is None else states
                    torch.testing.assert_close(streamed.warp_all(vector, states), resident.warp_all(vector, states), atol=tol, rtol=tol)
                    torch.testing.assert_close(streamed.warp_T_all(residual[first:last], states), resident.warp_T_all(residual[first:last], states), atol=tol, rtol=tol)
                for state in range(4):
                    torch.testing.assert_close(streamed.warp(vector, state), resident.warp(vector, state), atol=tol, rtol=tol)
                    torch.testing.assert_close(streamed.warp_T(vector, state), resident.warp_T(vector, state), atol=tol, rtol=tol)
                    assert streamed._gather_state_cache[0] == state
                assert '_gather_tables_cache' not in streamed.__dict__
                for batched in ('0', '1'):
                    with patch.dict('os.environ', {'GRICS_STATE_BATCH': batched}):
                        nv = image.numel()
                        y = torch.randn(smaps.shape[0] * 2 * nv, device='cuda', dtype=torch.complex128)
                        delta = torch.randn(alpha.shape, device='cuda', dtype=torch.complex128)
                        for cls, args, value in ((EncodingOperator, (smaps, nv, samples, 2), image),
                                                 (MotionPerturbationSimulator, (smaps, nv, samples, 2, image), delta)):
                            full, grouped = cls(*args, resident), cls(*args, streamed)
                            torch.testing.assert_close(grouped.forward(value), full.forward(value), atol=tol*10, rtol=tol*10)
                            torch.testing.assert_close(grouped.adjoint(y), full.adjoint(y), atol=tol*10, rtol=tol*10)
                            torch.testing.assert_close(grouped.normal(value), full.normal(value), atol=tol*20, rtol=tol*20)
                        cases += 1
    print(f'PASS: {cases} CUDA cases: resident/CPU-stored tables, both models, dimensions, precisions, state batching, forward/adjoint/normal, and selected state chunks.')


def peak_memory():
    alpha = torch.randn(3, 64, 48, 32, 2, device='cuda', dtype=torch.float64) * .02
    signal = torch.randn(12, 2, device='cuda', dtype=torch.float64)
    report = {}
    for name in ('resident', 'streamed'):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        if name == 'streamed':
            with limited_budget():
                motion = MotionOperator(64, 48, alpha, 'non-rigid', Nz=32, motion_signal=signal, compute_dtype=torch.complex64, warp_backend='gather')
        else:
            motion = MotionOperator(64, 48, alpha, 'non-rigid', Nz=32, motion_signal=signal, compute_dtype=torch.complex64, warp_backend='gather')
        image = torch.ones(alpha.shape[1:-1], device='cuda', dtype=torch.complex64).flatten()
        for state in range(12):
            motion.warp(image, state)
        torch.cuda.synchronize()
        report[name] = (torch.cuda.max_memory_allocated() - baseline) / 2**20
        del motion, image
    assert report['streamed'] < report['resident'] / 2
    print(f'GPU_PEAK_MiB {report}; includes construction and 12 warps; streamed budget is simulated, allocations measured.')


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(29)
    if torch.cuda.is_available():
        validate()
        peak_memory()
    else:
        print('CUDA checks skipped explicitly: GPU unavailable.')

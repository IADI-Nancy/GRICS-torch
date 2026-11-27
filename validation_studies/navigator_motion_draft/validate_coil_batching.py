"""Validate automatic coil batching, operator adjoints, and GPU peak memory.

Run from the repository root: python validation_studies/navigator_motion_draft/validate_coil_batching.py
GPU memory pressure is simulated through the planner's workspace budget; no
large filler allocation is used. Actual GPU peak allocations are measured.
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.reconstruction.coil_batching import CoilBatchPlanner
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator


def inputs(nz, dtype, backend, model, device, *, shape=(8, 6), coils=5):
    nx, ny = shape
    spatial = (nx, ny, nz) if nz > 1 else (nx, ny)
    smaps = torch.randn(coils, nx, ny, nz, dtype=torch.complex128, device=device)
    image = torch.randn(spatial, dtype=torch.complex128, device=device)
    signal = torch.randn(4, 2, dtype=torch.float64, device=device)
    alpha_shape = (3 if nz > 1 else 2, *spatial, 2) if model == 'non-rigid' else (6 if nz > 1 else 3, 4)
    alpha = torch.randn(alpha_shape, dtype=torch.float64, device=device) * .01
    motion = MotionOperator(nx, ny, alpha, model, Nz=nz, motion_signal=signal, compute_dtype=dtype, warp_backend=backend)
    phases = torch.arange(ny * nz, device=device)
    samples = [[(torch.arange(nx, device=device)[:, None] * (ny * nz) + phases[(phases + rep) % 3 == state][None, :]).flatten()
                if state < 3 else torch.empty(0, dtype=torch.long, device=device) for state in range(4)] for rep in range(2)]
    return smaps, image, alpha, motion, samples


def validate_policy():
    planner = CoilBatchPlanner(torch.zeros(5, 8, 6, 4), torch.complex64, 'policy test')
    with patch('torch.cuda.mem_get_info', side_effect=AssertionError('CPU must not query CUDA')):
        assert planner.batches() == [(0, 5)]
    planner.device = torch.device('cuda:0')
    with patch('torch.cuda.mem_get_info', return_value=(8 * 2**30, 40 * 2**30)), patch('torch.cuda.memory_reserved', return_value=4 * 2**30), patch('torch.cuda.memory_allocated', return_value=2**30):
        assert planner._memory_budget() == 5 * 2**30
    unit = planner.n_voxels * planner.element_size * planner.workspace_factor
    with patch.object(planner, '_memory_budget', return_value=2 * unit):
        assert planner.batches() == [(0, 2), (2, 4), (4, 5)]
        assert planner.state_budget(8 * unit) == 2 * unit // 8
    with patch.object(planner, '_memory_budget', return_value=unit):
        assert planner.batches() == [(i, i + 1) for i in range(5)]
    with patch.object(planner, '_memory_budget', return_value=100 * unit):
        assert planner.batches() == [(i, i + 1) for i in range(5)]  # Stable cap during one solve.
    with patch.object(planner, '_memory_budget', return_value=unit - 1):
        try:
            planner.batches()
        except MemoryError as exc:
            assert 'one FFT coil' in str(exc)
        else:
            raise AssertionError('Insufficient single-coil workspace must fail explicitly')


def validate_operators(device):
    cases = 0
    for nz in (1, 4):
        for dtype in (torch.complex128, torch.complex64):
            for backend in ('gather', 'sparse'):
                for state_batching in ('0', '1'):
                    for model in ('rigid', 'non-rigid'):
                        with patch.dict(os.environ, {'GRICS_STATE_BATCH': state_batching}):
                            smaps, image, alpha, motion, samples = inputs(nz, dtype, backend, model, device)
                            nv = image.numel()
                            for incomplete in (False, True):
                                indices = [[group.clone() for group in rep] for rep in samples]
                                if incomplete:
                                    indices[0][0] = indices[0][0][1:]
                                e_full = EncodingOperator(smaps, nv, indices, 2, motion)
                                e_chunk = EncodingOperator(smaps, nv, indices, 2, motion)
                                j_full = MotionPerturbationSimulator(smaps, nv, indices, 2, image, motion)
                                j_chunk = MotionPerturbationSimulator(smaps, nv, indices, 2, image, motion)
                                residual = torch.randn(smaps.shape[0] * 2 * nv, dtype=torch.complex128, device=device)
                                perturbation = torch.randn(alpha.shape, dtype=torch.complex128, device=device)
                                tol = 5e-5 if dtype == torch.complex64 else 1e-10
                                for full, chunk, vector in ((e_full, e_chunk, image), (j_full, j_chunk, perturbation)):
                                    unit = nv * torch.empty((), dtype=dtype).element_size() * CoilBatchPlanner.workspace_factor
                                    # Force an uneven last coil group, including the state-batched paths.
                                    budget = 2 * unit
                                    with patch.object(full._coil_batches, '_memory_budget', return_value=1000 * unit):
                                        expected = [full.forward(vector), full.adjoint(residual), full.normal(vector)]
                                    with patch.object(chunk._coil_batches, '_memory_budget', return_value=budget):
                                        actual = [chunk.forward(vector), chunk.adjoint(residual), chunk.normal(vector)]
                                    for a, b in zip(actual, expected):
                                        torch.testing.assert_close(a, b, atol=tol * 10, rtol=tol * 10)
                                    lhs = torch.vdot(actual[0], residual)
                                    rhs = torch.vdot(vector.flatten(), actual[1].flatten())
                                    assert abs(lhs - rhs) <= tol * (1 + abs(lhs) + abs(rhs)), (device, nz, dtype, backend, state_batching, model, incomplete, type(chunk).__name__, lhs, rhs)
                                    torch.testing.assert_close(actual[2].flatten(), chunk.adjoint(chunk.forward(vector)).flatten(), atol=tol * 20, rtol=tol * 20)
                                cases += 1
    print(f'PASS: {cases} {device} cases: full/chunked E and J, normal operators, adjoint identities, 2D/3D, rigid/non-rigid, single/double, sparse/gather, state batching, empty states, incomplete lines.')


def measure_gpu_peak():
    with patch.dict(os.environ, {'GRICS_STATE_BATCH': '0'}):
        smaps, image, _, motion, samples = inputs(32, torch.complex64, 'gather', 'non-rigid', 'cuda', shape=(128, 96), coils=32)
        operator = EncodingOperator(smaps, image.numel(), samples, 2, motion)
        unit = image.numel() * 8 * CoilBatchPlanner.workspace_factor
        report, reference = {}, None
        for name, budget in (('all_coils', None), ('limited_workspace', 4 * unit)):
            operator._coil_batches._batch_size = 32
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            context = patch.object(operator._coil_batches, '_memory_budget', return_value=budget) if budget is not None else patch.dict(os.environ, {})
            with context:
                result = operator.normal(image).cpu()
            torch.cuda.synchronize()
            report[name] = {'coil_batch_size': operator._coil_batches._batch_size,
                            'peak_increment_MiB': (torch.cuda.max_memory_allocated() - baseline) / 2**20}
            if reference is None:
                reference = result
            else:
                torch.testing.assert_close(result, reference, atol=2e-4, rtol=2e-4)
        assert report['all_coils']['coil_batch_size'] == 32
        assert report['limited_workspace']['coil_batch_size'] == 4
        assert report['limited_workspace']['peak_increment_MiB'] < report['all_coils']['peak_increment_MiB'] * .8
        print('GPU_PEAK_REPORT ' + json.dumps(report))


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)  # Same geometry precision as initialize_runtime().
    torch.manual_seed(43)
    validate_policy()
    validate_operators('cpu')
    if torch.cuda.is_available():
        validate_operators('cuda')
        measure_gpu_peak()
    else:
        print('GPU checks skipped explicitly: CUDA is unavailable.')

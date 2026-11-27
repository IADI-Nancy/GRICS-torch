"""Compare bounded warp workspace with direct full-volume gather/scatter formulas."""
import sys
from pathlib import Path
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reconstruction.MotionOperator import MotionOperator


def validate(device):
    cases = 0
    for nz in (1, 4):
        shape = (8, 6, nz) if nz > 1 else (8, 6)
        for dtype in (torch.complex64, torch.complex128):
            alpha = torch.randn(len(shape), *shape, 2, dtype=torch.float64, device=device) * .1
            motion = MotionOperator(8, 6, alpha, 'non-rigid', Nz=nz, motion_signal=torch.randn(3, 2, dtype=torch.float64, device=device), compute_dtype=dtype, warp_backend='gather')
            x = torch.randn(8*6*nz, dtype=dtype, device=device)
            y = torch.randn(3, x.numel(), dtype=dtype, device=device)
            tol = 2e-5 if dtype == torch.complex64 else 1e-12
            with patch.object(motion, 'gather_workspace_bytes', 1024):
                for state, (idx, w) in enumerate(motion.gatherMotionOperator):
                    torch.testing.assert_close(motion.warp(x, state), (w * x[idx]).sum(0), rtol=tol, atol=tol)
                    expected = torch.zeros_like(x).index_add_(0, idx.flatten(), (w*x[None]).flatten())
                    torch.testing.assert_close(motion.warp_T(x, state), expected, rtol=tol, atol=tol)
                    lhs = torch.vdot(motion.warp(x, state), y[state])
                    rhs = torch.vdot(x, motion.warp_T(y[state], state))
                    torch.testing.assert_close(lhs, rhs, rtol=tol, atol=tol)
                for states in (None, (1, 3)):
                    start, stop = (0, 3) if states is None else states
                    idx = torch.stack([item[0] for item in motion.gatherMotionOperator[start:stop]])
                    w = torch.stack([item[1] for item in motion.gatherMotionOperator[start:stop]])
                    expected = (w*x[idx]).sum(1)
                    torch.testing.assert_close(motion.warp_all(x, states), expected, rtol=tol, atol=tol)
                    expected = torch.zeros_like(x).index_add_(0, idx.flatten(), (w*y[start:stop, None]).flatten())
                    torch.testing.assert_close(motion.warp_T_all(y[start:stop], states), expected, rtol=tol, atol=tol)
            cases += 1
    print(f'PASS: {cases} {device} cases: forced voxel chunks, forward/adjoint identities, single/batched and selected states, both dimensions and precisions.')


def peak_memory():
    alpha = torch.zeros(3, 64, 64, 128, 1, dtype=torch.float64, device='cuda')
    motion = MotionOperator(64, 64, alpha, 'non-rigid', Nz=128, motion_signal=torch.ones(1, 1, dtype=torch.float64, device='cuda'), warp_backend='gather')
    idx, w = motion.gatherMotionOperator[0]
    x = torch.randn(alpha[0].numel(), dtype=torch.complex128, device='cuda')
    report = {}
    for adjoint in (False, True):
        reference = None
        for bounded in (False, True):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            if bounded:
                out = motion.warp_T(x, 0) if adjoint else motion.warp(x, 0)
            elif adjoint:
                out = torch.zeros_like(x).index_add_(0, idx.flatten(), (w*x[None]).flatten())
            else:
                out = (w*x[idx]).sum(0)
            torch.cuda.synchronize()
            key = ('adjoint' if adjoint else 'forward') + ('_bounded' if bounded else '_full')
            report[key] = (torch.cuda.max_memory_allocated() - baseline) / 2**20
            if reference is None:
                reference = out.cpu()
            else:
                torch.testing.assert_close(out.cpu(), reference, rtol=1e-11, atol=1e-11)
            del out
    assert report['adjoint_bounded'] < report['adjoint_full'] / 2
    assert report['forward_bounded'] < report['forward_full'] / 2
    print(f'WARP_GPU_PEAK_MiB {report}; additional allocation with inputs and tables already resident.')


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.manual_seed(31)
    validate('cpu')
    if torch.cuda.is_available():
        validate('cuda')
        peak_memory()
    else:
        print('CUDA checks skipped explicitly: GPU unavailable.')

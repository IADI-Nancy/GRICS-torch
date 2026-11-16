from __future__ import annotations

import atexit
import gc
import signal
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import sigpy as sp


_GUARDS_INSTALLED = False
_GPU_RUNTIME_ACTIVE = False


@dataclass(frozen=True)
class Runtime:
    """Resolved Torch device and optional ESPIRiT SigPy device for one workflow."""

    sp_device: sp.Device | None
    t_device: torch.device


def cleanup_runtime():
    """Best-effort cleanup without initializing GPU libraries for CPU runs."""
    gc.collect()
    if not _GPU_RUNTIME_ACTIVE:
        return

    cp = sys.modules.get("cupy")
    if cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    # is_initialized() is a local state check; unlike is_available(), it does
    # not initialize the CUDA driver during cleanup.
    if torch.cuda.is_initialized():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _signal_cleanup_handler(signum, frame):
    cleanup_runtime()
    raise SystemExit(128 + signum)


def _install_runtime_safety_guards():
    global _GUARDS_INSTALLED
    if _GUARDS_INSTALLED:
        return

    atexit.register(cleanup_runtime)
    # In notebook/ipykernel, custom SIGINT handlers can kill the kernel.
    # Keep only atexit cleanup there, and install signal handlers for scripts.
    if "ipykernel" not in sys.modules:
        signal.signal(signal.SIGINT, _signal_cleanup_handler)
        signal.signal(signal.SIGTERM, _signal_cleanup_handler)
    _GUARDS_INSTALLED = True


def initialize_runtime(params, print_gpu_info=False):
    """Initialize run state and return the resolved SigPy and Torch devices in Runtime."""
    global _GPU_RUNTIME_ACTIVE
    _install_runtime_safety_guards()
    from src.runtime.output_layout import RunOutputs
    if not hasattr(params, '_run_outputs') or params._run_outputs.closed:
        RunOutputs(params)

    runtime_device = str(params.runtime_device).lower()
    if runtime_device not in {"cpu", "gpu"}:
        raise ValueError("runtime_device must be 'cpu' or 'gpu'.")

    # CPU runs must not probe CUDA: is_available() initializes the NVIDIA
    # driver and can deadlock when many CPU reconstruction workers start.
    torch_cuda_ok = False
    if runtime_device == "gpu":
        torch_cuda_ok = torch.cuda.is_available()
        if not torch_cuda_ok:
            print("[runtime] GPU requested but PyTorch CUDA is unavailable. Falling back to CPU.", flush=True)
            runtime_device = "cpu"

    use_gpu = runtime_device == "gpu"
    _GPU_RUNTIME_ACTIVE = _GPU_RUNTIME_ACTIVE or use_gpu
    sp_device = None
    if params.coil_sensitivity_method == "espirit":
        from src.runtime.sigpy_backend import resolve_espirit_device

        sp_device = resolve_espirit_device(use_gpu)

    t_device = torch.device("cuda:0" if use_gpu else "cpu")
    params.runtime_device = runtime_device

    if use_gpu and torch_cuda_ok:
        torch.cuda.empty_cache()
        if print_gpu_info:
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Total GPU memory: {total_mem:.2f} GB")

    torch.set_default_dtype(torch.float64)
    if params.seed_enabled:
        torch.manual_seed(params.seed)
        if use_gpu and torch_cuda_ok:
            torch.cuda.manual_seed(params.seed)
            torch.cuda.manual_seed_all(params.seed)

    # Set both states explicitly so repeated notebook runs honor flag changes.
    deterministic = params.use_deterministic_algorithms
    torch.use_deterministic_algorithms(deterministic, warn_only=deterministic)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False

    params._run_outputs.snapshot(params)
    return Runtime(sp_device=sp_device, t_device=t_device)

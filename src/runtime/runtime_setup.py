import atexit
import gc
import shutil
import signal
import sys
import warnings
from pathlib import Path

import torch
import sigpy as sp


_GUARDS_INSTALLED = False


def cleanup_runtime():
    """Best-effort cleanup to release Python/CUDA memory between runs."""
    gc.collect()

    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    if torch.cuda.is_available():
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


def _clean_folder_contents(folder):
    path = Path(folder).expanduser().resolve()
    if str(path) in {"/", ""}:
        raise ValueError(f"Refusing to clean unsafe folder path: {folder}")

    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _clean_run_output_folders(params):
    _clean_folder_contents(params.debug_folder)
    _clean_folder_contents(params.logs_folder)
    _clean_folder_contents(params.results_folder)
    _clean_folder_contents(params.input_data_folder)


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
    _install_runtime_safety_guards()
    if params.clean_output_folders_before_run:
        _clean_run_output_folders(params)

    runtime_device = str(getattr(params, "runtime_device", "gpu")).lower()
    if runtime_device not in {"cpu", "gpu"}:
        raise ValueError("runtime_device must be 'cpu' or 'gpu'.")

    cupy_ok = False
    torch_cuda_ok = torch.cuda.is_available()

    if runtime_device == "gpu":
        try:
            import cupy as cp

            cupy_ok = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            cupy_ok = False

        if not torch_cuda_ok:
            warnings.warn(
                "runtime_device='gpu' requested but PyTorch CUDA is unavailable. Falling back to CPU.",
                RuntimeWarning,
            )
            runtime_device = "cpu"

    use_gpu = runtime_device == "gpu"
    # SigPy/CuPy can stay on CPU even when Torch runs on CUDA.
    if use_gpu and cupy_ok:
        sp_device = sp.Device(0)
    else:
        sp_device = sp.Device(-1)
        if use_gpu and not cupy_ok:
            warnings.warn(
                "CuPy/SigPy GPU backend is unavailable; using CPU for SigPy parts and CUDA for PyTorch parts.",
                RuntimeWarning,
            )

    t_device = torch.device("cuda:0" if use_gpu else "cpu")
    params.runtime_device = runtime_device

    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if print_gpu_info:
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Total GPU memory: {total_mem:.2f} GB")

    if params.debug_flag:
        torch.manual_seed(params.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if use_gpu and torch.cuda.is_available():
            torch.cuda.manual_seed(params.seed)
            torch.cuda.manual_seed_all(params.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return sp_device, t_device

import atexit
import gc
import shutil
import signal
import sys
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


def cleanup_notebook_namespace(
    namespace,
    aggressive=False,
    shutdown_kernel=False,
    restart_kernel=False,
):
    """
    Remove heavy objects from a notebook global namespace, then run runtime cleanup.
    """
    heavy_names = {
        "data",
        "recon",
        "jointReconstructor",
        "joint_reconstructor",
        "kspace",
        "kspace_corrupted",
        "image_corrupted",
        "image_ground_truth",
        "motion_model",
        "result",
    }
    for name in heavy_names:
        namespace.pop(name, None)

    if aggressive:
        for name, value in list(namespace.items()):
            if name.startswith("_"):
                continue
            if isinstance(value, torch.Tensor):
                namespace.pop(name, None)

    cleanup_runtime()

    if shutdown_kernel:
        try:
            from IPython import get_ipython

            ip = get_ipython()
            if ip is not None and getattr(ip, "kernel", None) is not None:
                ip.kernel.do_shutdown(restart=restart_kernel)
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


def clean_run_output_folders(params):
    _clean_folder_contents(params.debug_folder)
    _clean_folder_contents(params.logs_folder)
    _clean_folder_contents(params.results_folder)


def install_runtime_safety_guards():
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
    install_runtime_safety_guards()
    if getattr(params, "clean_output_folders_before_run", True):
        clean_run_output_folders(params)

    try:
        import cupy as cp
        cupy_ok = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        cupy_ok = False

    sp_device = sp.Device(0) if cupy_ok else sp.Device(-1)
    t_device = torch.device("cuda:0" if cupy_ok and torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if print_gpu_info:
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Total GPU memory: {total_mem:.2f} GB")

    if params.debug_flag:
        torch.manual_seed(params.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(params.seed)
            torch.cuda.manual_seed_all(params.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return sp_device, t_device

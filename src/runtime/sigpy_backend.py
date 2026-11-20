"""Resolve the optional SigPy backend when ESPIRiT is selected."""


def _setup_cupy_sigpy_compat(cp):
    """Bridge CuPy's cuDNN API before importing SigPy."""
    if hasattr(cp, "cudnn"):
        return
    try:
        import cupy.cuda.cudnn as cudnn
    except Exception:
        return
    cp.cudnn = cudnn


def resolve_espirit_device(use_gpu):
    """Keep ESPIRiT on CPU when the requested CuPy backend is unavailable."""
    cupy_ok = False
    # SigPy imports installed CuPy even on CPU. Apply its compatibility shim
    # here, without probing CUDA unless a GPU backend was requested.
    try:
        import cupy as cp

        _setup_cupy_sigpy_compat(cp)
        if use_gpu:
            cupy_ok = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        cupy_ok = False
    if use_gpu and not cupy_ok:
        print("[runtime] CuPy/SigPy GPU backend is unavailable; using CPU for SigPy and CUDA for PyTorch.", flush=True)

    import sigpy as sp

    return sp.Device(0 if use_gpu and cupy_ok else -1)

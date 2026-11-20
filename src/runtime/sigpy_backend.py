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
    reason = None
    # SigPy imports installed CuPy even on CPU. Apply its compatibility shim
    # here, without probing CUDA unless a GPU backend was requested.
    try:
        import cupy as cp

        _setup_cupy_sigpy_compat(cp)
        if use_gpu:
            cupy_ok = cp.cuda.runtime.getDeviceCount() > 0
        if cupy_ok:
            # ESPIRiT needs cuSOLVER for its SVD. CuPy loads it on first use,
            # which fails when libcusolver is not on the loader path.
            from cupy_backends.cuda.libs import cusolver  # noqa: F401
    except Exception as error:
        cupy_ok = False
        reason = error
    if use_gpu and not cupy_ok:
        detail = f" ({reason})" if reason is not None else ""
        print(f"[runtime] CuPy/SigPy GPU backend is unavailable{detail}; using CPU for SigPy and CUDA for PyTorch.", flush=True)

    import sigpy as sp

    return sp.Device(0 if use_gpu and cupy_ok else -1)

import os
import warnings
from pathlib import Path

import numpy as np


def _setup_env_guards():
    """Set only cache locations that project imports require.

    Jupyter owns its own configuration and runtime directories. Changing them
    from an imported package can disrupt an already-running kernel.
    """
    cache_dir = Path(os.environ.get("NUMBA_CACHE_DIR", "/tmp/numba_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    # Needed for deterministic PyTorch operations that route through CuBLAS.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _setup_warning_filters():
    visible_dep_warn = np.exceptions.VisibleDeprecationWarning
    warnings.filterwarnings(
        "ignore",
        message=r"This function is deprecated and will be removed in a future release\. "
        r"Use the cupy\.from_dlpack\(\) array constructor instead\.",
        category=visible_dep_warn,
        module=r"sigpy\.pytorch",
    )


def _setup_cupy_sigpy_compat():
    """
    Bridge CuPy API differences so sigpy can import cuDNN helpers across versions.
    """
    try:
        import cupy
    except Exception:
        return

    if hasattr(cupy, "cudnn"):
        return

    try:
        import cupy.cuda.cudnn as cudnn
    except Exception:
        return

    cupy.cudnn = cudnn


_setup_env_guards()
_setup_cupy_sigpy_compat()
_setup_warning_filters()

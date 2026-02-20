import os
from pathlib import Path


def _setup_env_guards():
    # Ensure numba caching has a writable location before sigpy/numba import.
    cache_dir = Path(os.environ.get("NUMBA_CACHE_DIR", "/tmp/numba_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))


_setup_env_guards()

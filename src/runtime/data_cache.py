"""Shared, source-keyed cache with process leases and atomic publication (POSIX)."""
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
import atexit
import fcntl
import hashlib
import json
import os
import uuid

DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[3] / 'data' / 'cache' / 'grics'
_LEASES = []
_LEASE_SCOPE = ContextVar("grics_cache_leases", default=None)


def cache_root(value=None):
    return Path(value).expanduser().resolve() if value and value != "auto" else DEFAULT_CACHE_ROOT


def source_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'ctime_ns': stat.st_ctime_ns}


def cache_key(sources, settings):
    identity = {'sources': [source_identity(path) for path in sources], 'settings': settings}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


class CacheLease:
    def __init__(self, path, lock, remove):
        self.path, self.lock, self.remove = path, lock, remove
        self.pid = os.getpid()
        self.closed = False

    def close(self):
        if self.closed or self.pid != os.getpid():
            return
        marker = self.path.with_suffix(self.path.suffix + '.remove')
        if self.remove:
            marker.touch()
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            if marker.exists():
                self.path.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
        self.lock.close()
        self.closed = True


def acquire_cached(root, category, key, suffix, builder, remove=True):
    """Return a leased artifact; builder writes only to the supplied temporary path."""
    if category not in {'converted', 'preprocessed'}:
        raise ValueError('Unknown cache category.')
    directory = cache_root(root) / category
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{key}{suffix}'
    lock = path.with_suffix(path.suffix + '.lock').open('a+b')
    try:
        # Hold a shared usage lock throughout building and reading. A separate
        # builder lock serializes misses without waiting for readers to finish.
        fcntl.flock(lock, fcntl.LOCK_SH)
        if not path.is_file():
            with path.with_suffix(path.suffix + '.build.lock').open('a+b') as build_lock:
                fcntl.flock(build_lock, fcntl.LOCK_EX)
                if not path.is_file():
                    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
                    try:
                        builder(temporary)
                        if not temporary.is_file() or temporary.stat().st_size == 0:
                            raise RuntimeError('Cache builder produced no data.')
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
    except BaseException:
        lock.close()
        raise
    lease = CacheLease(path, lock, remove)
    _LEASES.append(lease)
    scope = _LEASE_SCOPE.get()
    if scope is not None:
        scope.append(lease)
    return lease


def release_leases(leases=None):
    for lease in list(_LEASES if leases is None else leases):
        lease.close()
    _LEASES[:] = [lease for lease in _LEASES if not lease.closed]


@contextmanager
def lease_scope():
    leases = []
    token = _LEASE_SCOPE.set(leases)
    try:
        yield
    finally:
        release_leases(leases)
        _LEASE_SCOPE.reset(token)


def clear_cache(root=None):
    """Remove inactive artifacts; leave active files and stable lock files alone."""
    removed, busy = [], []
    for category, suffix in [('converted', '.mrd'), ('preprocessed', '.h5')]:
        directory = cache_root(root) / category
        if not directory.exists():
            continue
        for lock_path in directory.glob(f'*{suffix}.lock'):
            path = lock_path.with_suffix('')
            with lock_path.open('a+b') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    busy.append(str(path))
                    continue
                if path.exists():
                    path.unlink()
                    removed.append(str(path))
                path.with_suffix(path.suffix + '.remove').unlink(missing_ok=True)
                for temporary in directory.glob(f'.{path.name}.*.tmp'):
                    temporary.unlink(missing_ok=True)
    return removed, busy


atexit.register(release_leases)

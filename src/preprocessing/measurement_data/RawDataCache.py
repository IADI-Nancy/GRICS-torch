"""Process-safe cache for converted and preprocessed recorded data."""
from pathlib import Path
import atexit
import fcntl
import hashlib
import json
import os
import uuid


DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[4] / 'data' / 'cache' / 'grics'


class _CacheLease:
    def __init__(self, path, lock, remove_on_release):
        self.path = path
        self.lock = lock
        self.remove_on_release = remove_on_release
        self.pid = os.getpid()
        self.closed = False

    def close(self):
        if self.closed or self.pid != os.getpid():
            return
        marker = self.path.with_suffix(self.path.suffix + '.remove')
        if self.remove_on_release:
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


class RawDataCache:
    """Cache derived recorded-data files and lease them for one loader's lifetime."""

    def __init__(self, root=None, remove_on_release=True):
        self.root = Path(root).expanduser().resolve() if root and root != 'auto' else DEFAULT_CACHE_ROOT
        self.remove_on_release = remove_on_release
        self._leases = []
        self.pid = os.getpid()
        self.closed = False
        atexit.register(self.close)

    @staticmethod
    def source_identity(path):
        path = Path(path).expanduser().resolve()
        stat = path.stat()
        return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'ctime_ns': stat.st_ctime_ns}

    @classmethod
    def key(cls, sources, settings):
        identity = {'sources': [cls.source_identity(path) for path in sources], 'settings': settings}
        return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()

    def acquire_converted(self, key, builder):
        return self._acquire('converted', key, '.mrd', builder)

    def acquire_preprocessed(self, key, builder):
        return self._acquire('preprocessed', key, '.h5', builder)

    def _acquire(self, category, key, suffix, builder):
        if self.closed:
            raise RuntimeError('RawDataCache is closed.')
        directory = self.root / category
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{key}{suffix}'
        lock = path.with_suffix(path.suffix + '.lock').open('a+b')
        try:
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
        lease = _CacheLease(path, lock, self.remove_on_release)
        self._leases.append(lease)
        return lease

    def close(self):
        if self.closed or self.pid != os.getpid():
            return
        for lease in self._leases:
            lease.close()
        self._leases.clear()
        self.closed = True

    def clear(self):
        """Remove inactive cached data files while preserving active files and locks."""
        removed, busy = [], []
        for category, suffix in [('converted', '.mrd'), ('preprocessed', '.h5')]:
            directory = self.root / category
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

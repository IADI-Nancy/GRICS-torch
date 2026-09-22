"""Run-scoped output paths and provenance, shared by scripts and notebooks."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
import atexit
import fcntl
import json
import os
import subprocess
import time
import uuid

_ACTIVE_RUNS = ContextVar('grics_runs', default=None)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def bind_output_paths(params, folder):
    folder = Path(folder)
    params.reconstruction_folder = str(folder)
    params.initial_data_folder = str(folder / 'preprocessing')
    params.results_folder = str(folder / 'results')
    params.debug_folder = str(folder / 'diagnostics')
    params.logs_folder = str(folder)


def public_config(params):
    return {key: value for key, value in vars(params).items() if not key.startswith('_')}


class RunOutputs:
    def __init__(self, params):
        workflow_label = params.workflow_label
        if not workflow_label or Path(workflow_label).name != workflow_label or workflow_label in {'.', '..'}:
            raise ValueError('workflow_label must be a single directory name.')
        run_id = datetime.now().strftime('%Y%m%dT%H%M%S%f') + '-' + uuid.uuid4().hex[:8]
        self.root = Path(params.output_root).expanduser().resolve() / workflow_label / run_id
        self.root.mkdir(parents=True, exist_ok=False)
        self._lock = (self.root / ".run.lock").open("a+b")
        fcntl.flock(self._lock, fcntl.LOCK_SH)
        self.closed = False
        self.params = params
        self.pid = os.getpid()
        self.started = time.perf_counter()
        self.manifest = {'run_id': run_id, 'workflow_label': workflow_label, 'status': 'running',
                         'started_at': datetime.now(timezone.utc).isoformat(), 'reconstructions': {}}
        try:
            self.manifest['code_revision'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2],
                stderr=subprocess.DEVNULL, text=True).strip()
            self.manifest['code_dirty'] = bool(subprocess.check_output(
                ['git', 'status', '--porcelain'], cwd=Path(__file__).resolve().parents[2],
                stderr=subprocess.DEVNULL, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            self.manifest['code_revision'] = None
        params.run_folder = str(self.root)
        params._run_outputs = self
        unit = 'volume_001' if params.data_dimension == '3D' else 'slice_001'
        bind_output_paths(params, self.root / 'reconstructions' / unit)
        self.snapshot(params)
        self.flush()
        active = _ACTIVE_RUNS.get()
        if active is not None:
            active.append(self)
        atexit.register(self.close, 'incomplete')

    def snapshot(self, params):
        write_json(self.root / 'config_resolved.json', public_config(params))

    def flush(self):
        if os.getpid() == self.pid:
            write_json(self.root / 'manifest.json', self.manifest)

    def close(self, status='complete', error=None):
        if self.closed or os.getpid() != self.pid:
            return
        self.snapshot(self.params)
        self.manifest['status'] = status
        self.manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        self.manifest['elapsed_s'] = time.perf_counter() - self.started
        if error is not None:
            self.manifest['error'] = str(error)
        # Per-worker metadata avoids concurrent writes to the run manifest.
        for metadata in self.root.glob('reconstructions/*/.metadata.json'):
            self.manifest['reconstructions'][metadata.parent.name] = json.loads(metadata.read_text())
        self.manifest['outputs'] = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')
                                          if p.is_file() and not p.name.startswith('.'))
        self.flush()
        self.closed = True
        self._lock.close()


@contextmanager
def execution_scope():
    """Finalize manifests and release cache leases even on exceptions/interrupts."""
    from src.runtime.data_cache import lease_scope
    runs = []
    token = _ACTIVE_RUNS.set(runs)
    try:
        with lease_scope():
            try:
                yield
            except BaseException as exc:
                for run in runs:
                    run.close('failed', exc)
                raise
            else:
                for run in runs:
                    run.close()
    finally:
        _ACTIVE_RUNS.reset(token)


def managed_execution(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with execution_scope():
            return function(*args, **kwargs)
    return wrapped


def record_reconstruction(params, **metadata):
    folder = Path(params.reconstruction_folder)
    path = folder / '.metadata.json'
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(metadata)
    current['resolution_levels'] = params.ResolutionLevels
    current['configuration'] = public_config(params)
    write_json(path, current)

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
_EXECUTION_CLEANUPS = ContextVar('grics_execution_cleanups', default=None)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    # Replace the file atomically to avoid partially written JSON.
    temporary.replace(path)


def bind_output_paths(params, folder):
    folder = Path(folder)
    # Keep all outputs for one reconstruction under a common folder.
    params.reconstruction_folder = str(folder)
    params.initial_data_folder = str(folder / 'preprocessing')
    params.results_folder = str(folder / 'results')
    params.debug_folder = str(folder / 'diagnostics')
    params.logs_folder = str(folder)


def public_config(params):
    return {key: value for key, value in vars(params).items() if not key.startswith('_')}


def register_execution_cleanup(cleanup):
    cleanups = _EXECUTION_CLEANUPS.get()
    if cleanups is not None:
        cleanups.append(cleanup)


class RunOutputs:
    def __init__(self, params):
        workflow_label = params.workflow_label
        if not workflow_label or Path(workflow_label).name != workflow_label or workflow_label in {'.', '..'}:
            raise ValueError('workflow_label must be a single directory name.')
        # Name the run directory with the local date and time to the second.
        run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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
            # Record the code version and local changes for traceability.
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
        bind_output_paths(params, self.root)
        self.snapshot(params)
        self.flush()
        active = _ACTIVE_RUNS.get()
        if active is not None:
            active.append(self)
        # Mark runs incomplete if normal scope finalization never happens.
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
        metadata_files = [self.root / '.metadata.json', *self.root.glob('slice_*/.metadata.json')]
        for metadata in metadata_files:
            if metadata.exists():
                unit = metadata.parent.name if metadata.parent != self.root else (
                    'volume_001' if self.params.data_dimension == '3D' else 'slice_001')
                self.manifest['reconstructions'][unit] = json.loads(metadata.read_text())
        self.manifest['outputs'] = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')
                                          if p.is_file() and not p.name.startswith('.'))
        self.flush()
        self.closed = True
        self._lock.close()


@contextmanager
def execution_scope():
    """Finalize manifests and registered resources even on exceptions/interrupts."""
    runs = []
    cleanups = []
    runs_token = _ACTIVE_RUNS.set(runs)
    cleanups_token = _EXECUTION_CLEANUPS.set(cleanups)
    try:
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
        for cleanup in reversed(cleanups):
            cleanup()
        _EXECUTION_CLEANUPS.reset(cleanups_token)
        _ACTIVE_RUNS.reset(runs_token)


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


def reconstruction_axes(image, motion, motion_type):
    image_axes = ['x', 'y'] + (['z'] if image.ndim == 3 else [])
    motion_axes = (['component', 'motion_state'] if motion_type == 'rigid'
                   else ['component'] + image_axes[1:])
    if len(motion_axes) < motion.ndim:
        motion_axes.append('sensor')
    return image_axes, motion_axes


def write_reconstruction(params, image, motion, *, native_image=None,
                         image_stage='before_postprocessing', record_paths=False, **metadata):
    """Save tensors; callers may supply completion status and export provenance."""
    import torch

    folder = Path(params.results_folder)
    paths = {'output_dir': folder, 'image_file': None, 'native_image_file': None, 'motion_file': None}
    if params.save_reconstruction_tensors:
        folder.mkdir(parents=True, exist_ok=True)
        paths['image_file'] = folder / 'image_reconstructed.pt'
        paths['motion_file'] = folder / 'motion_parameters.pt'
        torch.save(image.detach().cpu(), paths['image_file'])
        if native_image is not None:
            paths['native_image_file'] = folder / 'image_reconstructed_native.pt'
            native_output = native_image.detach().cpu() if torch.is_tensor(native_image) else native_image
            torch.save(native_output, paths['native_image_file'])
        torch.save(motion.detach().cpu(), paths['motion_file'])
    image_axes, motion_axes = reconstruction_axes(image, motion, params.reconstruction_motion_type)
    details = dict(metadata)
    if record_paths:
        details.update(paths)
    details.setdefault('status', 'reconstructed')
    details['tensor_export'] = details.get('tensor_export', 'solver') if params.save_reconstruction_tensors else 'disabled'
    details.update(image_stage=image_stage,
                   image_shape=list(image.shape), motion_shape=list(motion.shape),
                   image_axes=image_axes, motion_axes=motion_axes,
                   motion_grid='reconstruction', motion_type=params.reconstruction_motion_type)
    if native_image is not None:
        details['native_image_shape'] = list(native_image.shape)
    record_reconstruction(params, **details)
    return paths

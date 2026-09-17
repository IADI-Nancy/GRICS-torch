"""Remove inactive run outputs: python -m src.utils.clear_runs [--dry-run]."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import shutil
import tomllib


def clear_runs(output_root, *, workflow_label=None, older_than_days=None, dry_run=False):
    """Remove recognized, inactive run directories; return selected and skipped paths."""
    if workflow_label is not None and (not workflow_label or Path(workflow_label).name != workflow_label
                                       or workflow_label in {'.', '..'}):
        raise ValueError('workflow_label must be a single directory name.')
    if older_than_days is not None and (not math.isfinite(older_than_days) or older_than_days < 0):
        raise ValueError('older_than_days must be a finite nonnegative number.')
    root = Path(output_root).expanduser().resolve()
    selected, skipped = [], []
    if not root.is_dir():
        return selected, skipped
    labels = [root / workflow_label] if workflow_label else sorted(root.iterdir())
    now = datetime.now(timezone.utc)
    for label in labels:
        if label.is_symlink() or not label.is_dir():
            continue
        for run in sorted(label.iterdir()):
            if run.is_symlink() or not run.is_dir():
                continue
            lock = None
            try:
                manifest_path = run / 'manifest.json'
                if manifest_path.is_symlink():
                    skipped.append((str(run), 'symlinked manifest'))
                    continue
                manifest = json.loads(manifest_path.read_text())
                if not isinstance(manifest, dict) or manifest.get('run_id') != run.name:
                    skipped.append((str(run), 'unrecognized manifest'))
                    continue
                lock_path = run / '.run.lock'
                if lock_path.is_symlink():
                    skipped.append((str(run), 'symlinked lock'))
                    continue
                if lock_path.exists():
                    lock = lock_path.open('r+b')
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        skipped.append((str(run), 'active run'))
                        continue
                elif manifest.get('status') not in {'complete', 'failed', 'incomplete'}:
                    skipped.append((str(run), 'running or unknown status without a lifecycle lock'))
                    continue
                if older_than_days is not None:
                    timestamp = manifest.get('finished_at') or manifest.get('started_at')
                    when = datetime.fromisoformat(timestamp)
                    if when.tzinfo is None:
                        raise ValueError('Run timestamp has no timezone')
                    if (now - when).total_seconds() < older_than_days * 86400:
                        continue
                if not dry_run:
                    shutil.rmtree(run)
                selected.append(str(run))
            except (OSError, ValueError, TypeError) as exc:
                skipped.append((str(run), str(exc)))
            finally:
                if lock is not None:
                    lock.close()
    return selected, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', help='Defaults to output_root in config/general.toml.')
    parser.add_argument('--workflow-label', help='Only remove runs under this label.')
    parser.add_argument('--older-than-days', type=float, help='Only remove runs older than this many days.')
    parser.add_argument('--dry-run', action='store_true', help='List matching runs without deleting them.')
    args = parser.parse_args()
    if args.output_root is None:
        config = Path(__file__).resolve().parents[2] / 'config' / 'general.toml'
        with config.open('rb') as handle:
            args.output_root = tomllib.load(handle)['paths']['output_root']
    try:
        selected, skipped = clear_runs(args.output_root, workflow_label=args.workflow_label,
                                      older_than_days=args.older_than_days, dry_run=args.dry_run)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Output root: {Path(args.output_root).expanduser().resolve()}')
    action = 'Would remove' if args.dry_run else 'Removed'
    print(f'{action} {len(selected)} runs; skipped {len(skipped)} directories.')
    for path in selected:
        print(f'{action}: {path}')
    for path, reason in skipped:
        print(f'Skipped: {path} ({reason})')


if __name__ == '__main__':
    main()

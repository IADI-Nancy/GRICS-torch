"""Empty inactive GRICS cache data: python -m src.utils.clear_cache [--cache-root PATH]."""
import argparse
from pathlib import Path
import tomllib
from src.preprocessing.real_data.RawDataCache import RawDataCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', default=None)
    args = parser.parse_args()
    if args.cache_root is None:
        config = Path(__file__).resolve().parents[2] / 'config' / 'general.toml'
        with config.open('rb') as handle:
            args.cache_root = tomllib.load(handle)['paths']['cache_root']
    cache = RawDataCache(args.cache_root, remove_on_release=False)
    removed, busy = cache.clear()
    print(f'Cache: {cache.root}')
    print(f'Removed {len(removed)} cached files; skipped {len(busy)} active files.')
    for path in busy:
        print(f'In use: {path}')


if __name__ == '__main__':
    main()

"""Empty inactive GRICS cache data: python -m src.utils.clear_cache [--cache-root PATH]."""
import argparse
from pathlib import Path
import tomllib
from src.runtime.data_cache import cache_root, clear_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', default=None)
    args = parser.parse_args()
    if args.cache_root is None:
        config = Path(__file__).resolve().parents[2] / 'config' / 'general.toml'
        with config.open('rb') as handle:
            args.cache_root = tomllib.load(handle)['paths']['cache_root']
    removed, busy = clear_cache(args.cache_root)
    print(f'Cache: {cache_root(args.cache_root)}')
    print(f'Removed {len(removed)} artifacts; skipped {len(busy)} active artifacts.')
    for path in busy:
        print(f'In use: {path}')


if __name__ == '__main__':
    main()

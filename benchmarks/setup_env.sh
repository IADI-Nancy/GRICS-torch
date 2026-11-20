#!/bin/bash
# Creates a Python 3.11+ virtual environment with the pinned benchmark dependencies and the IADI wrapperHDF5 package (h5Saec).
# Usage: benchmarks/setup_env.sh [venv_dir]   then:  source <venv_dir>/bin/activate; export PYTHONPATH=<venv_dir>/src/wrapperHDF5
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); V=${1:-$HERE/venv}
PYBIN=$(command -v python3.12 || command -v python3.11 || { echo "python >= 3.11 required (tomllib)"; exit 1; })
$PYBIN -m venv "$V"; "$V/bin/python" -m pip install --upgrade pip wheel setuptools
"$V/bin/python" -m pip install -r "$HERE/requirements.txt"
mkdir -p "$V/src"; [ -d "$V/src/wrapperHDF5" ] || git clone https://github.com/IADI-Nancy/wrapperHDF5 "$V/src/wrapperHDF5"
echo "done. Activate with: source $V/bin/activate; export PYTHONPATH=$V/src/wrapperHDF5"

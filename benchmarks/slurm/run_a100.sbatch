#!/bin/bash
#SBATCH -J grics_bench
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o grics_bench_%j.out
# One-GPU benchmark job. Submit from the repository root after `benchmarks/setup_env.sh` and `benchmarks/get_demo_data.sh`:
#   git clone <GRICS-torch 2.0> ../GRICS-torch-upstream
#   sbatch -A <account> -p <partition> benchmarks/slurm/run_a100.sbatch ../GRICS-torch-upstream
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd); REF=${1:?reference repo root}
V=${GRICS_VENV:-$REPO/benchmarks/venv}
export PYTHONPATH=$V/src/wrapperHDF5; export GRICS_BENCH_PY=$V/bin/python
"$GRICS_BENCH_PY" -c "import torch, sigpy, ismrmrd, h5py, numba, cupy; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" || exit 1
bash "$REPO/benchmarks/run_all.sh" "$REF" "$REPO"

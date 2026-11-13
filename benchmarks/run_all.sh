#!/bin/bash
# Reference-versus-batched comparison on one GPU: operator outputs, full reconstructions, roofline model, paper-size 2D slice.
# Usage: benchmarks/run_all.sh <reference_repo_root> [this_repo_root]
#   reference_repo_root: a checkout of upstream GRICS-torch, e.g. `git worktree add ../GRICS-torch-upstream upstream-main-4b9f2f5`
# Environment: GRICS_BENCH_BASE (output dir, default benchmarks/out), GRICS_DEMO_H5 (Zenodo demo file, default benchmarks/data/demo_breast_data.h5),
#              GRICS_BENCH_PY (python, default: python on PATH), PYTHONPATH must contain the wrapperHDF5 package (h5Saec).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/.." && pwd)
REF=$(cd "$1" && pwd); FAST=$(cd "${2:-$REPO}" && pwd)
export GRICS_BENCH_BASE=${GRICS_BENCH_BASE:-$HERE/out}; B=$GRICS_BENCH_BASE
export GRICS_DEMO_H5=${GRICS_DEMO_H5:-$HERE/data/demo_breast_data.h5}
PY=${GRICS_BENCH_PY:-python}
mkdir -p "$B/recon_reference" "$B/recon_compat" "$B/recon_batched_double" "$B/recon_batched_single"
cd "$HERE"
echo "=== node $(hostname) $(date)"; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader
echo "=== reference: $REF ($(git -C "$REF" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo "=== batched:   $FAST ($(git -C "$FAST" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo "=== operators $(date)"
$PY dump_ops.py "$REF" "$B/ops_reference.pt" 2>&1 | grep -v Warning
$PY dump_ops.py "$FAST" "$B/ops_batched_double.pt" 2>&1 | grep -v Warning
$PY dump_ops.py "$FAST" "$B/ops_batched_single_gather.pt" single gather 2>&1 | grep -v Warning
for t in batched_double batched_single_gather; do echo "--- operators $t vs reference"; $PY compare_dumps.py "$B/ops_reference.pt" "$B/ops_$t.pt"; done
echo "=== reconstructions $(date)"
for case in 2d_rigid 2d_nonrigid 3d_rigid 3d_nonrigid real; do
  $PY recon_dump.py "$REF" $case "$B/recon_reference/$case.pt" 2>&1 | grep -E "run\(\)|Error"
  GRICS_COMPAT=1 $PY recon_dump.py "$FAST" $case "$B/recon_compat/$case.pt" 2>&1 | grep -E "run\(\)|Error"
  $PY recon_dump.py "$FAST" $case "$B/recon_batched_double/$case.pt" 2>&1 | grep -E "run\(\)|Error"
  $PY recon_dump.py "$FAST" $case "$B/recon_batched_single/$case.pt" 20261101 single gather 2>&1 | grep -E "run\(\)|Error"
done
echo "--- reconstructions, upstream numerics (GRICS_COMPAT=1) vs reference: must be identical"
$PY compare_recons.py "$B/recon_reference" "$B/recon_compat" 2d_rigid 2d_nonrigid 3d_rigid 3d_nonrigid real
for t in batched_double batched_single; do echo "--- reconstructions $t (solver corrections on) vs reference: differences expected"; $PY compare_recons.py "$B/recon_reference" "$B/recon_$t" 2d_rigid 2d_nonrigid 3d_rigid 3d_nonrigid real; done
echo "=== roofline model $(date)"; $PY roofline.py "$REF" "$FAST" "$FAST" "$B/roofline.json" 2>&1 | grep -v Warning
echo "=== paper-size 2D slice (Zenodo breast slice, nonrigid_2d.toml, 3 repeats) $(date)"
$PY paper_size_bench.py "$REF" reference paper2d 2>&1 | grep -E "run |reference|Error"
$PY paper_size_bench.py "$FAST" batched_double paper2d 2>&1 | grep -E "run |batched|Error"
$PY paper_size_bench.py "$FAST" batched_single paper2d single gather 2>&1 | grep -E "run |batched|Error"
echo "=== summary"; $PY summarize_bench.py "$B" "$B/summary.md" && cat "$B/summary.md"
echo "=== done $(date)"

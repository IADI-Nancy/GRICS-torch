# Benchmarks: current implementation against GRICS-torch 2.0

Everything in `validation_studies/gpu_optimization_benchmarks.md` comes from the scripts in this folder. They compare a checkout of the GRICS-torch 2.0 code (the *reference*) with the current implementation on the same GPU, with the same seeds, and report the relative difference of the outputs together with the timings. One GPU and about 30 minutes are enough.

## 1. Environment

```bash
validation_studies/gpu_optimization_benchmarks/setup_env.sh                      # venv with validation_studies/gpu_optimization_benchmarks/requirements.txt + IADI wrapperHDF5 (h5Saec)
source validation_studies/gpu_optimization_benchmarks/venv/bin/activate
export PYTHONPATH=$PWD/validation_studies/gpu_optimization_benchmarks/venv/src/wrapperHDF5
validation_studies/gpu_optimization_benchmarks/get_demo_data.sh                  # Zenodo record 19086853: demo_breast_data.h5 (65 MB) -> validation_studies/gpu_optimization_benchmarks/data/
```

Python 3.11 or newer is required (`tomllib`). The pins in `requirements.txt` are the versions used for the published numbers; the code itself has no new dependency compared with GRICS-torch 2.0.

## 2. Reference checkout

Put a checkout of GRICS-torch 2.0 in a sibling directory, for example:

```bash
git clone <GRICS-torch 2.0> ../GRICS-torch-2.0
```

## 3. Run

```bash
validation_studies/gpu_optimization_benchmarks/run_all.sh ../GRICS-torch-2.0               # outputs in validation_studies/gpu_optimization_benchmarks/out/, summary in validation_studies/gpu_optimization_benchmarks/out/summary.md
```

or, on a SLURM cluster, `sbatch -A <account> -p <partition> validation_studies/gpu_optimization_benchmarks/slurm/run_a100.sbatch ../GRICS-torch-2.0`.

What `run_all.sh` does, in order:

| step | script | what is compared |
|---|---|---|
| operators | `dump_ops.py`, `compare_dumps.py` | `E.forward/adjoint/normal`, `J.forward/adjoint/normal` on fixed-seed synthetic problems (small cases for identity, 128² and 128×128×64 for timing); prints the max relative difference per operator |
| reconstructions | `recon_dump.py`, `compare_recons.py` | the four shipped synthetic validation settings (`rigid_2d`, `nonrigid_2d`, `rigid_3d`, `nonrigid_3d`) and the Zenodo breast slice; prints image and motion relative differences, `run()` times and an ACCEPTANCE line (NRMSE vs reference ≤ 1e-4) |
| roofline model | `roofline.py` | measured device ceilings (copy bandwidth, FP32/FP64 matmul, 3D FFT) and, per operator call, the bytes the batched formulation must move, hence the fraction of the bandwidth bound each code reaches |
| paper-size 2D | `paper_size_bench.py` | the Zenodo slice with `nonrigid_2d.toml` (the paper's 2D setting), 3 repeats, peak memory |

Each script can also be run on its own; the usage line is in its docstring. Two configurations of the current implementation are timed: the default settings (double precision, sparse warps) and `compute_precision = "single"` + `warp_backend = "gather"`, passed as the trailing `single gather` arguments. A third run with `GRICS_COMPAT=1` disables `jacobian_rotation_factor`, `motion_step_real`, and `warm_start_rescale`; the three runs compare these settings with the GRICS-torch 2.0 reference. Other solver, reader, and initialization changes remain active in each configuration.

Optional, not part of `run_all.sh`: `paper_size_bench.py <repo> <tag> paper3d|brain [single gather]` (synthetic 3D at the paper's size class and at 256×208×160) and `slice_concurrency.py` (several slices per GPU). `profile_recon.py` prints where the time of one reconstruction goes (operator calls, CG, operator builds).

## 4. Figures

`validation_studies/gpu_optimization_benchmarks/fig_main.py` draws the image-fidelity and timing figure from `validation_studies/gpu_optimization_benchmarks/out`:

```bash
python validation_studies/gpu_optimization_benchmarks/fig_main.py validation_studies/gpu_optimization_benchmarks/out recon_reference recon_batched_single \
       validation_studies/gpu_optimization_benchmarks/out/ops_reference.pt validation_studies/gpu_optimization_benchmarks/out/ops_batched_single_gather.pt figure_main "one NVIDIA A100 PCIe 40 GB"
```

`validation_studies/gpu_optimization_benchmarks/fig_schematic.py` and `validation_studies/gpu_optimization_benchmarks/fig_explainer.py` take only an output stem.

## 5. Published results

`results/a100_pcie_40gb_2026-11-13/` holds the raw outputs behind `validation_studies/gpu_optimization_benchmarks.md`: the SLURM log of job 10636891 (node with one dedicated A100 PCIe 40 GB, 2×AMD EPYC), `roofline.json` and the paper-size JSON files. In that run the tags were `reference`, `layer1` (commit 7e7f42b: batched operators only), `layer3dg` / `layer3_double_gather` (double precision, gather backend) and `layer3sg` / `layer3_single_gather` (single precision, gather backend).

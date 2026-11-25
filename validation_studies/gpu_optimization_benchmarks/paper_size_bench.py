"""End-to-end timings at the data sizes of the JOSS paper and at brain size, for one repository.
Usage: python paper_size_bench.py <repo_root> <tag> <case> [precision] [backend]
cases:
  paper3d  : synthetic 3D 192x192x56 (paper: 160x218x56), 24 coils, 16 non-rigid per-shot states, nonrigid_3d.toml iterations
  paper2d  : the Zenodo breast slice (320x532, 24 coils; paper: 288x475, 60-68 slices), nonrigid_2d.toml, 10 repeats -> per-slice time
  brain    : synthetic 3D 256x208x160 (our 7T matrices), 32 coils, 12 rigid per-shot states, rigid_3d.toml iterations
Writes <tag>_<case>.json next to this script's bench/ folder.
"""
import sys, os, time, json
from pathlib import Path
import torch
REPO = Path(sys.argv[1]).resolve(); TAG = sys.argv[2]; CASE = sys.argv[3]
PREC = sys.argv[4] if len(sys.argv) > 4 else None; BACKEND = sys.argv[5] if len(sys.argv) > 5 else None
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tests")); os.chdir(REPO)
S = os.environ.get("GRICS_BENCH_BASE", str(Path(__file__).resolve().parent / "out")); DEMO = os.environ.get("GRICS_DEMO_H5", str(Path(__file__).resolve().parent / "data" / "demo_breast_data.h5"))
OUT = Path(S) / "paper"; OUT.mkdir(parents=True, exist_ok=True)
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from src.preprocessing.DataPreparer import DataPreparer
from src.reconstruction.JointReconstructor import JointReconstructor

def run(data_type, recon, sampling, motion, overrides, filename=None, slice_idx=None, repeats=1):
    ov = {"workflow_label": f"paper_{TAG}_{CASE}", "output_root": str(OUT / "runs"), "runtime_device": "gpu",
          "seed_enabled": True, "seed": 20261101, "save_debug_plots": False, "save_reconstruction_logs": False,
          "save_reconstruction_tensors": False, "verbose": False, "print_to_console": False}
    if PREC: ov["compute_precision"] = PREC
    if BACKEND: ov["warp_backend"] = BACKEND
    ov.update(overrides)
    kw = dict(data_type=data_type, reconstruction_config=str(REPO / "config/reconstruction" / recon), overrides=ov)
    if sampling: kw["sampling_config"] = str(REPO / "config/sampling_simulation" / sampling)
    if motion: kw["motion_simulation_config"] = str(REPO / "config/motion_simulation" / motion)
    params = load_config(**kw)
    runtime = initialize_runtime(params)
    t0 = time.perf_counter()
    data = DataPreparer(params=params, runtime=runtime, filename=filename); data.load_input_data(); data.prepare_reconstructor_inputs(slice_idx=slice_idx)
    torch.cuda.synchronize(); t_prep = time.perf_counter() - t0
    times = []
    torch.cuda.reset_peak_memory_stats()
    for r in range(repeats):
        rec = JointReconstructor.from_prepared_data(data)
        torch.cuda.synchronize(); t1 = time.perf_counter(); img, mot = rec.run(); torch.cuda.synchronize(); times.append(time.perf_counter() - t1)
        print(f"  run {r+1}: {times[-1]:.1f} s", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    return {"prep_s": t_prep, "run_s": times, "peak_GiB": peak, "shape": [int(data.Nx), int(data.Ny), int(data.Nz)], "coils": int(data.smaps.shape[0]),
            "states": int(len(data.sampling_idx[0])), "image_abs_mean": float(img.abs().mean()), "motion_abs_max": float(mot.abs().max())}

if CASE == "paper3d":
    res = run("shepp-logan", "nonrigid_3d.toml", "random.toml", "nonrigid_per_shot_3d.toml",
              {"N_SheppLogan": 192, "Nz_SheppLogan": 56, "Ncoils_SheppLogan": 24, "NshotsPerNex": 16, "N_motion_states": 16})
elif CASE == "paper2d":
    res = run("preprocessed-real", "nonrigid_2d.toml", None, None, {}, filename=DEMO, slice_idx=0, repeats=3)
elif CASE == "brain":
    res = run("shepp-logan", "rigid_3d.toml", "random.toml", "rigid_per_shot_3d.toml",
              {"N_SheppLogan": 256, "Nz_SheppLogan": 160, "Ncoils_SheppLogan": 32, "NshotsPerNex": 12, "N_motion_states": 12})
else:
    raise SystemExit("unknown case")
res.update({"repo": REPO.name, "tag": TAG, "case": CASE, "precision": PREC or "double", "backend": BACKEND or "sparse",
            "gpu_util_during": os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader").read().strip()})
(OUT / f"{TAG}_{CASE}.json").write_text(json.dumps(res, indent=1))
print(f"{TAG} {CASE} {res['precision']}/{res['backend']}: prep {res['prep_s']:.1f} s, run {['%.1f' % t for t in res['run_s']]} s, peak {res['peak_GiB']:.1f} GiB, shape {res['shape']}, coils {res['coils']}, states {res['states']}", flush=True)

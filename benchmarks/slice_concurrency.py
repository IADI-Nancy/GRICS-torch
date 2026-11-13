"""Slice-level concurrency on one GPU: reconstruct the Zenodo breast slice in K concurrent processes and measure
throughput (slices per minute), the way the 2D multislice pipeline would run on the GPU instead of a CPU pool.
Usage: python slice_concurrency.py <repo_root> <tag> [precision] [backend]   (runs K = 1, 2, 4 in turn)
"""
import sys, os, time, json, subprocess
from pathlib import Path
REPO = Path(sys.argv[1]).resolve(); TAG = sys.argv[2]
PREC = sys.argv[3] if len(sys.argv) > 3 else None; BACKEND = sys.argv[4] if len(sys.argv) > 4 else None
S = os.environ.get("GRICS_BENCH_BASE", str(Path(__file__).resolve().parent / "out")); DEMO = os.environ.get("GRICS_DEMO_H5", str(Path(__file__).resolve().parent / "data" / "demo_breast_data.h5"))
PY = os.environ.get("GRICS_BENCH_PY", sys.executable); OUT = Path(S) / "slices"; OUT.mkdir(parents=True, exist_ok=True)
worker = r'''
import sys, os, time, torch
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/tests"); os.chdir(sys.argv[1])
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from src.preprocessing.DataPreparer import DataPreparer
from src.reconstruction.JointReconstructor import JointReconstructor
S = "%s"; DEMO = "%s"
ov = {"workflow_label": "slice_" + sys.argv[2], "output_root": S + "/runs", "runtime_device": "gpu", "seed_enabled": True, "seed": 1,
      "save_debug_plots": False, "save_reconstruction_logs": False, "save_reconstruction_tensors": False, "verbose": False, "print_to_console": False}
if sys.argv[3] != "none": ov["compute_precision"] = sys.argv[3]
if sys.argv[4] != "none": ov["warp_backend"] = sys.argv[4]
params = load_config(data_type="preprocessed-real", reconstruction_config=sys.argv[1] + "/config/reconstruction/nonrigid_2d.toml", overrides=ov)
runtime = initialize_runtime(params)
data = DataPreparer(params=params, runtime=runtime, filename=DEMO); data.load_input_data(); data.prepare_reconstructor_inputs(slice_idx=0)
n = int(sys.argv[5]); t0 = time.perf_counter()
for i in range(n):
    rec = JointReconstructor.from_prepared_data(data); img, mot = rec.run()
torch.cuda.synchronize(); print("WORKER_DONE", n, time.perf_counter() - t0, flush=True)
''' % (str(OUT), DEMO)
(OUT / "worker.py").write_text(worker)
results = {}
for K in (1, 2, 4):
    n_per = 3
    t0 = time.perf_counter()
    procs = [subprocess.Popen([PY, str(OUT / "worker.py"), str(REPO), f"{TAG}_{K}_{i}", PREC or "none", BACKEND or "none", str(n_per)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")}) for i in range(K)]
    outs = [p.communicate()[0] for p in procs]
    wall = time.perf_counter() - t0
    done = [l for o in outs for l in o.splitlines() if l.startswith("WORKER_DONE")]
    per_worker = [float(l.split()[2]) for l in done]
    results[K] = {"wall_s": wall, "slices": K * n_per, "slices_per_min": 60 * K * n_per / wall, "worker_recon_s": per_worker, "ok": len(done) == K}
    print(f"{TAG} K={K}: {K*n_per} slices in {wall:.1f} s wall (incl. start-up and preparation) -> {60*K*n_per/wall:.1f} slices/min; per-worker reconstruction loops {['%.1f' % t for t in per_worker]} s", flush=True)
    if len(done) != K: print("\n".join(o[-800:] for o in outs))
(OUT / f"{TAG}.json").write_text(json.dumps(results, indent=1))

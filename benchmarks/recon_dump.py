"""Full reconstruction with a given repository on the shipped synthetic validation settings (or the Zenodo
demo slice), saving image + motion + timing, for equivalence checks between the reference and the fast code.
Usage: python recon_dump.py <repo_root> <case> <out.pt> [seed]
cases: 2d_rigid | 2d_nonrigid | 3d_rigid | 3d_nonrigid | real
"""
import sys, os, time, json
from pathlib import Path
import torch
REPO = Path(sys.argv[1]).resolve(); CASE = sys.argv[2]; OUT = Path(sys.argv[3]); SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 20261101
PRECISION = sys.argv[5] if len(sys.argv) > 5 else None
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tests")); os.chdir(REPO)
S = os.environ.get("GRICS_BENCH_BASE", str(Path(__file__).resolve().parent / "out")); DEMO = os.environ.get("GRICS_DEMO_H5", str(Path(__file__).resolve().parent / "data" / "demo_breast_data.h5"))
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from src.preprocessing.DataPreparer import DataPreparer
from src.reconstruction.JointReconstructor import JointReconstructor

files = {"2d_rigid": ("shepp-logan", "rigid_2d.toml", "random.toml", "rigid_per_readout_2d.toml"),
         "2d_nonrigid": ("shepp-logan", "nonrigid_2d.toml", "random.toml", "nonrigid_per_readout_2d.toml"),
         "3d_rigid": ("shepp-logan", "rigid_3d.toml", "random.toml", "rigid_per_shot_3d.toml"),
         "3d_nonrigid": ("shepp-logan", "nonrigid_3d.toml", "random.toml", "nonrigid_per_shot_3d.toml"),
         "real": ("preprocessed-real", "nonrigid_2d.toml", None, None)}
data_type, recon, sampling, motion = files[CASE]
ov = {"workflow_label": f"recon_dump_{CASE}", "output_root": str(Path(S) / "runs"), "runtime_device": "gpu",
      "seed_enabled": True, "seed": SEED, "save_debug_plots": False, "save_reconstruction_logs": False,
      "save_reconstruction_tensors": False, "verbose": False, "print_to_console": False}
if PRECISION: ov["compute_precision"] = PRECISION
if len(sys.argv) > 6: ov["warp_backend"] = sys.argv[6]
if os.environ.get("GRICS_COMPAT") == "1":      # 2.0 numerics: solver corrections off
    ov.update({"jacobian_rotation_factor": False, "motion_step_real": False, "warm_start_rescale": False})
kw = dict(data_type=data_type, reconstruction_config=str(REPO / "config/reconstruction" / recon), overrides=ov)
if sampling: kw["sampling_config"] = str(REPO / "config/sampling_simulation" / sampling)
if motion: kw["motion_simulation_config"] = str(REPO / "config/motion_simulation" / motion)
params = load_config(**kw)
runtime = initialize_runtime(params)
data = DataPreparer(params=params, runtime=runtime, filename=DEMO if CASE == "real" else None)
data.load_input_data()
data.prepare_reconstructor_inputs(slice_idx=0 if CASE == "real" else None)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
rec = JointReconstructor.from_prepared_data(data)
torch.cuda.synchronize(); t0 = time.perf_counter()
img, mot = rec.run()
torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
gt = getattr(data, "image_ground_truth", None)
torch.save({"image": img.detach().cpu(), "motion": mot.detach().cpu(), "elapsed_s": elapsed,
            "gt": None if gt is None else gt.detach().cpu(), "uncorrected": getattr(data, "image_no_moco", None).detach().cpu() if getattr(data, "image_no_moco", None) is not None else None}, OUT)
print(f"{CASE} {REPO.name}: run() {elapsed:.1f} s; |image| {img.abs().mean().item():.4e}; motion max {mot.abs().max().item():.3f}", flush=True)

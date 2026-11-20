"""Where does the time go in one reconstruction? Monkey-patches timers around the operator calls, the CG solves,
the operator builds and the level preparation. Usage: python profile_recon.py <repo_root> <case> [precision]"""
import sys, os, time, functools, collections
from pathlib import Path
import torch
REPO = Path(sys.argv[1]).resolve(); CASE = sys.argv[2]; PREC = sys.argv[3] if len(sys.argv) > 3 else None
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tests")); os.chdir(REPO)
S = os.environ.get("GRICS_BENCH_BASE", str(Path(__file__).resolve().parent / "out")); DEMO = os.environ.get("GRICS_DEMO_H5", str(Path(__file__).resolve().parent / "data" / "demo_breast_data.h5"))
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime
from src.preprocessing.DataPreparer import DataPreparer
from src.reconstruction import JointReconstructor as JRmod
from src.reconstruction.JointReconstructor import JointReconstructor
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.joint_reconstructor_utils import operators as opmod

acc = collections.defaultdict(lambda: [0.0, 0])
def timed(name):
    def deco(fn):
        @functools.wraps(fn)
        def w(*a, **k):
            torch.cuda.synchronize(); t = time.perf_counter(); r = fn(*a, **k); torch.cuda.synchronize()
            acc[name][0] += time.perf_counter() - t; acc[name][1] += 1; return r
        return w
    return deco
for cls, names in [(EncodingOperator, ["forward", "adjoint", "normal"]), (MotionPerturbationSimulator, ["forward", "adjoint"])]:
    for n in names: setattr(cls, n, timed(f"{cls.__name__}.{n}")(getattr(cls, n)))
MotionOperator.__init__ = timed("MotionOperator.__init__")(MotionOperator.__init__)
ConjugateGradientSolver.cg = timed("CG.cg (total, includes operator calls)")(ConjugateGradientSolver.cg)
ConjugateGradientSolver._update_regularization_scale = timed("CG.reg_scale probes")(ConjugateGradientSolver._update_regularization_scale)
JointReconstructor._solve_image = timed("_solve_image (total)")(JointReconstructor._solve_image)
JointReconstructor._solve_motion = timed("_solve_motion (total)")(JointReconstructor._solve_motion)
JointReconstructor._prepare_resolution_level = timed("_prepare_resolution_level")(JointReconstructor._prepare_resolution_level)
JointReconstructor.gauss_newton_iteration = timed("gauss_newton_iteration (total)")(JointReconstructor.gauss_newton_iteration)
if hasattr(JointReconstructor, "_build_pseudo_gauss_seidel_motion_preconditioner"):
    JointReconstructor._build_pseudo_gauss_seidel_motion_preconditioner = timed("motion preconditioner build")(JointReconstructor._build_pseudo_gauss_seidel_motion_preconditioner)

files = {"2d_rigid": ("shepp-logan", "rigid_2d.toml", "random.toml", "rigid_per_readout_2d.toml"),
         "2d_nonrigid": ("shepp-logan", "nonrigid_2d.toml", "random.toml", "nonrigid_per_readout_2d.toml"),
         "3d_rigid": ("shepp-logan", "rigid_3d.toml", "random.toml", "rigid_per_shot_3d.toml"),
         "3d_nonrigid": ("shepp-logan", "nonrigid_3d.toml", "random.toml", "nonrigid_per_shot_3d.toml")}
data_type, recon, sampling, motion = files[CASE]
ov = {"workflow_label": f"profile_{CASE}", "output_root": S + "/profile_runs", "runtime_device": "gpu",
      "seed_enabled": True, "seed": 20261101, "save_debug_plots": False, "save_reconstruction_logs": False,
      "save_reconstruction_tensors": False, "verbose": False, "print_to_console": False}
if PREC: ov["compute_precision"] = PREC; ov["warp_backend"] = os.environ.get("GRICS_PROFILE_BACKEND", "gather")
params = load_config(data_type=data_type, reconstruction_config=str(REPO / "config/reconstruction" / recon),
                     sampling_config=str(REPO / "config/sampling_simulation" / sampling),
                     motion_simulation_config=str(REPO / "config/motion_simulation" / motion), overrides=ov)
runtime = initialize_runtime(params)
data = DataPreparer(params=params, runtime=runtime); data.load_input_data(); data.prepare_reconstructor_inputs()
rec = JointReconstructor.from_prepared_data(data)
torch.cuda.synchronize(); t0 = time.perf_counter(); img, mot = rec.run(); torch.cuda.synchronize(); total = time.perf_counter() - t0
print(f"{CASE} {REPO.name} {PREC or 'double'}: run() total {total:.2f} s  (GPU shared: {os.popen('nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader').read().strip()})")
for name, (t, n) in sorted(acc.items(), key=lambda kv: -kv[1][0]):
    print(f"  {name:44s} {t:7.2f} s  {n:5d} calls  {t/max(n,1)*1e3:8.2f} ms/call  {100*t/total:5.1f} %")

"""Dump operator outputs on fixed-seed synthetic problems for a given repository, for equivalence checks
between the reference and the fast implementation, plus per-call timings.
Usage: python dump_ops.py <repo_root> <out.pt>
"""
import os, sys, time, json
from pathlib import Path
import torch
ROOT = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[2])
SINGLE = len(sys.argv) > 3 and sys.argv[3] == "single"
MO_KW = {"compute_dtype": torch.complex64} if SINGLE else {}
if len(sys.argv) > 4 and sys.argv[4] == "gather": MO_KW["warp_backend"] = "gather"
DEV = torch.device("cuda")
torch.set_default_dtype(torch.float64)
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.preprocessing.MotionSamplingPreparer import MotionSamplingPreparer

def sync(): torch.cuda.synchronize()
def timeit(fn, reps=5):
    fn(); sync(); t = time.perf_counter()
    for _ in range(reps): fn()
    sync(); return (time.perf_counter() - t) / reps

def setup(Nx, Ny, Nz, Nc, Nm, Nex, motion, seed=0, empty_state=False):
    g = torch.Generator().manual_seed(seed)
    smaps = (torch.randn(Nc, Nx, Ny, Nz, generator=g) + 1j*torch.randn(Nc, Nx, Ny, Nz, generator=g)).to(torch.complex128).to(DEV)
    image = (torch.randn(Nx, Ny, Nz, generator=g) + 1j*torch.randn(Nx, Ny, Nz, generator=g)).to(torch.complex128).to(DEV)
    if Nz == 1: image = image[..., 0]
    bky, bkz = [], []
    for nex in range(Nex):
        if Nz > 1:
            pairs = torch.cartesian_prod(torch.arange(Ny), torch.arange(Nz)); pairs = pairs[torch.randperm(pairs.shape[0], generator=g)]
            ch = list(torch.tensor_split(pairs, Nm))
            if empty_state: ch[1] = ch[1][:0]
            bky.append([c[:, 0].to(DEV) for c in ch]); bkz.append([c[:, 1].to(DEV) for c in ch])
        else:
            ch = list(torch.tensor_split(torch.randperm(Ny, generator=g), Nm))
            if empty_state: ch[1] = ch[1][:0]
            bky.append([c.to(DEV) for c in ch])
    sampling = MotionSamplingPreparer.build_sampling_per_nex_per_motion(bky, DEV, Nx, Ny, Nz=Nz, binned_kz_indices=bkz if Nz > 1 else None)
    dim = 3 if Nz > 1 else 2
    if motion == "rigid":
        alpha = (0.7 * torch.randn(6 if dim == 3 else 3, Nm, generator=g)).to(DEV); alpha[dim:] *= 0.1
        ms = None
    else:
        shp = (dim, Nx, Ny, Nz, 2) if dim == 3 else (dim, Nx, Ny, 2)
        alpha = (0.5 * torch.randn(*shp, generator=g)).to(DEV); ms = torch.randn(Nm, 2, generator=g).to(DEV)
    return smaps, image, sampling, alpha, ms, g

cases = [
    ("2d_rigid_small", 24, 20, 1, 3, 3, 1, "rigid", False),
    ("2d_rigid_nex2_empty", 24, 20, 1, 3, 4, 2, "rigid", True),
    ("2d_nonrigid", 24, 20, 1, 3, 3, 1, "non-rigid", False),
    ("3d_rigid", 16, 14, 6, 3, 3, 1, "rigid", False),
    ("3d_rigid_nex2_empty", 16, 14, 6, 3, 4, 2, "rigid", True),
    ("3d_nonrigid", 16, 14, 6, 3, 3, 1, "non-rigid", False),
    ("2d_rigid_128", 128, 128, 1, 8, 8, 1, "rigid", False),
    ("3d_rigid_128x128x64", 128, 128, 64, 4, 8, 1, "rigid", False),
    ("2d_nonrigid_128", 128, 128, 1, 8, 8, 1, "non-rigid", False),
]
res, times = {}, {}
for name, Nx, Ny, Nz, Nc, Nm, Nex, motion, empty in cases:
    smaps, image, sampling, alpha, ms, g = setup(Nx, Ny, Nz, Nc, Nm, Nex, motion, empty_state=empty)
    mo = MotionOperator(Nx, Ny, alpha, motion, motion_signal=ms, Nz=Nz, **MO_KW)
    Ns = Nx * Ny * Nz
    E = EncodingOperator(smaps, Ns, sampling, Nex, mo)
    J_KW = {"rotate_gradients": False} if os.environ.get("GRICS_COMPAT") == "1" else {}   # 2.0 Jacobian (no rotation factor)
    J = MotionPerturbationSimulator(smaps, Ns, sampling, Nex, image, mo, "one_sided", **J_KW)
    x = image.flatten()
    y = (torch.randn(Nc * Nex * Ns, generator=g) + 1j * torch.randn(Nc * Nex * Ns, generator=g)).to(torch.complex128).to(DEV)
    d = (torch.randn(alpha.numel(), generator=g) + 1j * torch.randn(alpha.numel(), generator=g)).to(torch.complex128).to(DEV)
    out = {"E_forward": E.forward(x), "E_adjoint": E.adjoint(y), "E_normal": E.normal(x),
           "J_forward": J.forward(d), "J_adjoint": J.adjoint(y), "J_normal": J.normal(d)}
    res[name] = {k: v.cpu() for k, v in out.items()}
    if Nx >= 128:
        times[name] = {"E_forward": timeit(lambda: E.forward(x)), "E_adjoint": timeit(lambda: E.adjoint(y)),
                       "E_normal": timeit(lambda: E.normal(x)), "J_forward": timeit(lambda: J.forward(d)),
                       "J_adjoint": timeit(lambda: J.adjoint(y)), "J_normal": timeit(lambda: J.normal(d)),
                       "build_MotionOperator": timeit(lambda: MotionOperator(Nx, Ny, alpha, motion, motion_signal=ms, Nz=Nz, **MO_KW), reps=2)}
        print(name, {k: f"{v*1e3:.1f} ms" for k, v in times[name].items()}, flush=True)
    del E, J, mo; torch.cuda.empty_cache()
torch.save({"results": res, "times": times}, OUT)
print("saved", OUT)

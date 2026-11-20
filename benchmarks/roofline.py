"""Roofline-style analysis of the GRICS-torch operators on this GPU.
import os

1. Measures the machine ceilings: device memory bandwidth (copy), FP32/FP64 matmul throughput, FFT throughput.
2. For each operator call (E.forward/adjoint/normal, J.forward/adjoint) builds a traffic model: the minimum number of
   bytes that must cross device memory for the batched formulation (one read + one write per elementwise pass,
   gather traffic for the warp), hence a bandwidth-bound time, and compares it with the measured time of the
   reference code, the layer-1 code and the layer-2 code (single precision + gather).
Usage: python roofline.py <repo_reference> <repo_fast> <repo_fast2> <out.json>
"""
import sys, time, json, importlib, math
from pathlib import Path
import torch
dev = torch.device("cuda")
torch.set_default_dtype(torch.float64)
def sync(): torch.cuda.synchronize()
def timeit(fn, reps=5, warm=1):
    for _ in range(warm): fn()
    sync(); t = time.perf_counter()
    for _ in range(reps): fn()
    sync(); return (time.perf_counter() - t) / reps

# ---------------- machine ceilings ----------------
def ceilings():
    out = {}
    x = torch.empty(2**28, dtype=torch.float32, device=dev)  # 1 GiB
    y = torch.empty_like(x)
    t = timeit(lambda: y.copy_(x), reps=10)
    out["bandwidth_GBs_copy"] = 2 * x.numel() * 4 / t / 1e9
    t = timeit(lambda: x.mul_(1.0001), reps=10)
    out["bandwidth_GBs_inplace_scale"] = 2 * x.numel() * 4 / t / 1e9
    del x, y
    for dt, name in [(torch.float32, "fp32"), (torch.float64, "fp64")]:
        a = torch.randn(8192, 8192, dtype=dt, device=dev)
        t = timeit(lambda: a @ a, reps=3)
        out[f"matmul_TFLOPS_{name}"] = 2 * 8192**3 / t / 1e12
        del a
    for shape in [(128, 128, 64), (192, 192, 56), (256, 208, 160)]:
        for dt, name in [(torch.complex64, "c64"), (torch.complex128, "c128")]:
            x = torch.randn(4, *shape, dtype=dt, device=dev)
            t = timeit(lambda: torch.fft.fftn(x, dim=(1, 2, 3), norm="ortho"), reps=5)
            n = x.numel()
            out[f"fft3_{'x'.join(map(str, shape))}_{name}"] = {"ms_4coils": t * 1e3, "effective_GBs": 2 * n * x.element_size() / t / 1e9,
                                                           "GFLOPS": 5 * n * math.log2(math.prod(shape)) / t / 1e9}
            del x
    torch.cuda.empty_cache()
    return out

# ---------------- traffic model per operator call ----------------
def traffic_model(Nx, Ny, Nz, Nc, Nm, cbytes, K):
    """Bytes per call for the batched formulation. cbytes = bytes per complex element (8 or 16); K = 4 (2D) or 8 (3D).
    Gather warp: read x gathered K times (K*N*cbytes) + idx (K*N*8) + w (K*N*cbytes/2) + write N*cbytes.
    FFT over coils: 2 passes (read+write) per transform of Nc*N elements (lower bound: one pass)."""
    N = Nx * Ny * Nz
    warp = K * N * cbytes + K * N * 8 + K * N * cbytes // 2 + N * cbytes
    warp_T = warp  # index_add: same gather traffic plus the output read-modify-write (approximated as equal)
    coil_mult = N * cbytes + Nc * N * cbytes + Nc * N * cbytes          # read x, read smaps, write Nc images
    fft = 2 * Nc * N * cbytes                                           # one read + one write per transform (lower bound)
    sample = Nc * N * cbytes                                            # gather/scatter of samples (worst case all lines)
    mask = 2 * Nc * N * cbytes
    conj_sum = 2 * Nc * N * cbytes + N * cbytes
    E_forward = Nm * (warp + coil_mult + fft + sample)
    E_adjoint = Nm * (sample + fft + conj_sum + warp_T)
    E_normal = Nm * (warp + coil_mult + fft + mask + fft + conj_sum + warp_T)
    J_forward = Nm * (3 * N * cbytes + coil_mult + fft + sample)        # gradients cached: read Gx,Gy(,Gz) + du fields
    J_adjoint = Nm * (sample + fft + conj_sum + 3 * N * cbytes)
    return {"E_forward": E_forward, "E_adjoint": E_adjoint, "E_normal": E_normal, "J_forward": J_forward, "J_adjoint": J_adjoint}

# ---------------- operator timings for one repository ----------------
def bench_repo(repo, Nx, Ny, Nz, Nc, Nm, motion, single=False, gather=False):
    for m in [k for k in sys.modules if k == "src" or k.startswith("src.")]: del sys.modules[m]
    importlib.invalidate_caches()
    sys.path.insert(0, str(repo))
    MotionOperator = importlib.import_module("src.reconstruction.MotionOperator").MotionOperator
    EncodingOperator = importlib.import_module("src.reconstruction.EncodingOperator").EncodingOperator
    MPS = importlib.import_module("src.reconstruction.MotionPerturbationSimulator").MotionPerturbationSimulator
    MSP = importlib.import_module("src.preprocessing.MotionSamplingPreparer").MotionSamplingPreparer
    sys.path.remove(str(repo))
    g = torch.Generator().manual_seed(0)
    smaps = (torch.randn(Nc, Nx, Ny, Nz, generator=g) + 1j * torch.randn(Nc, Nx, Ny, Nz, generator=g)).to(torch.complex128).to(dev)
    image = (torch.randn(Nx, Ny, Nz, generator=g) + 1j * torch.randn(Nx, Ny, Nz, generator=g)).to(torch.complex128).to(dev)
    if Nz == 1: image = image[..., 0]
    if Nz > 1:
        pairs = torch.cartesian_prod(torch.arange(Ny), torch.arange(Nz)); pairs = pairs[torch.randperm(pairs.shape[0], generator=g)]
        ch = torch.tensor_split(pairs, Nm); bky = [[c[:, 0].to(dev) for c in ch]]; bkz = [[c[:, 1].to(dev) for c in ch]]
    else:
        ch = torch.tensor_split(torch.randperm(Ny, generator=g), Nm); bky = [[c.to(dev) for c in ch]]; bkz = None
    sampling = MSP.build_sampling_per_nex_per_motion(bky, dev, Nx, Ny, Nz=Nz, binned_kz_indices=bkz)
    dim = 3 if Nz > 1 else 2
    if motion == "rigid":
        alpha = (0.5 * torch.randn(6 if dim == 3 else 3, Nm, generator=g)).to(dev); alpha[dim:] *= 0.05; ms = None
    else:
        shp = (dim, Nx, Ny, Nz, 1) if dim == 3 else (dim, Nx, Ny, 1)
        alpha = (0.5 * torch.randn(*shp, generator=g)).to(dev); ms = torch.randn(Nm, 1, generator=g).to(dev)
    kw = {}
    if single: kw["compute_dtype"] = torch.complex64
    if gather: kw["warp_backend"] = "gather"
    torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
    mo = MotionOperator(Nx, Ny, alpha, motion, motion_signal=ms, Nz=Nz, **kw)
    mem_ops = (torch.cuda.memory_allocated() - m0) / 2**20
    Ns = Nx * Ny * Nz
    E = EncodingOperator(smaps, Ns, sampling, 1, mo); J = MPS(smaps, Ns, sampling, 1, image, mo, "one_sided")
    x = image.flatten(); y = E.forward(x); d = torch.randn(alpha.numel(), dtype=torch.complex128, device=dev)
    res = {"E_forward": timeit(lambda: E.forward(x)), "E_adjoint": timeit(lambda: E.adjoint(y)), "E_normal": timeit(lambda: E.normal(x)),
           "J_forward": timeit(lambda: J.forward(d)), "J_adjoint": timeit(lambda: J.adjoint(y)),
           "build_s": timeit(lambda: MotionOperator(Nx, Ny, alpha, motion, motion_signal=ms, Nz=Nz, **kw), reps=2),
           "operator_memory_MB": mem_ops, "peak_MB": torch.cuda.max_memory_allocated() / 2**20}
    del E, J, mo, smaps, image, x, y; torch.cuda.empty_cache()
    return res

def main():
    ref, fast, fast2, out = [Path(a) for a in sys.argv[1:4]] + [Path(sys.argv[4])]
    R = {"ceilings": ceilings()}
    bw = R["ceilings"]["bandwidth_GBs_copy"] * 1e9
    print("ceilings:", json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in R["ceilings"].items()}, indent=0)[:1200], flush=True)
    cases = [("2D 128x128, 8 coils, 8 states, rigid", 128, 128, 1, 8, 8, "rigid"),
             ("2D 320x532, 24 coils, 4 states, non-rigid (Zenodo slice size)", 320, 532, 1, 24, 4, "non-rigid"),
             ("3D 128x128x64, 4 coils, 8 states, rigid", 128, 128, 64, 4, 8, "rigid"),
             ("3D 192x192x56, 24 coils, 16 states, non-rigid (paper 3D size class)", 192, 192, 56, 24, 16, "non-rigid")]
    R["cases"] = {}
    for name, Nx, Ny, Nz, Nc, Nm, motion in cases:
        K = 8 if Nz > 1 else 4
        row = {"model_bytes_c128": traffic_model(Nx, Ny, Nz, Nc, Nm, 16, K), "model_bytes_c64": traffic_model(Nx, Ny, Nz, Nc, Nm, 8, K)}
        for tag, repo, single, gather in [("reference", ref, False, False), ("batched_double", fast, False, False), ("batched_single_gather", fast2, True, True)]:
            try:
                row[tag] = bench_repo(repo, Nx, Ny, Nz, Nc, Nm, motion, single, gather)
            except RuntimeError as e:
                row[tag] = {"error": str(e)[:120]}
            torch.cuda.empty_cache()
        R["cases"][name] = row
        print(f"\n{name}")
        for op in ["E_forward", "E_adjoint", "E_normal", "J_forward", "J_adjoint"]:
            b128 = row["model_bytes_c128"][op] / bw * 1e3; b64 = row["model_bytes_c64"][op] / bw * 1e3
            line = f"  {op:10s} roofline bound {b128:6.1f} ms (c128) / {b64:6.1f} ms (c64) |"
            for tag in ["reference", "batched_double", "batched_single_gather"]:
                r = row[tag]
                if "error" in r: line += f" {tag}: {r['error'][:30]} |"; continue
                bound = b64 if tag.endswith("gather") else b128
                line += f" {tag}: {r[op]*1e3:7.1f} ms ({100*bound/(r[op]*1e3):4.0f} % of roofline) |"
            print(line, flush=True)
        for tag in ["reference", "batched_double", "batched_single_gather"]:
            r = row[tag]
            if "error" not in r: print(f"  {tag:22s} operators {r['operator_memory_MB']:7.0f} MB, peak {r['peak_MB']:7.0f} MB, build {r['build_s']*1e3:6.1f} ms")
    out.write_text(json.dumps(R, indent=1, default=float)); print("saved", out)

if __name__ == "__main__":
    main()

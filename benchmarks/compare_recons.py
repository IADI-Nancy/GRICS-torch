"""Compare recon_dump.py outputs of two repositories: relative image difference, motion difference, timings.
Usage: python compare_recons.py <dir_ref> <dir_fast> case1 case2 ...
"""
import sys, torch
from pathlib import Path
ref, fast = Path(sys.argv[1]), Path(sys.argv[2])
for case in sys.argv[3:]:
    a, b = torch.load(ref / f"{case}.pt"), torch.load(fast / f"{case}.pt")
    rel_img = ((a["image"] - b["image"]).norm() / a["image"].norm()).item()
    rel_mot = ((a["motion"] - b["motion"]).norm() / (a["motion"].norm() + 1e-300)).item()
    line = f"{case:12s} image rel diff {rel_img:.2e} | motion rel diff {rel_mot:.2e} | run() {a['elapsed_s']:.1f} s -> {b['elapsed_s']:.1f} s ({a['elapsed_s']/b['elapsed_s']:.1f}x)"
    if a.get("gt") is not None:
        gt = a["gt"]; na = ((a["image"].abs() - gt.abs()).norm() / gt.abs().norm()).item(); nb = ((b["image"].abs() - gt.abs()).norm() / gt.abs().norm()).item()
        line += f" | raw NRMSE vs truth {na:.4f} / {nb:.4f}"
    print(line)
    print(f"{'':12s} ACCEPTANCE (NRMSE vs reference <= 1e-4): {'PASS' if rel_img <= 1e-4 else 'FAIL'}")

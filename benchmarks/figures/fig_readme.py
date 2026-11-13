"""README figure: time per reconstruction, upstream vs this code, on two GPUs.
Usage: python fig_readme.py <timings_rtx> <timings_a100> <out_stem>   (timings_* = benchmarks/results/<gpu>/recon_timings, written by recon_profile runs)"""
import sys, json, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = sys.argv[3]
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#b5b4ad", "#ffffff", "#e8e7e2"
VARIANTS = [("reference", "GRICS-torch (IADI-Nancy, main)", "#2a78d6"),
            ("batched_double_sparse", "this code, default settings (same images)", "#eb6834"),
            ("batched_single_gather", "this code, single precision + gather", "#eda100")]
CASES = [("2d_rigid", "2D rigid\n128²"), ("2d_nonrigid", "2D non-rigid\n128²"), ("3d_rigid", "3D rigid\n128×128×64"),
         ("3d_nonrigid", "3D non-rigid\n128×128×64"), ("real", "breast slice\n320×532, 24 coils")]
GPUS = [(sys.argv[1], "RTX PRO 6000 Blackwell"), (sys.argv[2], "A100 PCIe 40 GB")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def load(d):
    return {(r["tag"], r["case"]): r["elapsed_s"] for r in (json.load(open(p)) for p in glob.glob(str(Path(d) / "*.json")))}


fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURF)
for ax, (d, gpu) in zip(axes, GPUS):
    T = load(d)
    ax.set_facecolor(SURF); ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5); ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    n, w = len(CASES), 0.26
    for i, (tag, _, col) in enumerate(VARIANTS):
        vals = [T.get((tag, c), 0) for c, _ in CASES]
        bars = ax.bar([k + (i - 1) * (w + 0.02) for k in range(n)], vals, width=w, color=col, linewidth=0)
        for k, (b, v) in enumerate(zip(bars, vals)):
            if not v:
                continue
            ref = T.get(("reference", CASES[k][0]))
            label = f"{v:.1f} s" if tag == "reference" else f"{v:.1f} s\n{ref / v:.1f}×"
            ax.text(b.get_x() + b.get_width() / 2, v, label, ha="center", va="bottom", fontsize=7.5, color=INK2, linespacing=1.05)
    ax.set_xticks(range(n)); ax.set_xticklabels([l for _, l in CASES], fontsize=8.5, color=INK)
    ax.set_ylabel("time per reconstruction (s)", fontsize=9, color=INK2)
    ax.set_ylim(0, max(T.get(("reference", c), 0) for c, _ in CASES) * 1.25)
    ax.set_title(gpu, fontsize=10.5, color=INK, loc="left")
fig.legend(handles=[Patch(color=c, label=l) for _, l, c in VARIANTS], loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=300 if ext == "png" else None, facecolor=SURF)
print("saved", OUT)

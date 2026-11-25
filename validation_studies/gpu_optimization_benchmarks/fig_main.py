"""Single publication figure. (a, b) reconstructions: 3D rigid Shepp-Logan phantom (central slice) and the free-breathing
breast slice; columns: uncorrected, GRICS-torch, GRICS-torch-acc, |difference| with NRMSE. (c) reconstruction time per case.
(d) speed-up per operator call. Usage: python fig_main.py <bench_dir> <ref_subdir> <acc_subdir> <ops_ref.pt> <ops_acc.pt> <out_stem> [gpu label]"""
import sys, numpy as np, torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
B = Path(sys.argv[1]); REF, ACC = sys.argv[2], sys.argv[3]; OPS_REF, OPS_ACC = sys.argv[4], sys.argv[5]; OUT = sys.argv[6]
REF_NAME, ACC_NAME = "GRICS-torch 2.0", "Current"
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#b5b4ad", "#ffffff", "#e8e7e2"
C_REF, C_ACC = "#2a78d6", "#eb6834"
diff_cmap = LinearSegmentedColormap.from_list("blue_seq", ["#ffffff", "#9cc3ef", "#2a78d6", "#0b2f5e"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

def mag(t, transpose):
    a = t.abs().numpy()
    if a.ndim == 3: a = a[:, :, a.shape[2] // 2]
    return a.T if transpose else a

fig = plt.figure(figsize=(12.5, 9.6), facecolor=SURF)
gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.62, 1.0], hspace=0.18, wspace=0.05, left=0.075, right=0.985, top=0.95, bottom=0.085)
rows = [("3d_rigid", "a", "3D rigid motion\nShepp-Logan 128×128×64\ncentral slice", True),
        ("real", "b", "Breast MRI, free breathing\n320×532, 24 coils\nnon-rigid motion", False)]
col_titles = ["uncorrected", REF_NAME, ACC_NAME, f"|{ACC_NAME} − {REF_NAME}|"]
for r, (case, letter, label, tr) in enumerate(rows):
    a = torch.load(B / REF / f"{case}.pt"); c = torch.load(B / ACC / f"{case}.pt")
    ref = mag(a["image"], tr); acc = mag(c["image"], tr); unc = mag(a["uncorrected"], tr)
    d = mag(c["image"] - a["image"], tr); refmax = float(ref.max()); vmax = np.percentile(ref, 99.7)
    nrmse = float((c["image"] - a["image"]).norm() / a["image"].norm())
    for k, img in enumerate([unc, ref, acc, d]):
        ax = fig.add_subplot(gs[r, k]); ax.set_facecolor(SURF)
        if k < 3:
            ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        else:
            ax.imshow(img / refmax, cmap=diff_cmap, vmin=0, vmax=max(img.max() / refmax, 1e-16), interpolation="nearest")
            ax.text(0.03, 0.04, f"NRMSE = {nrmse:.1e}", transform=ax.transAxes, fontsize=8.5,
                    color=INK, va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor("#dddcd6")
        if r == 0: ax.set_title(col_titles[k], fontsize=9.5, color=INK, pad=8)
        if k == 0:
            ax.text(-0.12, 1.0, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, ha="center", va="top")
            ax.text(-0.12, 0.5, label, transform=ax.transAxes, rotation=90, fontsize=8.2, color=INK, ha="center", va="center", linespacing=1.3)

# ---- (c) reconstruction time per case ----
cases = [("2d_rigid", "2D rigid\n128²"), ("2d_nonrigid", "2D non-rigid\n128²"), ("3d_rigid", "3D rigid\n128×128×64"),
         ("3d_nonrigid", "3D non-rigid\n128×128×64"), ("real", "breast MRI\n320×532, 24 coils")]
T = {tag: [torch.load(B / d / f"{c}.pt")["elapsed_s"] for c, _ in cases] for tag, d in (("ref", REF), ("acc", ACC))}
def style(ax):
    ax.set_facecolor(SURF); ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5); ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
ax = fig.add_subplot(gs[2, 0:4]); style(ax); n, w = len(cases), 0.36
b1 = ax.bar([k - w / 2 - 0.01 for k in range(n)], T["ref"], width=w, color=C_REF, linewidth=0, label=REF_NAME)
b2 = ax.bar([k + w / 2 + 0.01 for k in range(n)], T["acc"], width=w, color=C_ACC, linewidth=0, label=ACC_NAME)
for k, (bb, tt) in enumerate(zip(b1, T["ref"])):
    ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.012 * max(T["ref"]), f"{tt:.0f} s", ha="center", va="bottom", fontsize=8, color=INK2)
for k, (bb, tt) in enumerate(zip(b2, T["acc"])):
    sp = T["ref"][k] / tt
    ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.012 * max(T["ref"]), f"{tt:.1f} s\n{sp:.0f}×" if sp >= 10 else f"{tt:.1f} s\n{sp:.1f}×", ha="center", va="bottom", fontsize=8, color=INK2, linespacing=1.1)
ax.set_xticks(range(n)); ax.set_xticklabels([l for _, l in cases], fontsize=8.5, color=INK)
ax.set_ylabel("time per reconstruction (s)", fontsize=9, color=INK2); ax.set_ylim(0, max(T["ref"]) * 1.2)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK, handlelength=1.2)
ax.text(-0.045, 1.02, "c", transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, ha="right", va="bottom")

GPU = sys.argv[7] if len(sys.argv) > 7 else "one GPU"
fig.text(0.075, 0.015, f"NRMSE: relative L2 norm of the difference, {REF_NAME} as reference. Timings on {GPU}, both codes run back to back on each case.", fontsize=7.6, color=INK2)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=600 if ext == "png" else None, facecolor=SURF)
print("saved", OUT)

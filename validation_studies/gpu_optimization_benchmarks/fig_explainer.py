"""Schematic: one picture per optimisation, for readers new to GPU programming.
GRICS-torch 2.0 (blue) vs the current implementation (orange). Usage: python fig_explainer.py <out_stem>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
OUT = sys.argv[1]
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#b5b4ad", "#ffffff"
BLUE, ORANGE, BLUE_L, ORANGE_L, BLUE_M, ORANGE_M, GREY_L = "#2a78d6", "#eb6834", "#e6f0fb", "#fdeee6", "#7fa9e0", "#f3a27d", "#f2f1ee"
REF, ACC = "GRICS-torch 2.0", "Current"
plt.rcParams.update({"font.family": "DejaVu Sans", "hatch.linewidth": 0.6})
FW, FH = 14.0, 11.6
fig = plt.figure(figsize=(FW, FH), facecolor=SURF)

def new_panel(left, bottom, width, height, letter, title):
    ax = fig.add_axes([left, bottom, width, height]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.kx, ax.ky = 100 / (width * FW), 100 / (height * FH)          # data units per inch, to draw true squares
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 99, 99, boxstyle="round,pad=0,rounding_size=1.2", fc="#fcfcfb", ec="#d6d5cf", lw=0.9))
    ax.text(2.5, 94, letter, fontsize=12, fontweight="bold", color=INK, ha="left", va="center")
    ax.text(6.5 if width < 0.6 else 4.3, 94, title, fontsize=10.5, fontweight="bold", color=INK, ha="left", va="center")
    return ax
def T(ax, x, y, s, fs=7.4, c=INK, ha="left", va="center", bold=False, ls=1.3, bbox=None):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, fontweight="bold" if bold else "normal", linespacing=ls, bbox=bbox)
def arr(ax, x0, y0, x1, y1, c=INK2, lw=0.9, ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw))
def bar(ax, x, y, w, h, label, fc, fs=6.2, tc="#ffffff", hatch=None, ec="none"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.6, hatch=hatch))
    if label:
        bb = dict(fc="#ffffff", ec="none", pad=0.8) if hatch else None
        ax.text(x + w / 2, y + h / 2, label, fontsize=fs, color=tc, ha="center", va="center", bbox=bb)
def grid(ax, x, y, n, side_in, fc, ec):
    dx, dy = side_in * ax.kx, side_in * ax.ky
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x + j * dx, y + i * dy), dx, dy, fc=fc, ec=ec, lw=0.5))
    return n * dx, n * dy
def rbox(ax, x, y, w, h, fc, ec, lw=0.9, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=lw, ls=ls))
def tag(ax, x, y, name, color): T(ax, x, y, name, fs=8.6, c=color, bold=True)

L1, L2, W2, GAP = 0.015, 0.51, 0.475, 0.014
H1 = 0.30; H3 = 0.33
Y1 = 1 - 0.012 - H1; Y2 = Y1 - GAP - H1; Y3 = Y2 - GAP - H3

# ===================================================================================== 1. batched coils
ax = new_panel(L1, Y1, W2, H1, "1", "Batched coils: every GPU call pays a fixed setup cost")
tag(ax, 3, 86, REF, BLUE)
for c in range(3):
    gx = 4 + c * 11
    gw, gh = grid(ax, gx, 68, 3, 0.14, BLUE_L, BLUE); T(ax, gx + gw / 2, 64.5, f"coil {c+1}", fs=6.4, ha="center")
    arr(ax, gx + gw / 2, 62, gx + gw / 2, 58.5)
    rbox(ax, gx - 0.4, 51, gw + 0.8, 6.5, BLUE, BLUE); T(ax, gx + gw / 2, 54.2, "FFT", fs=6.6, c="#ffffff", ha="center", bold=True)
T(ax, 3, 44, "time", fs=6.4, c=INK2)
x = 3
for c in range(3):
    bar(ax, x, 35, 5, 6, "setup", "#ffffff", fs=5.4, tc=BLUE, hatch="////", ec=BLUE); bar(ax, x + 5, 35, 5, 6, "work", BLUE, fs=5.4); x += 11
T(ax, 3, 26, "3 calls: 3 setups", fs=7.6, c=INK2)
tag(ax, 54, 86, ACC, ORANGE)
for c in range(3):
    gw, gh = grid(ax, 56 + c * 1.2, 68 + c * 2.0, 3, 0.14, ORANGE_L, ORANGE)
cxm = 56 + 1.2 + gw / 2
T(ax, cxm, 64.5, "all coils", fs=6.4, ha="center")
arr(ax, cxm, 62, cxm, 58.5)
rbox(ax, 56, 51, 11, 6.5, ORANGE, ORANGE); T(ax, 61.5, 54.2, "FFT", fs=6.6, c="#ffffff", ha="center", bold=True)
T(ax, 54, 44, "time", fs=6.4, c=INK2)
bar(ax, 54, 35, 5, 6, "setup", "#ffffff", fs=5.4, tc=ORANGE, hatch="////", ec=ORANGE); bar(ax, 59, 35, 15, 6, "work", ORANGE, fs=5.4)
T(ax, 54, 26, "1 call: 1 setup, same work", fs=7.6, c=INK2)
T(ax, 3, 12, "breast data, 24 coils: 24 FFT calls → 1 per motion state, in every operator call", fs=7.2, c=INK2)

# ===================================================================================== 2. compute once
ax = new_panel(L2, Y1, W2, H1, "2", "Compute once, reuse inside the CG loop")
def flow(ax, x, items, color, light, loop_idx, ys=(70, 59, 48)):
    for (lab, inside), cy in zip(items, ys):
        rbox(ax, x, cy - 3.5, 30, 7, color if inside else light, color)
        T(ax, x + 15, cy, lab, fs=6.8, c="#ffffff" if inside else INK, ha="center", bold=inside)
    for i in range(len(ys) - 1):
        arr(ax, x + 15, ys[i] - 3.5, x + 15, ys[i + 1] + 3.5)
    top, bot = ys[loop_idx[0]] + 3.5 + 1.8, ys[loop_idx[-1]] - 3.5 - 1.8
    rbox(ax, x - 1.8, bot, 33.6, top - bot, "none", color, lw=0.9, ls=(0, (3, 2)))
    T(ax, x + 33.5, (top + bot) / 2, "repeat\n≤ 10 ×", fs=7.0, c=color, bold=True)
tag(ax, 3, 86, REF, BLUE)
flow(ax, 4, [("warp the image  W·x", True), ("gradient  ∇(W·x)", True), ("CG step", True)], BLUE, BLUE_L, (0, 2))
T(ax, 3, 34, "motion step: warp and gradient redone at every CG step", fs=7.6, c=INK2)
tag(ax, 54, 86, ACC, ORANGE)
flow(ax, 55, [("warp the image  W·x", False), ("gradient  ∇(W·x)", False), ("CG step", True)], ORANGE, ORANGE_L, (2, 2))
T(ax, 54, 34, "done once per Gauss–Newton iteration", fs=7.6, c=INK2)
T(ax, 3, 12, "same for the transposed warp Wᵀ, which GRICS-torch rebuilt and re-sorted at every adjoint call", fs=7.2, c=INK2)

# ===================================================================================== 3. pre-sorted sparse warps
ax = new_panel(L1, Y2, W2, H1, "3", "Pre-sorted sparse warps: write the matrix entries in row order")
def sparse_matrix(ax, x, y, order, color):
    dx, dy = 0.3 * ax.kx, 0.3 * ax.ky
    for r in range(4):
        for c in range(5):
            on = (r, c) in order
            ax.add_patch(Rectangle((x + c * dx, y + (3 - r) * dy), dx, dy, fc=color if on else "#ffffff", ec=MUTED, lw=0.5))
            if on: T(ax, x + c * dx + dx / 2, y + (3 - r) * dy + dy / 2, str(order[(r, c)]), fs=7.4, c="#ffffff", ha="center", bold=True)
    for r in range(4): T(ax, x - 1.2, y + (3 - r) * dy + dy / 2, f"row {r}", fs=6.0, c=INK2, ha="right")
    T(ax, x + 2.5 * dx, y + 4 * dy + 2.5, "5 input pixels", fs=6.0, c=INK2, ha="center")
    return 5 * dx, 4 * dy
tag(ax, 3, 86, REF, BLUE)
sparse_matrix(ax, 10, 40, {(0, 0): 1, (1, 1): 2, (2, 2): 3, (3, 3): 4, (0, 1): 5, (1, 2): 6, (2, 3): 7, (3, 4): 8}, BLUE)
T(ax, 3, 33, "entries written 1 … 8 neighbour by neighbour", fs=7.0)
bar(ax, 3, 21, 14, 6, "sort", BLUE, fs=6.2); T(ax, 18.5, 24, "→", fs=8); bar(ax, 21, 21, 9, 6, "W·x", BLUE, fs=6.2)
T(ax, 3, 12, "sorted once at build (3D) or before every product (2D)", fs=7.2, c=INK2)
tag(ax, 54, 86, ACC, ORANGE)
sparse_matrix(ax, 61, 40, {(0, 0): 1, (0, 1): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5, (2, 3): 6, (3, 3): 7, (3, 4): 8}, ORANGE)
T(ax, 54, 33, "entries written 1 … 8 row by row", fs=7.0)
bar(ax, 54, 21, 9, 6, "W·x", ORANGE, fs=6.2)
T(ax, 54, 12, "already in order: no sort;  3D build 312 → 39 ms", fs=7.2, c=INK2)

# ===================================================================================== 4. asynchronous CG
ax = new_panel(L2, Y2, W2, H1, "4", "Asynchronous CG: one check per iteration instead of seven")
widths = [14, 4, 4, 7, 4, 4, 5]; labels = ["A·p", "dot", "‖p‖", "x, r", "‖r‖", "cmp", "rz"]
def lanes(ax, y, name, color, light, sync_all):
    tag(ax, 3, y + 17, name, color)
    T(ax, 10, y + 10.5, "CPU checks", fs=6.2, c=INK2, ha="right"); T(ax, 10, y + 3.0, "GPU works", fs=6.2, c=INK2, ha="right")
    ax.add_patch(Rectangle((11, y + 8.5), 78, 4, fc="#f4f3ef", ec="none")); ax.add_patch(Rectangle((11, y + 1), 78, 4, fc="#f4f3ef", ec="none"))
    x = 11
    for k, (w, lab) in enumerate(zip(widths, labels)):
        bar(ax, x, y + 1, w * 1.08, 4, lab, color, fs=5.4); x += w * 1.08 + 0.6
        if sync_all or k == 6:
            bar(ax, x, y + 1, 2.8, 4, "", "#ffffff", hatch="////", ec=color)
            arr(ax, x + 1.4, y + 5.3, x + 1.4, y + 8.5, c=color, lw=0.8, ms=5)
            bar(ax, x + 0.4, y + 8.5, 2.0, 4, "", light); x += 3.6
lanes(ax, 56, REF, BLUE, BLUE_M, True); T(ax, 91, 59, "7 stops", fs=7, c=BLUE, bold=True)
T(ax, 11, 49, "hatched: the GPU idles while the CPU reads one number and decides", fs=6.8, c=INK2)
lanes(ax, 22, ACC, ORANGE, ORANGE_M, False); T(ax, 91, 25, "1 stop", fs=7, c=ORANGE, bold=True)
T(ax, 11, 15, "all the numbers of the iteration are read at once: same decisions, identical result", fs=6.8, c=INK2)

# ===================================================================================== 5. storage and precision
ax = new_panel(L1, Y3, 2 * W2 + (L2 - L1 - W2), H3, "5", "Warp storage and precision")
cx, cy = 5.5, 46
dx, dy = 0.75 * ax.kx, 0.75 * ax.ky; ox, oy = 0.32 * ax.kx, 0.32 * ax.ky
front = [(cx, cy), (cx + dx, cy), (cx + dx, cy + dy), (cx, cy + dy)]; back = [(x + ox, y + oy) for x, y in front]
for a, b in zip(front, back): ax.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=0.8, zorder=1)
ax.add_patch(Polygon(front, closed=True, fc="none", ec=MUTED, lw=0.8)); ax.add_patch(Polygon(back, closed=True, fc="none", ec=MUTED, lw=0.8))
ax.plot([p[0] for p in front + back], [p[1] for p in front + back], "o", ms=6.5, color=BLUE, zorder=3)
ax.plot([cx + dx * 0.55 + ox * 0.5], [cy + dy * 0.45 + oy * 0.5], "o", ms=8, color=ORANGE, zorder=4)
T(ax, cx + dx + ox + 3, cy + dy * 0.55, "one voxel of the warped image\n= weighted sum of its 8 neighbours", fs=7.2)
T(ax, 2, 40, "stored per voxel and per motion state:", fs=7.2, c=INK2)
def membar(ax, y, name, color, segs, total, note=""):
    tag(ax, 2, y + 9, name, color)
    x = 2
    for lab, b, fc in segs:
        w = b / 256 * 36
        ax.add_patch(Rectangle((x, y), w, 5.5, fc=fc, ec="#ffffff", lw=0.6))
        T(ax, x + w / 2, y - 2.6, lab, fs=6.0, c=INK2, ha="center")
        x += w
    T(ax, x + 1.2, y + 2.7, total, fs=7.6, bold=True)
    if note: T(ax, x + 1.2 + len(total) * 1.15, y + 2.7, note, fs=7.0, c=INK2)
membar(ax, 26, REF, BLUE, [("8 values, complex128", 128, BLUE), ("8 × 2 indices, int64", 128, BLUE_M)], "256 bytes")
membar(ax, 9, ACC, ORANGE, [("8 indices", 64, ORANGE), ("8 weights", 32, ORANGE_M)], "96 bytes", "2.7× smaller:  2.0 GB → 0.77 GB  (128×128×64, 8 states)")
rbox(ax, 64, 8, 34, 66, GREY_L, MUTED)
T(ax, 66, 69, "precision", fs=8.6, bold=True)
T(ax, 66, 61, "inside the operators (warp, coils, FFT)", fs=7.2, c=INK2)
T(ax, 66, 55, "3.1415927", fs=10, c=ORANGE, bold=True); T(ax, 66, 50, "complex64, 8 bytes", fs=6.8, c=INK2)
T(ax, 66, 41, "in the solver (CG, dot products, motion)", fs=7.2, c=INK2)
T(ax, 66, 35, "3.141592653589793", fs=10, c=BLUE, bold=True); T(ax, 66, 30, "complex128, 16 bytes", fs=6.8, c=INK2)
T(ax, 66, 19, "images agree to 1e-6", fs=7.6, bold=True)
fig.text(0.015, 0.012, "1–4: same arithmetic, identical images.   5: optional (TOML keys compute_precision, warp_backend), images within 1e-6.", fontsize=7.8, color=INK2)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=600 if ext == "png" else None, facecolor=SURF)
print("saved", OUT)

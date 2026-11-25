"""Schematic of the GRICS-torch reconstruction pipeline and of the optimisations of the current implementation.
Usage: python fig_schematic.py <out_stem>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
OUT = sys.argv[1]
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#b5b4ad", "#ffffff"
BLUE, ORANGE, BLUE_L, ORANGE_L, GREY_L = "#2a78d6", "#eb6834", "#e6f0fb", "#fdeee6", "#f2f1ee"
REF, ACC = "GRICS-torch 2.0", "Current"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig = plt.figure(figsize=(13.5, 8.6), facecolor=SURF)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, text, fc=GREY_L, ec=MUTED, fs=8.6, color=INK, lw=0.9, bold=False, align="center", pad=0.3, radius=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size={radius}", fc=fc, ec=ec, lw=lw, mutation_aspect=1.0))
    if text:
        tx = x + w / 2 if align == "center" else x + 1.0
        ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=fs, color=color, fontweight="bold" if bold else "normal", linespacing=1.35)

def arrow(x0, y0, x1, y1, color=INK2, lw=1.1, style="-|>", rad=0.0, ms=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))

def label(x, y, t, fs=9.5, color=INK, bold=True, ha="left"):
    ax.text(x, y, t, fontsize=fs, color=color, fontweight="bold" if bold else "normal", ha=ha, va="center")

# ------------------------------------------------------------------ row 1: the pipeline
label(2.5, 98.0, "a   GRICS-torch: joint image and motion reconstruction (unchanged)")
y1, h1 = 85.5, 7.0
box(2.5, y1, 15, h1, "multi-coil k-space  y\nphysiological signal", fc=GREY_L)
arrow(17.8, y1 + h1 / 2, 20.7, y1 + h1 / 2)
box(21, y1, 16, h1, "binning into Nₘ\nmotion states", fc=GREY_L)
arrow(37.3, y1 + h1 / 2, 40.2, y1 + h1 / 2)
box(40.5, 81.5, 44.5, 14.5, "", fc="#fafaf8", ec=MUTED, lw=1.0)
ax.text(42, 94.4, "multi-resolution Gauss–Newton (levels ¼, ½, 1)", fontsize=8.8, color=INK, ha="left", va="center", fontweight="bold")
box(42.5, 84.0, 17.5, 7.5, "image step\nCG on  (EᴴE + λ) x = Eᴴ y", fc=BLUE_L, ec=BLUE, fs=8.2)
box(66.0, 84.0, 17.5, 7.5, "motion step\nCG on  (JᴴJ + μR) δα = Jᴴ r", fc=BLUE_L, ec=BLUE, fs=8.2)
arrow(60.4, 89.6, 65.6, 89.6); arrow(65.6, 85.9, 60.4, 85.9)
ax.text(63.0, 91.0, "r = y − E x", fontsize=7.2, color=INK2, ha="center", va="bottom")
ax.text(63.0, 84.5, "α ← α + δα", fontsize=7.2, color=INK2, ha="center", va="top")
arrow(85.3, y1 + h1 / 2, 88.2, y1 + h1 / 2)
box(88.5, y1, 9.5, h1, "image x\nmotion α", fc=GREY_L)
ax.text(50, 79.0, "every CG iteration applies EᴴE (image step) or JᴴJ (motion step): one warp, coil weighting, FFT and sampling per motion state and per coil",
        fontsize=8, color=INK2, ha="center", va="center", style="italic")

# ------------------------------------------------------------------ row 2: one operator call
label(2.5, 75.0, "b   One application of the encoding operator E, per motion state s")
y2, h2 = 65.5, 7.0
chain = ["image x", "warp  Wₛ\nbilinear / trilinear", "coil maps  Sc\nc = 1 … Nc", "FFT", "sampling  Pₛ\nacquired lines of state s", "k-space  yₛ"]
xs = [2.5, 16.0, 34.0, 52.0, 63.5, 85.0]; ws = [11, 15.5, 15.5, 9, 19, 12.5]
for t, x, w in zip(chain, xs, ws):
    box(x, y2, w, h2, t, fc=GREY_L, ec=MUTED, fs=8.4)
for i in range(len(xs) - 1):
    arrow(xs[i] + ws[i] + 0.3, y2 + h2 / 2, xs[i + 1] - 0.3, y2 + h2 / 2)
ax.text(50, 62.3, "E x = Σₛ Pₛ F S Wₛ x          EᴴE = Σₛ Wₛᵀ Sᴴ Fᴴ PₛᵀPₛ F S Wₛ          J: first-order change of E x for a motion increment δα",
        fontsize=8.2, color=INK, ha="center", va="center")
box(2.5, 50.0, 46, 9.2, "", fc=BLUE_L, ec=BLUE, lw=1.0)
ax.text(4, 57.3, REF, fontsize=8.8, color=BLUE, fontweight="bold", ha="left", va="center")
ax.text(4, 53.4, "for state s:  for repetition:  for coil c:  one warp product, one FFT, one gather per coil.\n"
                 "Warped image, its gradients and the transposed warp are rebuilt at every call;\n7 device→host reads per CG iteration.",
        fontsize=7.7, color=INK, ha="left", va="center", linespacing=1.45)
box(51.5, 50.0, 46, 9.2, "", fc=ORANGE_L, ec=ORANGE, lw=1.0)
ax.text(53, 57.3, ACC, fontsize=8.8, color=ORANGE, fontweight="bold", ha="left", va="center")
ax.text(53, 53.4, "for state s:  all coils in one tensor [Nc, N], one batched FFT; all states in one call\nwhen the problem is small. "
                  "Warped image, gradients and transposed warp computed once\nper Gauss–Newton iteration; 1 device→host read per CG iteration.",
        fontsize=7.7, color=INK, ha="left", va="center", linespacing=1.45)

# ------------------------------------------------------------------ row 3: the optimisations
label(2.5, 45.5, "c   The optimisations, where they act, and what they change")
cards = [
    ("1  Batched coils", "one FFT call per motion state over\nthe [Nc, Nx, Ny(, Nz)] tensor instead\nof one FFT per coil; the same for\nthe adjoint and the normal operator",
     "image step, motion step", "identical result"),
    ("2  Compute once, reuse", "the warped image Wₛx, its spatial\ngradients and the transposed warp Wₛᵀ\nare built once per Gauss–Newton\niteration, not at every CG iteration\nor operator call",
     "image step, motion step", "identical result"),
    ("3  Pre-sorted sparse warps", "COO entries emitted row by row with\nincreasing columns (c, c+1, c+Ny,\nc+Ny+1) and flagged coalesced: no\nsort at build time and none before\neach product",
     "warp Wₛ", "identical result\n3D operator build 312 → 39 ms"),
    ("4  Asynchronous CG", "the scalars of one iteration\n(denominator, residual norm,\nstagnation test) are read in a single\ndevice→host transfer; same\ndecisions, same iterates",
     "both CG solvers", "bit-identical result"),
    ("5  Precision and memory", "complex64 inside the operators,\ncomplex128 kept in the CG and the\ndot products; warps stored as\nneighbour indices and weights,\napplied by gather and index_add",
     "warp, FFT, coil weighting", "NRMSE ≤ 1e-6 vs GRICS-torch\nstored operators 2.7× smaller"),
]
cw, gap, cx0, cy, ch = 18.2, 1.0, 2.5, 11.0, 29.5
for i, (title, body, where, effect) in enumerate(cards):
    x = cx0 + i * (cw + gap)
    box(x, cy, cw, ch, "", fc="#fafaf8", ec=MUTED, lw=0.9)
    ax.add_patch(Rectangle((x, cy + ch - 1.0), cw, 1.0, fc=ORANGE, ec="none"))
    ax.text(x + 0.9, cy + ch - 3.6, title, fontsize=9.2, color=INK, fontweight="bold", ha="left", va="center")
    ax.text(x + 0.9, cy + ch - 5.8, body, fontsize=7.6, color=INK, ha="left", va="top", linespacing=1.45)
    ax.text(x + 0.9, cy + 9.4, "acts on", fontsize=7.2, color=INK2, ha="left", va="center")
    ax.text(x + 0.9, cy + 7.3, where, fontsize=7.9, color=BLUE, ha="left", va="center", fontweight="bold")
    ax.text(x + 0.9, cy + 4.5, "effect", fontsize=7.2, color=INK2, ha="left", va="center")
    ax.text(x + 0.9, cy + 2.1, effect, fontsize=7.9, color=ORANGE, ha="left", va="center", fontweight="bold", linespacing=1.3)
ax.text(2.5, 6.0, "Optimisations 1–4 leave the arithmetic unchanged and reproduce the GRICS-torch images exactly; 5 is optional and agrees to 1e-6. "
        "Algorithm, regularisation, iteration counts and stopping rules are untouched.", fontsize=8, color=INK2, ha="left", va="center")
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=600 if ext == "png" else None, facecolor=SURF)
print("saved", OUT)

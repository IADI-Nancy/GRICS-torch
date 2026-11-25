"""Compare two dump_ops.py outputs: max relative difference per case/operator and timing ratios.
Usage: python compare_dumps.py ref.pt fast.pt
"""
import sys, torch
a = torch.load(sys.argv[1]); b = torch.load(sys.argv[2])
worst = 0.0
for case in a["results"]:
    row = []
    for k, va in a["results"][case].items():
        vb = b["results"][case][k]
        rel = ((va - vb).norm() / (va.norm() + 1e-300)).item()
        worst = max(worst, rel); row.append(f"{k}={rel:.1e}")
    print(f"{case:24s} " + "  ".join(row))
print(f"WORST relative difference: {worst:.2e}")
for case in a["times"]:
    if case in b["times"]:
        print(f"{case:24s} " + "  ".join(f"{k}: {a['times'][case][k]*1e3:.1f}->{b['times'][case][k]*1e3:.1f} ms ({a['times'][case][k]/b['times'][case][k]:.1f}x)" for k in a["times"][case]))

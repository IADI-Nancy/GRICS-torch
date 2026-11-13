"""Turn the outputs of run_all.sh into a markdown summary. Usage: python summarize_bench.py <out_dir> <summary.md>"""
import sys, json, glob
from pathlib import Path
import torch
B = Path(sys.argv[1]); OUT = Path(sys.argv[2]); L = []
OPS = [("reference", "ops_reference.pt"), ("batched, double/sparse", "ops_batched_double.pt"), ("batched, single/gather", "ops_batched_single_gather.pt")]
REC = [("reference", "recon_reference"), ("batched, double/sparse", "recon_batched_double"), ("batched, single/gather", "recon_batched_single")]
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
L.append(f"# GRICS-torch reference vs GRICS-torch-batched on {gpu}\n")
ops = {lab: torch.load(B / f) for lab, f in OPS if (B / f).exists()}
if "reference" in ops:
    ref = ops["reference"]
    L.append("## Operator calls, ms per call (speed-up); identity = max relative difference of outputs vs reference\n")
    for case in ["2d_rigid_128", "2d_nonrigid_128", "3d_rigid_128x128x64"]:
        L.append(f"**{case}**\n\n| operator | " + " | ".join(ops) + " |\n|---|" + "---|" * len(ops))
        for op in ["E_forward", "E_adjoint", "E_normal", "J_forward", "J_adjoint", "J_normal"]:
            r = ref["times"][case][op] * 1e3
            L.append(f"| {op} | {r:.1f} | " + " | ".join(f"{d['times'][case][op]*1e3:.1f} ({r/(d['times'][case][op]*1e3):.1f}x)" for lab, d in ops.items() if lab != "reference") + " |")
        ident = [f"{lab}: {max(((ref['results'][case][k] - d['results'][case][k]).norm() / (ref['results'][case][k].norm() + 1e-300)).item() for k in ref['results'][case]):.1e}" for lab, d in ops.items() if lab != "reference"]
        L.append(f"\nIdentity vs reference: {', '.join(ident)}\n")
if (B / "recon_reference").exists():
    L.append("## Full reconstructions: seconds for run() (speed-up, relative image difference vs reference)\n")
    L.append("| case | " + " | ".join(lab for lab, _ in REC) + " |\n|---|" + "---|" * len(REC))
    for case in ["2d_rigid", "2d_nonrigid", "3d_rigid", "3d_nonrigid", "real"]:
        a = torch.load(B / "recon_reference" / f"{case}.pt"); cells = [f"{a['elapsed_s']:.1f} s"]
        for lab, d in REC[1:]:
            p = B / d / f"{case}.pt"
            if p.exists():
                b = torch.load(p); rel = ((a["image"] - b["image"]).norm() / a["image"].norm()).item()
                cells.append(f"{b['elapsed_s']:.1f} s ({a['elapsed_s']/b['elapsed_s']:.1f}x, diff {rel:.0e})")
            else: cells.append("-")
        L.append(f"| {case} | " + " | ".join(cells) + " |")
if (B / "roofline.json").exists():
    R = json.loads((B / "roofline.json").read_text()); ce = R["ceilings"]; bw = ce["bandwidth_GBs_copy"]
    L.append(f"\n## Roofline model (bandwidth-bound time = modelled bytes / measured copy bandwidth {bw:.0f} GB/s)\n")
    L.append("| case | operator | bound c128 (ms) | reference | batched double/sparse | batched single/gather (bound c64) |\n|---|---|---|---|---|---|")
    for name, row in R["cases"].items():
        for op in ["E_normal", "J_forward", "J_adjoint"]:
            b128 = row["model_bytes_c128"][op] / (bw * 1e9) * 1e3; b64 = row["model_bytes_c64"][op] / (bw * 1e9) * 1e3; cells = [f"{b128:.1f}"]
            for tag, bound in [("reference", b128), ("batched_double", b128), ("batched_single_gather", b64)]:
                r = row.get(tag, {}); cells.append(f"{r[op]*1e3:.1f} ms ({100*bound/(r[op]*1e3):.0f} %)" if op in r else str(r.get("error", "-"))[:30])
            L.append(f"| {name} | {op} | " + " | ".join(cells) + " |")
pj = sorted(glob.glob(str(B / "paper/*_paper2d.json")))
if pj:
    L.append("\n## Zenodo breast slice with the paper's nonrigid_2d.toml, seconds for run() over 3 repeats\n\n| run | shape | coils | states | precision/backend | run() s | peak GiB |\n|---|---|---|---|---|---|---|")
    for p in pj:
        d = json.loads(Path(p).read_text())
        L.append(f"| {d['tag']} | {'x'.join(map(str, d['shape']))} | {d['coils']} | {d['states']} | {d['precision']}/{d['backend']} | {', '.join('%.1f' % t for t in d['run_s'])} | {d['peak_GiB']:.1f} |")
OUT.write_text("\n".join(L) + "\n"); print("wrote", OUT)

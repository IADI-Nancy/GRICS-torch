"""Reconstruct the Siemens 2D phantom with standard parameters and one motion state.

Run from the repository root:
    python tests/validate_siemens_phantom_2d.py --device gpu

All slices run by default. Use --slice-index for a single-slice execution check.
A .physio.txt sidecar is generated beside the raw input, containing one zero
per retained imaging readout with -1 marking the absence of timestamps.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.siemens_phantom import PHANTOM_ROOT, run_phantom

RAW_DATA_FILE = PHANTOM_ROOT / "meas_MID00088_FID552355_AX_T2_TSE_HR_2NEX_SAT.dat"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-file", type=Path, default=RAW_DATA_FILE)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--slice-index", type=int, help="Zero-based slice index; default: reconstruct every slice")
    args = parser.parse_args()
    run_folder = run_phantom(args.raw_data_file, "2D", device=args.device, output_root=args.output_root, slice_idx=args.slice_index)
    print(f"[phantom] Outputs: {run_folder}")


if __name__ == "__main__":
    main()

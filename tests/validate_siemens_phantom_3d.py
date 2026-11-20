"""Reconstruct the Siemens 3D phantom with standard parameters and one motion state.

Run from the repository root:
    python tests/validate_siemens_phantom_3d.py --device gpu

Uses the Siemens pipeline configuration nonrigid_3d_breast.toml, including
its Odille spline coil calibration and full solver schedule.
A .physio.txt sidecar is generated beside the raw input, containing one zero
per retained imaging readout with -1 marking the absence of timestamps.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.siemens_phantom import PHANTOM_ROOT, run_phantom

RAW_DATA_FILE = PHANTOM_ROOT / "meas_MID00085_FID552352_AX_T1_EG_3D_FS_MASQUE_LR.dat"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-file", type=Path, default=RAW_DATA_FILE)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    run_folder = run_phantom(args.raw_data_file, "3D", device=args.device, output_root=args.output_root)
    print(f"[phantom] Outputs: {run_folder}")


if __name__ == "__main__":
    main()

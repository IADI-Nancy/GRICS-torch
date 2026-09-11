import argparse
import sys
from pathlib import Path

if "__file__" in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    _REPO_ROOT = Path.cwd()
sys.path.insert(0, str(_REPO_ROOT))

from src.preprocessing.RawDataPreparer import RawDataPreparer


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Export only slice 15 from raw data into demo_breast_data.h5."
    )
    parser.add_argument(
        "--ismrmrd-file",
        default="data/t2_1724.h5",
        help="Path to ISMRMRD file (default: data/t2_1724.h5).",
    )
    parser.add_argument(
        "--saec-file",
        default="data/2008-003 01-1724_S11_20210323_151329.h5",
        help="Path to SAEC file (default: data/2008-003 01-1724_S11_20210323_151329.h5).",
    )
    parser.add_argument(
        "--output",
        default="demo_breast_data.h5",
        help="Output H5 filename (default: demo_breast_data.h5).",
    )
    parser.add_argument(
        "--slice-idx",
        type=int,
        default=15,
        help="Slice index to export (default: 15).",
    )
    parser.add_argument(
        "--sensor-type",
        default="BELT",
        help="Physiological sensor type (default: BELT).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for reading pipeline (default: cuda).",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    ismrmrd_path = Path(args.ismrmrd_file)
    saec_path = Path(args.saec_file)
    if not ismrmrd_path.exists() or not saec_path.exists():
        missing = []
        if not ismrmrd_path.exists():
            missing.append(str(ismrmrd_path))
        if not saec_path.exists():
            missing.append(str(saec_path))
        raise FileNotFoundError(
            "Missing input file(s): "
            + ", ".join(missing)
            + ".\nRun from the folder containing them, or pass --ismrmrd-file/--saec-file."
        )

    reader = RawDataPreparer(
        ismrmrd_file=str(ismrmrd_path),
        physiological_file=str(saec_path),
        sensor_type=args.sensor_type,
        device=args.device,
    )

    print(
        f"[Export] Reading raw data and saving only slice {args.slice_idx} "
        f"to {args.output}..."
    )
    data = reader.read_data(
        h5filename=args.output,
        slice_idx=args.slice_idx,
    )
    print(
        "[Export] Done. Shapes: "
        f"kspace={data['kspace'].shape}, motion_data={data['motion_data'].shape}, "
        f"idx_ky={data['idx_ky'].shape}, idx_kz={data['idx_kz'].shape}, "
        f"idx_nex={data['idx_nex'].shape}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Reconstruct a complete Siemens breast 3D volume from a .dat file and SAEC.

Prepare the volume, run GRICS, normalize it when configured, zero-fill it to the
header's reconstruction grid, and export a DICOM series and configured tensors.
The default compute device is GPU; runtime settings can be set through overrides
when calling run_pipeline() from Python.

Call from the repository root:
    python pipelines/siemens_breast_3d_lowres.py subject.dat subject.saec
    python pipelines/siemens_breast_3d_lowres.py subject.dat subject.saec --output-root runs/volume --dicom-header-dir donor_dicoms --dicom-series-number 1001

--dicom-header-dir is optional; acquisition geometry is used without a donor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.DataPreparer import DataPreparer
from pipelines.utils.postprocessing import (
    grics_reference_image_for_normalization, grics_zero_fill_volume_shapes,
    normalize_reconstruction_by_grics_reference, zero_fill_loaded_reconstruction,
)
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.output_layout import managed_execution, write_reconstruction
from src.runtime.runtime_config import load_config, load_postprocessing_config
from src.runtime.runtime_setup import initialize_runtime
from src.utils.dicom_export import write_volume_dicoms

OUTPUT_ROOT = Path("runs/siemens_breast_3d_lowres")
RECONSTRUCTION_CONFIG = "config/reconstruction/nonrigid_3d_breast.toml"
POSTPROCESSING_CONFIG = "config/postprocessing/nonrigid_3d_breast.toml"
RUNTIME_DEVICE = "gpu"


@managed_execution
def run_pipeline(raw_data_file, saec_file, *, output_root=OUTPUT_ROOT, device=RUNTIME_DEVICE,
                 reconstruction_config=RECONSTRUCTION_CONFIG, postprocessing_config=POSTPROCESSING_CONFIG, overrides=None,
                 dicom_header_dir=None, dicom_series_number=1001) -> dict:
    """Load Siemens raw data and SAEC, reconstruct the volume, and save outputs."""
    raw_data_file, saec_file = Path(raw_data_file), Path(saec_file)
    if raw_data_file.suffix.lower() != ".dat":
        raise ValueError("raw_data_file must be a Siemens .dat file.")
    for path in (raw_data_file, saec_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_root = Path(output_root)
    params = load_config(
        data_type="siemens-saec", reconstruction_config=REPO_ROOT / reconstruction_config,
        overrides={"output_root": str(output_root.parent), "workflow_label": output_root.name,
                   "runtime_device": device, "flip_for_display": True,
                   "verbose": False, "print_to_console": False, **(overrides or {})},
    )
    postprocessing = load_postprocessing_config(REPO_ROOT / postprocessing_config)
    if postprocessing.normalize_image_by_grics_reference and params.coil_sensitivity_method != "odille-spline":
        raise ValueError("Reference-image normalization requires coil_sensitivity_method='odille-spline'.")
    runtime = initialize_runtime(params)
    data = DataPreparer(params=params, runtime=runtime, filename=(str(raw_data_file), str(saec_file)))
    data.load_input_data()
    if data.input_loader.Nz <= 1:
        raise ValueError("The 3D pipeline requires more than one encoded partition.")
    target_shape, encoded_shape = grics_zero_fill_volume_shapes(data.input_loader)
    data.prepare_reconstructor_inputs()

    reconstructor = JointReconstructor.from_prepared_data(data)
    image, motion = reconstructor.run()
    native_image = image.detach().cpu()
    if postprocessing.normalize_image_by_grics_reference:
        reference_image = grics_reference_image_for_normalization(data, image)
        image, _ = normalize_reconstruction_by_grics_reference(image, reference_image)
    image = zero_fill_loaded_reconstruction(image, target_shape, encoded_shape)
    image, motion = image.detach().cpu(), motion.detach().cpu()
    dicom_files, _ = write_volume_dicoms(
        image, Path(params.run_folder) / "exports/dicom", data,
        series_number=dicom_series_number, reference_dicom_path=dicom_header_dir)
    paths = write_reconstruction(
        params, image, motion, native_image=native_image, image_stage="after_postprocessing",
        record_paths=True, status="complete", tensor_export="pipeline", dicom_files=dicom_files)
    return {"run_folder": Path(params.run_folder), "image": image, "native_image": native_image, "motion": motion,
            "dicom_files": dicom_files, **paths}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_data_file", type=Path, help="Siemens .dat file.")
    parser.add_argument("saec_file", type=Path, help="SAEC physiological file.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dicom-header-dir", type=Path)
    parser.add_argument("--dicom-series-number", type=int, default=1001)
    args = parser.parse_args(argv)
    result = run_pipeline(
        args.raw_data_file, args.saec_file, output_root=args.output_root, dicom_header_dir=args.dicom_header_dir,
        dicom_series_number=args.dicom_series_number)
    print(f"[run] Outputs: {result['run_folder']}")
    return result


if __name__ == "__main__":
    main()

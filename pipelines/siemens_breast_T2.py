#!/usr/bin/env python
"""Reconstruct every Siemens breast T2 slice from a .dat file and SAEC.

Load the acquisition once, reconstruct all slices with parallel CPU workers,
apply configured normalization and zero-filling, and export a DICOM series
and configured tensors. GPU execution and slice selection are not supported.

Call from the repository root:
    python pipelines/siemens_breast_T2.py subject.dat subject.saec
    python pipelines/siemens_breast_T2.py subject.dat subject.saec --output-root runs/t2 --dicom-header-dir donor_dicoms --dicom-series-number 1001

--dicom-header-dir is optional; acquisition geometry is used without a donor.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from pydicom.uid import generate_uid

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.utils.parallel import reconstruct_slices_in_parallel
from pipelines.utils.postprocessing import (
    grics_reference_image_for_normalization, grics_zero_fill_shapes,
    normalize_reconstruction_by_grics_reference, zero_fill_loaded_reconstruction,
)
from src.preprocessing.DataPreparer import DataPreparer
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.output_layout import managed_execution, public_config, write_reconstruction
from src.runtime.runtime_config import load_config, load_postprocessing_config
from src.runtime.runtime_setup import initialize_runtime
from src.utils.dicom_export import write_reconstruction_dicom

OUTPUT_ROOT = Path("runs/siemens_breast_T2")
RECONSTRUCTION_CONFIG = "config/reconstruction/nonrigid_2d_breast.toml"
POSTPROCESSING_CONFIG = "config/postprocessing/nonrigid_2d_breast.toml"


def reconstruct_slice(slice_idx, input_data) -> dict:
    """Prepare, reconstruct, and postprocess one slice of the shared acquisition."""
    data = copy.copy(input_data)
    data.params = copy.copy(input_data.params)
    data.prepare_reconstructor_inputs(slice_idx=slice_idx)

    reconstructor = JointReconstructor.from_prepared_data(data)
    image, motion = reconstructor.run()
    native_image = image.detach().cpu()
    if data.postprocessing.normalize_image_by_grics_reference:
        reference_image = grics_reference_image_for_normalization(data, image)
        image, _ = normalize_reconstruction_by_grics_reference(image, reference_image)
    target_shape, encoded_shape = data.zero_fill_shapes
    image = zero_fill_loaded_reconstruction(image, target_shape, encoded_shape)
    return {"slice_idx": slice_idx, "image": image.detach().cpu(), "native_image": native_image,
            "motion": motion.detach().cpu(), "_configuration": public_config(data.params)}


@managed_execution
def run_pipeline(raw_data_file, saec_file, *, output_root=OUTPUT_ROOT, max_workers=None,
                 reconstruction_config=RECONSTRUCTION_CONFIG, postprocessing_config=POSTPROCESSING_CONFIG,
                 overrides=None, dicom_header_dir=None, dicom_series_number=1001) -> dict:
    """Load Siemens raw data once, reconstruct all slices on CPU, and export DICOMs."""
    if (overrides or {}).get("runtime_device", "cpu") != "cpu":
        raise ValueError("The 2D T2 pipeline only supports CPU execution.")
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
                   "runtime_device": "cpu", "flip_for_display": True,
                   "verbose": False, "print_to_console": False, **(overrides or {})},
    )
    postprocessing = load_postprocessing_config(REPO_ROOT / postprocessing_config)
    runtime = initialize_runtime(params)
    data = DataPreparer(params=params, runtime=runtime, filename=(str(raw_data_file), str(saec_file)))
    data.load_input_data()
    data.postprocessing = postprocessing
    data.zero_fill_shapes = grics_zero_fill_shapes(data.input_loader)
    indices = list(range(int(data.input_loader.Nz)))
    data.params._output_unit_count = len(indices)
    results = reconstruct_slices_in_parallel(data, indices, reconstruct_slice, max_workers)

    uids = {key: generate_uid() for key in
            ("study_instance_uid", "series_instance_uid", "frame_of_reference_uid")}
    for result in results:
        params = copy.copy(data.params)
        vars(params).update(result.pop("_configuration"))
        dicom_path = Path(data.params.run_folder) / "exports/dicom" / f"slice_{result['slice_idx'] + 1:03d}.dcm"
        result["dicom_file"] = write_reconstruction_dicom(
            result["image"], dicom_path, raw_data=data.input_loader, slice_index=result["slice_idx"],
            series_description="GRICS reconstruction RESEARCH ONLY", images_in_acquisition=len(indices),
            series_number=dicom_series_number, reference_dicom_path=dicom_header_dir, **uids)
        result.update(write_reconstruction(
            params, result["image"], result["motion"], native_image=result["native_image"],
            image_stage="after_postprocessing", record_paths=True, status="complete",
            tensor_export="pipeline", dicom_file=result["dicom_file"]))
    return {"run_folder": Path(data.params.run_folder), "reconstructions": results}


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

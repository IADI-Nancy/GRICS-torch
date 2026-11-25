"""Run timed article evaluations using the current preparation and solver APIs."""
import copy
from pathlib import Path
import time

from pipelines.utils.execution import (
    export_reconstruction, finish_run, reconstruction_overrides, synchronize, timed_reconstruction,
)
from pipelines.utils.inputs import data_type_from_raw_data_file, publish_preprocessed_cache
from pipelines.utils.parallel import reconstruct_slices_in_parallel
from pipelines.utils.postprocessing import (
    grics_reference_image_for_normalization, grics_zero_fill_shapes,
    normalize_reconstruction_by_grics_reference, zero_fill_loaded_reconstruction,
)
from src.preprocessing.DataPreparer import DataPreparer
from src.runtime.output_layout import managed_execution, public_config
from src.runtime.runtime_config import load_config, load_postprocessing_config
from src.runtime.runtime_setup import initialize_runtime


def reconstruct_slice(slice_idx, input_data):
    """Measure one slice and preserve native images for sharpness comparisons."""
    started = time.perf_counter()
    data = copy.copy(input_data)
    data.params = copy.copy(input_data.params)
    data.prepare_reconstructor_inputs(slice_idx=slice_idx)
    synchronize(data.kspace.device)
    preprocessing_seconds = time.perf_counter() - started
    image, motion, reconstruction_seconds = timed_reconstruction(data)
    native_image = image.detach().cpu()
    started = time.perf_counter()
    if data.postprocessing.normalize_image_by_grics_reference:
        reference_image = grics_reference_image_for_normalization(data, image)
        image, _ = normalize_reconstruction_by_grics_reference(image, reference_image)
    target_shape, encoded_shape = data.zero_fill_shapes
    image = zero_fill_loaded_reconstruction(image, target_shape, encoded_shape)
    synchronize(image.device)
    return {"slice_idx": slice_idx, "slice_number": slice_idx + 1,
            "image": image.detach().cpu(), "native_image": native_image, "motion": motion.detach().cpu(),
            "preprocessing_seconds": preprocessing_seconds, "reconstruction_seconds": reconstruction_seconds,
            "postprocessing_seconds": time.perf_counter() - started,
            "image_stage": "after_postprocessing", "_configuration": public_config(data.params)}


@managed_execution
def run_reconstruction(input_file, saec_file, *, output_root, reconstruction_config, device,
                       overrides=None, max_workers=None, prepared_output_file=None):
    """Evaluate raw or prepared data without DICOM export; retain article timing semantics."""
    started = time.perf_counter()
    input_file = Path(input_file)
    saec_file = Path(saec_file) if saec_file is not None else None
    data_type = data_type_from_raw_data_file(input_file, saec_file)
    settings = reconstruction_overrides(output_root, device, overrides,
                                       save_reconstruction_logs=True, save_reconstruction_tensors=True)
    if prepared_output_file is not None and data_type != "preprocessed-real":
        settings["cache_preprocessed_data"] = True
    params = load_config(data_type=data_type, reconstruction_config=reconstruction_config, overrides=settings)
    runtime = initialize_runtime(params)
    filename = str(input_file) if data_type == "preprocessed-real" else (str(input_file), str(saec_file))
    data = DataPreparer(params=params, runtime=runtime, filename=filename)
    data.load_input_data()
    if prepared_output_file is not None and data_type != "preprocessed-real":
        publish_preprocessed_cache(data.input_loader.realworld_h5_path, prepared_output_file)
        print(f"[input] Saved prepared data: {prepared_output_file}", flush=True)
    if params.data_dimension == "2D":
        root = Path(__file__).resolve().parents[2]
        data.postprocessing = load_postprocessing_config(root / "config/postprocessing/nonrigid_2d_breast.toml")
        data.zero_fill_shapes = grics_zero_fill_shapes(data.input_loader)
        indices = list(range(int(data.input_loader.Nz)))
        params._output_unit_count = len(indices)
        preprocessing_seconds = time.perf_counter() - started
        compute_started = time.perf_counter()
        if params.runtime_device == "cpu":
            results = reconstruct_slices_in_parallel(data, indices, reconstruct_slice, max_workers)
        else:
            if max_workers not in (None, 1):
                raise ValueError("GPU slice reconstruction requires max_workers=1.")
            results = [reconstruct_slice(index, data) for index in indices]
        timings = {"preprocessing_seconds": preprocessing_seconds,
                   "compute_wall_seconds": time.perf_counter() - compute_started,
                   "reconstruction_seconds_sum": sum(item["reconstruction_seconds"] for item in results)}
    else:
        if int(data.input_loader.Nz) <= 1:
            raise ValueError("3D evaluation requires more than one partition.")
        data.prepare_reconstructor_inputs()
        synchronize(data.kspace.device)
        preprocessing_seconds = time.perf_counter() - started
        image, motion, reconstruction_seconds = timed_reconstruction(data)
        image = image.detach().cpu()
        results = [{"image": image, "native_image": image, "motion": motion.detach().cpu(),
                    "reconstruction_seconds": reconstruction_seconds, "image_stage": "before_postprocessing"}]
        timings = {"preprocessing_seconds": preprocessing_seconds, "reconstruction_seconds": reconstruction_seconds}
    for result in results:
        saved_params = copy.copy(params)
        vars(saved_params).update(result.pop("_configuration", {}))
        export_reconstruction(saved_params, result)
    return finish_run(data, results, timings, return_tensors=True)

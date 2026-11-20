"""Selection, validation, and diagnostics for reconstructor input preparation."""

from src.preprocessing.SamplingSimulator import SamplingSimulator
from src.runtime.output_layout import record_reconstruction
from src.runtime.runtime_config import (
    REAL_DATA_TYPES,
    validate_calibration_size,
    validate_motion_readout_count,
    validate_reconstruction_size,
)


def ensure_input_loaded_and_reconstruction_not_prepared(input_loader, reconstruction_data_prepared):
    """Require loaded input data and prevent repeated preparation."""
    if not input_loader.is_loaded:
        raise RuntimeError("DataPreparer.prepare_reconstructor_inputs() requires load_input_data() first.")
    if reconstruction_data_prepared:
        raise RuntimeError("DataPreparer.prepare_reconstructor_inputs() was already called.")


def select_input_slice_if_needed(input_loader, params, slice_idx=None):
    """Return the index of a 2D input slice that still needs to be selected."""
    if (
        slice_idx is None and input_loader.slice_idx is None
        and params.data_dimension == "2D"
        and params.data_type in REAL_DATA_TYPES
    ):
        if int(input_loader.Nz) == 1:
            slice_idx = 0
        else:
            raise ValueError(
                f"The loaded 2D input contains {int(input_loader.Nz)} slices; "
                "slice_idx must be specified explicitly."
            )
    return slice_idx


def validate_input_for_reconstruction(input_data, params):
    """Validate the selected input shape and reconstruction requirements."""
    if params.data_dimension == "2D" and int(input_data.Nz) != 1:
        raise ValueError(
            "DataPreparer.prepare_reconstructor_inputs() requires exactly one 2D slice, "
            f"but the loaded data has {int(input_data.Nz)} slices. "
            "Call prepare_reconstructor_inputs(slice_idx=...) to select a slice."
        )
    spatial_shape = (
        (input_data.Nx, input_data.Ny, input_data.Nz)
        if params.data_dimension == "3D"
        else (input_data.Nx, input_data.Ny)
    )
    validate_reconstruction_size(params, spatial_shape)
    validate_calibration_size(params, spatial_shape)
    validate_motion_readout_count(params, input_data.ky_idx.numel())


def save_sampling_order_debug_plot_if_enabled(input_data, params):
    """Save the recorded sampling order when debug plotting is enabled."""
    if not params.save_debug_plots or params.kspace_sampling_type != "from-data":
        return
    SamplingSimulator._visualize_ky_order(
        [input_data.ky_idx.detach().cpu()], Ny=input_data.Ny,
        folder=params.initial_data_folder,
        fname=f"ky_order_acquisition_slice{input_data.slice_idx}.png")


def record_preprocessing_start_if_configured(params, filename, slice_idx):
    """Record preprocessing status when an output run has been initialized."""
    if not hasattr(params, "reconstruction_folder"):
        return
    record_reconstruction(
        params, inputs=filename, source_slice_index=slice_idx, status="preprocessing")

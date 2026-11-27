"""Image normalization and acquisition-grid restoration for breast pipelines."""
from collections.abc import Sequence
import math

import torch

from src.preprocessing.DataPreparer import DataPreparer
from src.preprocessing.InputDataLoader import InputDataLoader
from src.preprocessing.real_data.ISMRMRDReader import ISMRMRDReader
from src.utils.ismrmrd_io import acquisition_header
from src.utils.zero_fill import zero_fill_grics_image_to_shape


def grics_reference_image_for_normalization(data: DataPreparer, image: torch.Tensor) -> torch.Tensor:
    reference_image = getattr(data, "grics_reference_image", None)
    if reference_image is None:
        raise ValueError(
            "GRICS reference normalization requires the smoothed reference image "
            "prepared independently by CalibrationImageCalculator."
        )
    reference_image = reference_image.to(device=image.device)

    if tuple(reference_image.shape) == tuple(image.shape):
        return reference_image

    raise ValueError(
        "GRICS reference image shape is incompatible with reconstructed image: "
        f"reference={tuple(reference_image.shape)}, image={tuple(image.shape)}."
    )


def normalize_reconstruction_by_grics_reference(image: torch.Tensor, reference_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(image.shape) != tuple(reference_image.shape):
        raise ValueError(
            "GRICS reference normalization requires matching image shapes: "
            f"image={tuple(image.shape)}, reference={tuple(reference_image.shape)}."
        )

    reference_magnitude = torch.abs(reference_image).to(device=image.device, dtype=image.real.dtype)
    threshold = 1.0 * torch.mean(reference_magnitude)
    reference_magnitude = torch.clamp(reference_magnitude, min=threshold)
    return image / reference_magnitude, reference_magnitude


def _validate_working_grid_size(loaded_size, encoded_size, encoding, axis):
    """Accept the reader's explicit expansion, while rejecting unrelated oversize grids."""
    limit = getattr(encoding.encodingLimits, f'kspace_encoding_step_{axis}')
    factor = ISMRMRDReader._accel_factor_or_one(encoding.parallelImaging, f'kspace_encoding_step_{axis}')
    expected = max(encoded_size, ISMRMRDReader._expanded_matrix_size(ISMRMRDReader._encoding_limit_size(limit), factor))
    if loaded_size > encoded_size:
        if loaded_size != expected:
            raise ValueError(f'Loaded encoding-axis {axis} size {loaded_size} exceeds header matrix {encoded_size} and does not match the reader-expanded grid {expected}.')
        print(f'[postprocessing] Reader-expanded encoding axis {axis}: {loaded_size} -> {encoded_size}; restore the header grid by centered Fourier cropping before the image FOV crop.', flush=True)


def grics_zero_fill_shapes(input_data: InputDataLoader) -> tuple[tuple[int, int], tuple[int, int]]:
    """Read encoded and reconstruction matrix sizes from the ISMRMRD header."""
    header = acquisition_header(input_data)

    enc = header.encoding[0]
    recon = enc.reconSpace.matrixSize
    encoded = enc.encodedSpace.matrixSize
    target_y = int(recon.y)
    encoded_y = int(encoded.y)
    if encoded_y <= 0:
        raise ValueError(f"ISMRMRD encoded matrix y must be positive, got {encoded_y}.")
    if target_y == 0:
        # Some converted headers omit the reconstruction matrix. Preserve the
        # encoded pixel spacing when deriving the crop from the declared FOVs.
        recon_fov_y = float(enc.reconSpace.fieldOfView_mm.y)
        encoded_fov_y = float(enc.encodedSpace.fieldOfView_mm.y)
        if not all(math.isfinite(value) and value > 0 for value in (recon_fov_y, encoded_fov_y)):
            raise ValueError("A missing reconstruction matrix y requires positive, finite reconstruction and encoded FOV y.")
        target_y = round(encoded_y * recon_fov_y / encoded_fov_y)
        print(
            f"[input] ISMRMRD reconstruction matrix y is zero; "
            f"using {target_y} from the encoded matrix and field-of-view ratio.",
            flush=True,
        )
    elif target_y > encoded_y:
        target_y //= 2

    target_shape = (int(input_data.Nx), target_y)
    encoded_shape = (int(input_data.Nx), encoded_y)
    if any(size <= 0 for size in (*target_shape, *encoded_shape)):
        raise ValueError(
            f"GRICS zero-fill sizes must be positive: target_shape={target_shape}, encoded_shape={encoded_shape}."
        )
    _validate_working_grid_size(int(input_data.Ny), encoded_y, enc, 1)
    if any(target > encoded for target, encoded in zip(target_shape, encoded_shape)):
        raise ValueError(
            "GRICS zero-fill sizes are inconsistent: "
            f"target_shape={target_shape} is larger than encoded_shape={encoded_shape}."
        )
    return target_shape, encoded_shape


def grics_zero_fill_volume_shapes(input_data: InputDataLoader) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Extend the in-plane zero-fill layout with the header's 3D partition sizes."""
    target_shape, encoded_shape = grics_zero_fill_shapes(input_data)
    header = acquisition_header(input_data)
    enc = header.encoding[0]
    encoded_z = int(enc.encodedSpace.matrixSize.z)
    target_z = int(enc.reconSpace.matrixSize.z)
    if encoded_z <= 0:
        raise ValueError(f"ISMRMRD encoded matrix z must be positive, got {encoded_z}.")
    if target_z == 0:
        recon_fov_z = float(enc.reconSpace.fieldOfView_mm.z)
        encoded_fov_z = float(enc.encodedSpace.fieldOfView_mm.z)
        if not all(math.isfinite(value) and value > 0 for value in (recon_fov_z, encoded_fov_z)):
            raise ValueError("A missing reconstruction matrix z requires positive, finite reconstruction and encoded FOV z.")
        target_z = round(encoded_z * recon_fov_z / encoded_fov_z)
        print(f"[input] ISMRMRD reconstruction matrix z is zero; using {target_z} from the encoded matrix and field-of-view ratio.", flush=True)
    if not 1 < target_z <= encoded_z:
        raise ValueError(f"GRICS zero-fill z sizes are inconsistent: loaded={input_data.Nz}, target={target_z}, encoded={encoded_z}.")
    _validate_working_grid_size(int(input_data.Nz), encoded_z, enc, 2)
    return (*target_shape, target_z), (*encoded_shape, encoded_z)


def zero_fill_loaded_reconstruction(image: torch.Tensor, target_shape: Sequence[int],
                                   encoded_shape: Sequence[int]) -> torch.Tensor:
    """Restore the header Fourier grid, then crop the final spatial image axes."""
    return zero_fill_grics_image_to_shape(image, target_shape=target_shape, encoded_shape=encoded_shape,
                                          spatial_dims=tuple(range(-len(target_shape), 0)))

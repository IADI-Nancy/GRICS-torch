"""Global rigid registration for simulation evaluation, independent of reconstruction."""

import numpy as np
import torch
from scipy.ndimage import affine_transform, map_coordinates, spline_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def register_rigid_to_reference(reference, candidate, voxel_spacing=None):
    """Return a registered magnitude image and the fitted transform metadata.

    Fit only a single global translation and rotation: no spatial scaling,
    deformation, intensity normalization, or modification of reconstruction data.
    This removes a global pose offset from evaluation; it does not establish
    why that offset occurred. Apply the same procedure independently to both
    corrupted and corrected images before comparing their metrics.

    Coordinates follow tensor axis order. Rotation is about the image center
    in physical coordinates, using voxel_spacing (unit spacing by default).
    Fit by magnitude mean squared error with cubic interpolation and zero
    padding. Evaluate metrics over the entire original grid, including borders.
    Translation is bounded to 10% of each axis and rotation to +/-10 degrees
    per angle. Two deterministic sampling grids limit the cost for 3D volumes;
    the final image is resampled once from the original candidate.
    """
    fixed = reference.detach().abs().cpu().numpy().astype(np.float64)
    moving = candidate.detach().abs().cpu().numpy().astype(np.float64)
    ndim = fixed.ndim
    if ndim not in (2, 3) or fixed.shape != moving.shape:
        raise ValueError("Rigid registration requires matching 2D or 3D images.")
    if not np.isfinite(fixed).all() or not np.isfinite(moving).all():
        raise ValueError("Rigid registration requires finite images.")
    if np.linalg.norm(fixed) == 0 or np.linalg.norm(moving) == 0:
        raise ValueError("Rigid registration requires nonzero images.")
    spacing = np.ones(ndim) if voxel_spacing is None else np.asarray(voxel_spacing, dtype=float)
    if spacing.shape != (ndim,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("voxel_spacing must contain one positive finite value per image axis.")
    center = (np.asarray(fixed.shape) - 1) / 2
    coefficients = spline_filter(moving, order=3)
    angle_count = 1 if ndim == 2 else 3
    parameters = np.zeros(ndim + angle_count)
    limits = np.asarray(fixed.shape) * 0.1
    bounds = [(-limit, limit) for limit in limits] + [(-10., 10.)] * angle_count

    def sampling_transform(values):
        # Forward transform: rotate about center, then translate. SciPy samples
        # with its inverse: input_index = matrix @ output_index + offset.
        if ndim == 2:
            angle = np.deg2rad(values[ndim])
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        else:
            rotation = Rotation.from_euler("xyz", values[ndim:], degrees=True).as_matrix()
        matrix = rotation.T * spacing[None, :] / spacing[:, None]
        offset = center - matrix @ (center + values[:ndim])
        return matrix, offset

    optimizations = []
    for max_axis_samples in (32, 64):
        # Fit on voxel centers in the original grid, avoiding resize offsets.
        axes = [np.arange(0, size, max(1, int(np.ceil(size / max_axis_samples)))) for size in fixed.shape]
        coordinates = np.asarray(np.meshgrid(*axes, indexing="ij")).reshape(ndim, -1)
        target = fixed[tuple(coordinates)]
        denominator = max(float(np.mean(target ** 2)), np.finfo(float).tiny)

        def objective(values):
            matrix, offset = sampling_transform(values)
            sampled = map_coordinates(coefficients, matrix @ coordinates + offset[:, None],
                                      order=3, mode="constant", cval=0., prefilter=False)
            # Magnitudes cannot be negative; cubic interpolation may overshoot.
            return float(np.mean((np.maximum(sampled, 0.) - target) ** 2) / denominator)

        result = minimize(objective, parameters, method="Powell", bounds=bounds,
                          options={"xtol": 1e-4, "ftol": 1e-6, "maxiter": 200})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Global rigid registration failed: {result.message}")
        parameters = result.x
        optimizations.append({"max_axis_samples": max_axis_samples, "iterations": int(result.nit)})

    matrix, offset = sampling_transform(parameters)
    registered = affine_transform(coefficients, matrix, offset, output_shape=fixed.shape,
                                  order=3, mode="constant", cval=0., prefilter=False)
    registered = np.maximum(registered, 0.)
    # Identity is a valid rigid transform. Keep it explicitly if the sampled
    # fit would increase full-grid error; record this decision in the JSON.
    selected = "fitted"
    if np.sum((registered - fixed) ** 2) >= np.sum((moving - fixed) ** 2):
        registered = moving.copy()
        parameters = np.zeros_like(parameters)
        matrix, offset = sampling_transform(parameters)
        selected = "identity (fitted transform did not improve full-grid error)"
    metadata = {
        "method": "global rigid magnitude MSE; cubic interpolation; zero padding",
        "selected_transform": selected, "voxel_spacing": spacing.tolist(),
        "translation_voxels": parameters[:ndim].tolist(), "rotation_degrees": parameters[ndim:].tolist(),
        "output_to_input_matrix": matrix.tolist(), "output_to_input_offset": offset.tolist(),
        "translation_bound_voxels": limits.tolist(), "rotation_bound_degrees": 10.,
        "at_search_bound": bool(np.any(np.isclose(np.abs(parameters), np.r_[limits, [10.] * angle_count], atol=1e-3))),
        "optimization": optimizations,
    }
    return torch.from_numpy(registered), metadata

import torch


def to_cartesian_components(alpha_axis0, alpha_axis1):
    """
    Convert internal non-rigid components to Cartesian display components.

    Internal convention:
    - alpha_axis0: displacement along image axis 0 (rows)
    - alpha_axis1: displacement along image axis 1 (cols)

    Display convention (Cartesian):
    - x: horizontal (right positive)
    - y: vertical (up positive)

    Motion operator uses inverse (pull-back) displacements in array axes.
    X keeps the forward Cartesian sign, while Y is displayed with inverted sign
    per requested convention so positive values correspond to upward motion in
    the alpha_y maps.
    """
    alpha_x = -alpha_axis1
    alpha_y = -alpha_axis0
    return alpha_x, alpha_y


def flip_nonrigid_alpha_for_display(alpha_maps, flip_vertical):
    """Flip non-rigid alpha maps on image axis-0 for display consistency.

    Expects alpha maps shaped as [C, Nx, Ny] (2D) or [C, Nx, Ny, Nz] (3D).
    """
    if not bool(flip_vertical):
        return alpha_maps
    if alpha_maps.ndim not in (3, 4):
        raise ValueError("alpha_maps must have shape [C, Nx, Ny] or [C, Nx, Ny, Nz].")
    return torch.flip(alpha_maps, dims=[1])


def split_nonrigid_alpha_components(alpha_maps):
    """Return (axis0, axis1, axis2_or_none) from alpha maps.

    - 2D input [2, Nx, Ny] -> (alpha_x, alpha_y, None)
    - 3D input [3, Nx, Ny, Nz] -> (alpha_x, alpha_y, alpha_z)
    """
    if alpha_maps.ndim not in (3, 4):
        raise ValueError("alpha_maps must have shape [C, Nx, Ny] or [C, Nx, Ny, Nz].")
    if alpha_maps.shape[0] < 2:
        raise ValueError("alpha_maps must have at least 2 components.")

    alpha_x = alpha_maps[0]
    alpha_y = alpha_maps[1]
    alpha_z = alpha_maps[2] if alpha_maps.shape[0] > 2 else None
    return alpha_x, alpha_y, alpha_z

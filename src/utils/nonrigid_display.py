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

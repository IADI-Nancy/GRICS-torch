import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reconstruction.MotionOperator import MotionOperator


def _peak_xy(img):
    ny = img.shape[1]
    idx = torch.argmax(torch.abs(img)).item()
    x = idx // ny
    y = idx % ny
    return x, y


def test_meshgrid_construction_equivalence():
    # Check that the two formulations are equivalent:
    #   Y, X = meshgrid(coords_y, coords_x, indexing="xy")
    # and
    #   X, Y = meshgrid(coords_x, coords_y, indexing="ij")
    nx, ny = 7, 11
    coords_x = torch.arange(1, nx + 1, dtype=torch.float32)
    coords_y = torch.arange(1, ny + 1, dtype=torch.float32)

    y_xy, x_xy = torch.meshgrid(coords_y, coords_x, indexing="xy")
    x_ij, y_ij = torch.meshgrid(coords_x, coords_y, indexing="ij")

    assert torch.equal(x_xy, x_ij)
    assert torch.equal(y_xy, y_ij)


def test_u_x_shift_direction_matches_inverse_warp_convention():
    nx, ny = 9, 9
    img = torch.zeros((nx, ny), dtype=torch.complex64)
    img[4, 4] = 1.0

    ux = torch.ones((nx, ny), dtype=torch.float32)
    uy = torch.zeros((nx, ny), dtype=torch.float32)
    motion_op = MotionOperator.create_sparse_motion_operator(ux, uy)

    warped = (motion_op @ img.flatten()).reshape(nx, ny)
    peak_x, peak_y = _peak_xy(warped)

    # Positive Ux in inverse-warp convention shifts content toward -x.
    assert (peak_x, peak_y) == (3, 4)


def test_u_y_shift_direction_matches_inverse_warp_convention():
    nx, ny = 9, 9
    img = torch.zeros((nx, ny), dtype=torch.complex64)
    img[4, 4] = 1.0

    ux = torch.zeros((nx, ny), dtype=torch.float32)
    uy = torch.ones((nx, ny), dtype=torch.float32)
    motion_op = MotionOperator.create_sparse_motion_operator(ux, uy)

    warped = (motion_op @ img.flatten()).reshape(nx, ny)
    peak_x, peak_y = _peak_xy(warped)

    # Positive Uy in inverse-warp convention shifts content toward -y.
    assert (peak_x, peak_y) == (4, 3)


if __name__ == "__main__":
    test_meshgrid_construction_equivalence()
    test_u_x_shift_direction_matches_inverse_warp_convention()
    test_u_y_shift_direction_matches_inverse_warp_convention()
    print("Motion operator convention tests: OK")

"""Warm start between resolution levels and the units of an external initial motion."""
from types import SimpleNamespace

import torch

from src.reconstruction.joint_reconstructor_utils.resampling.upsample import upsample_data
from src.reconstruction.joint_reconstructor_utils.initialization import apply_external_initializer


def _level(n, device, image=None, motion=None):
    data = {"Nx": n, "Ny": n, "Nz": 1, "MotionStateIndices": torch.tensor([0, 1], device=device)}
    if image is not None:
        data["ReconstructedImage"] = image
    if motion is not None:
        data["RigidMotionModelFull"] = motion
        data["MotionModel"] = motion
    return data


def test_warm_start_is_rescaled_to_the_finer_level(device):
    coarse = _level(32, device, image=torch.ones((32, 32), dtype=torch.complex128, device=device),
                    motion=torch.zeros((3, 2), device=device))
    for rescale, expected in ((True, 0.5), (False, 1.0)):
        fine = _level(64, device)
        upsample_data(coarse, fine, SimpleNamespace(reconstruction_motion_type="rigid", warm_start_rescale=rescale), 3, device)
        assert abs(fine["ReconstructedImage"].abs().mean().item() - expected) < 1e-6


def test_initial_motion_is_given_in_full_resolution_pixels(device):
    params = SimpleNamespace(reconstruction_motion_type="rigid", N_motion_states=2, warm_start_rescale=True)
    level0 = _level(32, device, image=torch.zeros((32, 32), dtype=torch.complex128, device=device))
    apply_external_initializer(level0, initial_image=None, initial_motion=torch.tensor([[4.0, 4.0], [0.0, 0.0], [0.1, 0.1]]),
                               params=params, device=device, Nalpha=3, Nphysio=1, full_shape=(128, 128))
    assert level0["MotionModel"][0].tolist() == [1.0, 1.0]       # 4 px at 128 is 1 px at 32
    assert level0["MotionModel"][2].tolist() == [0.1, 0.1]       # angles are not rescaled
    level1, level2 = _level(64, device), _level(128, device)
    upsample_data(level0, level1, params, 3, device)
    level1["MotionModel"] = level1["RigidMotionModelFull"]
    upsample_data(level1, level2, params, 3, device)
    assert level2["MotionModel"][0].tolist() == [4.0, 4.0]       # back to the value that was supplied

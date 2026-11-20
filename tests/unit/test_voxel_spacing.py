"""Voxel spacing for voxel-aware regularisation can come from the configuration."""
from types import SimpleNamespace

import pytest

from src.preprocessing.DataPreparer import DataPreparer


def _preparer(**params):
    p = DataPreparer.__new__(DataPreparer)
    p.params = SimpleNamespace(**params)
    return p


def test_spacing_from_config():
    p = _preparer(regularization_scaling="voxel_aware", data_dimension="2D", voxel_spacing_mm=[0.7, 0.8])
    assert p.reconstruction_voxel_spacing() == (0.7, 0.8)


def test_wrong_length_is_rejected():
    p = _preparer(regularization_scaling="voxel_aware", data_dimension="3D", voxel_spacing_mm=[0.7, 0.8])
    with pytest.raises(ValueError):
        p.reconstruction_voxel_spacing()


def test_not_needed_without_voxel_aware():
    assert _preparer(regularization_scaling="direct", data_dimension="2D").reconstruction_voxel_spacing() is None

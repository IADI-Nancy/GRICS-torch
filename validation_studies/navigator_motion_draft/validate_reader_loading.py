"""Validate header filtering, acquired-line oversampling, and navigator reuse."""
import copy
import importlib.util
import sys
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import ismrmrd
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.preprocessing.real_data.ISMRMRDReader import ISMRMRDReader
from src.preprocessing.real_data.physiological_data.NavigatorEigenmodeCalculator import NavigatorEigenmodeCalculator
from src.utils.ismrmrd_io import ReadOnlyDataset
from src.utils.fftnc import fftnc, ifftnc

spec = importlib.util.spec_from_file_location('mosar_fixture', ROOT / 'validation_studies/navigator_motion_draft/validate_mosar_pipeline.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def validate_mosar(folder):
    path = folder / 'mosar.mrd'
    fixture.write_fixture(path)
    # Exercise twofold oversampling with the same known triplet layout.
    dataset = ismrmrd.Dataset(str(path), 'dataset', create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    header.encoding[0].reconSpace.matrixSize.x = 8
    dataset.write_xml_header(ismrmrd.xsd.ToXML(header))
    dataset.close()
    original = ReadOnlyDataset.read_acquisition
    decoded = []
    def read(dataset, index):
        decoded.append(index)
        return original(dataset, index)
    raw = []
    for set_index in (1, 0):
        decoded.clear()
        with patch.object(ReadOnlyDataset, 'read_acquisition', read):
            current = ISMRMRDReader(path, read_navigators=True, navigator_type='mosar', imaging_set=set_index).read_data()
        assert len(decoded) == 384 + 36 and len(set(decoded)) == len(decoded)
        assert current['kspace'].shape == (2, 1, 8, 10, 6)
        # Independent reference: grid original samples first, then crop image space.
        full = torch.zeros(2, 1, 16, 10, 6, dtype=torch.complex128)
        count = torch.zeros(10, 6)
        with ReadOnlyDataset(path) as dataset:
            nav = {record['acquisition_index'] for record in current['navigators']}
            for index in decoded:
                acq = dataset.read_acquisition(index)
                if index in nav:
                    continue
                ky, kz = int(acq.idx.kspace_encode_step_1) + 1, int(acq.idx.kspace_encode_step_2) + 1
                full[:, 0, :, ky, kz] += torch.from_numpy(acq.data.copy())
                count[ky, kz] += 1
        full /= count.clamp_min(1)[None, None, None]
        expected = fftnc(ifftnc(full, dims=(2,))[:, :, 4:12], dims=(2,))
        torch.testing.assert_close(current['kspace'], expected, atol=1e-11, rtol=1e-11)
        raw.append(current)
    with (ROOT / 'config/physiological_data/navigators.toml').open('rb') as stream:
        settings = tomllib.load(stream)['navigators']
    parameters = dict(navigator_type=settings["navigator_type"], navigator_set=settings["navigator_set"], imaging_set=settings["navigator_imaging_set"],
                      n_eigenmodes=settings["navigator_eigenmodes"], directions=settings["navigator_directions"],
                      startup_profiles=settings["navigator_startup_profiles"], regularization=settings["navigator_regularization"],
                      resolution_levels=settings["navigator_resolution_levels"], registration_loops=settings["navigator_registration_loops"],
                      extrapolation=settings["navigator_extrapolation"])
    first = NavigatorEigenmodeCalculator(**{**parameters, 'n_eigenmodes': 1, 'resolution_levels': (.5, 1.)})
    channels = first.read_channels(raw[0])
    second = NavigatorEigenmodeCalculator(**{**parameters, 'imaging_set': 0, 'n_eigenmodes': 1, 'resolution_levels': (.5, 1.)}, reference=first.reference)
    with patch.object(second, 'register_profiles', side_effect=AssertionError('Registration must be reused')):
        reused = second.read_channels(raw[1])
    np.testing.assert_array_equal(channels[1], reused[1])
    delta = (raw[0]['navigators'][0]['time_seconds'] - raw[1]['navigators'][0]['time_seconds'])
    np.testing.assert_allclose(channels[0][0] - reused[0][0], delta, atol=1e-12)
    assert second.metadata['imaging_set'] == 0
    bad = copy.deepcopy(raw[1])
    bad['navigators'][0]['data'][0, 0] += 1
    try:
        second.read_channels(bad)
    except ValueError as error:
        assert 'identical navigator' in str(error)
    else:
        raise AssertionError('Changed navigator samples must reject reuse')
    print('PASS: MOSAR header filtering halves decodes, acquired-line cropping matches grid-first reference, duplicate averaging preserved, modes reused with correct contrast clock, changed samples rejected.')


def validate_flags(folder):
    path = folder / 'flags.mrd'
    fixture.write_fixture(path)
    dataset = ismrmrd.Dataset(str(path), 'dataset', create_if_needed=False)
    for flag in (ismrmrd.ACQ_IS_NOISE_MEASUREMENT, ismrmrd.ACQ_IS_DUMMYSCAN_DATA, ismrmrd.ACQ_IS_NAVIGATION_DATA, ismrmrd.ACQ_IS_HPFEEDBACK_DATA):
        acq = ismrmrd.Acquisition.from_array(np.ones((2, 16), dtype=np.complex64))
        acq.setFlag(flag)
        acq.acquisition_time_stamp = 999999
        dataset.append_acquisition(acq)
    for ky, flag in enumerate((ismrmrd.ACQ_IS_PARALLEL_CALIBRATION, ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING)):
        acq = ismrmrd.Acquisition.from_array(np.ones((2, 16), dtype=np.complex64))
        acq.setFlag(flag)
        acq.idx.kspace_encode_step_1 = ky
        acq.acquisition_time_stamp = 1000010 + ky
        dataset.append_acquisition(acq)
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    header.encoding[0].reconSpace.matrixSize.x = 8
    dataset.write_xml_header(ismrmrd.xsd.ToXML(header))
    dataset.close()
    decoded = []
    original = ReadOnlyDataset.read_acquisition
    def read(dataset, index):
        decoded.append(index)
        return original(dataset, index)
    with patch.object(ReadOnlyDataset, 'read_acquisition', read):
        raw = ISMRMRDReader(path, read_navigators=True).read_data()
    assert decoded[-3:] == [842, 844, 845] and 840 not in decoded and 841 not in decoded and 843 not in decoded
    assert len(raw['navigators']) == 1
    expected = fftnc(ifftnc(torch.ones(2, 16, dtype=torch.complex128), dims=(-1,))[:, 4:12], dims=(-1,))
    for ky in (1, 2):
        torch.testing.assert_close(raw['reference_kspace'][:, 0, :, ky, 1], expected)

    decoded.clear()
    with patch.object(ReadOnlyDataset, 'read_acquisition', read):
        navigators = ISMRMRDReader(path).read_navigator_data()
    assert decoded == [842]
    assert navigators[0]['time_seconds'] == raw['navigators'][0]['time_seconds']
    print('PASS: generic flagged navigators, noise/dummy/feedback filtering, both calibration flags cropped correctly, navigator-only path decodes no imaging samples.')


if __name__ == '__main__':
    torch.set_num_threads(2)
    with TemporaryDirectory(prefix='reader-loading-') as directory:
        validate_mosar(Path(directory))
        validate_flags(Path(directory))

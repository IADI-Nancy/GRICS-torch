"""Validate navigator physiology, shared synchronization, caching, and preparation.

Run from the repository root:
    python validation_studies/navigator_motion_draft/validate_navigator_physiology.py

Uses a small temporary ISMRMRD fixture. No test input or cache is retained.
"""

import shutil
import struct
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ismrmrd
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.DataPreparer import DataPreparer
from src.preprocessing.real_data.ISMRMRDReader import ISMRMRDReader
from src.preprocessing.real_data.RawDataCache import RawDataCache
from src.preprocessing.real_data.RawDataPreparer import RawDataPreparer
from src.preprocessing.real_data.physiological_data.NavigatorEigenmodeCalculator import NavigatorEigenmodeCalculator
from src.preprocessing.real_data.physiological_data.PreprocessedPhysioReader import PreprocessedPhysioReader
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime


def write_fixture(path):
    xsd = ismrmrd.xsd
    space = xsd.encodingSpaceType(matrixSize=xsd.matrixSizeType(x=32, y=8, z=1), fieldOfView_mm=xsd.fieldOfViewMm(x=32., y=8., z=1.))
    limit = lambda maximum, center: xsd.limitType(minimum=0, maximum=maximum, center=center)
    encoding = xsd.encodingType(encodedSpace=space, reconSpace=space,
                                encodingLimits=xsd.encodingLimitsType(kspace_encoding_step_1=limit(7, 4), kspace_encoding_step_2=limit(0, 0), slice=limit(0, 0), repetition=limit(0, 0), set=limit(1, 0)),
                                trajectory=xsd.trajectoryType.CARTESIAN)
    header = xsd.ismrmrdHeader(experimentalConditions=xsd.experimentalConditionsType(H1resonanceFrequency_Hz=123000000),
                               measurementInformation=xsd.measurementInformationType(patientPosition=xsd.patientPositionType.HFS), encoding=[encoding])
    dataset = ismrmrd.Dataset(str(path), 'dataset', create_if_needed=True)
    dataset.write_xml_header(xsd.ToXML(header))
    counter = 0

    def append(values, set_id, ky, flags=()):
        nonlocal counter
        acquisition = ismrmrd.Acquisition.from_array(np.asarray(values, dtype=np.complex64))
        acquisition.acquisition_time_stamp = 10000 + counter * 4
        acquisition.scan_counter = counter + 1
        acquisition.idx.set, acquisition.idx.kspace_encode_step_1 = set_id, ky
        acquisition.center_sample = 16
        acquisition.read_dir[:] = (1, 0, 0)
        acquisition.phase_dir[:] = (0, 1, 0)
        acquisition.slice_dir[:] = (0, 0, 1)
        for flag in flags:
            acquisition.setFlag(flag)
        dataset.append_acquisition(acquisition)
        counter += 1

    x = np.arange(32.)
    phantom = np.exp(-((x[:, None] - 15) / 7) ** 2 - ((np.arange(8)[None, :] - 4) / 2) ** 2)
    image_kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(phantom)))
    append(np.ones((2, 32)), 0, 0, (ismrmrd.ACQ_IS_NOISE_MEASUREMENT,))
    for shot in range(12):
        for set_id in (0, 1):
            append(np.ones((1, 7)), 0, 0, (ismrmrd.ACQ_IS_NAVIGATION_DATA, ismrmrd.ACQ_IS_RTFEEDBACK_DATA))
            for direction in range(3):
                profiles = []
                for coil in range(2):
                    profile = np.exp(-((x - 10 - .8 * np.sin(shot * .3 + direction)) / 3) ** 2)
                    profile += .7 * np.exp(-((x - 22 - .5 * np.cos(shot * .5 + coil)) / 2) ** 2)
                    profile *= 1 + set_id
                    profiles.append(np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(profile))))
                append(profiles, set_id, 4)
            # Ordinary central imaging at a block boundary must not merge into
            # the next set's navigator triplet during classification.
            for ky in (0, 1, 2, 3, 5, 6, 7, 4):
                append(np.stack([image_kspace[:, ky], image_kspace[:, ky] * .8]), set_id, ky)
    dataset.close()


def run_validation():
    with TemporaryDirectory(prefix='grics-navigator-physiology-') as directory:
        folder = Path(directory)
        path = folder / 'fixture.mrd'
        write_fixture(path)
        reader = ISMRMRDReader(path, read_navigators=True, navigator_type='mosar')
        raw = reader.read_data()
        assert len(raw['navigators']) == 36
        assert len(raw['time_seconds']) == 96
        assert {record['counters']['set'] for record in raw['navigators']} == {1}
        assert all(record['data'].shape == (2, 32) for record in raw['navigators'])
        try:
            ISMRMRDReader(path, read_navigators=True, navigator_type='unknown')
        except ValueError as exc:
            assert 'navigator_type' in str(exc)
        else:
            raise AssertionError('Unsupported navigator types must be rejected explicitly')
        calculator = NavigatorEigenmodeCalculator(navigator_type="mosar")
        assert calculator.mri_reader_options['navigator_type'] == 'mosar'
        assert calculator.cache_parameters['navigator_type'] == 'mosar'
        # The calculator accepts the grouped-record contract independently of
        # acquisition-specific extraction and does not require three directions.
        grouped = {**raw, 'navigator_type': 'grouped-profiles', 'navigators': [record for record in raw['navigators'] if record['direction_index'] == 2]}
        generic = NavigatorEigenmodeCalculator(navigator_type='grouped-profiles', directions=(2,))
        assert len(generic.read_channels(grouped)[1]) == 2
        times, values, end, bounds = calculator.read_channels(raw)
        assert len(values) == 2 and values[0].shape == (12,)
        assert end == 0 and bounds == 'edge'
        np.testing.assert_allclose(np.asarray(values) @ np.asarray(values).T, np.eye(2), atol=1e-12)
        aligned = RawDataPreparer._synchronize_to_sequence_end(times, values, raw['time_seconds'].numpy(), source_sequence_end=end, bounds=bounds)
        np.testing.assert_allclose(aligned[-1], np.asarray(values)[:, -1], atol=1e-12)
        assert times[0][-1] < 0  # Last navigator is not falsely moved to MRI end.
        more = NavigatorEigenmodeCalculator(navigator_type="mosar", n_eigenmodes=3)
        _, more_values, _, _ = more.read_channels(raw)
        np.testing.assert_allclose(np.asarray(more_values)[:2], np.asarray(values), atol=1e-12)
        try:
            NavigatorEigenmodeCalculator(navigator_type="mosar", n_eigenmodes=100).read_channels(raw)
        except ValueError as exc:
            assert 'rank' in str(exc)
        else:
            raise AssertionError('Requested eigenmode count must be checked against measured rank')
        cache = RawDataCache(folder / 'cache', remove_on_release=False)
        preparer = RawDataPreparer(path, None, physiological_reader=NavigatorEigenmodeCalculator(navigator_type="mosar"), device='cpu')
        first = preparer.read_data(cache=cache)
        cached = RawDataPreparer(path, None, physiological_reader=NavigatorEigenmodeCalculator(navigator_type="mosar"), device='cpu')
        with patch.object(cached.reader, 'read_data', side_effect=AssertionError('Cache hit must not decode MRI again')):
            second = cached.read_data(cache=cache)
        np.testing.assert_array_equal(first['motion_data'], second['motion_data'])
        assert len(second['navigators']) == 36
        np.testing.assert_allclose(second['motion_data'].mean(axis=(0, 1)), 0, atol=1e-12)
        np.testing.assert_allclose(second['motion_data'].std(axis=(0, 1)), 1, atol=1e-12)
        # External physiology keeps its existing source and synchronization path.
        text = folder / 'physio.txt'
        np.savetxt(text, np.column_stack([np.zeros(96), -np.ones(96), np.linspace(0, 1, 96)]))
        external = PreprocessedPhysioReader('physio_text')
        assert external.read_channels(text)[1][0].shape == (96,)
        for overrides in ({'navigator_type': 'unknown'}, {'navigator_eigenmodes': 0}, {'navigator_directions': [0, 0]}, {'navigator_resolution_levels': [.5, .25, 1]}, {'physio_clock_drift_seconds': 1.0}):
            try:
                load_config(data_type='ismrmrd-navigators', reconstruction_config=ROOT / 'config/reconstruction/rigid_2d.toml', overrides=overrides)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Invalid navigator configuration accepted: {overrides}')
        # The loader uses the same prepared physiological interface as other data types.
        params = load_config(data_type='ismrmrd-navigators', reconstruction_config=ROOT / 'config/reconstruction/rigid_2d.toml',
                             overrides={'runtime_device': 'cpu', 'save_debug_plots': False, 'output_root': str(folder / 'runs'),
                                        'cache_root': str(folder / 'loader-cache'), 'ResolutionLevels': [1.0], 'GN_iterations_per_level': [1],
                                        'espirit_calibration_width': 4, 'espirit_kernel_width': 2, 'espirit_max_iter': 5, 'N_motion_states': 3})
        runtime = initialize_runtime(params)
        data = DataPreparer(params, runtime, filename=path).load_input_data()
        data.prepare_reconstructor_inputs()
        assert data.motion_signal.shape == (3, 2)
        assert data.kspace.shape[-2:] == (8, 1)
        assert torch.isfinite(data.motion_signal).all()
        # VB conversion must use its only measurement, not the old hard-coded 2.
        fake_raw = folder / 'fixture.dat'
        fake_raw.write_bytes(struct.pack('<II', 2933312, 6))
        siemens_params = load_config(data_type='siemens-navigators', reconstruction_config=ROOT / 'config/reconstruction/rigid_2d.toml',
                                     overrides={'runtime_device': 'cpu', 'save_debug_plots': False, 'cache_root': str(folder / 'siemens-cache')})
        from src.preprocessing.InputDataLoader import InputDataLoader
        loader = InputDataLoader(siemens_params, torch.device('cpu'), filename=fake_raw)

        def convert(command, **kwargs):
            assert command[command.index('-z') + 1] == '1'
            shutil.copyfile(path, command[command.index('-o') + 1])

        with patch('src.preprocessing.real_data.real_data_io.subprocess.run', side_effect=convert):
            loader.load_input_data()
        assert loader._input_motion_data.shape == (1, 96, 2)
        loader.raw_data_cache.close()
        data.input_loader.raw_data_cache.close()
        cache.close()
    print('PASS: MOSAR extraction, second-set isolation, eigenmode count/rank, shared scanner synchronization, normalization, cache reuse, complete reconstruction preparation, and Siemens VB dispatch')


if __name__ == '__main__':
    run_validation()

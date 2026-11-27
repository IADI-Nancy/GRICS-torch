"""Run a small, complete two-contrast MOSAR reconstruction and check DICOM export.

Run from the repository root: python validation_studies/navigator_motion_draft/validate_mosar_pipeline.py
"""
import sys
import shutil
import struct
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ismrmrd
import numpy as np
import pydicom
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.siemens_MP2RAGE_navigators import run_pipeline
from pipelines.utils.siemens_navigators import combine_mp2rage
from pipelines.utils.postprocessing import zero_fill_loaded_reconstruction
from src.preprocessing.real_data.ISMRMRDReader import ISMRMRDReader


def write_fixture(path):
    xsd = ismrmrd.xsd
    space = xsd.encodingSpaceType(matrixSize=xsd.matrixSizeType(x=16, y=8, z=4), fieldOfView_mm=xsd.fieldOfViewMm(x=16., y=8., z=4.))
    limit = lambda maximum, center: xsd.limitType(minimum=0, maximum=maximum, center=center)
    # Neither 8 ky lines nor 4 kz partitions is divisible by acceleration 3.
    # The reader expands to 10x6 and centres the acquired 8x4 grid inside it.
    encoding = xsd.encodingType(encodedSpace=space, reconSpace=space, trajectory=xsd.trajectoryType.CARTESIAN,
                               parallelImaging=xsd.parallelImagingType(accelerationFactor=xsd.accelerationFactorType(kspace_encoding_step_1=3, kspace_encoding_step_2=3)),
                               encodingLimits=xsd.encodingLimitsType(kspace_encoding_step_1=limit(7, 4), kspace_encoding_step_2=limit(3, 2), slice=limit(0, 0), set=limit(1, 0)))
    header = xsd.ismrmrdHeader(experimentalConditions=xsd.experimentalConditionsType(H1resonanceFrequency_Hz=123000000),
                               measurementInformation=xsd.measurementInformationType(patientPosition=xsd.patientPositionType.HFS),
                               sequenceParameters=xsd.sequenceParametersType(TI=[700., 2500.], TE=[3.], TR=[5000.], flipAngle_deg=[4., 5.]), encoding=[encoding])
    dataset = ismrmrd.Dataset(str(path), 'dataset', create_if_needed=True)
    dataset.write_xml_header(xsd.ToXML(header))
    x, y, z = np.meshgrid(np.arange(16.), np.arange(8.), np.arange(4.), indexing='ij')
    phantom = np.exp(-((x - 8) / 4) ** 2 - ((y - 4) / 2) ** 2 - ((z - 2) / 2) ** 2)
    coils = np.stack([phantom, phantom * .8 * np.exp(.2j)])
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(coils, axes=(1, 2, 3)), axes=(1, 2, 3)), axes=(1, 2, 3))
    counter = 0

    def append(values, set_index, ky, kz):
        nonlocal counter
        acq = ismrmrd.Acquisition.from_array(np.asarray(values, dtype=np.complex64))
        acq.acquisition_time_stamp, acq.scan_counter = 10000 + counter * 4, counter
        acq.idx.set, acq.idx.kspace_encode_step_1, acq.idx.kspace_encode_step_2 = set_index, ky, kz
        acq.center_sample = 8
        acq.read_dir[:], acq.phase_dir[:], acq.slice_dir[:] = (1, 0, 0), (0, 1, 0), (0, 0, 1)
        dataset.append_acquisition(acq)
        counter += 1

    order = [(ky, kz) for ky in range(8) for kz in range(4)]
    order.remove((4, 2))
    order.append((4, 2))  # Ordinary central imaging is retained at the block end.
    x = np.arange(16.)
    for shot in range(12):
        for set_index in (0, 1):
            for direction in range(3):
                profile = np.exp(-((x - 5 - .6 * np.sin(shot * .4 + direction)) / 2) ** 2)
                profile += .6 * np.exp(-((x - 11 - .4 * np.cos(shot * .7)) / 1.5) ** 2)
                samples = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(profile)))
                append(np.stack([samples, .8 * samples]), set_index, 4, 2)
            for ky, kz in order:
                append(kspace[:, :, ky, kz] * (-.6 if set_index == 0 else 1), set_index, ky, kz)
    dataset.close()


def run_validation(*, no_motion_correction=False):
    torch.set_num_threads(2)
    reference = torch.randn(16, 8, 4, dtype=torch.complex128)
    expanded = zero_fill_loaded_reconstruction(reference, (16, 10, 6), (16, 10, 6))
    restored = zero_fill_loaded_reconstruction(expanded, (16, 8, 4), (16, 8, 4))
    # Fourier grid restoration recovers the full FOV and complex values;
    # simply cropping the larger image array would not pass this check.
    torch.testing.assert_close(restored, reference, atol=1e-12, rtol=1e-12)
    first = torch.tensor([[[1 + 0j, -1 + 0j, 0j, 1j]]])
    second = torch.tensor([[[1 + 0j, 1 + 0j, 0j, 1 + 0j]]])
    torch.testing.assert_close(combine_mp2rage(first, second), torch.tensor([[[.5, -.5, 0., 0.]]]))
    with TemporaryDirectory(prefix='grics-mosar-pipeline-') as directory:
        folder = Path(directory)
        raw_file = folder / 'two_contrasts.mrd'
        write_fixture(raw_file)
        try:
            run_pipeline(raw_file)
        except ValueError as exc:
            assert 'Siemens .dat' in str(exc)
        else:
            raise AssertionError('The MOSAR pipeline must reject ISMRMRD inputs.')
        raw = ISMRMRDReader(raw_file, read_navigators=True, navigator_type='mosar', imaging_set=0).read_data()
        assert raw['kspace'].shape == (2, 1, 16, 10, 6)
        assert set(raw['idx_ky'].tolist()) == set(range(1, 9))
        assert set(raw['idx_kz'].tolist()) == set(range(1, 5))
        assert not torch.count_nonzero(raw['kspace'][..., [0, -1], :])
        assert not torch.count_nonzero(raw['kspace'][..., [0, -1]])
        assert len(raw['time_seconds']) == 384 and len(raw['navigators']) == 36
        assert raw['imaging_set'] == 0 and {record['counters']['set'] for record in raw['navigators']} == {1}
        assert raw['navigators'][0]['time_seconds'] > raw['time_seconds'][0]
        fake_raw = folder / 'two_contrasts.dat'
        fake_raw.write_bytes(struct.pack('<II', 2933312, 6))
        converted_files = []
        subprocess_run = subprocess.run

        def convert(command, **kwargs):
            if '-z' not in command:
                return subprocess_run(command, **kwargs)
            assert command[command.index('-z') + 1] == '1'
            output = Path(command[command.index('-o') + 1])
            shutil.copyfile(raw_file, output)
            converted_files.append(output)

        overrides = {
            'cache_root': str(folder / 'cache'), 'save_debug_plots': False, 'verbose': False, 'print_to_console': False,
            'ResolutionLevels': [1.], 'GN_iterations_per_level': [2], 'N_motion_states': 2,
            'max_iter_recon': 3, 'max_iter_motion': 1, 'cg_reg_scale_num_probes': 1,
            'espirit_calibration_width': 4, 'espirit_kernel_width': 2, 'espirit_max_iter': 5,
            'save_reconstruction_tensors': True}
        with patch('src.preprocessing.real_data.real_data_io.subprocess.run', side_effect=convert):
            result = run_pipeline(fake_raw, output_root=folder / 'runs/mosar', device='cpu',
                                  reconstruction_config=ROOT / 'config/reconstruction/nonrigid_3d.toml', overrides=overrides, no_motion_correction=no_motion_correction)
        if no_motion_correction:
            for contrast in result['contrasts'].values():
                assert torch.count_nonzero(torch.load(contrast['motion_file'], weights_only=True)).item() == 0
                log = Path(contrast['params'].logs_folder, 'reconstruction.log').read_text()
                assert 'Motion correction disabled' in log and 'Model optimization step:' not in log
            assert 'NO_MOCO' in pydicom.dcmread(result['dicom_series']['inv2']['files'][0]).SeriesDescription
        assert len(converted_files) == 1 and not converted_files[0].exists()
        assert not list((folder / 'cache/converted').glob('*.mrd'))
        assert set(result['dicom_series']) == {'inv1', 'inv2', 'combined'}
        datasets = {name: [pydicom.dcmread(file) for file in series['files']] for name, series in result['dicom_series'].items()}
        assert all(len(items) == 4 for items in datasets.values())
        assert len({str(items[0].SeriesInstanceUID) for items in datasets.values()}) == 3
        assert {int(items[0].SeriesNumber) for items in datasets.values()} == {1001, 1002, 1003}
        assert len({str(item.StudyInstanceUID) for items in datasets.values() for item in items}) == 1
        assert len({str(item.FrameOfReferenceUID) for items in datasets.values() for item in items}) == 1
        for name, items in datasets.items():
            image = result['combined_image'] if name == 'combined' else result['contrasts'][name]['image'].abs()
            assert torch.isfinite(image).all()
            for z, item in enumerate(items):
                decoded = item.pixel_array * float(item.RescaleSlope) + float(item.RescaleIntercept)
                # Reuse the established Siemens in-plane export orientation.
                expected = np.flip(image[..., z].numpy(), axis=(0, 1))
                np.testing.assert_allclose(decoded, expected, atol=float(item.RescaleSlope) * .51)
                if name == 'combined':
                    np.testing.assert_allclose(pydicom.pixels.apply_windowing(decoded, item), decoded, atol=1e-12)
                np.testing.assert_allclose(item.ImagePositionPatient, datasets['inv2'][z].ImagePositionPatient)
            assert len({float(item.RescaleSlope) for item in items}) == 1
        assert float(datasets['combined'][0].RescaleIntercept) == -.5
        assert datasets['combined'][0].VOILUTFunction == 'LINEAR_EXACT'
        assert datasets['combined'][0].ImageType[2] == 'R'
        assert float(result['combined_image'][8, 4, 2]) < -.3
        assert float(datasets['inv1'][0].InversionTime) == 700.
        assert float(datasets['inv2'][0].InversionTime) == 2500.
        assert 'InversionTime' not in datasets['combined'][0]
        assert result['manifest_file'].is_file()
        assert (result['contrasts']['inv2']['output_dir'] / 'image_mp2rage_combined.pt').is_file()
    print('PASS: two actual 3D reconstructions, second-set navigators for both contrasts, shared maps/motion reference, MP2RAGE, and three correctly scaled DICOM series')


if __name__ == '__main__':
    run_validation()

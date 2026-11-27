"""Validate first-set single navigators and one-image MPRAGE export.

Run from the repository root: python validation_studies/navigator_motion_draft/validate_mprage_pipeline.py
"""
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ismrmrd
import numpy as np
import pydicom
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.siemens_MPRAGE_navigators import run_pipeline
from src.preprocessing.real_data.ISMRMRDReader import ISMRMRDReader
from src.utils.ismrmrd_io import ReadOnlyDataset

spec = importlib.util.spec_from_file_location('mp2rage_fixture', Path(__file__).with_name('validate_mosar_pipeline.py'))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def write_fixture(path):
    """Keep one navigator per 32-readout shot; add central filler at the tail."""
    original = path.with_name('two_sets.mrd')
    fixture.write_fixture(original)
    with ReadOnlyDataset(original) as source:
        header = ismrmrd.xsd.CreateFromDocument(source.read_xml_header())
        header.encoding[0].encodingLimits.set.maximum = 0
        header.sequenceParameters.TI = [700.]
        header.sequenceParameters.flipAngle_deg = [4.]
        target = ismrmrd.Dataset(str(path), 'dataset', create_if_needed=True)
        try:
            target.write_xml_header(ismrmrd.xsd.ToXML(header))
            heads = source.read_acquisition_headers()
            selected = np.flatnonzero(heads['idx']['set'] == 0)
            for shot in range(12):
                block = selected[shot * 35:(shot + 1) * 35]
                indices = [block[0], *block[3:34]]
                for position, index in enumerate(indices):
                    # Extra central readouts must not become navigators.
                    if shot == 11 and position >= 30:
                        index = block[-1]
                    acquisition = source.read_acquisition(int(index))
                    acquisition.scan_counter = shot * 32 + position
                    acquisition.acquisition_time_stamp = 10000 + (shot * 32 + position) * 4
                    target.append_acquisition(acquisition)
        finally:
            target.close()


def run_validation():
    torch.set_num_threads(2)
    with TemporaryDirectory(prefix='grics-mprage-validation-') as directory:
        folder = Path(directory)
        mrd = folder / 'mprage.mrd'
        write_fixture(mrd)
        options = dict(read_navigators=True, navigator_type='mosar-mprage', navigator_set=0, imaging_set=0)
        raw = ISMRMRDReader(mrd, **options).read_data()
        assert len(raw['navigators']) == 12 and len(raw['time_seconds']) == 372
        assert all(record['counters']['set'] == 0 and record['direction_index'] == 0 for record in raw['navigators'])
        assert [record['acquisition_index'] for record in raw['navigators']] == list(range(0, 384, 32))
        # A later missing navigator must reject the entire cadence.
        with ReadOnlyDataset(mrd) as dataset:
            heads = dataset.read_acquisition_headers()
            limits = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header()).encoding[0].encodingLimits
        bad = heads.copy()
        bad['idx']['kspace_encode_step_1'][5 * 32] = 0
        try:
            ISMRMRDReader(mrd, **options)._mosar_mprage_navigator_indices(bad, limits, 0)
        except ValueError as error:
            assert 'later shot' in str(error)
        else:
            raise AssertionError('Missing navigator must reject extraction')
        raw_file = folder / 'mprage.dat'
        raw_file.write_bytes(struct.pack('<II', 2933312, 6))
        converted = []
        subprocess_run = subprocess.run

        def convert(command, **kwargs):
            if '-z' not in command:
                return subprocess_run(command, **kwargs)
            output = Path(command[command.index('-o') + 1])
            shutil.copyfile(mrd, output)
            converted.append(output)

        overrides = dict(cache_root=str(folder / 'cache'), save_debug_plots=False, print_to_console=False, verbose=False,
                         ResolutionLevels=[.5, 1.], GN_iterations_per_level=[2, 2], N_motion_states=2,
                         max_iter_recon=3, max_iter_motion=1, cg_reg_scale_num_probes=1,
                         espirit_calibration_width=4, espirit_kernel_width=2, espirit_max_iter=5, save_reconstruction_tensors=True)
        for no_motion_correction in (False, True):
            with patch('src.preprocessing.real_data.real_data_io.subprocess.run', side_effect=convert):
                result = run_pipeline(raw_file, output_root=folder / ('without_motion' if no_motion_correction else 'with_motion'),
                                      reconstruction_config=ROOT / 'config/reconstruction/rigid_3d.toml', device='cpu',
                                      overrides=overrides, no_motion_correction=no_motion_correction)
            assert set(result['contrasts']) == {'mprage'} and set(result['dicom_series']) == {'mprage'}
            assert result['combined_image'] is None
            contrast = result['contrasts']['mprage']
            assert contrast['params'].navigator_set == 0 and contrast['params'].navigator_directions == [0]
            assert torch.isfinite(contrast['image']).all()
            if no_motion_correction:
                assert not torch.count_nonzero(contrast['motion'])
            manifest = json.loads(result['manifest_file'].read_text())
            assert manifest['navigator_set'] == 0 and manifest['acquisition_type'] == 'MPRAGE'
            assert 'combination' not in manifest
            files = result['dicom_series']['mprage']['files']
            assert len(files) == 4
            datasets = [pydicom.dcmread(file) for file in files]
            assert len({item.SeriesInstanceUID for item in datasets}) == 1
            assert float(datasets[0].InversionTime) == 700.
            assert ('NO_MOCO' in datasets[0].SeriesDescription) == no_motion_correction
        assert len(converted) == 2 and not any(path.exists() for path in converted)
    print('PASS: first-set cadence, central filler retained, malformed cadence rejected, corrected/uncorrected multi-resolution MPRAGE, one image/DICOM series, and temporary conversion cleanup.')


if __name__ == '__main__':
    run_validation()

"""Native-grid 3D DICOM geometry, donor, and pipeline export checks."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pydicom
import torch

from src.runtime.hdf5_cache import write_tree
from src.utils.dicom_export import write_volume_dicoms
from pipelines import siemens_breast_3d_lowres as pipeline


def header_xml(n=8):
    return f'''<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
    <experimentalConditions><H1resonanceFrequency_Hz>123000000</H1resonanceFrequency_Hz></experimentalConditions>
    <encoding><encodedSpace><matrixSize><x>{n}</x><y>{n}</y><z>{n}</z></matrixSize>
    <fieldOfView_mm><x>80</x><y>80</y><z>24</z></fieldOfView_mm></encodedSpace>
    <reconSpace><matrixSize><x>{n}</x><y>{n}</y><z>{n}</z></matrixSize>
    <fieldOfView_mm><x>80</x><y>80</y><z>24</z></fieldOfView_mm></reconSpace>
    <encodingLimits/><trajectory>cartesian</trajectory></encoding></ismrmrdHeader>'''


def geometry():
    return {'position': [10., 20., 30.], 'read_dir': [1., 0., 0.],
            'phase_dir': [0., 0., 1.], 'slice_dir': [0., -1., 0.]}


class VolumeDicomTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = SimpleNamespace(ismrmrd_header=header_xml(),
                                   _source_slice_geometry={0: geometry(), 3: geometry()})
        self.image = torch.arange(2*8**3).reshape(2, 8, 8, 8).to(torch.complex128)

    def test_volume_positions_thickness_pixels_and_uids(self):
        paths, uids = write_volume_dicoms(self.image, self.root/'dicom', self.raw, series_number=123)
        datasets = [pydicom.dcmread(p) for p in paths]
        self.assertEqual(len(datasets), 8)
        self.assertEqual(len({ds.SOPInstanceUID for ds in datasets}), 8)
        self.assertEqual({ds.SeriesInstanceUID for ds in datasets}, {uids['series_instance_uid']})
        self.assertEqual({ds.StudyInstanceUID for ds in datasets}, {uids['study_instance_uid']})
        self.assertEqual({ds.FrameOfReferenceUID for ds in datasets}, {uids['frame_of_reference_uid']})
        self.assertEqual(len({ds.InstanceNumber for ds in datasets}), 8)
        for z, ds in enumerate(datasets):
            self.assertEqual(ds.SeriesNumber, 123)
            self.assertEqual(ds.MRAcquisitionType, '3D')
            self.assertEqual(ds.ImagesInAcquisition, 8)
            self.assertEqual(ds.SliceThickness, 3)
            self.assertEqual(ds.SpacingBetweenSlices, 3)
            np.testing.assert_allclose(ds.PixelSpacing, [10, 10])
            center = np.array([10, 20+(z-3.5)*3, 30])
            expected_ipp = center - np.array([35, 0, 35])
            np.testing.assert_allclose(ds.ImagePositionPatient, expected_ipp)
            np.testing.assert_allclose(ds.ImageOrientationPatient, [0, 0, 1, 1, 0, 0])
            self.assertEqual(ds.pixel_array.shape, (8, 8))
            self.assertEqual(ds.pixel_array.max(), 4095)
        self.assertEqual(self.raw._source_slice_geometry[0]['position'], [10, 20, 30])

    def test_axial_partition_pixels_follow_head_to_feet_positions(self):
        slab = {'position': [0., 0., 0.], 'read_dir': [0., 1., 0.],
                'phase_dir': [1., 0., 0.], 'slice_dir': [0., 0., 1.]}
        self.raw._source_slice_geometry = {0: slab}
        image = torch.zeros((1, 8, 8, 8), dtype=torch.complex128)
        # Distinct asymmetric landmarks identify the native first and last planes.
        image[0, 1, 2, 0] = 1
        image[0, 3, 4, -1] = 1
        paths, _ = write_volume_dicoms(image, self.root/'axial', self.raw)
        first, last = (pydicom.dcmread(paths[i]) for i in (0, -1))
        self.assertGreater(float(first.ImagePositionPatient[2]), float(last.ImagePositionPatient[2]))
        self.assertAlmostEqual(float(first.ImagePositionPatient[2]), 10.5)
        self.assertAlmostEqual(float(last.ImagePositionPatient[2]), -10.5)
        self.assertEqual(np.unravel_index(first.pixel_array.argmax(), (8, 8)), (6, 5))
        self.assertEqual(np.unravel_index(last.pixel_array.argmax(), (8, 8)), (4, 3))

    def test_donor_metadata_does_not_replace_partition_geometry(self):
        paths, _ = write_volume_dicoms(self.image, self.root/'donors', self.raw, series_number=12)
        donor = pydicom.dcmread(paths[0])
        donor.PatientName = 'Fixture^Donor'
        donor.ImagePositionPatient = [999, 999, 999]
        donor.PixelSpacing = [1, 1]
        donor.save_as(paths[0], enforce_file_format=True)
        paths, _ = write_volume_dicoms(self.image, self.root/'exports', self.raw,
                                      series_number=13, reference_dicom_path=paths[0])
        datasets = [pydicom.dcmread(p) for p in paths]
        self.assertEqual(str(datasets[0].PatientName), 'Fixture^Donor')
        self.assertEqual(len({tuple(ds.ImagePositionPatient) for ds in datasets}), 8)
        np.testing.assert_allclose(datasets[0].PixelSpacing, [10, 10])
        with self.assertRaisesRegex(ValueError, 'different'):
            write_volume_dicoms(self.image, self.root/'invalid', self.raw,
                                series_number=13, reference_dicom_path=paths[0])

    def test_requires_geometry(self):
        self.raw._source_slice_geometry = None
        with self.assertRaisesRegex(ValueError, 'slab geometry'):
            write_volume_dicoms(self.image, self.root/'invalid', self.raw)
        self.assertFalse((self.root/'invalid').exists())

    def test_pipeline_exports_after_solver_without_tensor_files(self):
        torch.set_num_threads(1)
        source = self.root/'subject.h5'
        with h5py.File(source, 'w') as f:
            f['kspace'] = np.random.default_rng(1).normal(size=(1, 1, 8, 8, 8)).astype(np.complex128)
            f['idx_ky'] = np.repeat(np.arange(8), 8)
            f['idx_kz'] = np.tile(np.arange(8), 8)
            f['idx_nex'] = np.zeros(64, dtype=np.int64)
            f['motion_data'] = np.sin(np.arange(64)/5).reshape(-1, 1)
            f['ismrmrd_header'] = header_xml()
            write_tree(f.create_group('slice_geometry'), {0: geometry()})
        real_timed = pipeline.timed_reconstruction
        completed = []
        real_export = pipeline.write_volume_dicoms
        def timed(data):
            result = real_timed(data)
            completed.append(True)
            self.assertFalse(list(Path(data.params.run_folder).rglob('*.dcm')))
            return result
        def export(*args, **kwargs):
            self.assertTrue(completed)
            return real_export(*args, **kwargs)
        with patch.object(pipeline, 'timed_reconstruction', side_effect=timed), patch.object(
            pipeline, 'write_volume_dicoms', side_effect=export,
        ):
            result = pipeline.run_pipeline(source, output_root=self.root/'runs/3d', device='cpu',
                save_reconstruction_tensors=False, export_dicom=True, dicom_series_number=123,
                overrides={'ResolutionLevels': [0.5, 1.0], 'GN_iterations_per_level': [2, 2],
                           'max_iter_recon': 2, 'max_iter_motion': 2})
        self.assertEqual(len(result['reconstructions'][0]['dicom_files']), 8)
        self.assertFalse(list(result['run_folder'].rglob('*.pt')))
        manifest = json.loads((result['run_folder']/'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['dicom_series_number'], 123)
        self.assertTrue(manifest['export_dicom'])


if __name__ == '__main__':
    unittest.main()

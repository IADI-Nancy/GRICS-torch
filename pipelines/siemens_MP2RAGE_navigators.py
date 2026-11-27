#!/usr/bin/env python
"""Reconstruct Siemens MP2RAGE data using second-set navigator triplets and export INV1, INV2, and their ratio combination."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.utils.siemens_navigators import run_cli, run_navigator_pipeline

OUTPUT_ROOT = Path('runs/siemens_MP2RAGE_navigators')
RECONSTRUCTION_CONFIG = 'config/reconstruction/nonrigid_3d_mosar.toml'


def run_pipeline(raw_data_file, *, output_root=OUTPUT_ROOT, device='gpu', reconstruction_config=RECONSTRUCTION_CONFIG,
                 overrides=None, dicom_header_dir=None, dicom_series_number=1001, no_motion_correction=False):
    return run_navigator_pipeline(raw_data_file, acquisition_type='MP2RAGE', output_root=output_root, device=device,
                                  reconstruction_config=reconstruction_config, overrides=overrides, dicom_header_dir=dicom_header_dir,
                                  dicom_series_number=dicom_series_number, no_motion_correction=no_motion_correction)


def main(argv=None):
    return run_cli(run_pipeline, description=__doc__, output_root=OUTPUT_ROOT, reconstruction_config=RECONSTRUCTION_CONFIG, argv=argv)


if __name__ == '__main__':
    main()

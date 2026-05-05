import sys
from pathlib import Path
if "__file__" in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    _REPO_ROOT = Path.cwd()
sys.path.insert(0, str(_REPO_ROOT))


saec_folder = "/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/SAEC/"
ismrmrd_folder = "/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/ISMRMRD/"
output_folder = "/home/pyuser/wkdir/data/GRICS-torch/article_dataset/"
sensor_type = 'BELT'
device = 'cpu'

import os
import glob
from src.preprocessing.RawDataReader import RawDataReader

saec_files = os.listdir(saec_folder)

for raw_file in saec_files:
    print(raw_file)

    reader = RawDataReader(
            ismrmrd_file=str(ismrmrd_folder + raw_file),
            saec_file=str(saec_folder + raw_file),
            sensor_type=sensor_type,
            device=device,
        )
    output = output_folder + raw_file
    data = reader.read_data_from_rawdata(h5filename=output)
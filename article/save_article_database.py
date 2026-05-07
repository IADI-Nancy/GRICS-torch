import sys
import os
import re
from pathlib import Path

if "__file__" in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    _REPO_ROOT = Path.cwd()

sys.path.insert(0, str(_REPO_ROOT))

from src.preprocessing.RawDataReader import RawDataReader


saec_folder = "/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/SAEC/"
ismrmrd_folder = "/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/ISMRMRD/"
output_folder = "/home/pyuser/wkdir/data/GRICS-torch/article_dataset/"

sensor_type = "BELT"
device = "cpu"

T2_H5_PATTERN = re.compile(r"^\d{4}_T2_[sm]\.h5$")

saec_files = os.listdir(saec_folder)

for raw_file in saec_files:
    if not T2_H5_PATTERN.match(raw_file):
        continue

    print(raw_file)

    reader = RawDataReader(
        ismrmrd_file=str(Path(ismrmrd_folder) / raw_file),
        saec_file=str(Path(saec_folder) / raw_file),
        sensor_type=sensor_type,
        device=device,
    )

    output = str(Path(output_folder) / raw_file)

    data = reader.read_data_from_rawdata(h5filename=output)
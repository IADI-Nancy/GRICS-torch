import numpy as np
import matplotlib
from pathlib import Path
import os
import re
from sharpness_index import sharpness_index
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRICS_plusplus_path = "/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/GRICS-BELT-3D"
no_correction_path = "/home/pyuser/wkdir/data/Breast-INNOV_GRICS_database/GRICS-BELT-3D-nomoco/"
reconstruction_times_file = GRICS_plusplus_path + '/' + "reconstruction_times.txt"
sharpness_enhansement_file = GRICS_plusplus_path + '/' + "sharpness_enhansement.txt"
Nsli_max = 120

def extract_matrix_size(log_text):
    nx = re.search(r'\bNx\s*=\s*(\d+)', log_text)
    ny = re.search(r'\bNy\s*=\s*(\d+)', log_text)
    nz = re.search(r'\bNz\s*=\s*(\d+)', log_text)

    if nx and ny and nz:
        return int(nx.group(1)), int(ny.group(1)), int(nz.group(1))
    else:
        return None
    
def extract_total_time(log_text):
    match = re.search(r'Total elapsed time\s*=\s*([\d.]+)', log_text)
    
    if match:
        return float(match.group(1))
    else:
        return None


def load_bin(file, data_type, output_size=None, endian="<"):
    """
    Parameters
    ----------
    file : str or Path
    data_type : str
        'complex-float', 'complex-double', or standard numpy dtype
    output_size : tuple or None
    endian : '<' little, '>' big
    """

    file = Path(file)

    if data_type == "complex-float":
        raw = np.fromfile(file, dtype=endian + "f4")
        data = raw[0::2] + 1j * raw[1::2]

    elif data_type == "complex-double":
        raw = np.fromfile(file, dtype=endian + "f8")
        data = raw[0::2] + 1j * raw[1::2]

    else:
        data = np.fromfile(file, dtype=endian + data_type)

    if output_size is not None:
        data = data.reshape(output_size)

    return data




with open(reconstruction_times_file, "w") as f:
    pass  # clears the file once at the beginning
with open(sharpness_enhansement_file, "w") as f:
    pass  # clears the file once at the beginning

# loop over subjects
for path_grics in sorted(Path(GRICS_plusplus_path).iterdir()):
    if not os.path.isdir(path_grics):
        break

    # Read the GRICS log
    file_log = path_grics / "grics.log.0"
    with open(file_log, "r") as f:
        log = f.read()

    matrix_size = extract_matrix_size(log)
    Ny, Nx, Nz = matrix_size

    # Read the GRICS reconstructed image
    file_grics = path_grics / "GricsRecon.dat.0000"
    file_no_moco = Path(no_correction_path) / path_grics.name / "GricsRecon.dat.0000"

    GricsRecon_grics = load_bin(file_grics, "complex-float", (Nz, Ny, Nx))
    GricsRecon_nomoco = load_bin(file_no_moco, "complex-float", (Nz, Ny, Nx))

    output_file = path_grics / "reconstructed_image.png"

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    im2show = abs(GricsRecon_grics[GricsRecon_grics.shape[0] // 2, :, :]) 
    ax.imshow(im2show, cmap="gray")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Add reconstruction time into the file
    total_time = extract_total_time(log)
    sharpness_enhansement = []
    for i_slice in range(GricsRecon_grics.shape[0]):
        sharpness_idc_grics = sharpness_index(torch.from_numpy(np.abs(GricsRecon_grics[i_slice])))
        sharpness_idc_nomoco = sharpness_index(torch.from_numpy(np.abs(GricsRecon_nomoco[i_slice])))
        sharpness_enhansement.append(200 * (sharpness_idc_grics - sharpness_idc_nomoco) / (sharpness_idc_grics + sharpness_idc_nomoco))
        


    with open(reconstruction_times_file, "a") as f:
        f.write(f"{total_time}\n")
    with open(sharpness_enhansement_file, "a") as f:
        f.write(f"{sum(sharpness_enhansement) / len(sharpness_enhansement)}\n")
        

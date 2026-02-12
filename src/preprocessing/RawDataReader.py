import h5py
import numpy as np
import ismrmrd
import bart
import threading
import matplotlib.pyplot as plt
import math
import os
import torch
from src.utils.fftnc import fftnc, ifftnc

# from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from src.preprocessing.RespiratoryDataReader import RespiratoryDataReader

def is_noise(acq):
    return acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)

class RawDataReader:
    def __init__(self):
        pass

    @staticmethod
    def remove_oversampling(kspace: torch.Tensor):
        """
        kspace shape:
        (coils, Nex, readout, Ny, Nslices)
        """

        device = kspace.device
        dtype = kspace.dtype

        coils, Nex, readout, Ny, Nsli = kspace.shape
        cropped_readout = readout // 2

        kspace_cropped = torch.zeros(
            (coils, Nex, cropped_readout, Ny, Nsli),
            dtype=dtype,
            device=device
        )

        for iz in range(Nsli):

            # Extract slice
            kspace_slice = kspace[..., iz]  # (coils, Nex, readout, Ny)

            # IFFT along readout dimension (dim=2)
            img = ifftnc(kspace_slice, dims=(2,))

            # Crop center (remove outer quarters)
            crop_start = readout // 4
            crop_end = 3 * readout // 4
            img_cropped = img[:, :, crop_start:crop_end, :]

            # Back to k-space
            kspace_cropped[..., iz] = fftnc(img_cropped, dims=(2,))

        return kspace_cropped

    @staticmethod
    def extract_kspace_info_torch(ismrmrd_path, device="cpu"):

        dset = ismrmrd.Dataset(ismrmrd_path, 'dataset', create_if_needed=False)

        timestamps = []
        slices = []
        idx_ky = []
        idx_kz = []
        idx_nex = []

        first_timestamp = 0

        header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())

        N_SLI = header.encoding[0].encodingLimits.slice.maximum + 1
        Nex = 1
        Ny = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1

        number_of_channels, number_of_samples = dset.read_acquisition(0).data.shape

        kspace = torch.zeros(
            (number_of_channels, Nex, number_of_samples, Ny, N_SLI),
            dtype=torch.complex64,
            device=device
        )

        for i in range(dset.number_of_acquisitions()):

            acq = dset.read_acquisition(i)

            if first_timestamp == 0:
                first_timestamp = acq.acquisition_time_stamp

            if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                continue

            timestamp = (
                float(acq.acquisition_time_stamp) - first_timestamp
            ) * 2.5e-3

            timestamps.append(timestamp)
            slices.append(acq.idx.slice)
            idx_ky.append(acq.idx.kspace_encode_step_1)
            idx_kz.append(acq.idx.kspace_encode_step_2)
            idx_nex.append(acq.idx.average)

            # Convert acquisition data to torch
            acq_data = torch.from_numpy(acq.data).to(device)

            kspace[:, 0, :, acq.idx.kspace_encode_step_1, acq.idx.slice] = acq_data

        return (
            kspace,
            torch.tensor(timestamps, device=device),
            torch.tensor(slices, device=device),
            torch.tensor(idx_ky, device=device),
            torch.tensor(idx_kz, device=device),
            torch.tensor(idx_nex, device=device),
        )

    def interpolate_signal_torch(time_original, signal_original, time_target):
        """
        All inputs must be torch tensors
        """
        return torch.interp(time_target, time_original, signal_original)


    @staticmethod
    def reshape_resp_data(respiratory_data_interpolated, slices, idx_ky, idx_kz=None, idx_nex=None):
        N_SLI = np.max(slices) + 1
        motion_data = np.zeros((N_SLI, int(len(respiratory_data_interpolated)/N_SLI)))
        line_idx_y = np.zeros((N_SLI, int(len(respiratory_data_interpolated)/N_SLI)))
        line_idx_z = np.zeros((N_SLI, int(len(respiratory_data_interpolated)/N_SLI)))
        line_idx_nex = np.zeros((N_SLI, int(len(respiratory_data_interpolated)/N_SLI)))
        for i_sli in range(N_SLI):
            tmp = respiratory_data_interpolated[np.array(slices) == i_sli]
            motion_data[i_sli] = np.squeeze(tmp)
            line_idx_y[i_sli] = idx_ky[np.array(slices) == i_sli]
            if idx_kz is not None:
                line_idx_z[i_sli] = idx_kz[np.array(slices) == i_sli]
            if idx_nex is not None:
                line_idx_nex[i_sli] = idx_nex[np.array(slices) == i_sli]
        return motion_data, line_idx_y, line_idx_z, line_idx_nex
    
    @staticmethod
    def interp1d_torch(x, y, x_new):
        """
        x: (N,) original time points (must be sorted)
        y: (N,) signal values
        x_new: (M,) target time points

        Returns:
            y_new: (M,) interpolated values
        """

        # Ensure 1D
        x = x.flatten()
        y = y.flatten()
        x_new = x_new.flatten()

        # Find insertion indices
        idx = torch.searchsorted(x, x_new)

        # Clamp indices to valid range
        idx0 = torch.clamp(idx - 1, 0, len(x) - 1)
        idx1 = torch.clamp(idx, 0, len(x) - 1)

        x0 = x[idx0]
        x1 = x[idx1]
        y0 = y[idx0]
        y1 = y[idx1]

        # Avoid division by zero
        denom = (x1 - x0)
        denom[denom == 0] = 1e-12

        weight = (x_new - x0) / denom

        y_new = y0 + weight * (y1 - y0)

        return y_new
    

    @staticmethod
    def reshape_resp_data_torch(
        respiratory_data_interpolated,
        slices,
        idx_ky,
        idx_kz=None,
        idx_nex=None,
    ):
        """
        All inputs must be torch tensors.
        slices must be integer tensor.
        """

        device = respiratory_data_interpolated.device

        # Number of slices
        N_SLI = int(torch.max(slices).item()) + 1

        # Count how many lines per slice (assumes equal distribution)
        lines_per_slice = respiratory_data_interpolated.numel() // N_SLI

        # Allocate outputs
        motion_data = torch.zeros(
            (N_SLI, lines_per_slice),
            dtype=respiratory_data_interpolated.dtype,
            device=device,
        )

        line_idx_y = torch.zeros(
            (N_SLI, lines_per_slice),
            dtype=idx_ky.dtype,
            device=device,
        )

        line_idx_z = None
        line_idx_nex = None

        if idx_kz is not None:
            line_idx_z = torch.zeros(
                (N_SLI, lines_per_slice),
                dtype=idx_kz.dtype,
                device=device,
            )

        if idx_nex is not None:
            line_idx_nex = torch.zeros(
                (N_SLI, lines_per_slice),
                dtype=idx_nex.dtype,
                device=device,
            )

        # Fill slice by slice
        for i_sli in range(N_SLI):

            mask = (slices == i_sli)

            motion_data[i_sli] = respiratory_data_interpolated[mask]
            line_idx_y[i_sli] = idx_ky[mask]

            if idx_kz is not None:
                line_idx_z[i_sli] = idx_kz[mask]

            if idx_nex is not None:
                line_idx_nex[i_sli] = idx_nex[mask]

        return motion_data, line_idx_y, line_idx_z, line_idx_nex

    @staticmethod
    def read_motion_and_kspace(ismrmrd_file, saec_file, sensor_type='BELT', device="cuda"):

        # Load physiological data (still numpy)
        time_saec, resp = RespiratoryDataReader.read_and_process_data(
            saec_file, sensor_type, dimension=2
        )

        time_saec = torch.tensor(time_saec, device=device)
        resp = torch.tensor(resp, device=device)

        # Load kspace
        kspace, time_kspace, slices, idx_ky, idx_kz, idx_nex = \
            RawDataReader.extract_kspace_info_torch(ismrmrd_file, device=device)

        # Interpolate
        respiratory_interpolated = RawDataReader.interp1d_torch(
            time_saec, resp, time_kspace
        )

        motion_data, line_idx_y, line_idx_z, line_idx_nex = RawDataReader.reshape_resp_data_torch(
        respiratory_interpolated,
        slices,
        idx_ky,
        idx_kz,
        idx_nex
        )

        # Remove oversampling
        kspace = RawDataReader.remove_oversampling(kspace)

        return {
            "kspace": kspace.detach().cpu().numpy(),
            "motion_data": motion_data.detach().cpu().numpy(),
            "idx_ky": line_idx_y.detach().cpu().numpy(),
            "idx_kz": line_idx_z.detach().cpu().numpy() if line_idx_z is not None else None,
            "idx_nex": line_idx_nex.detach().cpu().numpy() if line_idx_nex is not None else None
        }

    
    @staticmethod
    def read_data_from_rawdata(ismrmrd_file, saec_file, sensor_type='BELT', h5filename=None):
        data = RawDataReader.read_motion_and_kspace(ismrmrd_file, saec_file, sensor_type)
        
        if h5filename is not None:
            with h5py.File(h5filename, 'w') as f:
                f.create_dataset('motion_data', data=data['motion_data'])
                f.create_dataset('idx_ky', data=data['idx_ky'])
                f.create_dataset('idx_kz', data=data['idx_kz'])
                f.create_dataset('idx_nex', data=data['idx_nex'])
                f.create_dataset('kspace', data=data['kspace'])
        return data

    @staticmethod
    def read_kspace_and_motion_data_from_h5(h5_path):
        data = {}
        
        with h5py.File(h5_path, 'r') as f:
            data['motion_data'] = f['motion_data'][:]
            data['line_idx'] = f['line_idx'][:]
            data['kspace'] = f['kspace'][:]
        
        return data





        
        
    

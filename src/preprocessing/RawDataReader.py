import h5py
import numpy as np
import ismrmrd
import bart
import threading
import matplotlib.pyplot as plt
import math
import os

# from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from src.preprocessing.RespiratoryDataReader import RespiratoryDataReader

def is_noise(acq):
    return acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)

class RawDataReader:
    def __init__(self):
        pass

    @staticmethod
    def remove_oversampling_slice(kspace, kspace_cropped, iz):
        kspace_slice = kspace[:,:,:,:,iz]
        kspace_fft_readout = bart.bart(1, 'fft -u -i 4', kspace_slice)

        # Determine cropping indices
        readout_size = kspace_fft_readout.shape[2]
        crop_start = readout_size // 4
        crop_end = readout_size * 3 // 4
        
        # Crop the oversampled readout direction (remove outer quarters)
        kspace_fft_readout = kspace_fft_readout[:,:,crop_start:crop_end,:]
        kspace_cropped[:,:,:,:,[iz]] = np.expand_dims(bart.bart(1, 'fft -u 4', kspace_fft_readout), axis=4) # add back slice dimension

    @staticmethod
    def remove_oversampling(kspace):
        os.putenv("OMP_THREAD_LIMIT", "1")
        os.putenv("OMP_NUM_THREADS", "1")
        os.putenv("OPENBLAS_NUM_THREADS", "1")

        kspace_cropped = np.zeros((kspace.shape[0], kspace.shape[1], kspace.shape[2] // 2, kspace.shape[3], kspace.shape[4]), dtype=np.complex128)
        threads = list()
        for iz in range(kspace.shape[4]):
            x = threading.Thread(target=RawDataReader.remove_oversampling_slice, args=(kspace, kspace_cropped, iz,))
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join()
        
        return kspace_cropped

    @staticmethod
    def extract_kspace_info(ismrmrd_path):
        dset = ismrmrd.Dataset(ismrmrd_path, 'dataset', create_if_needed=False)
        timestamps = []
        slices = []
        idx_ky = []
        idx_kz = []
        idx_nex = []

        first_timestamp = 0

        # Load XML header to access encoding limits
        header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
        N_SLI = header.encoding[0].encodingLimits.slice.maximum + 1
        Nex = 1 # header.encoding[0].encodingLimits.average.maximum + 1 # workaround to have 1 Nex for the moment TODO
        Nz = header.encoding[0].encodingLimits.kspace_encoding_step_2.maximum + 1
        Rz = header.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2
        Nz = Rz * math.ceil(float(Nz) / float(Rz))
        Ny = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1
        Ry = header.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1
        Ny = Ry * math.ceil(float(Ny) / float(Ry))
        number_of_channels, number_of_samples = dset.read_acquisition(0).data.shape
        kspace = np.zeros(
                (Nex,
                number_of_channels,
                number_of_samples,
                Ny,                
                N_SLI),
                dtype=np.complex64
            )

        for i in range(dset.number_of_acquisitions()):
            acq = dset.read_acquisition(i)
            if first_timestamp == 0:
                first_timestamp = acq.acquisition_time_stamp
            if is_noise(acq):
                continue  # Skip noise acquisitions
            timestamp = (np.float64(acq.acquisition_time_stamp) - first_timestamp) * 2.5e-3
            timestamps.append(timestamp)
            slices.append(acq.idx.slice)
            kspace[0, :, :, acq.idx.kspace_encode_step_1, acq.idx.slice] = acq.data # TODO replace 0 with acq.idx.average for multiple Nex support
            idx_ky.append(acq.idx.kspace_encode_step_1)
            idx_kz.append(acq.idx.kspace_encode_step_2)
            idx_nex.append(acq.idx.average)
        # kspace = np.transpose(kspace, (0, 1, 4, 3, 2))

        return kspace, np.array(timestamps), np.array(slices), np.array(idx_ky), np.array(idx_kz), np.array(idx_nex)

    @staticmethod
    def interpolate_signal(time_original, signal_original, time_target):
        interpolator = interp1d(time_original, signal_original, kind='linear', bounds_error=False, fill_value='extrapolate')
        return interpolator(time_target)

    @staticmethod
    def reshape_resp_data(respiratory_data_interpolated, slices, idx_ky):
        N_SLI = np.max(slices) + 1
        motion_data = np.zeros((N_SLI, int(len(respiratory_data_interpolated)/N_SLI)))
        line_idx = np.zeros((N_SLI, int(len(respiratory_data_interpolated)/N_SLI)))
        for i_sli in range(N_SLI):
            tmp = respiratory_data_interpolated[np.array(slices) == i_sli]
            motion_data[i_sli] = np.squeeze(tmp)
            line_idx[i_sli] = idx_ky[np.array(slices) == i_sli]
        return motion_data, line_idx

    @staticmethod
    def read_motion_and_kspace(ismrmrd_path, saec_path, sensor_type='BELT'):
        # Load physiological data
        time_saec, respiratory_data_filtered = RespiratoryDataReader.read_and_process_data(saec_path, sensor_type, dimension=2)
        
        # Load MRI timestamps (in seconds)
        kspace, time_kspace, slices, idx_ky, idx_kz, idx_nex = RawDataReader.extract_kspace_info(ismrmrd_path)

        # Interpolate physiological signal to k-space timestamps
        respiratory_data_interpolated = RawDataReader.interpolate_signal(time_saec, respiratory_data_filtered, time_kspace)

        motion_data, line_idx = RawDataReader.reshape_resp_data(respiratory_data_interpolated, slices, idx_ky)

        return motion_data, line_idx, kspace

    @staticmethod
    def extract_central_kspace(kspace, size=32):
        nx, ny = kspace.shape[:2]
        cx, cy = nx // 2, ny // 2  # center indices
        half = size // 2
        kspace_central = kspace[cx - half:cx + half, cy - half:cy + half, :, :]
        return kspace_central
    
    @staticmethod
    def read_data_from_rawdata(ismrmrd_file, saec_file, sensor_type='BELT', h5filename=None):
        motion_data, line_idx, kspace = RawDataReader.read_motion_and_kspace(ismrmrd_file, saec_file, sensor_type)
        kspace = RawDataReader.remove_oversampling(kspace)

            # Prepare data dictionary
        data = {
            'motion_data': motion_data,
            'line_idx': line_idx,
            'kspace': kspace,
        }
        
        if h5filename is not None:
            with h5py.File(h5filename, 'w') as f:
                f.create_dataset('motion_data', data=motion_data)
                f.create_dataset('line_idx', data=line_idx)
                f.create_dataset('kspace', data=kspace)
        return data

    @staticmethod
    def read_kspace_and_motion_data_from_h5(h5_path):
        data = {}
        
        with h5py.File(h5_path, 'r') as f:
            data['motion_data'] = f['motion_data'][:]
            data['line_idx'] = f['line_idx'][:]
            data['kspace'] = f['kspace'][:]
        
        return data





        
        
    

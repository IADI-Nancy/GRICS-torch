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

class DataReader:
    def __init__(self):
        pass

    @staticmethod
    def remove_oversampling_slice(kspace, kspace_cropped, iz):
        kspace_slice = np.expand_dims(kspace[:,:,iz,:], axis=2)
        kspace_fft_readout = bart.bart(1, 'fft -u -i 1', kspace_slice)

        # Determine cropping indices
        readout_size = kspace_fft_readout.shape[0]
        crop_start = readout_size // 4
        crop_end = readout_size * 3 // 4
        
        # Crop the oversampled readout direction (remove outer quarters)
        kspace_fft_readout = kspace_fft_readout[crop_start:crop_end,:,:]
        kspace_cropped[:,:,[iz],:] = bart.bart(1, 'fft -u 1', kspace_fft_readout)

    @staticmethod
    def remove_oversampling(kspace):
        os.putenv("OMP_THREAD_LIMIT", "1")
        os.putenv("OMP_NUM_THREADS", "1")
        os.putenv("OPENBLAS_NUM_THREADS", "1")

        kspace_cropped = np.zeros((kspace.shape[0]//2, kspace.shape[1], kspace.shape[2], kspace.shape[3]), dtype=np.complex128)
        threads = list()
        for iz in range(kspace.shape[2]) :
            x = threading.Thread(target=DataReader.remove_oversampling_slice, args=(kspace, kspace_cropped, iz,))
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

        first_timestamp = 0

        # Load XML header to access encoding limits
        header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
        N_SLI = header.encoding[0].encodingLimits.slice.maximum + 1
        Ny = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1
        Ry = header.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1
        Ny = Ry * math.ceil(float(Ny) / float(Ry))
        number_of_channels, number_of_samples = dset.read_acquisition(0).data.shape
        kspace = np.zeros(
                (number_of_channels,
                N_SLI,
                Ny,
                number_of_samples),
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
            kspace[:, acq.idx.slice, acq.idx.kspace_encode_step_1, :] = acq.data
            idx_ky.append(acq.idx.kspace_encode_step_1)
        kspace = np.transpose(kspace, (3, 2, 1, 0))

        return np.array(timestamps), np.array(slices), np.array(idx_ky), kspace

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

    def prepare_motion_and_kspace(saec_path, ismrmrd_path, sensor_type='BELT'):
        # Load physiological data
        time_saec, respiratory_data_filtered = RespiratoryDataReader.read_and_process_data(saec_path, sensor_type, dimension=2)
        
        # Load MRI timestamps (in seconds)
        time_kspace, slices, idx_ky, kspace = DataReader.extract_kspace_info(ismrmrd_path)

        # Interpolate physiological signal to k-space timestamps
        respiratory_data_interpolated = DataReader.interpolate_signal(time_saec, respiratory_data_filtered, time_kspace)

        motion_data, line_idx = DataReader.reshape_resp_data(respiratory_data_interpolated, slices, idx_ky)

        return motion_data, line_idx, kspace

    @staticmethod
    def find_motion_bins(motion_data, line_idx, Nbins=8):
        NSLI, Ny = motion_data.shape
        binned_indices = np.empty((NSLI, Nbins), dtype=object)
        bin_centers = np.empty((NSLI, Nbins))
        for s in range(NSLI):
            # Get motion and line index for current slice
            motion = motion_data[s]
            lines = line_idx[s]

            # Define bin edges for this slice
            min_val, max_val = np.min(motion), np.max(motion)
            bins = np.linspace(min_val, max_val, Nbins + 1)
            bin_centers[s] = (bins[:-1] + bins[1:]) / 2.0

            # Digitize motion data into bins
            bin_ids = np.digitize(motion, bins) - 1
            bin_ids = np.clip(bin_ids, 0, Nbins - 1)  # Ensure all bin_ids are valid

            # Initialize list for each bin
            slice_bins = [[] for _ in range(Nbins)]

            # Assign line indices to corresponding bins
            for i in range(Ny):
                slice_bins[bin_ids[i]].append(lines[i])

            # Convert lists to arrays for consistency
            binned_indices[s] = [np.array(b, dtype=np.int32) for b in slice_bins]

        return binned_indices, bin_centers

    @staticmethod
    def extract_central_kspace(kspace, size=32):
        nx, ny = kspace.shape[:2]
        cx, cy = nx // 2, ny // 2  # center indices
        half = size // 2
        kspace_central = kspace[cx - half:cx + half, cy - half:cy + half, :, :]
        return kspace_central

    @staticmethod
    def calculate_sensitiviy_maps_BART(kspace, smap, iz):
        csm_ksp = np.expand_dims(kspace[:,:,iz,:], axis=2)
        smap[:,:,[iz],:] = bart.bart(1, 'ecalib -m1', csm_ksp)

    @staticmethod
    def calculate_sensitivity_maps_parallel(kspace):
        os.putenv("OMP_THREAD_LIMIT", "1")
        os.putenv("OMP_NUM_THREADS", "1")
        os.putenv("OPENBLAS_NUM_THREADS", "1")

        smap = np.zeros((kspace.shape[0], kspace.shape[1], kspace.shape[2], kspace.shape[3]), dtype=np.complex128)
        threads = list()
        for iz in range(kspace.shape[2]) :
            x = threading.Thread(target=DataReader.calculate_sensitiviy_maps_BART, args=(kspace, smap, iz,))
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join()
        
        return smap

    @staticmethod
    def calculate_prior_image(kspace, reg_size):
        coil_images = bart.bart(1, 'fft -u -i 3', kspace)
        calib_image_sos = np.sqrt(np.sum(np.square(np.abs(coil_images)), 3))

        # Filter calculation
        Hamming_window = np.hamming(reg_size)
        Hamming_window_x = np.pad(Hamming_window, (int(np.floor((calib_image_sos.shape[0] - reg_size)/2)), int(np.ceil((calib_image_sos.shape[0] - reg_size)/2))), \
                                'constant', constant_values=(0, 0))
        Hamming_window_y= np.pad(Hamming_window, (int(np.floor((calib_image_sos.shape[1] - reg_size)/2)), int(np.ceil((calib_image_sos.shape[1] - reg_size)/2))), \
                                'constant', constant_values=(0, 0))
        Hamming_window_xy = np.outer(Hamming_window_x, Hamming_window_y)
        
        # Filtering
        calib_image_sos_kspace = bart.bart(1, 'fft -u 3', calib_image_sos)
        calib_image_sos_kspace = np.multiply(calib_image_sos_kspace, np.tile(Hamming_window_xy[:,:,np.newaxis], (1,1,calib_image_sos.shape[2])))
        prior_image = bart.bart(1, 'fft -u -i 3', calib_image_sos_kspace)
        
        # Normalization and type casting
        prior_image = abs(prior_image)
        prior_image = prior_image * (prior_image.shape[0] * prior_image.shape[1] * prior_image.shape[2]) / np.sum(prior_image)
        prior_image = prior_image.astype(np.complex128)
        return prior_image
    
    @staticmethod
    def read_kspace_and_motion_data_from_rawdata(ismrmrd_file, saec_file, sensor_type='BELT', Nbins=8, h5filename=None):
        motion_data, line_idx, kspace = DataReader.prepare_motion_and_kspace(saec_file, ismrmrd_file, sensor_type)
        binned_indices, bin_centers = DataReader.find_motion_bins(motion_data, line_idx, Nbins=Nbins)

        smap = DataReader.calculate_sensitivity_maps_parallel(kspace)
        readout_size = smap.shape[0]
        crop_start = readout_size // 4
        crop_end = readout_size * 3 // 4
        smap = smap[crop_start:crop_end,:,:,:]

        kspace = DataReader.remove_oversampling(kspace)

        reg_size = 32
        # ToDo: parallelize (Fourier transforms separately)
        prior_image = DataReader.calculate_prior_image(kspace, reg_size)
        # save in .h5 formatThis

            # Prepare data dictionary
        data = {
            'motion_data': motion_data,
            'prior_image': prior_image,
            'line_idx': line_idx,
            'kspace': kspace,
            'smap': smap,
            'bin_centers': bin_centers,
            'binned_indices': binned_indices
        }
        
        if h5filename is not None:
            with h5py.File(h5filename, 'w') as f:
                f.create_dataset('motion_data', data=motion_data)
                f.create_dataset('prior_image', data=prior_image)
                f.create_dataset('line_idx', data=line_idx)
                f.create_dataset('kspace', data=kspace)
                f.create_dataset('smap', data=smap)
                f.create_dataset('bin_centers', data=bin_centers)
                # Save binned_indices as variable-length HDF5 dataset
                # Create a group to store binned indices for each slice and bin
                grp_binned = f.create_group('binned_indices')
                NSLI, Nbins = binned_indices.shape
                for s in range(NSLI):
                    grp_slice = grp_binned.create_group(f'slice_{s}')
                    for b in range(Nbins):
                        grp_slice.create_dataset(f'bin_{b}', data=binned_indices[s, b])
        return data

    @staticmethod
    def read_kspace_and_motion_data_from_h5(h5_path):
        """
        Read breast motion data from HDF5 file.
        
        Returns:
            dict: Contains motion_data, prior_image, line_idx, kspace, smap, bin_centers, binned_indices
        """
        data = {}
        
        with h5py.File(h5_path, 'r') as f:
            # Read simple datasets
            data['motion_data'] = f['motion_data'][:]
            data['prior_image'] = f['prior_image'][:]
            data['line_idx'] = f['line_idx'][:]
            data['kspace'] = f['kspace'][:]
            data['smap'] = f['smap'][:]
            data['bin_centers'] = f['bin_centers'][:]
            
            # Read binned_indices (nested group structure)
            binned_indices_group = f['binned_indices']
            NSLI = len(binned_indices_group)
            Nbins = len(binned_indices_group['slice_0'])
            
            binned_indices = np.empty((NSLI, Nbins), dtype=object)
            
            for s in range(NSLI):
                slice_group = binned_indices_group[f'slice_{s}']
                for b in range(Nbins):
                    binned_indices[s, b] = slice_group[f'bin_{b}'][:]
            
            data['binned_indices'] = binned_indices
        
        return data


# Usage example
if __name__ == '__main__':
    saec_file = 'data/2008-003 01-1724_S11_20210323_151329.h5'
    ismrmrd_file = 'data/t2_1724.h5'
    data = DataReader.read_kspace_and_motion_data_from_rawdata(ismrmrd_file, saec_file, \
                                                        sensor_type='BELT', Nbins=8,\
                                                        h5filename='data/breast_motion_data.h5')
    # Read the data
    data = DataReader.read_kspace_and_motion_data_from_h5('data/breast_motion_data.h5')
    
    # Access individual datasets
    motion_data = data['motion_data']
    prior_image = data['prior_image']
    line_idx = data['line_idx']
    kspace = data['kspace']
    smap = data['smap']
    bin_centers = data['bin_centers']
    binned_indices = data['binned_indices']
    
    # Print shapes to verify
    print(f"motion_data shape: {motion_data.shape}")
    print(f"prior_image shape: {prior_image.shape}")
    print(f"line_idx shape: {line_idx.shape}")
    print(f"kspace shape: {kspace.shape}")
    print(f"smap shape: {smap.shape}")
    print(f"bin_centers shape: {bin_centers.shape}")
    print(f"binned_indices shape: {binned_indices.shape}")
    
    # Access a specific binned index array
    print(f"\nExample - binned_indices[0, 0]: {binned_indices[0, 0]}")
    print(f"Type: {type(binned_indices[0, 0])}")





        
        
    

# Import SAEC reader
import os
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from h5wrapper.h5Saec import *

class RespiratoryDataReader:
    @staticmethod
    def get_respiration_from_saec(filename, sensor_type, flag_LR=False):
        SAECData = h5Saec.from_file(filename.strip())
        ticksTo1s = SAECData.attributes.ticksTo1s
        respiratory_data = []
        timestampsSAEC = []
        for attr, value in SAECData.__dict__.items():
            if sensor_type == 'BELT' :
                if 'SAEC_RESP' in attr:
                    respiratory_data.append(value.RESP.datas.values.astype(np.float64))
                    timestampsSAEC.append(value.RESP.timestamp.values)
            # elif sensor_type == '1MARMOT' or 'ALL_MARMOTs': # KISA !!!
            else:
                if 'MARMOT' in attr:
                    try:
                        if value.ACC.datas.values.astype(np.float64).size != 0:
                            respiratory_data.append(value.ACC.datas.values.astype(np.float64))
                            timestampsSAEC.append(value.ACC.timestamp.values)
                    except:
                        print("Accelerometer data was not found for a MARMOT")

            if 'SAEC_TRIGGER_SIEMENS' in attr:
                max_duration = 0
                # sequence_start = value.SeqStart.timestamp.values[0]
                for iStart in range(-1, -len(value.SeqStart.timestamp.values) - 1, -1) :
                    curr_stop = len(value.SeqStart.timestamp.values)
                    curr_duration = np.double(np.max(np.concatenate(timestampsSAEC, axis=0 ))) - np.double(np.min(np.concatenate(timestampsSAEC, axis=0 )))
                    for iStop in range(-1, -len(value.SeqStop.timestamp.values) - 1, -1) :
                        duration = np.double(value.SeqStop.timestamp.values[iStop]) - np.double(value.SeqStart.timestamp.values[iStart])
                        if duration > 0 and duration < curr_duration :
                            curr_stop = iStop
                            curr_duration = duration
                    if curr_stop < len(value.SeqStart.timestamp.values) :
                        if np.double(value.SeqStop.timestamp.values[curr_stop]) - np.double(value.SeqStart.timestamp.values[iStart]) > max_duration :
                            sequence_start = value.SeqStart.timestamp.values[iStart]
                            sequence_stop = value.SeqStop.timestamp.values[iStop]
                            max_duration = np.double(value.SeqStop.timestamp.values[curr_stop]) - np.double(value.SeqStart.timestamp.values[iStart])

        timestamps_in_sec = []
        for timestamp in timestampsSAEC:
            timestamp_in_sec = (np.float64(timestamp) - np.float64(sequence_start)) / ticksTo1s
            timestamps_in_sec.append(timestamp_in_sec)

        return np.asarray(timestamps_in_sec), respiratory_data

    @staticmethod
    def detect_MARMOT_displacement(timestamps, respiratory_data_filtered_lp, respiratory_data_filtered_hp, i_MARMOT):
        threshold = 0.002# 0.001
        displaced = np.zeros(respiratory_data_filtered_lp.shape[1])
        for i in range(respiratory_data_filtered_lp.shape[1]):
            MARMOT_lp = respiratory_data_filtered_lp[:, i] - np.mean(respiratory_data_filtered_lp[:, i])
            MARMOT_hp = respiratory_data_filtered_hp[:, i]

            MARMOT_dif = MARMOT_lp - MARMOT_hp

            time = 0
            segment_duration = int(50 / (timestamps[-1] - timestamps[0]) * len(timestamps))
            sigma = np.std(MARMOT_hp)
            while (time + segment_duration) < len(timestamps):
                sigma_tmp = np.std(MARMOT_hp[time:time + segment_duration])
                time = time + segment_duration
                if sigma_tmp < sigma:
                    sigma = sigma_tmp
            try:
                sigma_tmp = np.std(MARMOT_hp[time:time + segment_duration])
            except:
                sigma_tmp = sigma
            if sigma_tmp < sigma:
                sigma = sigma_tmp

            dif_norm = np.sqrt(np.sum(np.power(abs(MARMOT_dif),2))) / len(MARMOT_dif) / sigma 
            print(str(i_MARMOT+1) + '\t' + str(i+1) + '\t' + str(dif_norm))
            displaced[i] = 1 if dif_norm < threshold else 0
        return displaced

    @staticmethod
    def get_filtered_MARMOT_data(timestamps, respiratory_data, i_sensor):
        fsampling = np.float64(len(timestamps[i_sensor])) / (timestamps[i_sensor][-1] - timestamps[i_sensor][0])

        order = 1
        fcut = 0.3
        normal_cutoff = fcut / (fsampling / 2)

        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        respiratory_data_filtered_lp = np.zeros(respiratory_data[i_sensor].shape)
        for idx_track in range(3):
            respiratory_data_filtered_lp[:, idx_track] = filtfilt(b, a, respiratory_data[i_sensor][:, idx_track])

        order = 1
        fcut = 0.03
        normal_cutoff = fcut / (fsampling / 2)

        b, a = butter(order, normal_cutoff, btype='hp', analog=False)
        respiratory_data_filtered_hp = np.zeros(respiratory_data[i_sensor].shape)
        sigma = np.zeros(3)
        for idx_track in range(3):
            respiratory_data_filtered_hp[:, idx_track] = filtfilt(b, a, respiratory_data_filtered_lp[:, idx_track])
            sigma[idx_track] = np.std(respiratory_data_filtered_hp[:, idx_track])

        displaced = RespiratoryDataReader.detect_MARMOT_displacement(timestamps[i_sensor], respiratory_data_filtered_lp, respiratory_data_filtered_hp, i_sensor)
        sigma[displaced == 0] = 0

        return respiratory_data_filtered_hp, sigma

    @staticmethod
    def get_filtered_resp_data(timestamps, respiratory_data, sersor_type, dimension=2, path_to_graph=None):
        if sersor_type == 'BELT' :
            timestamps = np.squeeze(timestamps[0])
            respiratory_data = respiratory_data[0]
            order = 1
            fcut = 3.
            fsampling = np.float64(len(timestamps)) / (timestamps[-1] - timestamps[0])
            normal_cutoff = fcut / (fsampling / 2)

            b, a = butter(order, normal_cutoff, btype='low', analog=False)

            sigma1 = np.std(respiratory_data[:, 0])
            sigma2 = np.std(respiratory_data[:, 1])
            idx_track = 0 if sigma1 > sigma2 else 1

            respiratory_data_filtered = filtfilt(b, a, respiratory_data[:, idx_track])

            # Remove the outliers
            mean = np.mean(respiratory_data_filtered)
            sigma = np.std(respiratory_data_filtered)
            respiratory_data_within_2_sigma = np.where(respiratory_data_filtered > mean + 2 * sigma, mean + 2 * sigma, respiratory_data_filtered)
            respiratory_data_within_2_sigma = np.where(respiratory_data_filtered < mean - 2 * sigma, mean - 2 * sigma, respiratory_data_filtered)
            a2, a1, a0 = np.polyfit(timestamps, respiratory_data_within_2_sigma, 2)

            # Correction of the drift

            correction = -(a2 * timestamps**2 + a1 * timestamps + a0)
            respiratory_data_filtered = respiratory_data_filtered + correction

            sigma = np.std(respiratory_data_filtered)

            if path_to_graph is not None:
                plt.plot(timestamps, respiratory_data_filtered)
                plt.savefig(os.path.join(path_to_graph, path_to_graph))

            return respiratory_data_filtered

        elif sersor_type == '1MARMOT' : # not tested
            respiratory_data_filtered = []
            max_sigma = np.zeros(len(respiratory_data))
            tracks = np.zeros(len(respiratory_data))
            for i_sensor in range(len(respiratory_data)):
                respiratory_data_filtered_hp, sigma = RespiratoryDataReader.get_filtered_MARMOT_data(timestamps, respiratory_data, i_sensor)
                
                track_idx = np.argmax(sigma)
                max_sigma[i_sensor] = sigma[track_idx]
                tracks[i_sensor] = track_idx
                respiratory_data_filtered.append(respiratory_data_filtered_hp[:, track_idx])

            sensor_idx = np.argmax(max_sigma)
            respiratory_data_MARMOT = respiratory_data_filtered[sensor_idx]
            respiratory_data_MARMOT = respiratory_data_MARMOT / np.std(respiratory_data_MARMOT)
                    
            return np.squeeze(respiratory_data_MARMOT)
        else:
            Warning("Physiological sensor type is not correct")

    @staticmethod
    def read_and_process_data(saec_filename, sensor_type, dimension, path_to_graph=None):
        timestamps_saec, respiratory_data_saec = RespiratoryDataReader.get_respiration_from_saec(saec_filename, sensor_type)
        respiratory_data_filtered = RespiratoryDataReader.get_filtered_resp_data(timestamps_saec, respiratory_data_saec, sensor_type, dimension=dimension, path_to_graph=path_to_graph)
        return np.squeeze(timestamps_saec), np.squeeze(respiratory_data_filtered)
"""Prepare MRI and physiological recordings together, outside their readers."""

import h5py
import numpy as np
import torch

from src.preprocessing.ISMRMRDReader import ISMRMRDReader
from src.preprocessing.physiological_data.SAECReader import SAECReader
from src.preprocessing.physiological_data.PolarisInfraredTrackerReader import PolarisInfraredTrackerReader


class RawDataPreparer:
    """Coordinate reading, synchronization, slice grouping, and optional H5 export.

    polaris_channel_mode="all" uses tool Tx/Ty/Tz. "largest-amplitude" uses
    the axis with greatest peak-to-peak range at full-sequence MRI readout times,
    measured after low-pass filtering and before normalization. The caller selects the mode explicitly.
    Both physiological formats use the same synchronizer before slice selection.
    ISMRMRDReader is responsible only for MRI acquisition reading and mapping.
    """

    def __init__(self, ismrmrd_file, physiological_file, *, physiological_format,
                 sensor_type, device, print_raw_calibration_lines, polaris_channel_mode):
        if physiological_format not in {"SAEC", "PolarisInfraredTracker"}:
            raise ValueError("Unsupported physiological format.")
        # Readers own format-specific processing; the logging flag affects only raw calibration messages.
        self.polaris_channel_mode = polaris_channel_mode
        self.physiological_reader = (
            PolarisInfraredTrackerReader(channel_mode=polaris_channel_mode)
            if physiological_format == "PolarisInfraredTracker"
            else SAECReader(sensor_type=sensor_type))
        self.reader = ISMRMRDReader(ismrmrd_file, device=device, print_raw_calibration_lines=print_raw_calibration_lines)
        self.physiological_file = physiological_file
        self.physiological_format = physiological_format
        self.sensor_type = sensor_type
        self.device = device
        self.synchronization = None

    def _physiological_channels(self):
        return self.physiological_reader.read_channels(self.physiological_file)

    @property
    def selected_polaris_channels(self):
        return self.physiological_reader.metadata.get('polaris_channels')

    @property
    def polaris_peak_to_peak(self):
        return self.physiological_reader.metadata.get('polaris_peak_to_peak_xyz')

    @staticmethod
    def _synchronize_to_sequence_end(channel_times, channel_values, acquisition_times,
                                    *, source_sequence_end=None, bounds='raise'):
        """Align physiological channels and interpolate onto full-sequence MRI times.

        channel_times and channel_values are lists of 1D arrays, one per channel;
        channels may have different lengths and sampling rates. All times are seconds.
        acquisition_times are already relative to the full MRI sequence end and are
        never re-zeroed here (a selected slice may end earlier).

        source_sequence_end is the physiological sequence-end timestamp. Use zero
        for SAEC times already referenced to its recorded Siemens stop trigger. If
        None, each channel's last sample defines sequence end (Polaris convention).
        No clock drift correction is applied. Returns [readout, channel] float64 data.

        bounds='raise' rejects uncovered readouts. bounds='edge' holds endpoint values,
        preserving the existing SAEC interpolation behavior. Invalid values are
        rejected. Repeated source timestamps are expanded to a uniform grid,
        retaining every sample from packetized physiological sensors.
        """
        target = np.asarray(acquisition_times, dtype=np.float64)
        if target.ndim != 1 or not target.size or not np.isfinite(target).all():
            raise ValueError('Acquisition times must be a nonempty finite 1D array.')
        if bounds not in {'raise', 'edge'}:
            raise ValueError("bounds must be 'raise' or 'edge'.")
        if not len(channel_values) or len(channel_times) != len(channel_values):
            raise ValueError('Provide one timestamp array per physiological channel.')
        if source_sequence_end is not None and not np.isfinite(source_sequence_end):
            raise ValueError('Sequence end must be finite.')
        interpolated = []
        for index, (times, values) in enumerate(zip(channel_times, channel_values)):
            times = np.asarray(times, dtype=np.float64)
            values = np.asarray(values, dtype=np.float64)
            if times.ndim != 1 or not times.size or values.shape != times.shape:
                raise ValueError(f'Channel {index}: times and values must be matching nonempty 1D arrays.')
            if not np.isfinite(times).all() or not np.isfinite(values).all():
                raise ValueError(f'Channel {index}: invalid physiological times or values; resolve gaps first.')
            differences = np.diff(times)
            if np.any(differences < 0):
                raise ValueError(f'Channel {index}: timestamps must be nondecreasing.')
            if times.size < 2 or times[-1] <= times[0]:
                raise ValueError(
                    f'Channel {index}: at least two distinct timestamps are required.'
                )
            if np.any(differences == 0):
                # MARMOT and potentially BELT streams may be packetized: one
                # transport timestamp labels several uniformly sampled values.
                # Build one grid per input channel (and thus one grid shared by
                # all three axes of a MARMOT ACC sample), retaining all values.
                times = np.linspace(times[0], times[-1], num=times.size)
            end = times[-1] if source_sequence_end is None else source_sequence_end
            relative_times = times - end
            if bounds == 'raise' and (target.min() < relative_times[0] - 1e-9
                                      or target.max() > relative_times[-1] + 1e-9):
                raise ValueError(f'Channel {index}: physiological data do not cover MRI readouts; '
                                 'cannot synchronize without extrapolation.')
            interpolated.append(np.interp(target, relative_times, values))
        return np.column_stack(interpolated)

    def _prepare_data(self):
        raw = self.reader.read_data()
        times, values, end, bounds = self._physiological_channels()
        acquisition_times = raw["time_seconds"].detach().cpu().numpy()
        interpolated = self._synchronize_to_sequence_end(
            times, values, acquisition_times, source_sequence_end=end, bounds=bounds)
        self.synchronization = {
            "physiological_time_seconds": [t - (t[-1] if end is None else end) for t in times],
            "physiological_values": values,
            "acquisition_time_seconds": acquisition_times,
            "acquisition_values": interpolated,
            "slice_indices": raw["slice_indices"].detach().cpu().numpy(),
        }
        motion = self.physiological_reader.prepare_motion(interpolated)
        motion, ky, kz, nex = self._reshape_data_slicewise(
            torch.as_tensor(motion, device=self.device), raw["slice_indices"],
            raw["idx_ky"], raw["idx_kz"], raw["idx_nex"],
            group_by_z_index=not raw["uses_kz_as_volume_axis"])
        self._uses_kz_as_volume_axis = raw["uses_kz_as_volume_axis"]
        data = {
            "kspace": self.reader._remove_oversampling(raw["kspace"]).detach().cpu().numpy(),
            "motion_data": motion.detach().cpu().numpy(),
            "idx_ky": ky.detach().cpu().numpy(), "idx_kz": kz.detach().cpu().numpy(),
            "idx_nex": nex.detach().cpu().numpy(), "nex_source": raw["nex_source"],
            "nex_values": raw["nex_values"].detach().cpu().numpy(),
            "slice_geometry": raw["slice_geometry"],
        }
        if raw["reference_kspace"] is not None:
            data["reference_kspace"] = self.reader._remove_oversampling(
                raw["reference_kspace"]).detach().cpu().numpy()
        return data

    def _reshape_data_slicewise(self, respiratory_data_interpolated, z_indices,
        idx_ky, idx_kz, idx_nex, group_by_z_index=True):

        device = respiratory_data_interpolated.device

        if not group_by_z_index:
            # 3D slab acquisition: keep one row per readout and one column per physiological sensor.
            return (respiratory_data_interpolated, idx_ky.reshape(1, -1), idx_kz.reshape(1, -1), idx_nex.reshape(1, -1))

        N_SLI = int(torch.max(z_indices).item()) + 1

        counts = torch.bincount(z_indices, minlength=N_SLI)
        if torch.any(counts != counts[0]):
            raise ValueError("Acquisition lines per z-index are not uniform; cannot reshape into " "[Nz, Nlines] realworld format.")
        lines_per_slice = int(counts[0].item())

        motion_data = torch.zeros((N_SLI, lines_per_slice, respiratory_data_interpolated.shape[1]),
            dtype=respiratory_data_interpolated.dtype, device=device)

        line_idx_y = torch.zeros((N_SLI, lines_per_slice), dtype=idx_ky.dtype, device=device)
        line_idx_z = torch.zeros((N_SLI, lines_per_slice), dtype=idx_kz.dtype, device=device)
        line_idx_nex = torch.zeros((N_SLI, lines_per_slice), dtype=idx_nex.dtype, device=device)

        for i_sli in range(N_SLI):
            mask = (z_indices == i_sli)

            motion_data[i_sli] = respiratory_data_interpolated[mask]
            line_idx_y[i_sli] = idx_ky[mask]
            line_idx_z[i_sli] = idx_kz[mask]
            line_idx_nex[i_sli] = idx_nex[mask]

        return motion_data, line_idx_y, line_idx_z, line_idx_nex


    def read_data(self, output_h5_file=None, slice_idx=None):
        """Return reconstruction-ready arrays, optionally selecting a slice/exporting H5.

        Synchronization always uses the complete sequence before slice selection.
        ``self.synchronization`` retains the full aligned traces for display.
        """
        data = self._prepare_data()

        if slice_idx is not None:
            if self._uses_kz_as_volume_axis:
                raise ValueError("Slice selection requires a 2D multi-slice acquisition.")
            n_slices = int(data["kspace"].shape[-1])
            if slice_idx < 0 or slice_idx >= n_slices:
                raise ValueError(
                    f"slice_idx={slice_idx} is out of range for {n_slices} slices."
                )
            data = {
                **data,
                "kspace": data["kspace"][..., [slice_idx]],
                "motion_data": data["motion_data"][[slice_idx], :],
                "idx_ky": data["idx_ky"][[slice_idx], :],
                "idx_kz": data["idx_kz"][[slice_idx], :],
                "idx_nex": data["idx_nex"][[slice_idx], :],
                "slice_geometry": {0: data["slice_geometry"][slice_idx]},
            }
            if "reference_kspace" in data:
                data["reference_kspace"] = data["reference_kspace"][..., [slice_idx]]

        if output_h5_file is not None:
            with h5py.File(output_h5_file, 'w') as f:
                f.create_dataset('motion_data', data=data['motion_data'])
                f.create_dataset('idx_ky', data=data['idx_ky'])
                f.create_dataset('idx_kz', data=data['idx_kz'])
                f.create_dataset('idx_nex', data=data['idx_nex'])
                f.create_dataset('kspace', data=data['kspace'])
                if 'reference_kspace' in data:
                    f.create_dataset('reference_kspace', data=data['reference_kspace'])
                f.create_dataset('nex_values', data=data['nex_values'])
                f.attrs['nex_source'] = data['nex_source']
                for name, value in self.physiological_reader.metadata.items():
                    f.attrs[name] = value
            data['realworld_h5_path'] = output_h5_file

        return data

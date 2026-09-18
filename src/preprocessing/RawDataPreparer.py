"""Prepare MRI and physiological recordings together, outside their readers."""

import h5py
import hashlib
import json
import warnings
from pathlib import Path
from src.runtime.data_cache import acquire_cached, cache_key
from src.runtime.hdf5_cache import write_tree, read_tree
import numpy as np
import torch

from src.preprocessing.ISMRMRDReader import ISMRMRDReader
from src.preprocessing.physiological_data.SAECReader import SAECReader
from src.preprocessing.physiological_data.PreprocessedPhysioReader import PreprocessedPhysioReader
from src.preprocessing.physiological_data.PolarisInfraredTrackerReader import PolarisInfraredTrackerReader


class RawDataPreparer:
    """Coordinate reading, synchronization, slice grouping, and optional shared H5 caching.

    polaris_channel_mode="all" uses tool Tx/Ty/Tz. "largest-amplitude" uses
    the axis with greatest peak-to-peak range at full-sequence MRI readout times,
    measured after low-pass filtering and before normalization. The caller selects the mode explicitly.
    Timestamped physiological formats use the same synchronizer before slice selection.
    ISMRMRDReader is responsible only for MRI acquisition reading and mapping.
    """

    def __init__(self, ismrmrd_file, physiological_file, *, physiological_format,
                 sensor_type, device, print_raw_calibration_lines, polaris_channel_mode,
                 physio_clock_drift_seconds=0.0):
        if physiological_format not in {"SAEC", "PolarisInfraredTracker", "physio_text", "physio_array"}:
            raise ValueError("Unsupported physiological format.")
        # Readers own format-specific processing; the logging flag affects only raw calibration messages.
        if isinstance(physio_clock_drift_seconds, bool) or not isinstance(physio_clock_drift_seconds, (int, float)) or not np.isfinite(physio_clock_drift_seconds):
            raise ValueError("physio_clock_drift_seconds must be a finite number.")
        if physiological_format == "SAEC" and physio_clock_drift_seconds != 0:
            raise ValueError("Clock correction is supported only for Polaris, text and array physiology.")
        self.physio_clock_drift_seconds = float(physio_clock_drift_seconds)
        self.polaris_channel_mode = polaris_channel_mode
        if physiological_format in {"physio_text", "physio_array"}:
            self.physiological_reader = PreprocessedPhysioReader(physiological_format)
        elif physiological_format == "PolarisInfraredTracker":
            self.physiological_reader = PolarisInfraredTrackerReader(channel_mode=polaris_channel_mode)
        else:
            self.physiological_reader = SAECReader(sensor_type=sensor_type)
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
                                    *, source_sequence_end=None, bounds='raise',
                                    physio_clock_drift_seconds=0.0):
        """Align physiological channels and interpolate onto full-sequence MRI times.

        channel_times and channel_values are lists of 1D arrays, one per channel;
        channels may have different lengths and sampling rates. All times are seconds.
        acquisition_times are already relative to the full MRI sequence end and are
        never re-zeroed here (a selected slice may end earlier).

        source_sequence_end is the physiological sequence-end timestamp. Use zero
        for SAEC times already referenced to its recorded Siemens stop trigger. If
        None, each channel's last sample defines sequence end (Polaris convention).
        The clock correction is added after end alignment. Returns [readout, channel] float64 data.

        bounds='autoregression' extends up to one second at either edge.
        bounds='raise' rejects uncovered readouts. bounds='edge' holds endpoint values,
        preserving the existing SAEC interpolation behavior. Invalid values are
        rejected. Repeated source timestamps are expanded to a uniform grid,
        retaining every sample from packetized physiological sensors.
        """
        target = np.asarray(acquisition_times, dtype=np.float64)
        if target.ndim != 1 or not target.size or not np.isfinite(target).all():
            raise ValueError('Acquisition times must be a nonempty finite 1D array.')
        if not np.isfinite(physio_clock_drift_seconds):
            raise ValueError('physio_clock_drift_seconds must be finite.')
        if bounds not in {'raise', 'edge', 'autoregression'}:
            raise ValueError("bounds must be 'raise', 'edge' or 'autoregression'.")
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
            relative_times = times - end + physio_clock_drift_seconds
            if bounds == "autoregression":
                relative_times, values = RawDataPreparer._extend_autoregression(
                    relative_times, values, target, index)
            if bounds == 'raise' and (target.min() < relative_times[0] - 1e-9
                                      or target.max() > relative_times[-1] + 1e-9):
                raise ValueError(f'Channel {index}: physiological data do not cover MRI readouts; '
                                 'cannot synchronize without extrapolation.')
            interpolated.append(np.interp(target, relative_times, values))
        return np.column_stack(interpolated)

    @staticmethod
    def _extend_autoregression(times, values, target, channel):
        """Fit local AR(p) with an intercept; reverse history for left extrapolation.

        Fit on up to ten seconds of uniformly resampled history, with at most
        20 lags and at least roughly three observations per fitted lag.
        Keep original samples for interpolation within the measured interval.
        """
        left = max(0.0, times[0] - target.min())
        right = max(0.0, target.max() - times[-1])
        if max(left, right) > 1.0 + 1e-9:
            raise ValueError(f'Channel {channel}: not reasonable to extrapolate so far '
                             f'({max(left, right):.6g} seconds); maximum is 1 second.')
        if max(left, right) <= 1e-9:
            return times, values
        step = float(np.median(np.diff(times)))

        def predict(reverse, duration):
            history_span = min(10.0, times[-1] - times[0])
            size = max(2, int(np.floor(history_span / step)) + 1)
            grid = (np.linspace(times[0], times[0] + history_span, size) if reverse
                    else np.linspace(times[-1] - history_span, times[-1], size))
            history = np.interp(grid, times, values)
            if reverse:
                history = history[::-1]
            # Use the actual uniform spacing, including for short recordings.
            spacing = history_span / (size - 1)
            count = int(np.ceil(duration / spacing))
            offset = history.mean()
            history = history - offset
            order = min(20, max(1, (size - 1) // 3))
            if size == 2:
                coefficients = np.array([history[-1] - history[-2], 1.0])
            else:
                design = np.column_stack([np.ones(size - order)] + [
                    history[order - lag:size - lag] for lag in range(1, order + 1)])
                coefficients = np.linalg.lstsq(design, history[order:], rcond=None)[0]
            buffer = list(history)
            for _ in range(count):
                value = coefficients[0] + np.dot(coefficients[1:], buffer[-order:][::-1])
                if not np.isfinite(value):
                    raise ValueError(f'Channel {channel}: autoregressive extrapolation is nonfinite.')
                buffer.append(value)
            extension = np.asarray(buffer[-count:]) + offset
            distances = spacing * np.arange(1, count + 1)
            return ((times[0] - distances)[::-1], extension[::-1]) if reverse else (times[-1] + distances, extension)

        left_times, left_values = predict(True, left) if left > 1e-9 else ([], [])
        right_times, right_values = predict(False, right) if right > 1e-9 else ([], [])
        return (np.concatenate([left_times, times, right_times]),
                np.concatenate([left_values, values, right_values]))

    def _prepare_data(self):
        raw = self.reader.read_data()
        times, values, end, bounds = self._physiological_channels()
        acquisition_times = raw["time_seconds"].detach().cpu().numpy()
        if getattr(self.physiological_reader, 'already_synchronized', False):
            if self.physio_clock_drift_seconds != 0:
                raise ValueError('physio_clock_drift_seconds must be zero for already-synchronized '
                                 'physiological data (all timestamps are -1).')
            if any(len(channel) != len(acquisition_times) for channel in values):
                raise ValueError('Already-synchronized channels must contain exactly one value per '
                                 'retained MRI imaging readout in full acquisition order, before slice selection.')
            interpolated = np.column_stack(values)
            aligned_times = [acquisition_times.copy() for _ in times]
        else:
            interpolated = self._synchronize_to_sequence_end(
                times, values, acquisition_times, source_sequence_end=end,
                bounds=bounds if self.physiological_format == "SAEC" else "autoregression",
                physio_clock_drift_seconds=self.physio_clock_drift_seconds)
            aligned_times = [t - (t[-1] if end is None else end) + self.physio_clock_drift_seconds
                             for t in times]
        self.synchronization = {
            "physiological_time_seconds": aligned_times,
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
            "ismrmrd_header": raw["ismrmrd_header"],
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


    def read_data(self, output_h5_file=None, slice_idx=None, *, cache_h5=False,
                  cache_root=None, remove_temporary_data_after_run=True):
        """Return reconstruction-ready arrays, optionally selecting a slice/caching H5.

        ``ismrmrd_header`` contains the complete original XML header as text;
        Cached H5 files store it as a separate scalar UTF-8 dataset of the same name.
        The shared cache always contains all slices; selection happens after reading it.
        Synchronization always uses the complete sequence before slice selection.
        ``self.synchronization`` retains the full aligned traces for display.
        """
        if output_h5_file is not None:
            warnings.warn('output_h5_file now requests a shared cache entry; use cache_h5=True and '
                          'cache_root=... instead. The supplied filename is not written.',
                          DeprecationWarning, stacklevel=2)
            cache_h5 = True
        cached_path = None
        if cache_h5:
            source_dir = Path(__file__).parent
            implementation = hashlib.sha256(b''.join(path.read_bytes() for path in sorted(
                [Path(__file__), source_dir / 'ISMRMRDReader.py',
                 *source_dir.joinpath('physiological_data').glob('*.py'),
                 Path(__file__).parents[1] / 'runtime' / 'hdf5_cache.py']))).hexdigest()
            physiology_files = (list(self.physiological_file)
                                if self.physiological_format == 'physio_array'
                                else [self.physiological_file])
            key = cache_key([self.reader.ismrmrd_file, *physiology_files],
                            {'implementation': implementation, 'format': self.physiological_format,
                             'sensor': self.sensor_type, 'polaris_mode': self.polaris_channel_mode,
                             'physio_clock_drift_seconds': self.physio_clock_drift_seconds})
            def build(path):
                arrays = self._prepare_data()
                with h5py.File(path, 'w') as handle:
                    write_tree(handle, arrays)
                    metadata = handle.create_group('_cache')
                    metadata.attrs['kind'] = 'dict'
                    entries = {'synchronization': self.synchronization,
                               'physiological_metadata': self.physiological_reader.metadata,
                               'uses_kz_as_volume_axis': self._uses_kz_as_volume_axis}
                    metadata.attrs['keys'] = json.dumps(list(entries))
                    write_tree(metadata, entries)
            lease = acquire_cached(cache_root, 'preprocessed', key, '.h5', build,
                                   remove=remove_temporary_data_after_run)
            cached_path = str(lease.path)
            with h5py.File(lease.path, 'r') as handle:
                data = read_tree(handle)
            metadata = data.pop('_cache')
            self.synchronization = metadata['synchronization']
            self.physiological_reader.metadata = metadata['physiological_metadata']
            self._uses_kz_as_volume_axis = metadata['uses_kz_as_volume_axis']
        else:
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

        if cached_path is not None:
            data['realworld_h5_path'] = cached_path

        return data

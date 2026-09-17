"""Generic physiological text and NumPy files, with one channel per sensor/track."""

import numpy as np


class PreprocessedPhysioReader:
    def __init__(self, data_format):
        if data_format not in {'physio_text', 'physio_array'}:
            raise ValueError(f'Unsupported generic physiological format: {data_format!r}.')
        self.data_format = data_format
        self.metadata = {}
        self.already_synchronized = False

    @staticmethod
    def _real_array(array, name):
        if array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
            raise ValueError(f'{name} must contain finite real numeric values.')
        return np.asarray(array, dtype=np.float64)

    def _read_arrays(self, files):
        if not isinstance(files, (tuple, list)) or len(files) != 2:
            raise ValueError('physio_array requires (timestamps_file, values_file).')
        times = self._real_array(np.load(files[0], allow_pickle=False), 'Timestamps')
        values = self._real_array(np.load(files[1], allow_pickle=False), 'Values')
        if times.ndim != 3 or times.shape[-1] != 1 or any(n == 0 for n in times.shape):
            raise ValueError('Timestamps must have nonempty shape (Nsensors, Nsamples, 1).')
        if values.ndim != 3 or values.shape[:2] != times.shape[:2] or values.shape[-1] == 0:
            raise ValueError('Values must have matching nonempty shape (Nsensors, Nsamples, Ntracks).')
        pairs = [(s, t) for s in range(values.shape[0]) for t in range(values.shape[2])]
        return ([times[s, :, 0] for s, t in pairs],
                [values[s, :, t] for s, t in pairs], pairs)

    def _read_text(self, filename):
        """Read ``SENSOR TIMESTAMP VALUE1 [VALUE2 ...]`` records.

        Each row is one sample from one sensor. Value columns are that sensor's
        tracks, so every data row must contain the same positive number of
        tracks. Sensor records may be interleaved in the file, but rows for an
        individual sensor must remain in timestamp order.
        """
        rows_by_sensor = {}
        n_tracks = None
        saw_data = False
        with open(filename, encoding='utf-8-sig') as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.split('#', 1)[0].split()
                if not fields:
                    continue
                if not saw_data and len(fields) >= 2 and [field.upper() for field in fields[:2]] == ['SENSOR', 'TIMESTAMP']:
                    continue
                if len(fields) < 3:
                    raise ValueError(
                        f'Physiological text line {line_number}: expected SENSOR TIMESTAMP VALUE1 [VALUE2 ...].'
                    )
                try:
                    sensor_float = float(fields[0])
                    timestamp = float(fields[1])
                    values = [float(field) for field in fields[2:]]
                except ValueError as exc:
                    raise ValueError(f'Physiological text line {line_number}: expected numeric values.') from exc
                if not np.isfinite([sensor_float, timestamp, *values]).all():
                    raise ValueError(f'Physiological text line {line_number}: values must be finite.')
                if sensor_float < 0 or sensor_float != np.floor(sensor_float):
                    raise ValueError(f'Physiological text line {line_number}: SENSOR must be a nonnegative integer.')
                if n_tracks is None:
                    n_tracks = len(values)
                elif len(values) != n_tracks:
                    raise ValueError(
                        f'Physiological text line {line_number}: expected {n_tracks} value columns, got {len(values)}.'
                    )
                sensor = int(sensor_float)
                rows_by_sensor.setdefault(sensor, []).append((timestamp, values))
                saw_data = True
        if not saw_data:
            raise ValueError('Physiological text must contain at least one sample.')
        sensors = sorted(rows_by_sensor)
        times = []
        values = []
        pairs = []
        for sensor in sensors:
            rows = rows_by_sensor[sensor]
            sensor_times = np.asarray([row[0] for row in rows], dtype=np.float64)
            sensor_values = np.asarray([row[1] for row in rows], dtype=np.float64)
            for track in range(n_tracks):
                times.append(sensor_times)
                values.append(sensor_values[:, track])
                pairs.append((sensor, track))
        return times, values, pairs

    def read_channels(self, files):
        times, values, pairs = (self._read_arrays(files) if self.data_format == 'physio_array'
                                else self._read_text(files))
        synchronized = [bool(np.all(t == -1)) for t in times]
        if any(synchronized) and not all(synchronized):
            raise ValueError('Do not mix timestamped and already-synchronized channels.')
        self.already_synchronized = all(synchronized)
        if not self.already_synchronized:
            for index, t in enumerate(times):
                if t.size < 2 or np.any(np.diff(t) <= 0):
                    raise ValueError(f'Channel {index}: at least two strictly increasing timestamps are required.')
        self.metadata = {'physio_channels': [list(pair) for pair in pairs],
                         'already_synchronized': self.already_synchronized}
        return times, values, None, 'raise'

    @staticmethod
    def prepare_motion(interpolated):
        # Generic inputs are supplied in the units/scaling desired by the caller.
        return interpolated

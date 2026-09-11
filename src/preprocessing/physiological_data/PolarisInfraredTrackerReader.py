"""Read single-tool NDI ToolBox tracking TSV exports (such as Vienna R1).

The companion .tbr is recording metadata, not the coordinate time series.
Coordinates are preserved in export units; the TSV does not declare their unit.
"""

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class TrackingData:
    """Arrays indexed by sample, with XYZ and scalar-first Q0/Qx/Qy/Qz order.

    Invalid tracking states are retained, with corresponding poses set to NaN
    so plots show gaps. Marker positions have shape (samples, markers, 3).
    """

    timestamps: np.ndarray
    time_seconds: np.ndarray
    frames: np.ndarray
    tool_positions: np.ndarray
    quaternions: np.ndarray
    errors: np.ndarray
    tool_states: np.ndarray
    marker_positions: np.ndarray
    marker_states: np.ndarray


class PolarisInfraredTrackerReader:
    """Reader for the fixed-marker, single-tool layout exported by ToolBox."""

    def __init__(self, channel_mode="all"):
        if channel_mode not in {"all", "largest-amplitude"}:
            raise ValueError("polaris_channel_mode must be 'all' or 'largest-amplitude'.")
        self.channel_mode = channel_mode
        self.metadata = {}

    def read_channels(self, filename):
        """Return filtered XYZ channels and end-alignment settings for MRI preparation."""
        tracking = self.read(filename)
        positions = self._lowpass(tracking.time_seconds, tracking.tool_positions)
        return [tracking.time_seconds] * 3, list(positions.T), None, "raise"

    def prepare_motion(self, interpolated):
        """Select and normalize axes at full-sequence MRI times, before slice selection."""
        peak_to_peak = np.ptp(interpolated, axis=0)
        # np.argmax resolves ties in X, Y, Z order.
        indices = ([int(np.argmax(peak_to_peak))]
                   if self.channel_mode == "largest-amplitude" else [0, 1, 2])
        self.metadata = {
            'polaris_channel_mode': self.channel_mode,
            'polaris_channels': [["Tx", "Ty", "Tz"][i] for i in indices],
            'polaris_peak_to_peak_xyz': peak_to_peak,
        }
        motion = interpolated[:, indices]
        motion = motion - motion.mean(axis=0)
        scale = motion.std(axis=0).max()
        if scale > 0:
            motion = motion / scale
        return motion

    @staticmethod
    def _lowpass(times, positions):
        """Smooth position data with a 1.0 Hz, order-1 zero-phase filter."""
        times = np.asarray(times, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        if (times.ndim != 1 or times.size < 7
                or positions.shape != (times.size, 3)):
            raise ValueError('Polaris low-pass filtering requires at least 7 XYZ samples.')
        if not np.isfinite(times).all() or not np.isfinite(positions).all():
            raise ValueError('Polaris: invalid physiological times or values; resolve gaps first.')
        if np.any(np.diff(times) <= 0):
            raise ValueError('Polaris timestamps must be strictly increasing.')
        # Match the sampling-rate convention used by the Marmot filter.
        sampling_rate = times.size / (times[-1] - times[0])
        if sampling_rate <= 2 * 1.0:
            raise ValueError('Polaris sampling rate must exceed twice the 1.0 Hz cutoff.')
        b, a = butter(1, 1.0 / (sampling_rate / 2), btype='lowpass')
        return filtfilt(b, a, positions, axis=0)

    @staticmethod
    def read(filename):
        """Read a TSV, rejecting unsupported layouts and non-increasing times.

        Elapsed seconds start at zero. Decimal subtraction before conversion to
        float preserves subsecond precision in the large absolute timestamps.
        """
        path = Path(filename)
        if path.suffix.lower() != '.tsv':
            raise ValueError('Read the .tsv coordinate export, not the .tbr metadata file.')
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.reader(stream, delimiter='\t')
            header = next(reader, [])
            prefix = ['Frame', 'Time [sec]', 'Face', 'State', 'Q0', 'Qx',
                      'Qy', 'Qz', 'Tx', 'Ty', 'Tz', 'Error', 'Markers']
            if (header[:1] != ['Tools'] or header[2:15] != prefix
                    or (len(header) - 15) % 4
                    or header[15:] != ['State', 'Tx', 'Ty', 'Tz'] * ((len(header) - 15) // 4)):
                raise ValueError('Unsupported NDI tracking TSV header.')
            marker_count = (len(header) - 15) // 4
            rows = []
            times = []
            for line, row in enumerate(reader, start=2):
                if not row:
                    continue
                try:
                    if len(row) != len(header):
                        raise ValueError('column count does not match header')
                    if int(row[0]) != 1 or int(row[14]) != marker_count:
                        raise ValueError('expected one tool and a fixed marker count')
                    stamp = Decimal(row[3])
                    if not stamp.is_finite() or (times and stamp <= times[-1]):
                        raise ValueError('timestamps must be finite and strictly increasing')
                    # Unavailable poses may contain nonnumeric placeholders.
                    pose = [float(v) for v in row[6:14]] if row[5] == 'OK' else [np.nan] * 8
                    markers = []
                    states = []
                    for start in range(15, len(row), 4):
                        states.append(row[start])
                        markers.append([float(v) for v in row[start + 1:start + 4]]
                                       if row[start] == 'OK' else [np.nan] * 3)
                    rows.append((int(row[2]), pose, row[5], markers, states))
                    times.append(stamp)
                except (ValueError, InvalidOperation) as exc:
                    raise ValueError(f'{path.name}, line {line}: {exc}') from exc
        if not rows:
            raise ValueError(f'{path.name}: no tracking samples.')
        poses = np.asarray([r[1] for r in rows], dtype=float)
        return TrackingData(
            timestamps=np.asarray([float(t) for t in times]),
            time_seconds=np.asarray([float(t - times[0]) for t in times]),
            frames=np.asarray([r[0] for r in rows], dtype=np.int64),
            tool_positions=poses[:, 4:7],
            quaternions=poses[:, :4],
            errors=poses[:, 7],
            tool_states=np.asarray([r[2] for r in rows]),
            marker_positions=np.asarray([r[3] for r in rows], dtype=float).reshape(len(rows), marker_count, 3),
            marker_states=np.asarray([r[4] for r in rows]).reshape(len(rows), marker_count),
        )

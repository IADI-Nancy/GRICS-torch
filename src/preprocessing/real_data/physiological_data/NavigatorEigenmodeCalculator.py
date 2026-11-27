"""Extract timestamped physiological eigenmodes from decoded navigator groups."""

import hashlib

import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import map_coordinates


class NavigatorEigenmodeCalculator:
    """Non-rigid profile registration followed by one joint, uncentered SVD.

    Implements the physiological reader interface using decoded MRI records.
    Navigators come from the second contrast (zero-based set counter 1);
    imaging readouts can come from either contrast on the shared scanner clock.
    All processing settings must be supplied explicitly from configuration.
    """

    uses_mri_data = True
    supports_clock_correction = False
    already_synchronized = False

    def __init__(self, *, navigator_type, imaging_set, n_eigenmodes, directions, startup_profiles, regularization, resolution_levels, registration_loops, extrapolation, reference=None):
        if not isinstance(navigator_type, str) or not navigator_type:
            raise ValueError("navigator_type must be a nonempty string identifying the reader's extraction strategy.")
        if type(imaging_set) is not int or imaging_set not in (0, 1):
            raise ValueError("imaging_set must be 0 or 1.")
        self.imaging_set = imaging_set
        if type(n_eigenmodes) is not int or n_eigenmodes < 1:
            raise ValueError("n_eigenmodes must be a positive integer.")
        if not directions or any(type(value) is not int or value not in (0, 1, 2) for value in directions) or len(set(directions)) != len(directions):
            raise ValueError("directions must contain distinct direction indices from 0, 1, 2.")
        if type(startup_profiles) is not int or startup_profiles < 0 or type(registration_loops) is not int or registration_loops < 1:
            raise ValueError("startup_profiles must be nonnegative and registration_loops positive integers.")
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive.")
        if not resolution_levels or any(not np.isfinite(level) or level <= 0 or level > 1 for level in resolution_levels) or resolution_levels[-1] != 1 or any(a >= b for a, b in zip(resolution_levels, resolution_levels[1:])):
            raise ValueError("resolution_levels must increase strictly in (0, 1] and end at 1.")
        if extrapolation not in {"edge", "raise", "autoregression"}:
            raise ValueError("extrapolation must be edge, raise, or autoregression.")
        self.navigator_type = navigator_type
        self.n_eigenmodes = n_eigenmodes
        self.directions = tuple(directions)
        self.startup_profiles = startup_profiles
        self.regularization = float(regularization)
        self.resolution_levels = tuple(resolution_levels)
        self.registration_loops = registration_loops
        self.extrapolation = extrapolation
        self.reference = reference
        self.metadata = {}
        self.profiles = {}
        self.displacements = {}

    @property
    def mri_reader_options(self):
        return {"read_navigators": True, "navigator_type": self.navigator_type, "navigator_set": 1, "imaging_set": self.imaging_set}

    @property
    def cache_parameters(self):
        return {"format": "navigator-eigenmodes", "n_eigenmodes": self.n_eigenmodes, "directions": list(self.directions),
                "startup_profiles": self.startup_profiles, "regularization": self.regularization,
                "resolution_levels": list(self.resolution_levels), "registration_loops": self.registration_loops,
                "extrapolation": self.extrapolation, **self.mri_reader_options}

    @staticmethod
    def source_files(filename):
        return []  # The MRI file is already included by RawDataPreparer.

    @staticmethod
    def prepare_motion(interpolated):
        return interpolated

    def read_channels(self, raw):
        """Return eigenmode channels on the MRI clock for shared synchronization."""
        if raw.get("navigator_type") != self.navigator_type or raw.get("navigator_set") != 1:
            raise ValueError("Eigenmode extraction requires explicitly decoded second-set navigator groups of the configured type; flagged feedback is not a substitute.")
        records = raw.get("navigators")
        if not records:
            raise ValueError("No navigator records were supplied.")
        groups = {}
        for record in records:
            if record["counters"]["set"] != 1:
                raise ValueError("Only set counter 1 may contribute to navigator eigenmodes.")
            group = groups.setdefault(record["group_index"], {})
            direction = record["direction_index"]
            if direction in group:
                raise ValueError("Duplicate direction in a navigator group.")
            group[direction] = record
        ordered = [groups[key] for key in sorted(groups)]
        available_directions = set(ordered[0])
        if not set(self.directions) <= available_directions or any(set(group) != available_directions for group in ordered):
            raise ValueError("Navigator groups must contain consistent directions and include every selected direction.")
        times = np.asarray([group[min(available_directions)]["time_seconds"] for group in ordered], dtype=np.float64)
        if times.size <= self.startup_profiles or times.size < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Navigator times must increase strictly and outnumber the startup profiles.")
        oversampling = raw["readout_oversampling_factor"]
        if oversampling not in (1, 2):
            raise ValueError("Navigator readout oversampling must be explicitly 1 or 2.")
        parameters = {key: value for key, value in self.cache_parameters.items() if key != 'imaging_set'}
        fingerprint = hashlib.sha256()
        fingerprint.update(str(oversampling).encode())
        for record in records:
            fingerprint.update(np.asarray([record['acquisition_index'], record['acquisition_time_stamp'], record['group_index'], record['direction_index']], dtype=np.int64).tobytes())
            samples = record['data'].detach().cpu().numpy()
            fingerprint.update(str((samples.shape, samples.dtype)).encode())
            fingerprint.update(samples.tobytes())
        fingerprint = fingerprint.hexdigest()
        if self.reference is not None:
            if self.reference['parameters'] != parameters or self.reference['fingerprint'] != fingerprint:
                raise ValueError('Navigator reuse requires identical navigator samples, scanner timestamps, extraction indices, and processing settings.')
            previous = self.reference['metadata']
            modes = previous['navigator_temporal_modes']
            singular_values = previous['navigator_singular_values']
            rank = previous['navigator_numerical_rank']
            print('[navigators] Reusing verified second-set registration and SVD; synchronizing to this contrast on the scanner clock.', flush=True)
        else:
            fields = []
            for direction in self.directions:
                samples = np.stack([group[direction]["data"].detach().cpu().numpy() for group in ordered], axis=-1)
                if samples.ndim != 3 or not np.isfinite(samples).all():
                    raise ValueError("Navigator samples must be finite [coil, sample, time] arrays.")
                if oversampling == 2:
                    if samples.shape[1] % 2:
                        raise ValueError("Twofold navigator oversampling requires an even sample count.")
                    image = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(samples, axes=1), axis=1), axes=1)
                    size = samples.shape[1] // 2
                    start = samples.shape[1] // 2 - size // 2
                    samples = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(image[:, start:start + size], axes=1), axis=1), axes=1)
                profiles = np.abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(samples, axes=1), axis=1), axes=1))
                if any(int(profiles.shape[axis] * level) < 2 for axis in (1, 2) for level in self.resolution_levels):
                    raise ValueError("Navigator registration pyramid would have fewer than two spatial or temporal samples; adjust navigator_resolution_levels.")
                profiles[..., :self.startup_profiles] = profiles[..., self.startup_profiles:self.startup_profiles + 1]
                displacement = []
                for coil, profile in enumerate(profiles):
                    try:
                        field, _, _ = self.register_profiles(profile, self.resolution_levels, self.registration_loops, self.regularization)
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        raise ValueError(f"Navigator registration failed for ordinal {direction}, coil {coil}: {exc}") from exc
                    displacement.append(field)
                self.profiles[direction] = profiles
                self.displacements[direction] = np.asarray(displacement)
                fields.append(self.displacements[direction].reshape(-1, times.size))
                print(f"[navigators] Registered second-set ordinal {direction}: {profiles.shape[0]} coils, {times.size} profiles.", flush=True)
            matrix = np.concatenate(fields, axis=0)
            if not np.isfinite(matrix).all():
                raise ValueError("Navigator displacement matrix contains non-finite values.")
            _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
            tolerance = np.finfo(matrix.dtype).eps * max(matrix.shape) * singular_values[0]
            rank = int(np.count_nonzero(singular_values > tolerance))
            if self.n_eigenmodes > rank:
                raise ValueError(f"Requested {self.n_eigenmodes} navigator eigenmodes but the displacement matrix has numerical rank {rank}.")
            modes = vh[:self.n_eigenmodes].T.copy()
            # Fix arbitrary SVD signs for repeatable saved signals; scaling is unchanged.
            for mode in range(modes.shape[1]):
                if modes[np.argmax(np.abs(modes[:, mode])), mode] < 0:
                    modes[:, mode] *= -1
        acquisition_times = raw["time_seconds"].detach().cpu().numpy()
        before = int(np.count_nonzero(acquisition_times < times[0]))
        after = int(np.count_nonzero(acquisition_times > times[-1]))
        if self.extrapolation == "edge" and (before or after):
            print(f"[navigators] Endpoint holding explicitly selected: {before} imaging readouts before the first navigator and {after} after the last ({max(0., acquisition_times[-1] - times[-1]):.4f} s at the end). Interior synchronization uses linear interpolation.", flush=True)
        self.metadata = {"navigator_type": self.navigator_type, "navigator_set": 1, "imaging_set": self.imaging_set, "navigator_directions": list(self.directions), "navigator_time_seconds": times,
                         "navigator_temporal_modes": modes, "navigator_singular_values": singular_values,
                         "navigator_numerical_rank": rank, "navigator_startup_profiles": self.startup_profiles,
                         "navigator_acquisition_indices": [[group[direction]["acquisition_index"] for direction in sorted(available_directions)] for group in ordered],
                         "navigator_extrapolation": self.extrapolation, "navigator_endpoint_readouts": [before, after]}
        self.reference = {'parameters': parameters, 'fingerprint': fingerprint, 'metadata': self.metadata.copy()}
        print(f"[navigators] Keeping {self.n_eigenmodes} joint SVD modes from the second set; no temporal smoothing.", flush=True)
        # Times are already relative to the retained MRI end, so end=0 preserves
        # their shared clock instead of moving the last navigator to sequence end.
        return [times.copy() for _ in range(self.n_eigenmodes)], list(modes.T), 0.0, self.extrapolation

    @staticmethod
    def register_profiles(profiles, levels, loops, regularization):
        """Port registration_SSD_nonrigid_1DT for V0=Inf and first-order smoothing.

        The reference is the final profile. With V0=Inf the MATLAB temporal penalty
        vanishes, so the linear systems are independent spatial tridiagonal blocks.
        The image pyramid still resamples both space and time, as in MATLAB.
        """
        image = np.asarray(profiles, dtype=np.float64)
        maximum = image.max()
        if maximum <= 0 or not np.isfinite(image).all():
            raise ValueError("Registration requires finite, nonzero magnitude profiles.")
        image = image / maximum
        nx, nt = image.shape
        shapes = [(max(2, int(nx * level)), max(2, int(nt * level))) for level in levels]

        def resize(values, shape):
            axes = [(old - 1 - (new - 1) * old / new) / 2 + np.arange(new) * old / new
                    for old, new in zip(values.shape, shape)]
            return map_coordinates(values, np.asarray(np.meshgrid(*axes, indexing="ij")), order=1, mode="constant", cval=0., prefilter=False)

        pyramid = [None] * len(levels)
        references = [None] * len(levels)
        pyramid[-1], references[-1] = image, image[:, -1:]
        for level in range(len(levels) - 2, -1, -1):
            pyramid[level] = resize(pyramid[level + 1], shapes[level])
            references[level] = resize(references[level + 1], (shapes[level][0], 1))[:, 0:1]
        best = None
        history = []
        for level, (nxl, ntl) in enumerate(shapes):
            displacement = np.zeros((nxl, ntl)) if best is None else resize(best, (nxl, ntl)) * nxl / best.shape[0]
            if best is not None and ntl > best.shape[1]:
                displacement[:, 0] = displacement[:, 1]
                displacement[:, -1] = displacement[:, -2]
            diagonal = np.full(nxl, 2.)
            diagonal[[0, -1]] = 1.
            derivative_scale = (nx / nxl) ** 2
            best_residual = np.inf
            for _ in range(loops):
                coordinates = np.asarray(np.meshgrid(np.arange(nxl), np.arange(ntl), indexing="ij"), dtype=float)
                coordinates[0] += displacement
                registered = map_coordinates(pyramid[level], coordinates, order=1, mode="constant", cval=0., prefilter=False)
                residual = registered - references[level]
                score = float(np.linalg.norm(residual))
                if score >= best_residual:
                    break
                best_residual, best = score, displacement.copy()
                gradient = np.gradient(registered, axis=0)
                rhs = -gradient * residual
                weight = regularization * np.linalg.norm(rhs)
                if weight == 0:
                    break
                penalty = weight * derivative_scale
                spatial_laplacian = diagonal[:, None] * displacement
                spatial_laplacian[1:] -= displacement[:-1]
                spatial_laplacian[:-1] -= displacement[1:]
                rhs -= penalty * spatial_laplacian
                # MATLAB column-major vectorization places spatial blocks consecutively.
                banded = np.zeros((3, nxl * ntl))
                banded[1] = (gradient ** 2 + penalty * diagonal[:, None]).ravel(order="F")
                banded[0, 1:] = -penalty
                banded[2, :-1] = -penalty
                banded[0, np.arange(ntl) * nxl] = 0.
                banded[2, np.arange(1, ntl + 1) * nxl - 1] = 0.
                increment = solve_banded((1, 1), banded, rhs.ravel(order="F"), check_finite=True)
                displacement += increment.reshape((nxl, ntl), order="F")
            history.append(best_residual)
        coordinates = np.asarray(np.meshgrid(np.arange(nx), np.arange(nt), indexing="ij"), dtype=float)
        coordinates[0] += best
        registered = map_coordinates(image, coordinates, order=1, mode="constant", cval=0., prefilter=False) * maximum
        return best, registered, history

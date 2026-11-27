from src.utils.ismrmrd_io import ReadOnlyDataset
import ismrmrd
import torch
import math
import warnings
import numpy as np
from time import perf_counter
from src.utils.fftnc import fftnc, ifftnc


def _is_noise(acq):
    return acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)


def _has_ismrmrd_flag(acq, flag_name):
    flag = getattr(ismrmrd, flag_name, None)
    return (flag is not None) and acq.isFlagSet(flag)


def _is_non_imaging(acq):
    return (
        _has_ismrmrd_flag(acq, "ACQ_IS_DUMMYSCAN_DATA")
        or _has_ismrmrd_flag(acq, "ACQ_IS_PHASECORR_DATA")
        or _has_ismrmrd_flag(acq, "ACQ_IS_NAVIGATION_DATA")
        or _has_ismrmrd_flag(acq, "ACQ_IS_HPFEEDBACK_DATA")
        or _has_ismrmrd_flag(acq, "ACQ_IS_RTFEEDBACK_DATA")
    )


def _is_parallel_calibration(acq):
    return (
        _has_ismrmrd_flag(acq, "ACQ_IS_PARALLEL_CALIBRATION")
        or _has_ismrmrd_flag(acq, "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING")
    )

class ISMRMRDReader:
    """Decode one ISMRMRD acquisition into MRI tensors and acquisition metadata.

    This reader handles ISMRMRD-specific mapping and readout oversampling removal.
    It does not synchronize physiological signals or select reconstruction slices.
    """

    def __init__(self, ismrmrd_file, device="cpu", *, read_navigators=False, navigator_type=None, navigator_set=1, imaging_set=None):
        if type(read_navigators) is not bool:
            raise ValueError("read_navigators must be a boolean.")
        self.ismrmrd_file = ismrmrd_file
        self.device = device
        self.read_navigators = read_navigators
        if navigator_type not in (None, 'mosar', 'mosar-mprage'):
            raise ValueError('navigator_type must be None, mosar, or mosar-mprage.')
        if navigator_type is not None and not read_navigators:
            raise ValueError('Explicit navigator extraction requires read_navigators=True.')
        if type(navigator_set) is not int or navigator_set < 0:
            raise ValueError('navigator_set must be a nonnegative integer.')
        self.navigator_type = navigator_type
        self.navigator_set = navigator_set
        if imaging_set is not None and (type(imaging_set) is not int or imaging_set < 0 or navigator_type is None):
            raise ValueError('imaging_set requires a navigator type and a nonnegative integer.')
        self.imaging_set = navigator_set if imaging_set is None else imaging_set

    def _mosar_navigator_indices(self, heads, limits, set_index):
        """Identify central triplets in one set, explicitly excluding feedback.

        Every set/repetition block must begin with its triplet; further central
        imaging lines must be isolated. Ambiguous sequences are rejected.
        Ordinals describe acquisition order, not anatomical directions.
        """
        names = ('ACQ_IS_NOISE_MEASUREMENT', 'ACQ_IS_NAVIGATION_DATA', 'ACQ_IS_DUMMYSCAN_DATA',
                 'ACQ_IS_PHASECORR_DATA', 'ACQ_IS_HPFEEDBACK_DATA', 'ACQ_IS_RTFEEDBACK_DATA')
        excluded = sum(1 << (getattr(ismrmrd, name) - 1) for name in names)
        eligible = np.flatnonzero((heads['flags'] & excluded) == 0)
        selected = eligible[heads['idx']['set'][eligible] == set_index]
        if not len(selected):
            raise ValueError(f'MOSAR extraction found no imaging acquisitions in set counter {set_index}.')
        if len(np.unique(heads['idx']['slice'][selected])) != 1:
            raise ValueError('MOSAR central-triplet extraction currently requires a single encoded slice/slab.')
        cy, cz = limits.kspace_encoding_step_1.center, limits.kspace_encoding_step_2.center
        if cy is None or cz is None:
            raise ValueError('MOSAR navigator extraction requires declared ky/kz centres.')
        keys = np.column_stack([heads['idx'][name][eligible] for name in ('set', 'repetition', 'slice')])
        boundaries = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1, len(eligible)]
        groups = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            block = eligible[start:stop]
            if int(heads['idx']['set'][block[0]]) != set_index:
                continue
            if len(block) < 4:
                raise ValueError('A MOSAR set block must contain a navigator triplet followed by imaging readouts.')
            central = (heads['idx']['kspace_encode_step_1'][block] == cy) & (heads['idx']['kspace_encode_step_2'][block] == cz)
            if not np.all(central[:3]):
                raise ValueError(f'MOSAR set block starts at acquisition {block[0]} without a central triplet.')
            remaining = np.flatnonzero(central[3:])
            if np.any(np.diff(remaining) == 1):
                raise ValueError('Ambiguous repeated central lines inside a MOSAR imaging block; navigator layout is not supported.')
            groups.append(block[:3])
        if len(groups) < 2:
            raise ValueError('MOSAR extraction requires at least two set blocks with central navigator triplets.')
        reference_shape = (int(heads['active_channels'][groups[0][0]]), int(heads['number_of_samples'][groups[0][0]]))
        result = {}
        for group, indices in enumerate(groups):
            for direction, index in enumerate(indices):
                shape = (int(heads['active_channels'][index]), int(heads['number_of_samples'][index]))
                if shape != reference_shape:
                    raise ValueError('MOSAR navigator triplets have inconsistent channel/sample dimensions.')
                result[int(index)] = (group, direction)
        print(f'[ismrmrd] Explicit MOSAR extraction: {len(groups)} central triplets in set counter {set_index}; feedback flags are excluded and triplets are removed from imaging.', flush=True)
        return result

    def _mosar_mprage_navigator_indices(self, heads, limits, set_index):
        """Validate the single-set sequence's regular one-readout navigator cadence."""
        if set_index != 0:
            raise ValueError('MOSAR MPRAGE requires the first set (counter 0).')
        names = ('ACQ_IS_NOISE_MEASUREMENT', 'ACQ_IS_NAVIGATION_DATA', 'ACQ_IS_DUMMYSCAN_DATA',
                 'ACQ_IS_PHASECORR_DATA', 'ACQ_IS_HPFEEDBACK_DATA', 'ACQ_IS_RTFEEDBACK_DATA')
        excluded = sum(1 << (getattr(ismrmrd, name) - 1) for name in names)
        selected = np.flatnonzero((heads['flags'] & excluded) == 0)
        if not len(selected) or np.any(heads['idx']['set'][selected] != 0):
            raise ValueError('MOSAR MPRAGE requires imaging acquisitions in set counter 0 only.')
        if any(len(np.unique(heads['idx'][name][selected])) != 1 for name in ('slice', 'repetition')):
            raise ValueError('MOSAR MPRAGE cadence extraction supports one slab and one repetition.')
        cy, cz = limits.kspace_encoding_step_1.center, limits.kspace_encoding_step_2.center
        if cy is None or cz is None:
            raise ValueError('MOSAR MPRAGE requires declared ky/kz centres.')
        central = np.flatnonzero((heads['idx']['kspace_encode_step_1'][selected] == cy) & (heads['idx']['kspace_encode_step_2'][selected] == cz))
        if len(central) < 3 or central[0] != 0:
            raise ValueError('MOSAR MPRAGE requires at least three central navigators starting at the first eligible readout.')
        period = int(central[1])
        if period <= 1 or central[2] != 2 * period or len(selected) % period:
            raise ValueError('MOSAR MPRAGE initial central lines do not establish a regular single-navigator shot cadence with complete blocks.')
        positions = np.arange(0, len(selected), period)
        if not np.isin(positions, central).all():
            raise ValueError('MOSAR MPRAGE navigator cadence fails at a later shot; refusing partial extraction.')
        indices = selected[positions]
        shapes = np.column_stack([heads[name][indices] for name in ('active_channels', 'number_of_samples')])
        if np.any(shapes != shapes[0]):
            raise ValueError('MOSAR MPRAGE navigator readouts have inconsistent channel/sample dimensions.')
        extra = len(central) - len(indices)
        print(f'[ismrmrd] Explicit MOSAR MPRAGE extraction: {len(indices)} first-set single navigators, one every {period} eligible readouts; {extra} additional central lines remain in imaging. Feedback is excluded.', flush=True)
        return {int(index): (group, 0) for group, index in enumerate(indices)}

    def _navigator_record(self, acq, i):
        return {
            "data": torch.from_numpy(acq.data.copy()).to(self.device),
            "acquisition_index": i,
            "acquisition_time_stamp": int(acq.acquisition_time_stamp),
            "scan_counter": int(acq.scan_counter),
            "flags": int(acq.flags),
            "counters": {name: int(getattr(acq.idx, name)) for name in (
                "kspace_encode_step_1", "kspace_encode_step_2", "average", "slice",
                "contrast", "phase", "repetition", "set", "segment")},
            "user_counters": list(acq.idx.user),
            "position": list(acq.position), "read_dir": list(acq.read_dir),
            "phase_dir": list(acq.phase_dir), "slice_dir": list(acq.slice_dir),
            "sample_time_us": float(acq.sample_time_us),
            "center_sample": int(acq.center_sample),
            "discard_pre": int(acq.discard_pre), "discard_post": int(acq.discard_post),
            "trajectory": torch.from_numpy(acq.traj.copy()).to(self.device),
        }

    def read_navigator_data(self):
        """Read flagged navigators without gridding imaging or calibration data.

        This explicit navigator-only entry point also supports acquisitions whose
        imaging layout is not supported by read_data(). Timestamps retain the
        same origin: the last retained non-noise, non-navigation imaging readout.
        """
        dset = ReadOnlyDataset(self.ismrmrd_file)
        navigators = []
        last_imaging_timestamp = None
        try:
            heads = dset.read_acquisition_headers()
            names = ('ACQ_IS_NOISE_MEASUREMENT', 'ACQ_IS_NAVIGATION_DATA', 'ACQ_IS_DUMMYSCAN_DATA',
                     'ACQ_IS_PHASECORR_DATA', 'ACQ_IS_HPFEEDBACK_DATA', 'ACQ_IS_RTFEEDBACK_DATA')
            excluded = sum(1 << (getattr(ismrmrd, name) - 1) for name in names)
            imaging = np.flatnonzero((heads['flags'] & excluded) == 0)
            if len(imaging):
                last_imaging_timestamp = int(heads['acquisition_time_stamp'][imaging[-1]])
            selected = ((heads['flags'] & (1 << (ismrmrd.ACQ_IS_NAVIGATION_DATA - 1))) != 0) & ((heads['flags'] & (1 << (ismrmrd.ACQ_IS_NOISE_MEASUREMENT - 1))) == 0)
            for i in np.flatnonzero(selected):
                navigators.append(self._navigator_record(dset.read_acquisition(int(i)), int(i)))
        finally:
            dset.close()
        if last_imaging_timestamp is None:
            raise ValueError("Cannot reference navigator times: no retained imaging readout found.")
        for navigator in navigators:
            navigator["time_seconds"] = (navigator["acquisition_time_stamp"] - last_imaging_timestamp) * 2.5e-3
        if not navigators:
            print("[ismrmrd] Navigator reading requested, but no acquisitions carry ACQ_IS_NAVIGATION_DATA.", flush=True)
        return navigators

    @staticmethod
    def _encoding_limit_size(limit_obj):
        if limit_obj is None:
            return 1
        if getattr(limit_obj, "maximum", None) is None:
            return 1
        return int(limit_obj.maximum) + 1

    @staticmethod
    def _accel_factor_or_one(parallel_obj, attr_name):
        if parallel_obj is None:
            return 1
        val = getattr(parallel_obj.accelerationFactor, attr_name, None)
        if val is None:
            return 1
        return max(1, int(val))

    @staticmethod
    def _center_shift(limit_obj, size):
        """Index offset that moves the declared k-space centre of an encoding axis to size // 2."""
        center = getattr(limit_obj, "center", None) if limit_obj is not None else None
        if center is None:
            return 0
        return size // 2 - int(center)

    @staticmethod
    def _expanded_matrix_size(n_lines, accel):
        acquired_lines = math.ceil(float(n_lines) / float(accel))
        return int(2 * math.ceil((float(accel) * float(acquired_lines)) / 2.0))

    @staticmethod
    def _readout_oversampling_factor(enc):
        """Use the encoded/reconstruction matrix ratio, or warn and assume 2x."""
        try:
            encoded_x = int(enc.encodedSpace.matrixSize.x)
            recon_x = int(enc.reconSpace.matrixSize.x)
            if encoded_x <= 0 or recon_x <= 0 or encoded_x % recon_x != 0 or encoded_x // recon_x not in (1, 2):
                raise ValueError(f"Unsupported readout oversampling: encodedSpace.x={encoded_x}, reconSpace.x={recon_x}.")
            return encoded_x // recon_x
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            warnings.warn(f"Cannot determine readout oversampling from the ISMRMRD header ({exc}); falling back to 2x readout oversampling.", RuntimeWarning, stacklevel=2)
            return 2

    @staticmethod
    def _remove_oversampling(readouts):
        """Crop readout image space for a batch with samples on the last axis."""
        readout = readouts.shape[-1]
        if readout % 2:
            raise ValueError('Twofold readout oversampling requires an even sample count.')
        cropped_readout = readout // 2
        crop_start = readout // 2 - cropped_readout // 2
        image = ifftnc(readouts, dims=(-1,))
        return fftnc(image[..., crop_start:crop_start + cropped_readout], dims=(-1,))

    def _extract_mri_data(self):
        started = perf_counter()
        dset = ReadOnlyDataset(self.ismrmrd_file)
        try:
            xml_header = dset.read_xml_header()
            self.ismrmrd_header = (xml_header.decode("utf-8")
                                   if isinstance(xml_header, bytes) else xml_header)
            header = ismrmrd.xsd.CreateFromDocument(self.ismrmrd_header)
            enc = header.encoding[0]
            limits = enc.encodingLimits
            heads = dset.read_acquisition_headers()
            if self.navigator_type == 'mosar':
                navigator_indices = self._mosar_navigator_indices(heads, limits, self.navigator_set)
                imaging_navigator_indices = (navigator_indices if self.imaging_set == self.navigator_set else
                                             self._mosar_navigator_indices(heads, limits, self.imaging_set))
            elif self.navigator_type == 'mosar-mprage':
                navigator_indices = self._mosar_mprage_navigator_indices(heads, limits, self.navigator_set)
                if self.imaging_set != self.navigator_set:
                    raise ValueError('MOSAR MPRAGE requires imaging and navigators from set counter 0.')
                imaging_navigator_indices = navigator_indices
            elif self.navigator_type is None:
                navigator_indices = {}
            else:
                raise ValueError(f'Unsupported navigator type: {self.navigator_type!r}.')
            self._oversampling_factor = self._readout_oversampling_factor(enc)
            if self._oversampling_factor == 1:
                print("[ismrmrd] no readout oversampling declared (encodedSpace.x == reconSpace.x): the readout is kept whole.", flush=True)

            N_SLI = self._encoding_limit_size(limits.slice)
            # Nex currently follows repetition, not the Siemens average counter.
            # Our Siemens protocol uses two complementary averages at R=2 to
            # form one fully sampled image for coil-sensitivity estimation.
            # Combining those averages is intentional for that protocol.
            # TODO: distinguish it from true multiple-Nex acquisitions, whose
            # averages must remain separate; this ambiguity is unresolved.
            Nex = self._encoding_limit_size(limits.repetition)
            Nex = max(1, Nex)

            Nz_native = self._encoding_limit_size(limits.kspace_encoding_step_2)

            Ny = self._encoding_limit_size(limits.kspace_encoding_step_1)
            Ry = self._accel_factor_or_one(getattr(enc, "parallelImaging", None), "kspace_encoding_step_1")
            Ny = max(self._expanded_matrix_size(Ny, Ry), int(enc.encodedSpace.matrixSize.y))
            # Place the declared k-space centre at Ny // 2, where the centred FFT expects it.
            # 2.0 ignored encodingLimits.*.center; that only matters for asymmetric
            # (partial-Fourier) index ranges that do not start at the grid origin.
            shift_y = self._center_shift(limits.kspace_encoding_step_1, Ny)

            # Heuristic: multi-slice acquisitions are treated as 2D stacks,
            # while slab acquisitions (typically one encoded slice with kz partitions)
            # use kspace_encode_step_2 as the volume axis.
            use_kz_as_volume_axis = (N_SLI <= 1) and (Nz_native > 1)
            shift_z = 0
            if use_kz_as_volume_axis:
                Rz = self._accel_factor_or_one(
                    getattr(enc, "parallelImaging", None), "kspace_encoding_step_2"
                )
                Nz = max(self._expanded_matrix_size(Nz_native, Rz), int(enc.encodedSpace.matrixSize.z))
                z_size = Nz
                shift_z = self._center_shift(limits.kspace_encoding_step_2, Nz)
            else:
                Nz = 1
                z_size = N_SLI
            if shift_y or shift_z:
                print(f"[ismrmrd] k-space centre from the header: line indices shifted by {shift_y} (ky) and {shift_z} (kz) onto the {Ny} x {Nz} grid.", flush=True)

            self._raw_uses_kz_as_volume_axis = use_kz_as_volume_axis
            self._raw_n_slices = N_SLI

            kspace = None
            reference_kspace = None
            reference_line_seen = None
            nex_values_seen = set()
            timestamps = []
            z_indices = []
            idx_ky = []
            idx_kz = []
            idx_nex = []
            slice_geometry = {}
            navigators = [] if self.read_navigators else None

            names = ('ACQ_IS_NOISE_MEASUREMENT', 'ACQ_IS_NAVIGATION_DATA', 'ACQ_IS_DUMMYSCAN_DATA',
                     'ACQ_IS_PHASECORR_DATA', 'ACQ_IS_HPFEEDBACK_DATA', 'ACQ_IS_RTFEEDBACK_DATA')
            excluded = sum(1 << (getattr(ismrmrd, name) - 1) for name in names)
            imaging = (heads['flags'] & excluded) == 0
            if self.navigator_type is not None:
                imaging &= heads['idx']['set'] == self.imaging_set
                imaging[list(imaging_navigator_indices)] = False
                navigator_mask = np.zeros(len(heads), dtype=bool)
                navigator_mask[list(navigator_indices)] = True
            else:
                navigator_mask = ((heads['flags'] & (1 << (ismrmrd.ACQ_IS_NAVIGATION_DATA - 1))) != 0) & ((heads['flags'] & (1 << (ismrmrd.ACQ_IS_NOISE_MEASUREMENT - 1))) == 0)
                navigator_mask &= self.read_navigators
            selected = np.flatnonzero(imaging | navigator_mask)
            print(f'[ismrmrd] Decoding {len(selected)}/{len(heads)} acquisitions after header filtering; acquired-readout FFT input batches target 32 MiB.', flush=True)
            pending = []

            def flush_readouts():
                nonlocal reference_kspace
                if not pending:
                    return
                readouts = torch.stack([entry[0] for entry in pending]).to(device=self.device, dtype=torch.complex128)
                if self._oversampling_factor == 2:
                    readouts = self._remove_oversampling(readouts)
                for entry, acq_data in zip(pending, readouts):
                    _, nex, ky, z, calibration, index = entry
                    if line_count[nex, ky, z] > 0:
                        kspace[:, nex, :, ky, z] += acq_data
                    else:
                        kspace[:, nex, :, ky, z] = acq_data
                    line_count[nex, ky, z] += 1
                    if calibration:
                        if reference_kspace is None:
                            reference_kspace = torch.zeros((kspace.shape[0], 1, *kspace.shape[2:]), dtype=kspace.dtype, device=kspace.device)
                        if reference_line_seen[ky, z]:
                            raise ValueError(f'Duplicate parallel-calibration acquisition found for ky={ky}, z={z} in acquisition {index}. Expected at most one calibration line per (ky, z).')
                        reference_kspace[:, 0, :, ky, z] = acq_data
                        reference_line_seen[ky, z] = True
                pending.clear()

            for i in selected:
                i = int(i)
                acq = dset.read_acquisition(i)
                if self.navigator_type is not None:
                    if i in navigator_indices:
                        navigator = self._navigator_record(acq, i)
                        navigator['group_index'], navigator['direction_index'] = navigator_indices[i]
                        navigators.append(navigator)
                        continue
                    if int(acq.idx.set) != self.imaging_set or i in imaging_navigator_indices:
                        continue
                if not _is_noise(acq) and _has_ismrmrd_flag(acq, "ACQ_IS_NAVIGATION_DATA"):
                    if self.read_navigators and self.navigator_type is None:
                        navigators.append(self._navigator_record(acq, i))
                    continue
                if _is_noise(acq) or _is_non_imaging(acq):
                    continue

                ky = int(acq.idx.kspace_encode_step_1) + shift_y
                kz = int(acq.idx.kspace_encode_step_2) + shift_z
                rep = int(acq.idx.repetition)
                sli = int(acq.idx.slice)
                ts = float(acq.acquisition_time_stamp)
                if not 0 <= ky < Ny or (use_kz_as_volume_axis and not 0 <= kz < Nz):
                    raise ValueError(
                        f"Acquisition {i}: line (ky={ky - shift_y}, kz={kz - shift_z}) falls outside the "
                        f"{Ny} x {Nz} encoded grid after centring; check encodingLimits in the header.")

                nex = rep
                if use_kz_as_volume_axis:
                    z = kz
                else:
                    z = sli

                acq_data = torch.from_numpy(acq.data.copy())

                if kspace is None:
                    ncha, nsamp = acq.data.shape
                    if self._oversampling_factor == 2:
                        if nsamp % 2:
                            raise ValueError('Twofold readout oversampling requires an even sample count.')
                        nsamp //= 2
                    batch_size = max(1, (32 * 2**20) // (ncha * acq.data.shape[1] * 16))
                    kspace = torch.zeros(
                        (ncha, Nex, nsamp, Ny, z_size),
                        dtype=torch.complex128,
                        device=self.device,
                    )
                    reference_line_seen = torch.zeros(
                        (Ny, z_size),
                        dtype=torch.bool,
                        device=self.device,
                    )
                    line_count = torch.zeros((Nex, Ny, z_size), dtype=torch.int32, device=self.device)

                timestamps.append(ts)
                z_indices.append(z)
                idx_ky.append(ky)
                idx_kz.append(kz)
                idx_nex.append(nex)
                if z not in slice_geometry:
                    patient_position = getattr(header.measurementInformation, "patientPosition", None)
                    slice_geometry[z] = {
                        "position": list(acq.position),
                        "read_dir": list(acq.read_dir),
                        "phase_dir": list(acq.phase_dir),
                        "slice_dir": list(acq.slice_dir),
                        "patient_position": str(patient_position).split(".")[-1] if patient_position is not None else None,
                    }
                nex_values_seen.add(nex)
                pending.append((acq_data, nex, ky, z, _is_parallel_calibration(acq), i))
                if len(pending) >= batch_size:
                    flush_readouts()
            flush_readouts()

            if kspace is None or len(timestamps) == 0:
                raise ValueError("No acquisitions were mapped into the output tensor.")

            repeated = line_count > 1
            if bool(torch.any(repeated)):
                print(f"[ismrmrd] {int(repeated.sum().item())} (ky, kz, repetition) lines were acquired more than once; "
                      "their readouts are averaged. Each readout keeps its own physiological sample.", flush=True)
                kspace.div_(line_count.clamp(min=1).reshape(1, Nex, 1, Ny, z_size))

            if navigators is not None:
                for navigator in navigators:
                    navigator["time_seconds"] = (navigator["acquisition_time_stamp"] - timestamps[-1]) * 2.5e-3
                if not navigators:
                    print("[ismrmrd] Navigator reading requested, but no acquisitions carry ACQ_IS_NAVIGATION_DATA.", flush=True)
            self.navigators = navigators
            timestamps = torch.tensor(timestamps, device=self.device, dtype=torch.float64)
            timestamps = (timestamps - timestamps[-1]) * 2.5e-3

            if reference_line_seen is None or not torch.any(reference_line_seen):
                reference_kspace = None
            self.reference_kspace = reference_kspace

            nex_values = torch.tensor(sorted(nex_values_seen), device=self.device, dtype=torch.int64)
            print(f'[ismrmrd] Header selection, sample decoding, oversampling removal, and gridding completed in {perf_counter() - started:.2f} s.', flush=True)

            return (
                kspace,
                timestamps,
                torch.tensor(z_indices, device=self.device),
                torch.tensor(idx_ky, device=self.device),
                torch.tensor(idx_kz, device=self.device),
                torch.tensor(idx_nex, device=self.device),
                "repetition",
                nex_values,
                slice_geometry,
            )
        finally:
            dset.close()


    def read_data(self):
        """Read decoded MRI tensors and full-sequence acquisition metadata.

        Readout oversampling is removed from imaging and reference k-space using
        the header matrix ratio (1x or 2x), with a warning and 2x fallback when invalid.
        Physiological processing and slice selection are handled downstream.
        Navigators are returned as acquisition-ordered records when requested,
        or None otherwise. Their samples and trajectories remain as acquired;
        time_seconds shares the imaging time origin (last retained imaging readout).
        """
        (kspace, times, slices, ky, kz, nex, nex_source, nex_values,
         geometry) = self._extract_mri_data()
        reference_kspace = self.reference_kspace
        return {
            "ismrmrd_header": self.ismrmrd_header,
            "kspace": kspace, "time_seconds": times, "slice_indices": slices,
            "idx_ky": ky, "idx_kz": kz, "idx_nex": nex,
            "nex_source": nex_source, "nex_values": nex_values,
            "slice_geometry": geometry, "reference_kspace": reference_kspace,
            "uses_kz_as_volume_axis": self._raw_uses_kz_as_volume_axis,
            "navigators": self.navigators,
            "readout_oversampling_factor": self._oversampling_factor,
            "navigator_type": self.navigator_type,
            "navigator_set": self.navigator_set if self.navigator_type is not None else None,
            "imaging_set": self.imaging_set if self.navigator_type is not None else None,
        }

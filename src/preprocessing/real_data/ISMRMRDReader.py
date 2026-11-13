from src.utils.ismrmrd_io import ReadOnlyDataset
import ismrmrd
import torch
import math
import warnings
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

    def __init__(self, ismrmrd_file, device="cpu"):
        self.ismrmrd_file = ismrmrd_file
        self.device = device

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

    def _remove_oversampling(self, kspace: torch.Tensor):

        device = kspace.device
        dtype = kspace.dtype

        coils, Nex, readout, Ny, Nsli = kspace.shape
        cropped_readout = readout // 2

        kspace_cropped = torch.zeros(
            (coils, Nex, cropped_readout, Ny, Nsli),
            dtype=dtype, device=device)

        for iz in range(Nsli):

            kspace_slice = kspace[..., iz]

            img = ifftnc(kspace_slice, dims=(2,))

            crop_start = readout // 2 - cropped_readout // 2
            crop_end = crop_start + cropped_readout
            img_cropped = img[:, :, crop_start:crop_end, :]

            kspace_cropped[..., iz] = fftnc(img_cropped, dims=(2,))

        return kspace_cropped

    def _extract_mri_data(self):
        dset = ReadOnlyDataset(self.ismrmrd_file)
        try:
            xml_header = dset.read_xml_header()
            self.ismrmrd_header = (xml_header.decode("utf-8")
                                   if isinstance(xml_header, bytes) else xml_header)
            header = ismrmrd.xsd.CreateFromDocument(self.ismrmrd_header)
            enc = header.encoding[0]
            limits = enc.encodingLimits
            self._oversampling_factor = self._readout_oversampling_factor(enc)

            # Readout oversampling from the header; upstream assumed 2x for every file and
            # cropped half of the field of view when there was none.
            encoded_x, recon_x = int(enc.encodedSpace.matrixSize.x), int(enc.reconSpace.matrixSize.x)
            if recon_x <= 0 or encoded_x % recon_x != 0 or encoded_x // recon_x not in (1, 2):
                raise ValueError(f"Unsupported readout oversampling: encodedSpace.x={encoded_x}, reconSpace.x={recon_x}.")
            self._oversampling_factor = encoded_x // recon_x
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
            # Upstream ignored encodingLimits.*.center; that only matters for asymmetric
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

            num_acq = dset.number_of_acquisitions()
            for i in range(num_acq):
                acq = dset.read_acquisition(i)
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

                acq_data = torch.from_numpy(acq.data).to(self.device)

                if kspace is None:
                    ncha, nsamp = acq.data.shape
                    kspace = torch.zeros(
                        (ncha, Nex, nsamp, Ny, z_size),
                        dtype=torch.complex128,
                        device=self.device,
                    )
                    reference_kspace = torch.zeros(
                        (ncha, 1, nsamp, Ny, z_size),
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
                # A line acquired more than once in the same repetition (averages) is averaged;
                # upstream silently kept the last one.
                if line_count[nex, ky, z] > 0:
                    kspace[:, nex, :, ky, z] += acq_data
                else:
                    kspace[:, nex, :, ky, z] = acq_data
                line_count[nex, ky, z] += 1

                if _is_parallel_calibration(acq):
                    if reference_line_seen[ky, z]:
                        raise ValueError(
                            "Duplicate parallel-calibration acquisition found for "
                            f"ky={ky}, z={z} in acquisition {i}. "
                            "Expected at most one calibration line per (ky, z)."
                        )
                    reference_kspace[:, 0, :, ky, z] = acq_data
                    reference_line_seen[ky, z] = True

            if kspace is None or len(timestamps) == 0:
                raise ValueError("No acquisitions were mapped into the output tensor.")

            repeated = line_count > 1
            if bool(torch.any(repeated)):
                print(f"[ismrmrd] {int(repeated.sum().item())} (ky, kz, repetition) lines were acquired more than once; "
                      "their readouts are averaged. Each readout keeps its own physiological sample.", flush=True)
                kspace = kspace / line_count.clamp(min=1).to(kspace.dtype).reshape(1, Nex, 1, Ny, z_size)

            timestamps = torch.tensor(timestamps, device=self.device)
            timestamps = (timestamps - timestamps[-1]) * 2.5e-3

            if reference_line_seen is None or not torch.any(reference_line_seen):
                reference_kspace = None
            self.reference_kspace = reference_kspace

            nex_values = torch.tensor(sorted(nex_values_seen), device=self.device, dtype=torch.int64)

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
        """
        (kspace, times, slices, ky, kz, nex, nex_source, nex_values,
         geometry) = self._extract_mri_data()
        reference_kspace = self.reference_kspace
        if self._oversampling_factor == 2:
            kspace = self._remove_oversampling(kspace)
            reference_kspace = None if reference_kspace is None else self._remove_oversampling(reference_kspace)
        return {
            "ismrmrd_header": self.ismrmrd_header,
            "kspace": kspace, "time_seconds": times, "slice_indices": slices,
            "idx_ky": ky, "idx_kz": kz, "idx_nex": nex,
            "nex_source": nex_source, "nex_values": nex_values,
            "slice_geometry": geometry, "reference_kspace": reference_kspace,
            "uses_kz_as_volume_axis": self._raw_uses_kz_as_volume_axis,
        }

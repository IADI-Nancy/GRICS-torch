import ismrmrd
import torch
import math
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

# CODEX: to rename this class to ISMRMRDReader
class RawDataReader:

    def __init__(self, ismrmrd_file, device="cpu", print_raw_calibration_lines=False):
        self.ismrmrd_file = ismrmrd_file
        self.device = device
        # Log each flagged calibration acquisition; does not control data extraction.
        self.print_raw_calibration_lines = bool(print_raw_calibration_lines)

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
    def _expanded_matrix_size(n_lines, accel):
        acquired_lines = math.ceil(float(n_lines) / float(accel))
        return int(2 * math.ceil((float(accel) * float(acquired_lines)) / 2.0))

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

            crop_start = readout // 4
            crop_end = 3 * readout // 4
            img_cropped = img[:, :, crop_start:crop_end, :]

            kspace_cropped[..., iz] = fftnc(img_cropped, dims=(2,))

        return kspace_cropped

# CODEX: make this function less vertical (remove separation to multiple lines)
    def _extract_mri_data(self):
        dset = ismrmrd.Dataset(self.ismrmrd_file, 'dataset', create_if_needed=False)
        try:
            header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
            enc = header.encoding[0]
            limits = enc.encodingLimits

            N_SLI = self._encoding_limit_size(limits.slice)
            # Use repetition as Nex source for these 3D raw datasets.
            Nex = self._encoding_limit_size(limits.repetition)
            Nex = max(1, Nex)

            Nz_native = self._encoding_limit_size(limits.kspace_encoding_step_2)

            Ny = self._encoding_limit_size(limits.kspace_encoding_step_1)
            Ry = self._accel_factor_or_one(getattr(enc, "parallelImaging", None), "kspace_encoding_step_1")
            Ny = self._expanded_matrix_size(Ny, Ry)

            # Heuristic: multi-slice acquisitions are treated as 2D stacks,
            # while slab acquisitions (typically one encoded slice with kz partitions)
            # use kspace_encode_step_2 as the volume axis.
            use_kz_as_volume_axis = (N_SLI <= 1) and (Nz_native > 1)
            if use_kz_as_volume_axis:
                Rz = self._accel_factor_or_one(
                    getattr(enc, "parallelImaging", None), "kspace_encoding_step_2"
                )
                Nz = self._expanded_matrix_size(Nz_native, Rz)
                z_size = Nz
            else:
                Nz = 1
                z_size = N_SLI

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

                ky = int(acq.idx.kspace_encode_step_1)
                kz = int(acq.idx.kspace_encode_step_2)
                rep = int(acq.idx.repetition)
                sli = int(acq.idx.slice)
                ts = float(acq.acquisition_time_stamp)

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
                kspace[:, nex, :, ky, z] = acq_data

                if _is_parallel_calibration(acq):
                    if self.print_raw_calibration_lines:
                        print(
                            "[RawDataReader] parallel calibration line: "
                            f"acquisition={i}, ky={ky}, z={z}, repetition={rep}"
                        )
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
        """Read MRI arrays and full-sequence times, without physiological processing.

        K-space is returned before readout oversampling removal. Acquisition
        mapping and timestamps follow the existing scanner-reading implementation.
        """
        (kspace, times, slices, ky, kz, nex, nex_source, nex_values,
         geometry) = self._extract_mri_data()
        return {
            "kspace": kspace, "time_seconds": times, "slice_indices": slices,
            "idx_ky": ky, "idx_kz": kz, "idx_nex": nex,
            "nex_source": nex_source, "nex_values": nex_values,
            "slice_geometry": geometry, "reference_kspace": self.reference_kspace,
            "uses_kz_as_volume_axis": self._raw_uses_kz_as_volume_axis,
        }

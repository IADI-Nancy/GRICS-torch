import h5py
import ismrmrd
import torch
import math
import os
from src.utils.fftnc import fftnc, ifftnc
from src.preprocessing.RespiratoryDataReader import RespiratoryDataReader


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


class RawDataReader:

    def __init__(self, ismrmrd_file, saec_file, sensor_type='BELT', device="cuda"):
        self.ismrmrd_file = ismrmrd_file
        self.saec_file = saec_file
        self.sensor_type = sensor_type
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
    def _expanded_matrix_size(n_lines, accel):
        return int(2 * math.ceil((float(accel) * float(n_lines)) / 2.0))

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


    def _extract_mri_info(self):

        dset = ismrmrd.Dataset(self.ismrmrd_file, 'dataset', create_if_needed=False)
        try:
            header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
            enc = header.encoding[0]
            limits = enc.encodingLimits
            acq_sys = getattr(header, "acquisitionSystemInformation", None)

            N_SLI = self._encoding_limit_size(limits.slice)
            # Use repetition as Nex source for these 3D raw datasets.
            Nex = self._encoding_limit_size(limits.repetition)
            Nex = max(1, Nex)

            Nz = self._encoding_limit_size(limits.kspace_encoding_step_2)
            Rz = self._accel_factor_or_one(getattr(enc, "parallelImaging", None), "kspace_encoding_step_2")
            Nz = self._expanded_matrix_size(Nz, Rz)

            Ny = self._encoding_limit_size(limits.kspace_encoding_step_1)
            Ry = self._accel_factor_or_one(getattr(enc, "parallelImaging", None), "kspace_encoding_step_1")
            Ny = self._expanded_matrix_size(Ny, Ry)

            use_kz_as_volume_axis = Nz > 1
            z_size = Nz if use_kz_as_volume_axis else N_SLI

            ncha_header = None
            if acq_sys is not None and getattr(acq_sys, "receiverChannels", None) is not None:
                ncha_header = int(acq_sys.receiverChannels)

            kspace = None
            filled = None
            nex_values_seen = set()
            timestamps = []
            z_indices = []
            idx_ky = []
            idx_kz = []
            idx_nex = []

            num_acq = dset.number_of_acquisitions()
            for i in range(num_acq):
                acq = dset.read_acquisition(i)
                if _is_noise(acq) or _is_non_imaging(acq):
                    continue

                ncha, nsamp = acq.data.shape
                ky = int(acq.idx.kspace_encode_step_1)
                kz = int(acq.idx.kspace_encode_step_2)
                rep = int(acq.idx.repetition)
                sli = int(acq.idx.slice)
                ts = float(acq.acquisition_time_stamp)

                nex = rep
                z = kz if use_kz_as_volume_axis else sli

                acq_data = torch.from_numpy(acq.data).to(self.device)
                if ncha_header is not None and acq_data.shape[0] != ncha_header:
                    continue

                if kspace is None:
                    ncha_use = ncha_header if ncha_header is not None else int(acq_data.shape[0])
                    kspace = torch.zeros(
                        (ncha_use, Nex, nsamp, Ny, z_size),
                        dtype=torch.complex128,
                        device=self.device,
                    )
                    filled = torch.zeros((Nex, Ny, z_size), dtype=torch.bool, device=self.device)

                if acq_data.shape[0] != kspace.shape[0] or acq_data.shape[1] != kspace.shape[2]:
                    continue

                # Keep first valid sample per (nex, ky, kz/slice) to avoid silent overwrite corruption.
                if filled[nex, ky, z]:
                    continue

                timestamps.append(ts)
                z_indices.append(z)
                idx_ky.append(ky)
                idx_kz.append(kz)
                idx_nex.append(nex)
                nex_values_seen.add(nex)
                kspace[:, nex, :, ky, z] = acq_data
                filled[nex, ky, z] = True

            if kspace is None or len(timestamps) == 0:
                raise ValueError("No acquisitions were mapped into the output tensor.")

            timestamps = torch.tensor(timestamps, device=self.device)
            timestamps = (timestamps - timestamps[-1]) * 2.5e-3

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
            )
        finally:
            close = getattr(dset, "close", None)
            if callable(close):
                close()


    def _interp1d_torch(self, x, y, x_new):

        x = x.flatten()
        y = y.flatten()
        x_new = x_new.flatten()

        idx = torch.searchsorted(x, x_new)

        idx0 = torch.clamp(idx - 1, 0, len(x) - 1)
        idx1 = torch.clamp(idx, 0, len(x) - 1)

        x0 = x[idx0]
        x1 = x[idx1]
        y0 = y[idx0]
        y1 = y[idx1]

        denom = (x1 - x0)
        denom[denom == 0] = 1e-12

        weight = (x_new - x0) / denom

        y_new = y0 + weight * (y1 - y0)

        return y_new


    def _reshape_data_slicewise(self, respiratory_data_interpolated, z_indices, idx_ky, idx_kz, idx_nex):

        device = respiratory_data_interpolated.device

        N_SLI = int(torch.max(z_indices).item()) + 1

        counts = torch.bincount(z_indices, minlength=N_SLI)
        if torch.any(counts != counts[0]):
            raise ValueError(
                "Acquisition lines per z-index are not uniform; cannot reshape into "
                "[Nz, Nlines] realworld format."
            )
        lines_per_slice = int(counts[0].item())

        motion_data = torch.zeros(
            (N_SLI, lines_per_slice),
            dtype=respiratory_data_interpolated.dtype,
            device=device)

        line_idx_y = torch.zeros(
            (N_SLI, lines_per_slice),
            dtype=idx_ky.dtype,
            device=device)

        line_idx_z = torch.zeros(
            (N_SLI, lines_per_slice),
            dtype=idx_kz.dtype,
            device=device)

        line_idx_nex = torch.zeros(
            (N_SLI, lines_per_slice),
            dtype=idx_nex.dtype,
            device=device)

        for i_sli in range(N_SLI):
            mask = (z_indices == i_sli)

            motion_data[i_sli] = respiratory_data_interpolated[mask]
            line_idx_y[i_sli] = idx_ky[mask]
            line_idx_z[i_sli] = idx_kz[mask]
            line_idx_nex[i_sli] = idx_nex[mask]

        return motion_data, line_idx_y, line_idx_z, line_idx_nex


    def _read_motion_and_kspace(self, slice_idx=None):

        time_saec, resp = RespiratoryDataReader._read_and_process_data(
            self.saec_file, self.sensor_type)

        time_saec = torch.tensor(time_saec, device=self.device)
        resp = torch.tensor(resp, device=self.device)

        kspace, time_kspace, z_indices, idx_ky, idx_kz, idx_nex, nex_source, nex_values = \
            self._extract_mri_info()

        respiratory_interpolated = self._interp1d_torch(
            time_saec, resp, time_kspace)

        motion_data, line_idx_y, line_idx_z, line_idx_nex = \
            self._reshape_data_slicewise(
                respiratory_interpolated, z_indices, idx_ky, idx_kz, idx_nex)

        kspace = self._remove_oversampling(kspace)

        if slice_idx is not None:
            n_slices = int(kspace.shape[-1])
            if slice_idx < 0 or slice_idx >= n_slices:
                raise ValueError(
                    f"slice_idx={slice_idx} is out of range for {n_slices} slices."
                )
            kspace = kspace[..., [slice_idx]]
            motion_data = motion_data[[slice_idx], :]
            line_idx_y = line_idx_y[[slice_idx], :]
            line_idx_z = line_idx_z[[slice_idx], :]
            line_idx_nex = line_idx_nex[[slice_idx], :]

        return {
            "kspace": kspace.detach().cpu().numpy(),
            "motion_data": motion_data.detach().cpu().numpy(),
            "idx_ky": line_idx_y.detach().cpu().numpy(),
            "idx_kz": line_idx_z.detach().cpu().numpy(),
            "idx_nex": line_idx_nex.detach().cpu().numpy(),
            "nex_source": nex_source,
            "nex_values": nex_values.detach().cpu().numpy(),
        }


    def _read_data_from_rawdata(self, h5filename=None, slice_idx=None):

        data = self._read_motion_and_kspace(slice_idx=slice_idx)

        if h5filename is None:
            base = os.path.splitext(os.path.basename(self.ismrmrd_file))[0]
            h5filename = os.path.join(
                os.path.dirname(self.ismrmrd_file),
                f"{base}_realworld_from_raw.h5",
            )

        with h5py.File(h5filename, 'w') as f:
            f.create_dataset('motion_data', data=data['motion_data'])
            f.create_dataset('idx_ky', data=data['idx_ky'])
            f.create_dataset('idx_kz', data=data['idx_kz'])
            f.create_dataset('idx_nex', data=data['idx_nex'])
            f.create_dataset('kspace', data=data['kspace'])
            f.create_dataset('nex_values', data=data['nex_values'])
            f.attrs['nex_source'] = data['nex_source']

        data['realworld_h5_path'] = h5filename

        return data

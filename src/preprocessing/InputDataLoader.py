"""Load recorded or synthetic acquisition data before reconstruction preparation."""

import os
from pathlib import Path

import torch

from src.preprocessing.SamplingSimulator import SamplingSimulator
from src.preprocessing.SyntheticDataGenerator import SyntheticDataGenerator
from src.preprocessing.measurement_data.ISMRMRDReader import ISMRMRDReader
from src.preprocessing.measurement_data.RawDataPreparer import RawDataPreparer
from src.preprocessing.measurement_data.physiological_data.PolarisInfraredTrackerReader import PolarisInfraredTrackerReader
from src.preprocessing.measurement_data.physiological_data.PreprocessedPhysioReader import PreprocessedPhysioReader
from src.preprocessing.measurement_data.physiological_data.SAECReader import SAECReader
from src.preprocessing.measurement_data.real_data_io import convert_siemens_to_ismrmrd, read_preprocessed_data
from src.runtime.output_layout import bind_output_paths, record_reconstruction
from src.runtime.runtime_config import REAL_DATA_TYPES, ISMRMRD_READER_DATA_TYPES, validate_sampling_size


class InputDataLoader:
    """Choose an input source and prepare reconstruction-facing input data.

    The loader converts arrays to tensors. For 2D multi-slice data, it retains
    all slices and selects one slice at a time; for 3D data, it keeps the full
    volume active. It also prepares sampling and motion metadata. Format-specific
    decoding and physiological synchronization are delegated to dedicated classes.
    """

    def __init__(self, params, t_device=None, filename=None, slice_idx=None):
        self.params = params
        self._one_motion_state_per_readout = (params.simulated_motion_type.endswith("-per-shot")
                                             and params.kspace_sampling_type == "from-data")
        self.t_device = t_device
        self.filename = filename
        self.raw_data_preparer = None
        self.realworld_h5_path = None
        self.rawdata_filenames = None
        self.siemens_filenames = None
        self.slice_idx = slice_idx
        self.motion_data = None
        self.reference_kspace = None
        self.ky_idx = None
        self.kz_idx = None
        self.nex_idx = None
        self.ky_per_motion_state = None
        self.kz_per_motion_state = None
        self._input_data_loaded = False
        self._input_kspace = None
        self._input_reference_kspace = None
        self._input_motion_data = None
        self._input_idx_ky = None
        self._input_idx_kz = None
        self._input_idx_nex = None
        self._input_slice_geometry = None
        self.source_ismrmrd_file = None
        self.ismrmrd_header = None
        self._validate_inputs()

    @property
    def is_loaded(self):
        return self._input_data_loaded

    def _validate_inputs(self):
        if self.params.data_type != "shepp-logan" and self.filename is None:
            raise ValueError("filename is required when data_type is not 'shepp-logan'.")

        is_3d = self.params.data_dimension == "3D"
        supports_slice_idx = self.params.data_type in REAL_DATA_TYPES

        if self.slice_idx is not None:
            self.slice_idx = int(self.slice_idx)

        if self.params.data_type in ISMRMRD_READER_DATA_TYPES and self.params.data_type.startswith("ismrmrd-"):
            self.rawdata_filenames = self._resolve_rawdata_filenames()
        elif self.params.data_type in ISMRMRD_READER_DATA_TYPES and self.params.data_type.startswith("siemens-"):
            self.siemens_filenames = self._resolve_siemens_filenames()
        if self.params.data_type.endswith("-saec") and not self.params.rawdata_sensor_type:
            raise ValueError(f"rawdata_sensor_type must be set for data_type={self.params.data_type!r}.")

        if self.slice_idx is not None and not supports_slice_idx:
            raise ValueError(
                "slice_idx is only supported for 2D preprocessed-real or raw MRI/physiology inputs. "
                f"Do not provide slice_idx when data_type='{self.params.data_type}'."
            )

        if self.slice_idx is not None and self.slice_idx < 0:
            raise ValueError("slice_idx must be >= 0.")

        if is_3d and self.slice_idx is not None:
            raise ValueError(
                "slice_idx is only supported for 2D preprocessed-real or raw MRI/physiology inputs. "
                "Do not provide slice_idx when data_dimension='3D'."
            )

    def load_input_data(self):
        if self._input_data_loaded:
            raise RuntimeError("InputDataLoader.load_input_data() was called after input data was already loaded.")
        if hasattr(self.params, 'run_folder'):
            bind_output_paths(self.params, Path(self.params.run_folder))
        self._load_input_data()
        self._input_data_loaded = True
        return self

    def select_slice(self, slice_idx, params):
        """Select one slice while retaining the complete loaded acquisition."""
        if not self.is_loaded:
            raise RuntimeError("InputDataLoader.select_slice() requires load_input_data() first.")
        self.params = params
        self._select_and_prepare_2d_slice(slice_idx)

    def _use_synthetic_data(self, generator):
        # Real and synthetic inputs provide the same acquisition attributes to DataPreparer.
        for name in ("kspace", "Ncha", "Nx", "Ny", "Nz", "ky_idx", "nex_idx",
                     "ky_per_motion_state", "kz_idx", "kz_per_motion_state", "smaps_generated"):
            setattr(self, name, getattr(generator, name))
        if hasattr(generator, "phantom_generated"):
            self.phantom_generated = generator.phantom_generated

    def _load_input_data(self):
        print(f"[InputDataLoader] Loading input data (data_type={self.params.data_type})...")
        if self.params.data_type == 'shepp-logan':
            generator = SyntheticDataGenerator(self.params, self.t_device)
            self._use_synthetic_data(generator.generate_shepp_logan(
                N=self.params.N_SheppLogan, Ncoils=self.params.Ncoils_SheppLogan,
                Nz=self.params.Nz_SheppLogan, random_phase=True,
            ))
        elif self.params.data_type == 'preprocessed-real':
            self._load_realworld_data(self.filename, slice_idx=self.slice_idx)
        elif self.params.data_type in ISMRMRD_READER_DATA_TYPES and self.params.data_type.startswith("ismrmrd-"):
            path_to_ismrm, path_to_physiology = self.rawdata_filenames
            self.source_ismrmrd_file = path_to_ismrm
            self._load_realworld_data_from_ismrm_and_physiology(path_to_ismrm, path_to_physiology, slice_idx=self.slice_idx)
        elif self.params.data_type in ISMRMRD_READER_DATA_TYPES and self.params.data_type.startswith("siemens-"):
            path_to_siemens, path_to_physiology = self.siemens_filenames
            path_to_ismrm = self._convert_siemens_to_ismrmrd(path_to_siemens)
            self.source_ismrmrd_file = path_to_ismrm
            self._load_realworld_data_from_ismrm_and_physiology(path_to_ismrm, path_to_physiology, slice_idx=self.slice_idx)
        elif self.params.data_type == 'from_image':
            generator = SyntheticDataGenerator(self.params, self.t_device)
            self._use_synthetic_data(generator.load_image(self.filename))
        else:
            raise ValueError("Unknown data_type")

    def _configure_realworld_motion_inputs(self, motion_data, kz_all=None):
        # Retain kz coordinates and flatten ky/repetition indices for readout grouping.
        self.kz_idx = kz_all
        ky_idx = self.ky_idx.reshape(-1)
        nex_idx = self.nex_idx.reshape(-1)
        # Group ky readouts by repetition, preserving their acquisition order.
        self.ky_per_motion_state = [
            [value.reshape(1) for value in ky_idx[nex_idx == nex]]
            for nex in range(self.params.Nex)
        ]
        if self._one_motion_state_per_readout:
            # With recorded sampling, assign one simulated motion state to each readout.
            requested_states = self.params.N_motion_states
            self.params.N_motion_states = ky_idx.numel()
            self.params._motion_states_per_level = [self.params.N_motion_states] * len(self.params.ResolutionLevels)
            if requested_states != self.params.N_motion_states:
                print(f"[config] Per-shot simulation: N_motion_states changed from {requested_states} to {self.params.N_motion_states} (recorded readout count).", flush=True)
        # Retain the aligned physiological signals for downstream motion preparation.
        self.motion_data = motion_data

    def _resolve_rawdata_filenames(self):
        return self._resolve_mri_physiology_filenames("ismrmrd_file")

    def _resolve_siemens_filenames(self):
        return self._resolve_mri_physiology_filenames("siemens_raw_file")

    def _resolve_mri_physiology_filenames(self, mri_key):
        suffix = self.params.data_type.split('-', 1)[1]
        physiology_keys = {
            'saec': ('saec_file',), 'polaris': ('polaris_file',),
            'physio_text': ('physio_file',),
            'physio_array': ('physio_timestamps_file', 'physio_values_file'),
        }[suffix]
        keys = (mri_key, *physiology_keys)
        paths = None
        resampling = self.params.kspace_sampling_type != "from-data"
        if resampling and isinstance(self.filename, (str, os.PathLike)):
            return self.filename, None
        if isinstance(self.filename, (tuple, list)) and len(self.filename) == len(keys):
            paths = tuple(self.filename)
        elif isinstance(self.filename, dict):
            unknown = set(self.filename) - set(keys)
            if unknown:
                raise ValueError(f"Unknown input filename keys: {sorted(unknown)}.")
            paths = tuple(self.filename.get(key) for key in keys)
        required = paths[:1] if resampling and paths is not None else paths
        if required is None or any(not isinstance(value, (str, os.PathLike)) or not os.fspath(value)
                                   for value in required):
            if resampling:
                raise ValueError(f"Reordered raw input requires an MRI path or a dict containing {mri_key!r}.")
            description = "both files" if len(keys) == 2 else "all three files"
            raise ValueError(
                f"For data_type={self.params.data_type!r}, filename must contain {description}: "
                f"a {len(keys)}-item tuple/list or a dict with keys {keys}."
            )
        physiology = paths[1:] if suffix == 'physio_array' else paths[1]
        return paths[0], physiology

    def _convert_siemens_to_ismrmrd(self, path_to_siemens):
        return convert_siemens_to_ismrmrd(path_to_siemens, self.params)

    def _initialize_realworld_data(self, data, slice_idx=None):
        is_3d = self.params.data_dimension == "3D"

        self._input_kspace = torch.from_numpy(data['kspace']).to(self.t_device, dtype=torch.cdouble)
        reference_np = data.get('reference_kspace')
        self._input_reference_kspace = (
            None if reference_np is None else torch.from_numpy(reference_np).to(self.t_device, dtype=torch.cdouble)
        )

        if self.params.kspace_sampling_type == "from-data":
            self._input_motion_data = torch.from_numpy(data['motion_data']).to(self.t_device)
            self._input_idx_ky = torch.from_numpy(data['idx_ky']).to(self.t_device, dtype=torch.int64)
            self._input_idx_kz = torch.from_numpy(data['idx_kz']).to(self.t_device, dtype=torch.int64)
            self._input_idx_nex = torch.from_numpy(data['idx_nex']).to(self.t_device, dtype=torch.int64)
        self._input_slice_geometry = data.get('slice_geometry')
        self.ismrmrd_header = data.get('ismrmrd_header')

        self.kspace = self._input_kspace
        self.reference_kspace = self._input_reference_kspace
        self._validate_and_set_kspace_dimensions()

        if is_3d:
            self._prepare_3d_sampling_and_motion_inputs()
        elif slice_idx is not None:
            self._select_and_prepare_2d_slice(slice_idx)

    def _validate_and_set_kspace_dimensions(self):
        if self.kspace.ndim != 5 or any(size < 1 for size in self.kspace.shape):
            raise ValueError("Real kspace must have nonempty [coil, repetition, x, y, z] dimensions.")
        self.Ncha, n_repetitions, self.Nx, self.Ny, self.Nz = self.kspace.shape
        if self.params.kspace_sampling_type == "from-data":
            self.params.Nex = n_repetitions
        elif self.params.Nex != n_repetitions:
            raise ValueError("Configured Nex must match the repetitions in the real k-space array.")

    def _build_resampled_realworld_inputs(self):
        if self.params.simulated_motion_type == "as-it-is":
            raise ValueError("Reordered real data requires simulated motion.")
        validate_sampling_size(self.params, self.Ny, self.Nz)
        simulator = SamplingSimulator(self.Ny, self.params, self.t_device)
        (self.ky_idx, self.nex_idx, self.ky_per_motion_state,
         self.kz_idx, self.kz_per_motion_state) = simulator._build_phase_encode_indices_and_nex(Nz=self.Nz)
        self.motion_data = None

    def _prepare_3d_sampling_and_motion_inputs(self):
        if self.params.kspace_sampling_type != "from-data":
            self._build_resampled_realworld_inputs()
            return
        idx_ky_flat = self._input_idx_ky.reshape(-1)
        if self._input_motion_data.ndim != 2 or self._input_motion_data.shape[0] != idx_ky_flat.numel():
            raise ValueError(
                "3D motion_data must have shape [Nreadout, Nsensor]. "
                f"Got {tuple(self._input_motion_data.shape)}; expected first dimension {idx_ky_flat.numel()}."
            )
        self.ky_idx = idx_ky_flat
        self.nex_idx = self._input_idx_nex.reshape(-1)
        kz_all = self._input_idx_kz.reshape(-1)
        self._configure_realworld_motion_inputs(self._input_motion_data, kz_all=kz_all)

    def _select_and_prepare_2d_slice(self, slice_idx):
        """Select a 2D slice and update its k-space, dimensions, sampling indices, and motion inputs.

        Keep the complete acquisition in the cached input arrays for later slice selections.
        """
        # Validate the acquisition, slice index, and recorded motion array shape.
        if self.params.data_dimension == "3D":
            raise ValueError("slice_idx must not be provided when data_dimension='3D'.")
        if self._input_kspace is None:
            raise RuntimeError("No loaded input k-space is available for slice selection.")

        slice_idx = int(slice_idx)
        n_slices = int(self._input_kspace.shape[-1])
        if slice_idx < 0 or slice_idx >= n_slices:
            raise ValueError(f"slice_idx={slice_idx} is out of range for {n_slices} slices.")
        if self.params.kspace_sampling_type == "from-data" and self._input_motion_data.ndim not in {2, 3}:
            raise ValueError(
                "2D motion_data must have shape [Nslice, Nreadout] or "
                "[Nslice, Nreadout, Nsensor]. "
                f"Got {tuple(self._input_motion_data.shape)}."
            )

        # Update the selected slice and its output paths.
        self.slice_idx = slice_idx
        if hasattr(self.params, 'run_folder'):
            folder = Path(self.params.run_folder)
            if getattr(self.params, '_output_unit_count', 1) > 1:
                folder /= f'slice_{slice_idx + 1:03d}'
            bind_output_paths(self.params, folder)
            record_reconstruction(self.params, inputs=self.filename, source_slice_index=slice_idx)
        # Select the slice while retaining the singleton slice dimension.
        self.kspace = self._input_kspace[..., slice_idx:slice_idx + 1]
        self.reference_kspace = (
            None
            if self._input_reference_kspace is None
            else self._input_reference_kspace[..., slice_idx:slice_idx + 1]
        )
        self._validate_and_set_kspace_dimensions()
        # Build new sampling indices when the recorded sampling order is not used.
        if self.params.kspace_sampling_type != "from-data":
            self._build_resampled_realworld_inputs()
            return
        # Select the chosen slice's motion signals and sampling indices.
        motion_data = self._input_motion_data[slice_idx]
        if motion_data.ndim == 1:
            motion_data = motion_data.unsqueeze(-1)
        self.ky_idx = self._input_idx_ky[slice_idx]
        self.kz_idx = None
        self.nex_idx = self._input_idx_nex[slice_idx]
        self._configure_realworld_motion_inputs(motion_data, kz_all=None)

    def _load_realworld_data_from_ismrm_and_physiology(self, path_to_ismrm, path_to_physiology, slice_idx=None):
        # Resampled workflows need decoded MRI data but do not use recorded physiology.
        if self.params.kspace_sampling_type != "from-data":
            reader = ISMRMRDReader(path_to_ismrm, device="cpu",
                                   print_raw_calibration_lines=self.params.print_raw_calibration_lines)
            raw_data = reader.read_data()
            data = {"kspace": raw_data["kspace"].cpu().numpy(),
                    "slice_geometry": raw_data["slice_geometry"],
                    "ismrmrd_header": raw_data.get("ismrmrd_header")}
            if raw_data["reference_kspace"] is not None:
                data["reference_kspace"] = raw_data["reference_kspace"].cpu().numpy()
            self._initialize_realworld_data(data, slice_idx=slice_idx)
            return
        # From-data workflows synchronize the selected physiological source to MRI readouts.
        physiological_reader = self._create_physiological_reader()
        preparer = RawDataPreparer(
            ismrmrd_file=path_to_ismrm, physiological_file=path_to_physiology,
            physiological_reader=physiological_reader,
            physio_clock_drift_seconds=self.params.physio_clock_drift_seconds,
            device="cpu", print_raw_calibration_lines=self.params.print_raw_calibration_lines)
        self.raw_data_preparer = preparer
        data = preparer.read_data(
            cache_h5=self.params.cache_preprocessed_data, cache_root=self.params.cache_root,
            remove_temporary_data_after_run=self.params.remove_temporary_data_after_run)
        # Convert the complete prepared acquisition into the loader's tensor representation.
        self.realworld_h5_path = data.get("realworld_h5_path")
        self._initialize_realworld_data(data, slice_idx=slice_idx)

    def _create_physiological_reader(self):
        """Construct the physiological reader selected by the runtime configuration."""
        physiological_format = self.params.data_type.split("-", 1)[1]
        if physiological_format == "polaris":
            return PolarisInfraredTrackerReader(
                channel_mode=self.params.polaris_channel_mode, lowpass_cutoff_hz=self.params.polaris_lowpass_cutoff_hz)
        if physiological_format == "saec":
            return SAECReader(
                sensor_type=self.params.rawdata_sensor_type, belt_lowpass_cutoff_hz=self.params.saec_belt_lowpass_cutoff_hz,
                marmot_lowpass_cutoff_hz=self.params.saec_marmot_lowpass_cutoff_hz,
                marmot_highpass_cutoff_hz=self.params.saec_marmot_highpass_cutoff_hz)
        if physiological_format in {"physio_text", "physio_array"}:
            return PreprocessedPhysioReader(physiological_format)
        raise ValueError(f"Unsupported physiological format: {physiological_format!r}.")

    def _load_realworld_data(self, path_to_data, slice_idx=None):
        data = read_preprocessed_data(
            path_to_data, from_data=self.params.kspace_sampling_type == "from-data",
        )
        self._initialize_realworld_data(data, slice_idx=slice_idx)

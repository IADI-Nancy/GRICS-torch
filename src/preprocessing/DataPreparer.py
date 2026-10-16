import copy
import gc
import os
import time
import torch
from typing import Dict, Optional, Sequence, Union

from src.preprocessing.InputDataLoader import InputDataLoader
from src.preprocessing.MotionSimulator import MotionSimulator
from src.utils.fftnc import ifftnc
from src.preprocessing.MotionSamplingPreparer import MotionSamplingPreparer
from src.preprocessing.CoilSensitivityCalculator import CoilSensitivityCalculator
from src.preprocessing.utils.reconstructor_input_preparation import (
    ensure_input_loaded_and_reconstruction_not_prepared,
    record_preprocessing_start_if_configured,
    save_sampling_order_debug_plot_if_enabled,
    select_input_slice_if_needed,
    validate_input_for_reconstruction,
)
from src.utils.plotting import (
    show_and_save_image, compute_motion_plot_y_limits,
    save_nonrigid_alpha_plots,
)
from src.utils.nonrigid_display import to_cartesian_components


class DataPreparer:
    """Transform one loaded slice or volume into reconstruction inputs.

    InputDataLoader owns source loading and slice selection. This class applies
    normalization, coil-map estimation, motion import or simulation, and motion
    binning before reconstruction.
    """

    def __init__(self, params, sp_device=None, t_device=None,
                 filename: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
                 slice_idx=None):
        """
        Configure input loading and reconstruction preparation.

        Call load_input_data() and prepare_reconstructor_inputs() after construction.

        Args:
            params: Runtime/config parameters.
            sp_device: SigPy device.
            t_device: Torch device.
            filename: Input path, or a pair (MRI file, physiological file) for raw data.
                Dictionaries use ismrmrd_file or siemens_raw_file, plus saec_file,
                polaris_file, or physio_file. Generic array input uses three paths:
                (MRI file, timestamps .npy, values .npy), or dictionary keys
                physio_timestamps_file and physio_values_file alongside the MRI key.
            slice_idx: Slice/partition to load for 2D real-data inputs.
        """
        self.input_loader = InputDataLoader(params, t_device=t_device, filename=filename, slice_idx=slice_idx)
        self.params = params
        self.sp_device = sp_device
        self.t_device = t_device
        self.filename = filename
        self.slice_idx = self.input_loader.slice_idx
        self.motion_plot_context = None
        self._motion_curve_for_binning = None
        self._motion_plot_kwargs = {}
        self._motion_plot_y_limits = None
        self.kz_idx_chronological = None
        self.grics_reference_image = None
        self.kspace_scale = 1.0
        self._reconstruction_data_prepared = False

    def load_input_data(self):
        """Load or generate input data once for subsequent reconstruction preparation."""
        self.input_loader.load_input_data()
        return self

    def _set_input_data_from_loader(self, loader):
        """Set input data and metadata used by reconstruction preparation.

        Tensors are shared initially; preparation may replace self.kspace without
        changing the loader's original acquisition.
        """
        self.kspace = loader.kspace
        self.reference_kspace = loader.reference_kspace
        self.Ncha, self.Nx, self.Ny, self.Nz = loader.Ncha, loader.Nx, loader.Ny, loader.Nz
        self.ky_idx, self.kz_idx, self.nex_idx = loader.ky_idx, loader.kz_idx, loader.nex_idx
        self.ky_per_motion_state = loader.ky_per_motion_state
        self.kz_per_motion_state = loader.kz_per_motion_state
        self.motion_data = loader.motion_data
        self.slice_idx = loader.slice_idx
        self.raw_data_preparer = loader.raw_data_preparer
        self.realworld_h5_path = loader.realworld_h5_path
        self.source_ismrmrd_file = loader.source_ismrmrd_file
        self.ismrmrd_header = loader.ismrmrd_header
        self._input_slice_geometry = loader._input_slice_geometry
        if hasattr(loader, "smaps_generated"):
            self.smaps_generated = loader.smaps_generated
        if hasattr(loader, "phantom_generated"):
            self.phantom_generated = loader.phantom_generated

    def prepare_reconstructor_inputs(self, slice_idx=None):
        """Prepare reconstructor inputs for one selected slice or volume."""

        ensure_input_loaded_and_reconstruction_not_prepared(self.input_loader, self._reconstruction_data_prepared)
        slice_idx = select_input_slice_if_needed(self.input_loader, self.params, slice_idx)

        if slice_idx is not None:
            # Multi-slice pipelines shallow-copy one loaded DataPreparer for each
            # slice. Those copies share input_loader, but should have separate slice data.
            self.input_loader = copy.copy(self.input_loader)
            self.input_loader.select_slice(slice_idx, self.params)

        # Check input dimensions, calibration support, and motion readout count.
        validate_input_for_reconstruction(self.input_loader, self.params)
        self._set_input_data_from_loader(self.input_loader)
        save_sampling_order_debug_plot_if_enabled(self.input_loader, self.params)
        record_preprocessing_start_if_configured(self.params, self.filename, self.slice_idx)
        self._normalize_kspace_if_enabled()
        self._compute_coil_sensitivity_maps()
        self._compute_ifft_image_from_input_kspace()
        # Import recorded motion or simulate motion and corrupt the input data.
        motionSimulator = self._apply_or_import_motion()
        # Set a shared display range for initial and final motion plots.
        self._prepare_motion_plot_y_limits()
        # Group acquired samples into the motion states consumed by reconstruction.
        self._build_binned_sampling_indices()
        # Collect motion and sampling metadata used by reconstruction plots.
        self._prepare_motion_plot_context()
        # Save optional images and motion diagnostics from preprocessing.
        self._save_initial_data()
        # Release the potentially large simulator before reconstruction starts.
        del motionSimulator
        self._reconstruction_data_prepared = True
        return self

    def _compute_coil_sensitivity_maps(self):
        """Estimate coil maps and retain an optional GRICS calibration image."""
        method = self.params.coil_sensitivity_method
        slice_label = "all" if self.slice_idx is None else f"{self.slice_idx + 1:03d}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp}] [csm] slice {slice_label} calculating coil sensitivity maps "
            f"(method={method})",
            flush=True,
        )
        calculator = CoilSensitivityCalculator(self.params, sp_device=self.sp_device)
        self.smaps = calculator.calculate(
            self.kspace, reference_kspace=self.reference_kspace,
        )
        self.grics_reference_image = getattr(calculator, "grics_reference_image", None)

    def _compute_ifft_image_from_input_kspace(self):
        """Inverse FFT and coil-combine the loaded input k-space."""
        averaged_kspace = torch.mean(self.kspace, dim=1)
        coil_ifft_images = ifftnc(averaged_kspace, dims=(-3, -2, -1))
        input_ifft_image = torch.sum(coil_ifft_images * self.smaps.conj(), dim=0)
        if self.params.data_dimension == "2D":
            input_ifft_image = input_ifft_image.squeeze(-1)
        self.input_ifft_image = input_ifft_image.to(self.t_device)
        del coil_ifft_images

    def _prepare_motion_simulator_inputs(self, motion_sim_device):
        input_ifft_image = self.input_ifft_image.to(motion_sim_device)
        smaps = self.smaps.to(motion_sim_device)
        ky_idx = self.ky_idx.to(motion_sim_device)
        nex_idx = self.nex_idx.to(motion_sim_device)
        ky_per_motion_state = [
            [ky.to(motion_sim_device) for ky in ky_per_nex]
            for ky_per_nex in self.ky_per_motion_state
        ]

        kz_idx = getattr(self, "kz_idx", None)
        if kz_idx is not None:
            kz_idx = kz_idx.to(motion_sim_device)

        kz_per_motion_state = getattr(self, "kz_per_motion_state", None)
        if kz_per_motion_state is not None:
            kz_per_motion_state = [
                [kz.to(motion_sim_device) for kz in kz_per_nex]
                for kz_per_nex in kz_per_motion_state
            ]

        return input_ifft_image, smaps, ky_idx, nex_idx, ky_per_motion_state, kz_idx, kz_per_motion_state

    def _simulate_motion_on_device(self, motion_sim_device):
        inputs = self._prepare_motion_simulator_inputs(motion_sim_device)
        input_ifft_image, smaps, ky_idx, nex_idx, ky_per_motion_state, kz_idx, kz_per_motion_state = inputs
        print("[DataPreparer] Starting motion simulation "
            f"(type={self.params.simulated_motion_type}, device={motion_sim_device})...", flush=True)
        motion_simulator = MotionSimulator(input_ifft_image, smaps, ky_idx, nex_idx, ky_per_motion_state,
            kz_idx=kz_idx, kz_per_motion_state=kz_per_motion_state, params=self.params,
            sp_device=self.sp_device, t_device=motion_sim_device)

        simulation_methods = {
            "rigid-per-shot": motion_simulator.simulate_discrete_rigid_motion,
            "rigid-per-readout": motion_simulator.simulate_per_readout_rigid_motion,
            "non-rigid-per-shot": motion_simulator.simulate_discrete_non_rigid_motion,
            "non-rigid-per-readout": motion_simulator.simulate_per_readout_non_rigid_motion,
        }
        simulation_methods[self.params.simulated_motion_type]()
        return motion_simulator

    @staticmethod
    def _is_cuda_out_of_memory(error):
        return isinstance(error, torch.cuda.OutOfMemoryError) or "cuda out of memory" in str(error).lower()

    def _simulate_motion_with_cpu_fallback(self):
        motion_sim_device = torch.device(self.t_device)
        retry_on_cpu = False
        try:
            motion_simulator = self._simulate_motion_on_device(motion_sim_device)
        except RuntimeError as error:
            if motion_sim_device.type != "cuda" or not self._is_cuda_out_of_memory(error):
                raise
            retry_on_cpu = True

        if retry_on_cpu:
            gc.collect()
            torch.cuda.empty_cache()
            print("[DataPreparer] CUDA memory is insufficient for motion simulation; "
                "switching to CPU and restarting the simulation.", flush=True)
            motion_simulator = self._simulate_motion_on_device(torch.device("cpu"))
        return motion_simulator

    def _apply_or_import_motion(self):
        # Real motion case
        if self.params.simulated_motion_type == 'as-it-is':
            self._motion_curve_for_binning = self.motion_data
            self.image_no_moco = self.input_ifft_image
            return None

        # Only a simulated-motion workflow treats the input IFFT image as its ground truth.
        self.image_ground_truth = self.input_ifft_image
        motionSimulator = self._simulate_motion_with_cpu_fallback()

        self.kspace = motionSimulator.get_corrupted_kspace().to(self.t_device)
        self.image_no_moco = motionSimulator.get_corrupted_image().to(self.t_device)
        if hasattr(motionSimulator, "alpha_maps"):
            self.alpha_maps_ground_truth = motionSimulator.alpha_maps.to(self.t_device)

        if self.params.simulated_motion_type.startswith("rigid-"):
            if self.Nz > 1:
                motion_curve, tx, ty, tz, rx, ry, rz = motionSimulator.get_rigid_motion_information_3d()
                self._motion_plot_kwargs = {"tx": tx, "ty": ty, "tz": tz, "rx": rx, "ry": ry, "rz": rz}
            else:
                motion_curve, tx, ty, phi = motionSimulator.get_rigid_motion_information_2d()
                self._motion_plot_kwargs = {"tx": tx, "ty": ty, "phi": phi}
        else:
            motion_curve, _ = motionSimulator.get_nonrigid_motion_information()
            self._motion_plot_kwargs = {}

        # Motion binning expects [readouts, signals]; simulated motion provides
        # one scalar navigator value per readout.
        self._motion_curve_for_binning = motion_curve.reshape(-1, 1)
        return motionSimulator

    def _prepare_motion_plot_context(self):
        if self._motion_curve_for_binning is None:
            self.motion_plot_context = None
            return

        motion_plot_context = {"motion_curve": self._motion_curve_for_binning, "labels": self.motion_labels,
            "ky_idx": self.ky_idx_chronological, "nex_idx": self.nex_idx_chronological,
            "kz_idx": self.kz_idx_chronological, "resolution_levels": self.params.ResolutionLevels,
            "data_type": self.params.data_type, "y_limits": self._motion_plot_y_limits, "alpha_visual_scale": None}

        if (
            self.params.simulated_motion_type.startswith("non-rigid-")
            and hasattr(self, "alpha_maps_ground_truth")
            and self.alpha_maps_ground_truth is not None
        ):
            alpha = self.alpha_maps_ground_truth
            if alpha.ndim >= 3 and alpha.shape[0] >= 2:
                alpha_axis0 = alpha[0].real if torch.is_complex(alpha[0]) else alpha[0]
                alpha_axis1 = alpha[1].real if torch.is_complex(alpha[1]) else alpha[1]
                # For 3D volumes, compute scale over the full volume.
                alpha_x, alpha_y = to_cartesian_components(alpha_axis0, alpha_axis1)
                amp_max = float(torch.max(torch.sqrt(alpha_axis0 * alpha_axis0 + alpha_axis1 * alpha_axis1)).item())
                alpha_abs_max_x = float(torch.max(torch.abs(alpha_x)).item())
                alpha_abs_max_y = float(torch.max(torch.abs(alpha_y)).item())
                motion_plot_context["alpha_visual_scale"] = {
                    "alpha_abs_max_x": max(alpha_abs_max_x, 1e-12),
                    "alpha_abs_max_y": max(alpha_abs_max_y, 1e-12),
                    "amp_max": max(amp_max, 1e-12),
                }

        self.motion_plot_context = motion_plot_context

    def _prepare_motion_plot_y_limits(self):
        if self._motion_curve_for_binning is not None:
            self._motion_plot_y_limits = compute_motion_plot_y_limits(
                self._motion_curve_for_binning, **self._motion_plot_kwargs,
            )
        else:
            self._motion_plot_y_limits = None

    def _build_binned_sampling_indices(self):
        prepared = MotionSamplingPreparer.prepare(
            self._motion_curve_for_binning, self.ky_idx, self.kz_idx, self.nex_idx,
            self.t_device, self.params, self.Nx, self.Ny, Nz=self.Nz,
            kspace=self.kspace, y_limits=self._motion_plot_y_limits,
            plot_kwargs=self._motion_plot_kwargs,
        )
        self.binned_ky_indices = prepared.binned_ky_indices
        self.binned_kz_indices = prepared.binned_kz_indices
        self.motion_signal = prepared.motion_signal
        self.motion_labels = prepared.motion_labels
        self.ky_idx_chronological = prepared.chronological_ky
        self.kz_idx_chronological = prepared.chronological_kz
        self.nex_idx_chronological = prepared.chronological_nex
        self.sampling_idx = prepared.sampling_indices["all"]

    def _normalize_kspace_if_enabled(self):
        self.kspace_scale = 1.0
        if not self.params.normalize_kspace:
            return

        k = self.kspace
        if self.params.kspace_norm_mode == "max":
            s = torch.max(torch.abs(k))
        elif self.params.kspace_norm_mode == "rms":
            s = torch.linalg.norm(k.flatten()) / (k.numel() ** 0.5)
        else:
            raise ValueError(f"Unknown kspace_norm_mode: {self.params.kspace_norm_mode}")

        s = torch.clamp(s.real if torch.is_complex(s) else s, min=self.params.kspace_norm_eps)
        self.kspace_scale = float(s.item())
        self.kspace = self.kspace / self.kspace_scale
        if self.reference_kspace is not None:
            self.reference_kspace = self.reference_kspace / self.kspace_scale

    def _save_initial_data(self):
        folder = self.params.initial_data_folder
        if not folder or not self.params.save_debug_plots:
            return
        os.makedirs(folder, exist_ok=True)

        flip_for_display = self.params.flip_for_display

        # Simulation motion case - save ground truth image
        if (self._has_simulated_motion() and hasattr(self, "image_ground_truth") and self.image_ground_truth is not None):
            show_and_save_image(self.image_ground_truth, "image_ground_truth", folder,
                flip_for_display=flip_for_display, jupyter_display=False)
        # Save corrupted image
        if hasattr(self, "image_no_moco") and self.image_no_moco is not None:
            show_and_save_image(self.image_no_moco, "image_corrupted", folder,
                flip_for_display=flip_for_display, jupyter_display=False)

        if (hasattr(self, "alpha_maps_ground_truth") and self.alpha_maps_ground_truth is not None
            and self.params.simulated_motion_type.startswith("non-rigid-") and self.alpha_maps_ground_truth.ndim >= 3
            and self.alpha_maps_ground_truth.shape[0] >= 2):

            scale = (self.motion_plot_context or {}).get("alpha_visual_scale", None)
            save_nonrigid_alpha_plots(self.alpha_maps_ground_truth[..., 0], self.image_ground_truth,
                "simulated_input", folder, flip_vertical=flip_for_display,
                abs_max_x=None if scale is None else scale.get("alpha_abs_max_x"),
                abs_max_y=None if scale is None else scale.get("alpha_abs_max_y"),
                amp_max=None if scale is None else scale.get("amp_max"))

    def _has_simulated_motion(self):
        return self.params.simulated_motion_type != "as-it-is"

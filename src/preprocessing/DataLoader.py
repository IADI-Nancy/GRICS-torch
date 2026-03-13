import numpy as np
import os
import torch
from skimage.data import shepp_logan_phantom
from skimage import io as skio
from skimage.transform import resize
import math
import h5py
import sigpy as sp
import sigpy.mri as spmri

from src.preprocessing.RawDataReader import RawDataReader
from src.preprocessing.MotionSimulator import MotionSimulator
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions
from src.preprocessing.SamplingSimulator import SamplingSimulator
from src.preprocessing.MotionBinner import MotionBinner
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.utils.plotting import (
    show_and_save_image, compute_motion_y_limits,
    save_nonrigid_alpha_plots,
)
from src.utils.nonrigid_display import to_cartesian_components


class DataLoader:
    @staticmethod
    def _to_device_recursive(obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, list):
            return [DataLoader._to_device_recursive(x, device) for x in obj]
        if isinstance(obj, tuple):
            return tuple(DataLoader._to_device_recursive(x, device) for x in obj)
        return obj

    def __init__(self, params, sp_device=None, t_device=None, filename=None, slice_idx=0):
        self.params = params
        self.sp_device = sp_device
        self.t_device = t_device
        self.kspace_scale = 1.0
        self.filename = filename
        self.slice_idx = slice_idx
        self.motion_plot_context = None

        if self.params.data_type != "shepp-logan" and self.filename is None:
            raise ValueError("filename is required when data_type is not 'shepp-logan'.")
        
        if self.params.data_type == 'shepp-logan': # Generation of Shepp-Logan phantom with coil sensitivities + sampling simulation   
            self._generate_shepp_logan(N=self.params.N_SheppLogan, Ncoils=self.params.Ncoils_SheppLogan, Nz=self.params.Nz_SheppLogan, random_phase=True)
        elif self.params.data_type == 'real-world': # Real-world data with acquisition order and motion data
            self._load_realworld_data(self.filename, slice_idx=self.slice_idx)
        elif self.params.data_type == 'raw-data': # Real-world data with acquisition order and motion data, loaded from raw data files
            if isinstance(self.filename, (tuple, list)) and len(self.filename) == 2:
                path_to_ismrm, path_to_saec = self.filename
            elif isinstance(self.filename, dict):
                path_to_ismrm = self.filename.get("ismrmrd_file")
                path_to_saec = self.filename.get("saec_file")
            else:
                raise ValueError(
                    "For data_type='raw-data', filename must be a 2-item tuple/list "
                    "(ismrmrd_file, saec_file) or a dict with keys "
                    "'ismrmrd_file' and 'saec_file'."
                )
            self._load_realworld_data_from_ismrm_and_saec(path_to_ismrm, path_to_saec, slice_idx=self.slice_idx)
        elif self.params.data_type == 'from_image':
            self._load_from_image(self.filename)
        elif self.params.data_type == 'from_dicom':
            self._load_from_dicom(self.filename)
        else:
            raise ValueError("Unknown data_type")

        self._normalize_kspace_if_enabled()
        
        # Keep a copy of motion-free k-space before any simulated corruption.
        self.kspace_nomotion = self.kspace.clone()

        # Calculate ESPIRiT maps and input image
        self.smaps = self._calc_espirit_maps()
        self.img_cplx = ifftnc(self.kspace, dims=(-3, -2, -1))
        smaps_replicated = self.smaps.unsqueeze(1).expand(-1, self.params.Nex, -1, -1, -1)
        self.image_ground_truth = torch.sum(self.img_cplx*smaps_replicated.conj(), dim=0).to(self.t_device)

        motion_sim_device = self.t_device
        if (
            self.params.motion_type == "rigid"
            and self.params.motion_state_mode == "realistic"
        ):
            motion_sim_device = torch.device("cpu")

        motionSimulator = MotionSimulator(
            self._to_device_recursive(self.image_ground_truth, motion_sim_device),
            self._to_device_recursive(self.smaps, motion_sim_device),
            self._to_device_recursive(self.ky_idx, motion_sim_device),
            self._to_device_recursive(self.nex_idx, motion_sim_device),
            self._to_device_recursive(self.ky_per_motion, motion_sim_device),
            params=self.params,
            sp_device=self.sp_device,
            t_device=motion_sim_device,
        )
        
        if self.params.motion_simulation_type == 'as-it-is':
            if self.params.data_type in ['real-world', 'raw-data']:
                self.image_no_moco = self.image_ground_truth.clone()
            else:
                raise ValueError("Simulation type 'as-it-is' is only compatible with real-world or raw-data, which already contain motion. Please choose a different simulation type or data type.")
        else:
            if self.params.motion_simulation_type == 'no-motion-data':
                motionSimulator._simulate_no_motion()
                self.image_no_moco = self.image_ground_truth.clone()
            else:      
                if self.params.motion_type == "rigid":
                    if self.params.motion_state_mode == "per-shot":
                        motionSimulator._simulate_discrete_rigid_motion()
                    elif self.params.motion_state_mode == "realistic":
                        motionSimulator._simulate_realistic_rigid_motion()
                    else:
                        raise ValueError("Unknown motion_state_mode for rigid motion.")
                elif self.params.motion_type == "non-rigid":
                    if self.params.motion_state_mode == "per-shot":
                        motionSimulator._simulate_discrete_non_rigid_motion()
                    elif self.params.motion_state_mode == "realistic":
                        motionSimulator._simulate_realistic_non_rigid_motion()
                    else:
                        raise ValueError("Unknown motion_state_mode for non-rigid motion.")
                else:
                    raise ValueError("Unknown motion_type")
                self.kspace = motionSimulator._get_corrupted_kspace().to(self.t_device)
                self.image_no_moco = motionSimulator._get_corrupted_image().to(self.t_device)
                if hasattr(motionSimulator, "alpha_maps"):
                    self.alpha_maps_true = motionSimulator.alpha_maps.to(self.t_device)

            motion_curve, tx, ty, phi = motionSimulator._get_motion_information()
            tz = getattr(motionSimulator, "tz", None)
            rx = getattr(motionSimulator, "rx", None)
            ry = getattr(motionSimulator, "ry", None)
            rz = getattr(motionSimulator, "rz", None)
            # In 3D rigid mode, phi is a legacy compatibility alias (mapped to rz).
            # Hide it from input plots to avoid duplicate rotational traces.
            phi_for_plot = None if (self.Nz > 1 and rz is not None) else phi

            y_limits = compute_motion_y_limits(
                motion_curve, tx=tx, ty=ty, phi=phi_for_plot, tz=tz, rx=rx, ry=ry, rz=rz
            )
            (
                self.binned_indices,
                self.motion_signal,
                self.motion_labels,
                self.ky_idx_chronological,
                self.nex_idx_chronological,
            ) = MotionBinner._bin_motion(
                motion_curve, self.ky_idx, self.nex_idx, self.t_device, self.params,
                tx=tx, ty=ty, phi=phi_for_plot, tz=tz, rx=rx, ry=ry, rz=rz,
                y_limits=y_limits, return_debug_data=True,
            )
            self.motion_plot_context = {
                "motion_curve": motion_curve,
                "labels": self.motion_labels,
                "ky_idx": self.ky_idx_chronological,
                "nex_idx": self.nex_idx_chronological,
                "resolution_levels": self.params.ResolutionLevels,
                "data_type": self.params.data_type,
                "y_limits": y_limits,
                "alpha_visual_scale": None,
            }
            if (
                self.params.motion_type == "non-rigid"
                and hasattr(self, "alpha_maps_true")
                and self.alpha_maps_true is not None
            ):
                alpha = self.alpha_maps_true
                if alpha.ndim >= 3 and alpha.shape[0] >= 2:
                    alpha_axis0 = alpha[0].real if torch.is_complex(alpha[0]) else alpha[0]
                    alpha_axis1 = alpha[1].real if torch.is_complex(alpha[1]) else alpha[1]
                    # For 3D volumes, compute scale over the full volume.
                    alpha_x, alpha_y = to_cartesian_components(alpha_axis0, alpha_axis1)
                    amp_max = float(torch.max(torch.sqrt(alpha_axis0 * alpha_axis0 + alpha_axis1 * alpha_axis1)).item())
                    alpha_abs_max_x = float(torch.max(torch.abs(alpha_x)).item())
                    alpha_abs_max_y = float(torch.max(torch.abs(alpha_y)).item())
                    self.motion_plot_context["alpha_visual_scale"] = {
                        "alpha_abs_max_x": max(alpha_abs_max_x, 1e-12),
                        "alpha_abs_max_y": max(alpha_abs_max_y, 1e-12),
                        "amp_max": max(amp_max, 1e-12),
                    }
        
        self.sampling_idx = SamplingSimulator._build_sampling_per_nex_per_motion(
            self.binned_indices, self.t_device, self.Nx, self.Ny,
            Nz=self.Nz, kspace_sampling_type=getattr(self.params, "kspace_sampling_type", "from-data")  # [Nex][Nmotion]
        )

        self._save_input_data_artifacts()

        if self.params.debug_flag and self._has_simulated_motion():
            self._debug_check_true_motion_image_reconstruction(motionSimulator)

    def _normalize_kspace_if_enabled(self):
        if not self.params.normalize_kspace:
            self.kspace_scale = 1.0
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
        print(
            f"[DataLoader] k-space normalized ({self.params.kspace_norm_mode}), "
            f"scale={self.kspace_scale:.6e}"
        )

    def _save_input_data_artifacts(self):
        folder = self.params.input_data_folder
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)

        flip_for_display = self.params.flip_for_display

        if (
            self._has_simulated_motion()
            and hasattr(self, "image_ground_truth")
            and self.image_ground_truth is not None
        ):
            show_and_save_image(
                self.image_ground_truth[0], "image_ground_truth", folder,
                flip_for_display=flip_for_display, jupyter_display=False,
            )
        if hasattr(self, "image_no_moco") and self.image_no_moco is not None:
            show_and_save_image(
                self.image_no_moco[0], "image_corrupted", folder,
                flip_for_display=flip_for_display, jupyter_display=False,
            )

        if (
            hasattr(self, "alpha_maps_true")
            and self.alpha_maps_true is not None
            and self.params.motion_type == "non-rigid"
            and self.alpha_maps_true.ndim >= 3
            and self.alpha_maps_true.shape[0] >= 2
        ):
            scale = (self.motion_plot_context or {}).get("alpha_visual_scale", None)
            save_nonrigid_alpha_plots(
                self.alpha_maps_true, self.image_ground_truth[0],
                "simulated_input", folder,
                flip_vertical=flip_for_display,
                abs_max_x=None if scale is None else scale.get("alpha_abs_max_x"),
                abs_max_y=None if scale is None else scale.get("alpha_abs_max_y"),
                amp_max=None if scale is None else scale.get("amp_max"),
            )

    def _has_simulated_motion(self):
        return self.params.motion_simulation_type not in {"no-motion-data", "as-it-is"}

    @staticmethod
    def _normalize_real_image(arr):
        arr = np.asarray(arr)
        if arr.ndim > 2:
            # RGB(A) -> grayscale luminance
            arr = arr[..., :3]
            arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        arr = np.squeeze(arr).astype(np.float64, copy=False)
        if np.iscomplexobj(arr):
            arr = np.abs(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D image after conversion, got shape {arr.shape}.")
        arr -= np.min(arr)
        denom = np.max(arr)
        if denom > 0:
            arr /= denom
        return arr

    def _create_synthetic_coil_maps(self, Nx, Ny, Ncoils, Nz=1, random_phase=True):
        X, Y = torch.meshgrid(
            torch.arange(1, Nx + 1, device=self.t_device, dtype=torch.float64),
            torch.arange(1, Ny + 1, device=self.t_device, dtype=torch.float64),
            indexing="ij",
        )

        sigma = max(Nx, Ny) / 4.0
        grid_size = math.ceil(math.sqrt(Ncoils))
        xs = torch.linspace(Nx / 4.0, 3.0 * Nx / 4.0, grid_size, device=self.t_device)
        ys = torch.linspace(Ny / 4.0, 3.0 * Ny / 4.0, grid_size, device=self.t_device)
        centers = [(x.item(), y.item()) for y in ys for x in xs][:Ncoils]

        smaps_2d = torch.empty((Ncoils, Nx, Ny), dtype=torch.cdouble, device=self.t_device)
        for c, (x0, y0) in enumerate(centers):
            profile = torch.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma ** 2))
            phase = (
                torch.exp(1j * 2.0 * math.pi * torch.rand(1, device=self.t_device))
                if random_phase else 1.0
            )
            smaps_2d[c] = profile * phase

        smaps = smaps_2d.unsqueeze(-1).expand(Ncoils, Nx, Ny, Nz)
        self.smaps_generated = smaps.clone()
        return smaps

    def _build_synthetic_kspace_from_reference_image(self, image_2d):
        image_2d = image_2d.to(self.t_device, dtype=torch.float64)
        Nx, Ny = image_2d.shape
        Nz = 1
        Ncoils = int(self.params.Ncoils_input)

        smaps = self._create_synthetic_coil_maps(Nx, Ny, Ncoils, Nz=Nz, random_phase=True)
        ref = image_2d.unsqueeze(-1).expand(Nx, Ny, Nz).unsqueeze(0)  # (1, Nx, Ny, Nz)
        coil_imgs = ref * smaps  # (Ncoils, Nx, Ny, Nz)
        coil_imgs = coil_imgs.unsqueeze(1).expand(-1, self.params.Nex, -1, -1, -1).contiguous()

        self.kspace = fftnc(coil_imgs, dims=(-3, -2, -1))
        self.Ncha, _, self.Nx, self.Ny, self.Nz = self.kspace.shape

        samplingSimulator = SamplingSimulator(self.Ny, self.params, self.t_device)
        self.ky_idx, self.nex_idx, self.ky_per_motion = samplingSimulator._build_ky_and_nex()

    def _apply_resize_factor(self, img_np):
        factor = float(self.params.image_resize_factor)
        if factor <= 0:
            raise ValueError("image_resize_factor must be > 0.")
        if factor == 1.0:
            return img_np
        h, w = img_np.shape
        new_h = max(1, int(round(h * factor)))
        new_w = max(1, int(round(w * factor)))
        return resize(img_np, (new_h, new_w), anti_aliasing=True, preserve_range=True)

    def _load_from_image(self, path_to_image):
        ext = os.path.splitext(path_to_image)[1].lower()
        if ext == ".npy":
            img_np = np.load(path_to_image)
        elif ext == ".npz":
            npz = np.load(path_to_image)
            keys = list(npz.keys())
            if not keys:
                raise ValueError(f"No arrays found in NPZ file: {path_to_image}")
            img_np = npz[keys[0]]
        else:
            img_np = skio.imread(path_to_image)

        img_np = self._normalize_real_image(img_np)
        img_np = self._apply_resize_factor(img_np)
        img_t = torch.from_numpy(img_np).to(self.t_device, dtype=torch.float64)
        self._build_synthetic_kspace_from_reference_image(img_t)

    def _load_from_dicom(self, path_to_dicom):
        try:
            import pydicom
        except ImportError as e:
            raise ImportError(
                "pydicom is required for data_type='from_dicom'. "
                "Install it with: pip install pydicom"
            ) from e

        ds = pydicom.dcmread(path_to_dicom)
        px = ds.pixel_array.astype(np.float64, copy=False)
        if px.ndim > 2:
            # Use first frame if multi-frame DICOM is provided.
            px = px[0]
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        px = px * slope + intercept
        img_np = self._normalize_real_image(px)
        img_np = self._apply_resize_factor(img_np)
        img_t = torch.from_numpy(img_np).to(self.t_device, dtype=torch.float64)
        self._build_synthetic_kspace_from_reference_image(img_t)


    def _build_shepp_logan_2d(self, N, fill_fraction):
        fill_fraction = min(max(fill_fraction, 0.1), 1.0)
        phantom_native = shepp_logan_phantom()
        h, w = phantom_native.shape
        # Add margins by padding the native phantom before the final resize.
        # This preserves the phantom shape definition without pre-shrinking it.
        canvas_h = max(h, int(round(h / fill_fraction)))
        canvas_w = max(w, int(round(w / fill_fraction)))
        pad_top = (canvas_h - h) // 2
        pad_bottom = canvas_h - h - pad_top
        pad_left = (canvas_w - w) // 2
        pad_right = canvas_w - w - pad_left
        phantom_padded = np.pad(
            phantom_native,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0,
        )
        phantom_np_2d = resize(
            phantom_padded,
            (N, N),
            anti_aliasing=True,
        )
        return phantom_np_2d

    def _build_shepp_logan_3d(self, N, Nz, fill_fraction):
        # 3D modified Shepp-Logan approximation using a sum of ellipsoids.
        fill_fraction = min(max(float(fill_fraction), 0.1), 1.0)
        x = np.linspace(-1.0, 1.0, N, dtype=np.float64) / fill_fraction
        y = np.linspace(-1.0, 1.0, N, dtype=np.float64) / fill_fraction
        z = np.linspace(-1.0, 1.0, Nz, dtype=np.float64) / fill_fraction
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        phantom = np.zeros((N, N, Nz), dtype=np.float64)

        # (A, a, b, c, x0, y0, z0, phi_deg), with in-plane rotation phi.
        ellipsoids = [
            (1.0,   0.69,   0.92,   0.90,   0.0,    0.0,    0.0,    0.0),
            (-0.8,  0.6624, 0.874,  0.88,   0.0,   -0.0184, 0.0,    0.0),
            (-0.2,  0.11,   0.31,   0.22,   0.22,   0.0,    0.0,  -18.0),
            (-0.2,  0.16,   0.41,   0.28,  -0.22,   0.0,    0.0,   18.0),
            (0.1,   0.21,   0.25,   0.41,   0.0,    0.35,  -0.15,   0.0),
            (0.1,   0.046,  0.046,  0.05,   0.0,    0.10,   0.25,   0.0),
            (0.1,   0.046,  0.046,  0.05,   0.0,   -0.10,   0.25,   0.0),
            (0.1,   0.046,  0.023,  0.05,  -0.08,  -0.605,  0.0,    0.0),
            (0.1,   0.023,  0.023,  0.02,   0.0,   -0.606,  0.0,    0.0),
            (0.1,   0.023,  0.046,  0.02,   0.06,  -0.605,  0.0,    0.0),
        ]

        for A, a, b, c, x0, y0, z0, phi_deg in ellipsoids:
            phi = np.deg2rad(phi_deg)
            x_shift = X - x0
            y_shift = Y - y0
            z_shift = Z - z0
            x_rot = x_shift * np.cos(phi) + y_shift * np.sin(phi)
            y_rot = -x_shift * np.sin(phi) + y_shift * np.cos(phi)
            inside = (x_rot / a) ** 2 + (y_rot / b) ** 2 + (z_shift / c) ** 2 <= 1.0
            phantom[inside] += A

        # Match in-plane orientation of the canonical 2D Shepp-Logan (major axis vertical).
        phantom = np.swapaxes(phantom, 0, 1)

        # Keep the expected non-negative intensity profile.
        phantom = np.clip(phantom, 0.0, None)
        max_val = float(phantom.max())
        if max_val > 0:
            phantom /= max_val
        return phantom

    # Ncoils should be a perfect square (or close) for coil map generation
    def _generate_shepp_logan(self, N=128, Ncoils=4, Nz=1, random_phase=True):
        # 1. Create phantom (Nx, Ny, Nz)
        fill_fraction = float(self.params.SheppLoganFillFraction)
        if int(Nz) > 1:
            phantom_np = self._build_shepp_logan_3d(N, Nz, fill_fraction)
            phantom = torch.tensor(phantom_np, dtype=torch.float64, device=self.t_device)
        else:
            phantom_np_2d = self._build_shepp_logan_2d(N, fill_fraction)
            phantom_2d = torch.tensor(phantom_np_2d, dtype=torch.float64, device=self.t_device)
            phantom = phantom_2d.unsqueeze(-1).expand(N, N, Nz)  # (Nx, Ny, Nz)
        self.phantom_generated = phantom.clone()

        # 2. Create coil sensitivity maps (Ncoils, Nx, Ny, Nz)
        X, Y, Z = torch.meshgrid(
            torch.arange(1, N + 1, device=self.t_device),
            torch.arange(1, N + 1, device=self.t_device),
            torch.arange(1, Nz + 1, device=self.t_device),
            indexing="ij",
        )

        sigma_xy = N / 4
        # Large z-spread to keep meaningful sensitivity over the full slab.
        sigma_z = max(float(Nz) / 2.5, 1.0)
        z0 = (Nz + 1) / 2.0

        # Coil centers on (approximately) square grid
        grid_size = math.ceil(math.sqrt(Ncoils))
        xs = torch.linspace(N / 4, 3 * N / 4, grid_size, device=self.t_device)
        ys = torch.linspace(N / 4, 3 * N / 4, grid_size, device=self.t_device)
        centers = [(x.item(), y.item()) for y in ys for x in xs][:Ncoils]

        # Allocate coil maps directly in (Ncoils, Nx, Ny, Nz)
        smaps = torch.empty((Ncoils, N, N, Nz), dtype=torch.cdouble, device=self.t_device)

        for c, (x0, y0) in enumerate(centers):
            profile_xy = torch.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma_xy ** 2))
            profile_z = torch.exp(-((Z - z0) ** 2) / (2 * sigma_z ** 2))
            profile = profile_xy * profile_z
            phase = (
                torch.exp(1j * 2 * math.pi * torch.rand(1, device=self.t_device))
                if random_phase else 1.0
            )
            smaps[c] = profile * phase
        self.smaps_generated = smaps.clone()

        # 3. Generate coil images directly in (Ncoils, Nx, Ny, Nz)
        phantom_for_coils = phantom.unsqueeze(0)          # (1, Nx, Ny, Nz)
        coil_imgs = phantom_for_coils * smaps             # (Ncoils, Nx, Ny, Nz)
        coil_imgs = coil_imgs.unsqueeze(1).expand(-1, self.params.Nex, -1, -1, -1) # add Nex dimension: (Ncoils, Nex, Nx, Ny, Nz)

        coil_imgs = coil_imgs.contiguous()       # important for FFT speed

        # 4. FFT → k-space (Ncoils, Nx, Ny, Nz)
        self.kspace = fftnc(coil_imgs, dims=(-3, -2, -1))
        self.Ncha, _, self.Nx, self.Ny, self.Nz = self.kspace.shape

        # 5. Sampling
        samplingSimulator = SamplingSimulator(self.Ny, self.params, self.t_device)
        self.ky_idx, self.nex_idx, self.ky_per_motion = samplingSimulator._build_ky_and_nex()

    def _load_realworld_data_from_ismrm_and_saec(self, path_to_ismrm, path_to_saec, slice_idx=0):
        reader = RawDataReader(ismrmrd_file=path_to_ismrm, saec_file=path_to_saec, sensor_type="BELT", device="cuda")
        data = reader._read_data_from_rawdata()

        kspace_np = data['kspace']
        is_3d = self.params.data_dimension == "3D"
        if is_3d:
            # 3D acquisition: keep all kz partitions for volumetric reconstruction.
            self.kspace = torch.from_numpy(kspace_np).to(self.t_device, dtype=torch.cdouble)
        else:
            self.kspace = torch.from_numpy(kspace_np).to(self.t_device, dtype=torch.cdouble)[:, :, :, :, [slice_idx]]
        self.params.Nex = int(self.kspace.shape[1])
        self.params.NshotsPerNex = int(self.kspace.shape[3])
        self.params.Nshots = int(self.params.Nex) * int(self.params.NshotsPerNex)
        if getattr(self.params, "motion_state_mode", None) == "per-shot":
            self.params.N_motion_states = self.params.Nshots
        # For 3D, ky/nex/motion indices are the same across all kz partitions;
        # use partition 0 as the representative ordering.
        z_sel = 0 if is_3d else slice_idx
        self.ky_idx = torch.from_numpy(data['idx_ky'][z_sel]).to(self.t_device, dtype=torch.int64)
        self.nex_idx = torch.from_numpy(data['idx_nex'][z_sel]).to(self.t_device, dtype=torch.int64)
        motion_data = data['motion_data'][z_sel, :]
        motion_data = torch.from_numpy(motion_data).to(self.t_device)
        y_limits = compute_motion_y_limits(motion_data)
        (
            self.ky_per_motion,
            self.motion_signal,
            self.motion_labels,
            self.ky_idx_chronological,
            self.nex_idx_chronological,
        ) = MotionBinner._bin_motion(
            motion_data, self.ky_idx, self.nex_idx, self.t_device, self.params,
            y_limits=y_limits, return_debug_data=True
        )
        self.motion_plot_context = {
            "motion_curve": motion_data,
            "labels": self.motion_labels,
            "ky_idx": self.ky_idx_chronological,
            "nex_idx": self.nex_idx_chronological,
            "resolution_levels": self.params.ResolutionLevels,
            "data_type": self.params.data_type,
            "y_limits": y_limits,
        }
        self.binned_indices = self.ky_per_motion
        self.Ncha, _, self.Nx, self.Ny, self.Nz = self.kspace.shape

        SamplingSimulator._visualize_ky_order(
            [self.ky_idx.detach().cpu()], Ny=self.Ny,
            folder=self.params.input_data_folder, fname=f"ky_order_rawdata_slice{slice_idx}.png"
        )
        

    def _load_realworld_data(self, path_to_data, slice_idx=0):
        data = {}
        with h5py.File(path_to_data, 'r') as f:
            data['motion_data'] = f['motion_data'][:]
            data['idx_ky'] = f['idx_ky'][:]
            data['idx_kz'] = f['idx_kz'][:]
            data['idx_nex'] = f['idx_nex'][:]
            data['kspace'] = f['kspace'][:]

        kspace_np = data['kspace']
        is_3d = self.params.data_dimension == "3D"
        if is_3d:
            # 3D acquisition: keep all kz partitions for volumetric reconstruction.
            self.kspace = torch.from_numpy(kspace_np).to(self.t_device, dtype=torch.cdouble)
        else:
            self.kspace = torch.from_numpy(kspace_np).to(self.t_device, dtype=torch.cdouble)[:, :, :, :, [slice_idx]]
        self.params.Nex = int(self.kspace.shape[1])
        self.params.NshotsPerNex = int(self.kspace.shape[3])
        self.params.Nshots = int(self.params.Nex) * int(self.params.NshotsPerNex)
        if getattr(self.params, "motion_state_mode", None) == "per-shot":
            self.params.N_motion_states = self.params.Nshots
        # For 3D, ky/nex/motion indices are shared across kz partitions.
        z_sel = 0 if is_3d else slice_idx
        ky_dx = data['idx_ky'][z_sel]
        self.ky_idx = torch.from_numpy(ky_dx).to(self.t_device, dtype=torch.int64)
        self.nex_idx = torch.from_numpy(data['idx_nex'][z_sel]).to(self.t_device, dtype=torch.int64)
        motion_data = data['motion_data'][z_sel, :]
        motion_data = torch.from_numpy(motion_data).to(self.t_device)
        y_limits = compute_motion_y_limits(motion_data)
        (
            self.ky_per_motion,
            self.motion_signal,
            self.motion_labels,
            self.ky_idx_chronological,
            self.nex_idx_chronological,
        ) = MotionBinner._bin_motion(
            motion_data, self.ky_idx, self.nex_idx, self.t_device, self.params,
            y_limits=y_limits, return_debug_data=True
        )
        self.motion_plot_context = {
            "motion_curve": motion_data,
            "labels": self.motion_labels,
            "ky_idx": self.ky_idx_chronological,
            "nex_idx": self.nex_idx_chronological,
            "resolution_levels": self.params.ResolutionLevels,
            "data_type": self.params.data_type,
            "y_limits": y_limits,
        }
        self.binned_indices = self.ky_per_motion
        self.Ncha, _, self.Nx, self.Ny, self.Nz = self.kspace.shape

        SamplingSimulator._visualize_ky_order(
            [self.ky_idx.detach().cpu()], Ny=self.Ny,
            folder=self.params.input_data_folder, fname=f"ky_order_realworld_slice{slice_idx}.png"
        )

    def _calc_espirit_maps(self):
        acs = int(self.params.acs)
        kernel_width = int(self.params.kernel_width)
        espirit_max_iter = self.params.espirit_max_iter
        sp_device = self.sp_device
        kspace = self.kspace
        device = kspace.device             # torch device of input
        
        use_gpu = device.type == "cuda"

        if sp_device is None:
            sp_device = sp.Device(0 if use_gpu else -1)

        Ncha, _, Nx, Ny, Nz = kspace.shape

        # For measured data, calibrate maps from all repeats to avoid unstable
        # Nex-specific map estimates that can look noise-like in recon outputs.
        if self.params.data_type in {"real-world", "raw-data"} and kspace.shape[1] > 1:
            kspace_calib = torch.mean(kspace, dim=1)
        else:
            kspace_calib = kspace[:, 0]

        # Bound calibration settings by actual data size to avoid oversized
        # calibration matrices and excessive memory usage.
        if Nz > 1:
            calib_width_eff = max(1, min(acs, Nx, Ny, Nz))
        else:
            calib_width_eff = max(1, min(acs, Nx, Ny))
        kernel_width_eff = max(1, min(kernel_width, calib_width_eff))

        # 3D calibration for volumetric data, 2D calibration for single-slice data.
        if Nz > 1:
            # ---- GPU path ----
            if use_gpu:
                import cupy as cp

                try:
                    # Use complex64 for calibration to reduce temporary GPU memory.
                    kspace_cp = cp.asarray(kspace_calib[:, :, :, :].contiguous(), dtype=cp.complex64)
                    maps_cp = spmri.app.EspiritCalib(
                        kspace_cp,
                        calib_width=calib_width_eff,
                        kernel_width=kernel_width_eff,
                        max_iter=espirit_max_iter,
                        device=sp_device,
                    ).run()
                    maps_cp = maps_cp.astype(cp.complex64, copy=False)
                    maps_cp = cp.ascontiguousarray(maps_cp)
                    maps_t = torch.view_as_real(torch.utils.dlpack.from_dlpack(maps_cp))
                    espirit_maps = torch.complex(maps_t[..., 0], maps_t[..., 1]).to(torch.complex128)
                except cp.cuda.memory.OutOfMemoryError:
                    # Fallback to CPU calibration when GPU memory is insufficient.
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    torch.cuda.empty_cache()

                    kspace_np = kspace_calib[:, :, :, :].detach().cpu().numpy().astype(np.complex64, copy=False)
                    maps_np = spmri.app.EspiritCalib(
                        kspace_np,
                        calib_width=calib_width_eff,
                        kernel_width=kernel_width_eff,
                        max_iter=espirit_max_iter,
                        device=sp.Device(-1),
                    ).run()
                    maps_np = maps_np.astype(np.complex128, copy=False)
                    maps_t = torch.from_numpy(np.stack([maps_np.real, maps_np.imag], axis=-1))
                    espirit_maps = torch.complex(maps_t[..., 0], maps_t[..., 1]).to(device)
                finally:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    torch.cuda.empty_cache()

            # ---- CPU path ----
            else:
                kspace_np = kspace_calib[:, :, :, :].cpu().numpy().astype(np.complex64, copy=False)
                maps_np = spmri.app.EspiritCalib(
                    kspace_np,
                    calib_width=calib_width_eff,
                    kernel_width=kernel_width_eff,
                    max_iter=espirit_max_iter,
                    device=sp_device,
                ).run()
                maps_np = maps_np.astype(np.complex128, copy=False)
                maps_t = torch.from_numpy(np.stack([maps_np.real, maps_np.imag], axis=-1))
                espirit_maps = torch.complex(maps_t[..., 0], maps_t[..., 1]).to(device)

            return espirit_maps

        espirit_maps = torch.zeros((Ncha, Nx, Ny, Nz), dtype=torch.complex128, device=device)

        # Legacy 2D slice calibration (single-slice data).
        for z in range(Nz):
            if use_gpu:
                import cupy as cp

                kspace_cp = cp.asarray(kspace_calib[:, :, :, z].contiguous(), dtype=cp.complex64)
                maps_cp = spmri.app.EspiritCalib(
                    kspace_cp,
                    calib_width=calib_width_eff,
                    kernel_width=kernel_width_eff,
                    max_iter=espirit_max_iter,
                    device=sp_device,
                ).run()
                maps_cp = maps_cp.astype(cp.complex64, copy=False)
                maps_cp = cp.ascontiguousarray(maps_cp)
                maps_t = torch.view_as_real(torch.utils.dlpack.from_dlpack(maps_cp))
            else:
                kspace_np = kspace_calib[:, :, :, z].cpu().numpy().astype(np.complex64, copy=False)
                maps_np = spmri.app.EspiritCalib(
                    kspace_np,
                    calib_width=calib_width_eff,
                    kernel_width=kernel_width_eff,
                    max_iter=espirit_max_iter,
                    device=sp_device,
                ).run()
                maps_np = maps_np.astype(np.complex128, copy=False)
                maps_t = torch.from_numpy(np.stack([maps_np.real, maps_np.imag], axis=-1))

            espirit_maps[:, :, :, z] = torch.complex(maps_t[..., 0], maps_t[..., 1]).to(device)

        if use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()

        return espirit_maps

    def _debug_check_true_motion_image_reconstruction(self, motionSimulator):
        # This consistency check is meaningful only for simulated non-rigid data.
        if self.params.motion_type != "non-rigid":
            return
        if not self._has_simulated_motion():
            return
        if not hasattr(motionSimulator, "alpha_maps"):
            return
        if not hasattr(self, "motion_signal"):
            return

        with torch.no_grad():
            alpha_true = motionSimulator.alpha_maps.to(self.t_device)
            # Use exactly the signal/sampling that are fed to GN.
            signal_true = self.motion_signal.to(self.t_device)
            sampling_true = self.sampling_idx
            nsamples_true = self.Nx * self.Ny * self.Nz

            motion_op_true = MotionOperator(
                self.Nx, self.Ny, alpha_true, self.params.motion_type,
                motion_signal=signal_true, Nz=self.Nz,
            )

            encoding_true = EncodingOperator(
                self.smaps, nsamples_true, sampling_true, self.params.Nex, motion_op_true
            )

            kspace_vec = self.kspace.reshape(self.Ncha, self.params.Nex, nsamples_true).flatten()
            b = encoding_true.adjoint(kspace_vec)
            x0 = torch.zeros_like(b)
            solver = ConjugateGradientSolver(
                encoding_true, reg_lambda=0.0, verbose=False,
                early_stopping=self.params.cg_early_stopping, max_stag_steps=self.params.cg_max_stag_steps,
                max_more_steps=self.params.cg_max_more_steps, use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
                reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
            )
            img_vec = solver._solve_cg(b.flatten(), x0=x0.flatten(), max_iter=80, tol=1e-6)
            if self.Nz > 1:
                img_back = img_vec.reshape(self.params.Nex, self.Nx, self.Ny, self.Nz)
            else:
                img_back = img_vec.reshape(self.params.Nex, self.Nx, self.Ny)
            img_ref = self.image_ground_truth

            num = torch.linalg.norm((img_back - img_ref).flatten())
            den = torch.linalg.norm(img_ref.flatten()) + 1e-12
            rel_err = (num / den).item()

            print(f"[DEBUG] GN-input consistency check (true alpha, clustered signal/sampling): rel_err={rel_err:.6e}")
            show_and_save_image(
                img_back[0], "gn_input_consistency_recovered_image", self.params.debug_folder,
                flip_for_display=self.params.flip_for_display,
            )






        

    





    
    
        

import numpy as np
import os
import torch
from skimage.data import shepp_logan_phantom
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
from src.utils.show_and_save_image import show_and_save_image
from src.utils.save_alpha_component_map import save_alpha_component_map
from src.utils.save_clustered_motion_plots import compute_motion_y_limits
from src.utils.save_nonrigid_quiver_with_contours import save_nonrigid_quiver_with_contours
from src.utils.nonrigid_display import to_cartesian_components


class DataLoader:
    def __init__(
        self,
        params,
        sp_device=None,
        t_device=None,
        filename=None,
        slice_idx=0,
        recalculate_n_motion_states_from_navigator=False,
    ):
        self.params = params
        self.sp_device = sp_device
        self.t_device = t_device
        self.kspace_scale = 1.0
        self.filename = filename
        self.slice_idx = slice_idx
        self.motion_plot_context = None
        self.recalculate_n_motion_states_from_navigator = recalculate_n_motion_states_from_navigator

        if (
            self.recalculate_n_motion_states_from_navigator
            and self.params.data_type in {"real-world", "raw-data"}
            and self.params.motion_type == "rigid"
        ):
            # TODO: detect rigid motion events from navigator/belt signal and update
            # params.N_motion_states automatically before motion binning/reconstruction.
            raise NotImplementedError(
                "recalculate_n_motion_states_from_navigator is not implemented yet for "
                "real-world/raw-data rigid motion."
            )

        if self.params.data_type != "shepp-logan" and self.filename is None:
            raise ValueError("filename is required when data_type is not 'shepp-logan'.")
        
        if self.params.data_type == 'shepp-logan': # Generation of Shepp-Logan phantom with coil sensitivities + sampling simulation   
            self.generate_shepp_logan(N=self.params.N_SheppLogan, Ncoils=self.params.Ncoils_SheppLogan, Nz=self.params.Nz_SheppLogan, random_phase=True)
        elif self.params.data_type == 'real-world': # Real-world data with acquisition order and motion data
            self.load_realworld_data(self.filename, slice_idx=self.slice_idx)
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
            self.load_realworld_data_from_ismrm_and_saec(path_to_ismrm, path_to_saec, slice_idx=self.slice_idx)
        else:
            raise ValueError("Unknown data_type")

        self._normalize_kspace_if_enabled()
        
        # Keep a copy of motion-free k-space before any simulated corruption.
        self.kspace_nomotion = self.kspace.clone()

        # Calculate ESPIRiT maps and input image
        self.smaps = self.calc_espirit_maps()
        self.img_cplx = ifftnc(self.kspace, dims=(-3, -2, -1))
        smaps_replicated = self.smaps.unsqueeze(1).expand(-1, self.params.Nex, -1, -1, -1)
        self.image_ground_truth = torch.sum(self.img_cplx*smaps_replicated.conj(), dim=0).to(self.t_device)

        motionSimulator = MotionSimulator(
            self.image_ground_truth,
            self.smaps,
            self.ky_idx,
            self.nex_idx,
            self.ky_per_motion,
            params=self.params,
            sp_device=self.sp_device,
            t_device=self.t_device,
        )
        
        if self.params.motion_simulation_type == 'as-it-is':
            if self.params.data_type in ['real-world', 'raw-data']:
                self.image_no_moco = self.image_ground_truth.clone()
            else:
                raise ValueError("Simulation type 'as-it-is' is only compatible with real-world or raw-data, which already contain motion. Please choose a different simulation type or data type.")
        else:
            if self.params.motion_simulation_type == 'no-motion-data':
                motionSimulator.simulate_no_motion()
                self.image_no_moco = self.image_ground_truth.clone()
            else:      
                if self.params.motion_simulation_type == 'discrete-rigid':
                    motionSimulator.simulate_discrete_rigid_motion()
                elif self.params.motion_simulation_type == 'rigid':
                    motionSimulator.simulate_realistic_rigid_motion()
                elif self.params.motion_simulation_type == 'discrete-non-rigid':
                    motionSimulator.simulate_discrete_non_rigid_motion()
                elif self.params.motion_simulation_type == 'non-rigid':
                    motionSimulator.simulate_realistic_non_rigid_motion()
                else:
                    raise ValueError("Unknown motion_simulation_type")
                self.kspace = motionSimulator.get_corrupted_kspace()
                self.image_no_moco = motionSimulator.get_corrupted_image()
                if hasattr(motionSimulator, "alpha_maps"):
                    self.alpha_maps_true = motionSimulator.alpha_maps

            motion_curve, tx, ty, phi = motionSimulator.get_motion_information()
            y_limits = compute_motion_y_limits(motion_curve, tx=tx, ty=ty, phi=phi)
            (
                self.binned_indices,
                self.motion_signal,
                self.motion_labels,
                self.ky_idx_chronological,
                self.nex_idx_chronological,
            ) = MotionBinner.bin_motion(
                motion_curve,
                self.ky_idx,
                self.nex_idx,
                self.t_device,
                self.params,
                tx=tx,
                ty=ty,
                phi=phi,
                y_limits=y_limits,
                return_debug_data=True,
            )
            self.motion_plot_context = {
                "motion_curve": motion_curve,
                "labels": self.motion_labels,
                "ky_idx": self.ky_idx_chronological,
                "nex_idx": self.nex_idx_chronological,
                "resolution_levels": getattr(self.params, "ResolutionLevels", None),
                "data_type": getattr(self.params, "data_type", None),
                "y_limits": y_limits,
                "alpha_visual_scale": None,
            }
            if (
                self.params.motion_simulation_type in {"discrete-non-rigid", "non-rigid"}
                and hasattr(self, "alpha_maps_true")
                and self.alpha_maps_true is not None
            ):
                alpha = self.alpha_maps_true
                if alpha.ndim == 3 and alpha.shape[0] >= 2:
                    alpha_axis0 = alpha[0].real if torch.is_complex(alpha[0]) else alpha[0]
                    alpha_axis1 = alpha[1].real if torch.is_complex(alpha[1]) else alpha[1]
                    alpha_x, alpha_y = to_cartesian_components(alpha_axis0, alpha_axis1)
                    amp_max = float(torch.max(torch.sqrt(alpha_axis0 * alpha_axis0 + alpha_axis1 * alpha_axis1)).item())
                    alpha_abs_max_x = float(torch.max(torch.abs(alpha_x)).item())
                    alpha_abs_max_y = float(torch.max(torch.abs(alpha_y)).item())
                    self.motion_plot_context["alpha_visual_scale"] = {
                        "alpha_abs_max_x": max(alpha_abs_max_x, 1e-12),
                        "alpha_abs_max_y": max(alpha_abs_max_y, 1e-12),
                        "amp_max": max(amp_max, 1e-12),
                    }
        
        self.sampling_idx = SamplingSimulator.build_sampling_per_nex_per_motion(
            self.binned_indices,  # [Nex][Nmotion]
            self.Nx, self.Ny,
            self.t_device
        )

        self._save_input_data_artifacts()

        if self.params.debug_flag and self.params.motion_simulation_type in ['discrete-non-rigid', 'rigid', 'discrete-rigid']:
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
        folder = getattr(self.params, "input_data_folder", None)
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)

        flip_for_display = getattr(
            self.params, "flip_for_display", self.params.data_type in {"real-world", "raw-data"}
        )

        if (
            self._has_simulated_motion()
            and hasattr(self, "image_ground_truth")
            and self.image_ground_truth is not None
        ):
            show_and_save_image(
                self.image_ground_truth[0],
                "img_ground_truth",
                folder,
                flip_for_display=flip_for_display,
                jupyter_display=False,
            )
        if hasattr(self, "image_no_moco") and self.image_no_moco is not None:
            show_and_save_image(
                self.image_no_moco[0],
                "img_corrupted",
                folder,
                flip_for_display=flip_for_display,
                jupyter_display=False,
            )

        if (
            hasattr(self, "alpha_maps_true")
            and self.alpha_maps_true is not None
            and self.params.motion_simulation_type in {"discrete-non-rigid", "non-rigid"}
        ):
            alpha = self.alpha_maps_true
            if alpha.ndim == 3 and alpha.shape[0] >= 2:
                alpha_axis0 = alpha[0].real if torch.is_complex(alpha[0]) else alpha[0]
                alpha_axis1 = alpha[1].real if torch.is_complex(alpha[1]) else alpha[1]
                alpha_x, alpha_y = to_cartesian_components(alpha_axis0, alpha_axis1)
                scale = (self.motion_plot_context or {}).get("alpha_visual_scale", None)
                alpha_abs_max_x = None if scale is None else scale.get("alpha_abs_max_x")
                alpha_abs_max_y = None if scale is None else scale.get("alpha_abs_max_y")
                amp_max = None if scale is None else scale.get("amp_max")
                save_alpha_component_map(
                    alpha_x,
                    "simulated_alpha_x_input",
                    os.path.join(folder, "simulated_alpha_x_input.png"),
                    flip_vertical=flip_for_display,
                    abs_max=alpha_abs_max_x,
                )
                save_alpha_component_map(
                    alpha_y,
                    "simulated_alpha_y_input",
                    os.path.join(folder, "simulated_alpha_y_input.png"),
                    flip_vertical=flip_for_display,
                    abs_max=alpha_abs_max_y,
                )
                save_nonrigid_quiver_with_contours(
                    alpha_axis0,
                    alpha_axis1,
                    self.image_ground_truth[0],
                    "simulated_motion_quiver_input",
                    os.path.join(folder, "simulated_motion_quiver_input.png"),
                    flip_vertical=flip_for_display,
                    amp_vmax=amp_max,
                )

    def _has_simulated_motion(self):
        return self.params.motion_simulation_type in {
            "rigid",
            "non-rigid",
            "discrete-rigid",
            "discrete-non-rigid",
        }


    # Ncoils should be a perfect square (or close) for coil map generation
    def generate_shepp_logan(self, N=128, Ncoils=4, Nz=1, random_phase=True):
        # 1. Create phantom (Nx, Ny, Nz)
        fill_fraction = float(getattr(self.params, "SheppLoganFillFraction", 0.82))
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
        phantom_2d = torch.tensor(
            phantom_np_2d,
            dtype=torch.float64,
            device=self.t_device
        )

        phantom = phantom_2d.unsqueeze(-1).expand(N, N, Nz)  # (Nx, Ny, Nz)

        # 2. Create coil sensitivity maps (Ncoils, Nx, Ny, Nz)
        X, Y = torch.meshgrid(
            torch.arange(1, N + 1, device=self.t_device),
            torch.arange(1, N + 1, device=self.t_device),
            indexing="ij"
        )

        sigma = N / 4

        # Coil centers on (approximately) square grid
        grid_size = math.ceil(math.sqrt(Ncoils))
        xs = torch.linspace(N / 4, 3 * N / 4, grid_size, device=self.t_device)
        ys = torch.linspace(N / 4, 3 * N / 4, grid_size, device=self.t_device)
        centers = [(x.item(), y.item()) for y in ys for x in xs][:Ncoils]

        # Allocate coil maps directly in (Ncoils, Nx, Ny)
        smaps_2d = torch.empty(
            (Ncoils, N, N),
            dtype=torch.cdouble,
            device=self.t_device
        )

        for c, (x0, y0) in enumerate(centers):
            profile = torch.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
            phase = (
                torch.exp(1j * 2 * math.pi * torch.rand(1, device=self.t_device))
                if random_phase else 1.0
            )
            smaps_2d[c] = profile * phase

        # Expand to Nz: (Ncoils, Nx, Ny, Nz)
        smaps = smaps_2d.unsqueeze(-1).expand(Ncoils, N, N, Nz)
        self.smaps_generated = smaps.clone()

        # 3. Generate coil images directly in (Ncoils, Nx, Ny, Nz)
        phantom = phantom.unsqueeze(0)          # (1, Nx, Ny, Nz)
        coil_imgs = phantom * smaps             # (Ncoils, Nx, Ny, Nz)
        coil_imgs = coil_imgs.unsqueeze(1).expand(-1, self.params.Nex, -1, -1, -1) # add Nex dimension: (Ncoils, Nex, Nx, Ny, Nz)

        coil_imgs = coil_imgs.contiguous()       # important for FFT speed

        # 4. FFT → k-space (Ncoils, Nx, Ny, Nz)
        self.kspace = fftnc(coil_imgs, dims=(-3, -2, -1))
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape

        # 5. Sampling
        samplingSimulator = SamplingSimulator(self.Ny, self.params, self.t_device)
        self.ky_idx, self.nex_idx, self.ky_per_motion = samplingSimulator.build_ky_and_nex()

    def load_realworld_data_from_ismrm_and_saec(self, path_to_ismrm, path_to_saec, slice_idx=0):
        reader = RawDataReader(
            ismrmrd_file=path_to_ismrm,
            saec_file=path_to_saec,
            sensor_type="BELT",
            device="cuda"
        )
        data = reader.read_data_from_rawdata()

        self.kspace = torch.from_numpy(data['kspace']).to(self.t_device, dtype=torch.cdouble)[:, :, :, :, [slice_idx]]
        self.params.Nex = int(self.kspace.shape[1])
        self.params.NshotsPerNex = int(self.kspace.shape[3])
        self.params.Nshots = int(self.params.Nex) * int(self.params.NshotsPerNex)
        if self.params.motion_simulation_type in ["discrete-rigid", "discrete-non-rigid"]:
            self.params.N_motion_states = self.params.Nshots
        self.ky_idx = torch.from_numpy(data['idx_ky'][slice_idx]).to(self.t_device, dtype=torch.int64)
        self.nex_idx = torch.zeros_like(self.ky_idx, device=self.t_device)
        motion_data = data['motion_data'][slice_idx, :]
        motion_data = torch.from_numpy(motion_data).to(self.t_device)
        y_limits = compute_motion_y_limits(motion_data)
        (
            self.ky_per_motion,
            self.motion_signal,
            self.motion_labels,
            self.ky_idx_chronological,
            self.nex_idx_chronological,
        ) = MotionBinner.bin_motion(
            motion_data, self.ky_idx, self.nex_idx, self.t_device, self.params
            , y_limits=y_limits
            , return_debug_data=True
        )
        self.motion_plot_context = {
            "motion_curve": motion_data,
            "labels": self.motion_labels,
            "ky_idx": self.ky_idx_chronological,
            "nex_idx": self.nex_idx_chronological,
            "resolution_levels": getattr(self.params, "ResolutionLevels", None),
            "data_type": getattr(self.params, "data_type", None),
            "y_limits": y_limits,
        }
        self.binned_indices = self.ky_per_motion
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape

        SamplingSimulator.visualize_ky_order(
            [self.ky_idx.detach().cpu()],
            Ny=self.Ny,
            folder=getattr(self.params, "input_data_folder", self.params.debug_folder),
            fname=f"ky_order_rawdata_slice{slice_idx}.png"
        )
        

    def load_realworld_data(self, path_to_data, slice_idx=0):
        data = {}
        with h5py.File(path_to_data, 'r') as f:
            data['motion_data'] = f['motion_data'][:]
            data['idx_ky'] = f['idx_ky'][:]
            data['idx_kz'] = f['idx_kz'][:]
            data['idx_nex'] = f['idx_nex'][:]
            data['kspace'] = f['kspace'][:]

        self.kspace = torch.from_numpy(data['kspace']).to(self.t_device, dtype=torch.cdouble)[:, :, :, :, [slice_idx]]
        self.params.Nex = int(self.kspace.shape[1])
        self.params.NshotsPerNex = int(self.kspace.shape[3])
        self.params.Nshots = int(self.params.Nex) * int(self.params.NshotsPerNex)
        if self.params.motion_simulation_type in ["discrete-rigid", "discrete-non-rigid"]:
            self.params.N_motion_states = self.params.Nshots
        ky_dx = data['idx_ky'][slice_idx]
        self.ky_idx = torch.from_numpy(ky_dx).to(self.t_device, dtype=torch.int64)
        self.nex_idx = torch.zeros_like(self.ky_idx, device=self.t_device) # TODO Add multiple Nex
        motion_data = data['motion_data'][slice_idx, :]
        motion_data = torch.from_numpy(motion_data).to(self.t_device)
        y_limits = compute_motion_y_limits(motion_data)
        (
            self.ky_per_motion,
            self.motion_signal,
            self.motion_labels,
            self.ky_idx_chronological,
            self.nex_idx_chronological,
        ) = MotionBinner.bin_motion(
            motion_data, self.ky_idx, self.nex_idx, self.t_device, self.params
            , y_limits=y_limits
            , return_debug_data=True
        )
        self.motion_plot_context = {
            "motion_curve": motion_data,
            "labels": self.motion_labels,
            "ky_idx": self.ky_idx_chronological,
            "nex_idx": self.nex_idx_chronological,
            "resolution_levels": getattr(self.params, "ResolutionLevels", None),
            "data_type": getattr(self.params, "data_type", None),
            "y_limits": y_limits,
        }
        self.binned_indices = self.ky_per_motion
        self.Ncha, _, self.Nx, self.Ny, self.Nsli = self.kspace.shape

        SamplingSimulator.visualize_ky_order(
            [self.ky_idx.detach().cpu()],
            Ny=self.Ny,
            folder=getattr(self.params, "input_data_folder", self.params.debug_folder),
            fname=f"ky_order_realworld_slice{slice_idx}.png"
        )

    def calc_espirit_maps(self):
        acs=self.params.acs
        kernel_width=self.params.kernel_width
        espirit_max_iter = getattr(self.params, "espirit_max_iter", 100)
        sp_device=self.sp_device
        kspace = self.kspace
        device = kspace.device             # torch device of input
        
        use_gpu = device.type == "cuda"

        if sp_device is None:
            sp_device = sp.Device(0 if use_gpu else -1)

        nCha, _, nX, nY, nSlices = kspace.shape

        espirit_maps = torch.zeros(
            (nCha, nX, nY, nSlices),
            dtype=torch.complex128,
            device=device
        )

        for i in range(nSlices):

            # ---- GPU path ----
            if use_gpu:
                import cupy as cp
                kspace_cp = cp.asarray(kspace[:, 0, :, :, i].contiguous())
                maps_cp = spmri.app.EspiritCalib(
                    kspace_cp, calib_width=acs,
                    kernel_width=kernel_width,
                    max_iter=espirit_max_iter,
                    device=sp_device
                ).run()
                maps_cp = maps_cp.astype(cp.complex128, copy=False)
                maps_cp = cp.ascontiguousarray(maps_cp)
                maps_t = torch.view_as_real(torch.utils.dlpack.from_dlpack(maps_cp))

            # ---- CPU path ----
            else:
                kspace_np = kspace[:, 0, :, :, i].cpu().numpy()
                maps_np = spmri.app.EspiritCalib(
                    kspace_np, calib_width=acs,
                    kernel_width=kernel_width,
                    max_iter=espirit_max_iter,
                    device=sp_device
                ).run()
                maps_np = maps_np.astype(np.complex128, copy=False)
                maps_t = torch.from_numpy(np.stack([maps_np.real, maps_np.imag], axis=-1))

            espiritual = torch.complex(maps_t[..., 0], maps_t[..., 1])
            espirit_maps[:, :, :, i] = espiritual

            if use_gpu:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                torch.cuda.empty_cache()

        return espirit_maps

    def _debug_check_true_motion_image_reconstruction(self, motionSimulator):
        # This consistency check is meaningful only for simulated non-rigid data.
        if self.params.motion_type != "non-rigid":
            return
        if self.params.motion_simulation_type not in {"discrete-non-rigid", "non-rigid"}:
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
            nsamples_true = self.Nx * self.Ny

            motion_op_true = MotionOperator(
                self.Nx,
                self.Ny,
                alpha_true,
                self.params.motion_type,
                motion_signal=signal_true,
            )

            encoding_true = EncodingOperator(
                self.smaps,
                nsamples_true,
                sampling_true,
                self.params.Nex,
                motion_op_true,
            )

            kspace_vec = self.kspace[..., 0].reshape(self.Ncha, self.params.Nex, nsamples_true).flatten()
            b = encoding_true.adjoint(kspace_vec)
            x0 = torch.zeros_like(b)
            solver = ConjugateGradientSolver(
                encoding_true,
                reg_lambda=0.0,
                verbose=False,
                early_stopping=self.params.cg_early_stopping,
                max_stag_steps=self.params.cg_max_stag_steps,
                max_more_steps=self.params.cg_max_more_steps,
                use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
                reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
            )
            img_vec = solver.solve_cg(b.flatten(), x0=x0.flatten(), max_iter=80, tol=1e-6)
            img_back = img_vec.reshape(self.params.Nex, self.Nx, self.Ny)
            img_ref = self.image_ground_truth[..., 0]

            num = torch.linalg.norm((img_back - img_ref).flatten())
            den = torch.linalg.norm(img_ref.flatten()) + 1e-12
            rel_err = (num / den).item()

            print(f"[DEBUG] GN-input consistency check (true alpha, clustered signal/sampling): rel_err={rel_err:.6e}")
            show_and_save_image(
                img_back[0],
                "gn_input_consistency_recovered_image",
                self.params.debug_folder,
                flip_for_display=getattr(
                    self.params,
                    "flip_for_display",
                    self.params.data_type in {"real-world", "raw-data"},
                ),
            )






        

    





    
    
        

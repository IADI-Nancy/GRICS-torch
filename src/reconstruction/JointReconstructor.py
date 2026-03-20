import torch
import torch.nn.functional as F
import time
from contextlib import nullcontext
from tqdm.auto import tqdm

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.utils.plotting import show_and_save_image
from src.utils.save_final_motion_plots import save_final_nonrigid_alpha_maps, save_final_rigid_motion_plots
from src.utils.joint_reconstructor_utils import (
    _assign_cached_reg_scale,
    _append_run_log,
    _console,
    _format_cg_info,
    _init_run_logging,
    _initialize_level_tracking,
    _parse_gn_iterations_per_level,
    _save_nonrigid_motion_debug,
    _save_run_residual_plots,
)

# --------------------------------------------------------------------------
# Class that performs joint image–motion reconstruction
# --------------------------------------------------------------------------
class JointReconstructor:

    def __init__(self, KspaceData, smaps, SamplingIndices, motion_signal, params, kspace_scale=1.0, motion_plot_context=None):
        Ncoils, Nx_full, Ny_full, Nz_full = smaps.shape

        # Parameters constant for all resolutions        
        self.params = params
        self.Ncoils = Ncoils
        self.Nz_full = int(Nz_full)
        self.device = KspaceData.device
        if self.params.reconstruction_motion_type == "rigid":
            self.Nalpha = 6 if self.Nz_full > 1 else 3
        else:
            self.Nalpha = 3 if self.Nz_full > 1 else 2
        self.kspace_scale = float(kspace_scale)
        if motion_signal is None:
            raise ValueError("motion_signal must be provided.")
        self.motion_signal = motion_signal.to(self.device)
        self.motion_plot_context = motion_plot_context or {}
        self._last_image_cg_info = None
        self._last_motion_cg_info = None
        self._current_level_idx = 0

        # Data changing with resolution
        self.Data_full = {}
        self.Data_full["Nx"] = Nx_full
        self.Data_full["Ny"] = Ny_full
        self.Data_full["Nz"] = self.Nz_full
        self.Data_full["SensitivityMaps"] = smaps
        self.Data_full["KspaceData"] = KspaceData
        self.Data_full["Nsamples"] = sum(
            SamplingIndices[0][ms].numel()
            for ms in range(len(SamplingIndices[0]))
        )
        self.Data_full["SamplingIndices"] = SamplingIndices

    def _resize_img_xy(self, img, new_size):
        is_complex = img.is_complex()
        target_3d = len(new_size) == 3

        # ---------- Helper: interpolate real/imag ----------
        def interp_part(x):
            """Interpolate real-valued tensor in 2D or 3D spatial coordinates."""
            if target_3d:
                nx_new, ny_new, nz_new = new_size
                if x.ndim == 3:
                    # [Nx, Ny, Nz] -> [1, 1, Nz, Nx, Ny]
                    xv = x.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                    out = F.interpolate(xv, size=(nz_new, nx_new, ny_new), mode="trilinear", align_corners=False)
                    return out[0, 0].permute(1, 2, 0)  # [Nx, Ny, Nz]
                elif x.ndim == 4:
                    # [C, Nx, Ny, Nz] -> [1, C, Nz, Nx, Ny]
                    xv = x.permute(0, 3, 1, 2).unsqueeze(0)
                    out = F.interpolate(xv, size=(nz_new, nx_new, ny_new), mode="trilinear", align_corners=False)
                    return out[0].permute(0, 2, 3, 1)  # [C, Nx, Ny, Nz]
                else:
                    raise ValueError(f"Unexpected shape {x.shape} for 3D resize.")

            if x.ndim == 2:
                x = x.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
                out = F.interpolate(x, size=new_size, mode="bilinear", align_corners=False)
                return out[0, 0]

            elif x.ndim == 3:
                C = x.shape[0]
                out_list = []
                for c in range(C):
                    xc = x[c].unsqueeze(0).unsqueeze(0)
                    rc = F.interpolate(xc, size=new_size, mode="bilinear", align_corners=False)
                    out_list.append(rc[0, 0])
                return torch.stack(out_list, dim=0)
            elif x.ndim == 4 and x.shape[-1] == 1:
                # 2D-with-single-z convention: [C, Nx, Ny, 1] -> [C, Nx_new, Ny_new, 1]
                C = x.shape[0]
                out_list = []
                for c in range(C):
                    xc = x[c, :, :, 0].unsqueeze(0).unsqueeze(0)
                    rc = F.interpolate(xc, size=new_size, mode="bilinear", align_corners=False)
                    out_list.append(rc[0, 0])
                return torch.stack(out_list, dim=0).unsqueeze(-1)
            else:
                raise ValueError(f"Unexpected shape {x.shape}")

        # ---------- Real tensor case ----------
        if not is_complex:
            return interp_part(img)

        # ---------- Complex case ----------
        real = interp_part(img.real)
        imag = interp_part(img.imag)
        return torch.complex(real, imag)

    def _downsample_sampling_indices(self, Sampling_full, Nx_res, Ny_res, Nz_res=1):
        Nx_full, Ny_full = self.Data_full["Nx"], self.Data_full["Ny"]
        Nz_full = int(self.Data_full.get("Nz", 1))

        # central crop coordinates
        x0 = (Nx_full - Nx_res) // 2
        y0 = (Ny_full - Ny_res) // 2
        z0 = (Nz_full - Nz_res) // 2

        Sampling_res = []

        for nex in range(self.params.Nex):
            Sampling_res.append([])
            for indices in Sampling_full[nex]:
                if Nz_full > 1:
                    # Decode flattened 3D index: idx = ((x * Ny) + y) * Nz + z
                    z = indices % Nz_full
                    xy = indices // Nz_full
                    x = xy // Ny_full
                    y = xy % Ny_full
                else:
                    # compute x,y coordinates for 2D flattening
                    x = indices // Ny_full
                    y = indices % Ny_full

                # mask inside central region
                if Nz_full > 1:
                    mask = (
                        (x >= x0) & (x < x0 + Nx_res)
                        & (y >= y0) & (y < y0 + Ny_res)
                        & (z >= z0) & (z < z0 + Nz_res)
                    )
                else:
                    mask = (x >= x0) & (x < x0 + Nx_res) & (y >= y0) & (y < y0 + Ny_res)

                # keep only those indices
                x_crop = x[mask] - x0
                y_crop = y[mask] - y0

                if Nz_full > 1:
                    z_keep = z[mask] - z0
                    # re-flatten for Nx_res × Ny_res × Nz_full grid
                    new_inds = (x_crop * Ny_res + y_crop) * Nz_res + z_keep
                else:
                    # re-flatten for Nx_res × Ny_res grid
                    new_inds = x_crop * Ny_res + y_crop

                Sampling_res[nex].append(new_inds)

        return Sampling_res

    def _downsample_kspace(self, Nx_res, Ny_res, Nz_res=1):
        Nx_full, Ny_full = self.Data_full["Nx"], self.Data_full["Ny"]
        Nz_full = int(self.Data_full.get("Nz", 1))
        kspace_full = self.Data_full["KspaceData"]

        # central crop coordinates
        x0 = (Nx_full - Nx_res) // 2
        y0 = (Ny_full - Ny_res) // 2
        z0 = (Nz_full - Nz_res) // 2

        if Nz_full > 1:
            kspace_res = kspace_full[:, :, x0:x0 + Nx_res, y0:y0 + Ny_res, z0:z0 + Nz_res]
        else:
            kspace_res = kspace_full[:, :, x0:x0 + Nx_res, y0:y0 + Ny_res, :]
        kspace_res = kspace_res.reshape(kspace_full.shape[0], kspace_full.shape[1], -1)

        return kspace_res   

    def _downsample_data(self, res_factor):    
        Nx = int(round(self.Data_full["Nx"] * res_factor))
        Ny = int(round(self.Data_full["Ny"] * res_factor))
        Nz_full = int(self.Data_full.get("Nz", 1))
        Nz = int(round(Nz_full * res_factor)) if Nz_full > 1 else 1
        Nz = max(Nz, 1)

        Data_res = {}
        Data_res["Nx"] = Nx
        Data_res["Ny"] = Ny
        Data_res["Nz"] = Nz

        resize_shape = (Nx, Ny, Nz) if Nz > 1 else (Nx, Ny)
        Data_res["SensitivityMaps"] = self._resize_img_xy(self.Data_full["SensitivityMaps"], resize_shape)
        Data_res["SamplingIndices"] = self._downsample_sampling_indices(self.Data_full["SamplingIndices"], Nx, Ny, Nz_res=Nz)
        Data_res["KspaceData"] = self._downsample_kspace(Nx, Ny, Nz_res=Nz)
        Data_res["Nsamples"] = Data_res["KspaceData"].shape[2]

        return Data_res
    
    def _upsample_data(self, Data_prev, Data_res):
        img_prev = Data_prev["ReconstructedImage"]
        resize_shape = (
            (Data_res["Nx"], Data_res["Ny"], Data_res["Nz"])
            if int(Data_res.get("Nz", 1)) > 1 else
            (Data_res["Nx"], Data_res["Ny"])
        )
        img_res = self._resize_img_xy(img_prev, resize_shape)
        Data_res["ReconstructedImage"] = img_res

        mot_prev = Data_prev["MotionModel"]
        if self.params.reconstruction_motion_type == "rigid":
            Data_res["MotionModel"] = torch.zeros((self.Nalpha, self.params.N_motion_states), device=self.device)
            Data_res["MotionModel"][0,:] = mot_prev[0,:] * Data_res["Nx"] / Data_prev["Nx"]  # scale translations
            Data_res["MotionModel"][1,:] = mot_prev[1,:] * Data_res["Ny"] / Data_prev["Ny"]  # scale translations
            if self.Nalpha > 3:
                Data_res["MotionModel"][2,:] = mot_prev[2,:] * Data_res.get("Nz", 1) / max(1, Data_prev.get("Nz", 1))
                Data_res["MotionModel"][3:,:] = mot_prev[3:,:]
            else:
                Data_res["MotionModel"][2,:] = mot_prev[2,:]  # rotations remain the same
        else:
            resize_shape = (
                (Data_res["Nx"], Data_res["Ny"], Data_res["Nz"])
                if int(Data_res.get("Nz", 1)) > 1 else
                (Data_res["Nx"], Data_res["Ny"])
            )
            mot_res = self._resize_img_xy(mot_prev, resize_shape)
            mot_res[0] = mot_res[0] * Data_res["Nx"] / Data_prev["Nx"]
            mot_res[1] = mot_res[1] * Data_res["Ny"] / Data_prev["Ny"]
            if mot_res.shape[0] > 2 and int(Data_res.get("Nz", 1)) > 1:
                mot_res[2] = mot_res[2] * Data_res["Nz"] / max(1, Data_prev["Nz"])
            Data_res["MotionModel"] = mot_res

    def _build_motion_operator(self, Data_res):
        Nx, Ny = Data_res["Nx"], Data_res["Ny"]
        alpha = Data_res["MotionModel"]
        if self.params.reconstruction_motion_type == "rigid":
            motionOperator = MotionOperator(
                Nx, Ny, alpha, self.params.reconstruction_motion_type, Nz=Data_res.get("Nz", 1)
            )
        else:
            motion_signal = self.motion_signal
            motionOperator = MotionOperator(
                Nx, Ny, alpha, self.params.reconstruction_motion_type,
                motion_signal=motion_signal.to(dtype=alpha.dtype), Nz=Data_res.get("Nz", 1)
            )
        return motionOperator

    def _build_encoding_operator(self, Data_res):
        E = EncodingOperator(Data_res["SensitivityMaps"], Data_res["Nsamples"], Data_res["SamplingIndices"],
                             self.params.Nex, Data_res["MotionOperator"])
        return E
    
    def _build_motion_perturbation_simulator(self, Data_res):
        J = MotionPerturbationSimulator(Data_res["SensitivityMaps"], Data_res["Nsamples"], Data_res["SamplingIndices"],
                                        self.params.Nex, Data_res["ReconstructedImage"], Data_res["MotionOperator"])
        return J

    def _lambda_r_for_level(self):
        lambda_r = self.params.lambda_r
        if isinstance(lambda_r, (list, tuple)):
            if len(lambda_r) == 0:
                raise ValueError("lambda_r list/tuple cannot be empty.")
            if len(lambda_r) != len(self.params.ResolutionLevels):
                raise ValueError(
                    "Inconsistent config: "
                    f"lambda_r has {len(lambda_r)} values, "
                    f"but ResolutionLevels has {len(self.params.ResolutionLevels)} values."
                )
            return float(lambda_r[self._current_level_idx])
        return float(lambda_r)

    def _solve_image(self, Data_res):
        x0 = Data_res["ReconstructedImage"]
        x0 = x0.to(self.device)
        E = Data_res["E"]

        b = E.adjoint(Data_res["KspaceData"])
        solver = ConjugateGradientSolver(
            E, reg_lambda=self._lambda_r_for_level(), verbose=self.params.verbose, early_stopping=self.params.cg_early_stopping,
            true_residual_interval=self.params.cg_true_residual_interval, max_stag_steps=self.params.cg_max_stag_steps,
            max_more_steps=self.params.cg_max_more_steps, use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
            reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
        )
        _assign_cached_reg_scale(self.params, Data_res, "image", solver, b.flatten())

        img_vec = solver.cg(b.flatten(), x0=x0.flatten(), max_iter=self.params.max_iter_recon, tol=self.params.tol_recon)
        self._last_image_cg_info = solver.last_info

        if int(Data_res.get("Nz", 1)) > 1:
            img = img_vec.reshape(self.params.Nex, Data_res["Nx"], Data_res["Ny"], Data_res["Nz"])
        else:
            img = img_vec.reshape(self.params.Nex, Data_res["Nx"], Data_res["Ny"])
        return img

    def _n_motion_params(self, Data_res):
        if self.params.reconstruction_motion_type == "rigid":
            return self.Nalpha * self.params.N_motion_states
        return self.Nalpha * Data_res["Nx"] * Data_res["Ny"] * int(Data_res.get("Nz", 1))

    def _solve_motion(self, Data_res, residual):
        Nparams = self._n_motion_params(Data_res)
        J = Data_res["J"]
        b_data = J.adjoint(residual)
        x0 = torch.zeros(Nparams, dtype=b_data.dtype, device=residual.device)

        if self.params.reconstruction_motion_type == "non-rigid":
            reg_shape = (
                (self.Nalpha, Data_res["Nx"], Data_res["Ny"], int(Data_res.get("Nz", 1)))
                if int(Data_res.get("Nz", 1)) > 1
                else (self.Nalpha, Data_res["Nx"], Data_res["Ny"])
            )
            solver = ConjugateGradientSolver(
                J, reg_lambda=self.params.lambda_m, regularizer="Tikhonov_gradient",
                regularization_shape=reg_shape, verbose=self.params.verbose,
                early_stopping=self.params.cg_early_stopping, true_residual_interval=self.params.cg_true_residual_interval,
                max_stag_steps=self.params.cg_max_stag_steps, max_more_steps=self.params.cg_max_more_steps,
                use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy, reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
            )
            # Unscaled _regularization:
            # _A(dm) = J^H J dm + mu * GhG(dm)
            # b     = J^H r    - mu * GhG(alpha_current)
            _assign_cached_reg_scale(self.params, Data_res, "motion_nonrigid", solver, b_data.flatten())
            b = b_data - solver._effective_lambda() * solver._regularization(Data_res["MotionModel"].flatten())
            mot_pert_vec = solver.cg(b.flatten(), x0=x0.flatten(), max_iter=self.params.max_iter_motion, tol=self.params.tol_motion)
        else:
            solver = ConjugateGradientSolver(
                J, reg_lambda=self.params.lambda_m, verbose=self.params.verbose, early_stopping=self.params.cg_early_stopping,
                true_residual_interval=self.params.cg_true_residual_interval, max_stag_steps=self.params.cg_max_stag_steps,
                max_more_steps=self.params.cg_max_more_steps, use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
                reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
            )
            _assign_cached_reg_scale(self.params, Data_res, "motion_rigid", solver, b_data.flatten())
            mot_pert_vec = solver.cg(b_data.flatten(), x0=x0.flatten(), max_iter=self.params.max_iter_motion, tol=self.params.tol_motion)
        self._last_motion_cg_info = solver.last_info

        if self.params.reconstruction_motion_type == "rigid":
            motion_perturb = mot_pert_vec.reshape(self.Nalpha, self.params.N_motion_states)
        else:
            if int(Data_res.get("Nz", 1)) > 1:
                motion_perturb = mot_pert_vec.reshape(self.Nalpha, Data_res["Nx"], Data_res["Ny"], Data_res["Nz"])
            else:
                motion_perturb = mot_pert_vec.reshape(self.Nalpha, Data_res["Nx"], Data_res["Ny"])
        return motion_perturb
    
    def _prepare_resolution_level(self, idx_res, r):
        _console(self.params, f"\n=== Resolution level {idx_res+1}: factor {r} ===")

        # Prepare low-resolution dataset
        Data_res = self._downsample_data(r)

        # Initialize image and motion model
        if idx_res == 0:
            if int(Data_res.get("Nz", 1)) > 1:
                Data_res["ReconstructedImage"] = torch.zeros(
                    (self.params.Nex, Data_res["Nx"], Data_res["Ny"], Data_res["Nz"]),
                    dtype=torch.complex128, device=self.device
                )
            else:
                Data_res["ReconstructedImage"] = torch.zeros((self.params.Nex, Data_res["Nx"], Data_res["Ny"]), dtype=torch.complex128, device=self.device)
            
            if self.params.reconstruction_motion_type == "rigid":
                Data_res["MotionModel"] = torch.zeros((self.Nalpha, self.params.N_motion_states), device=self.device)
            elif self.params.reconstruction_motion_type == "non-rigid":
                if int(Data_res.get("Nz", 1)) > 1:
                    Data_res["MotionModel"] = torch.zeros(
                        (self.Nalpha, Data_res["Nx"], Data_res["Ny"], Data_res["Nz"]), device=self.device
                    )
                else:
                    Data_res["MotionModel"] = torch.zeros((self.Nalpha, Data_res["Nx"], Data_res["Ny"]), device=self.device)
        return Data_res

    @staticmethod
    def _strip_level_runtime_state(data):
        if data is None:
            return None
        for key in ("MotionOperator", "E", "J"):
            data.pop(key, None)
        return data

    @staticmethod
    def _make_next_level_initializer(data):
        if data is None:
            return None
        return {
            "Nx": data["Nx"],
            "Ny": data["Ny"],
            "Nz": data.get("Nz", 1),
            "ReconstructedImage": data["ReconstructedImage"],
            "MotionModel": data["MotionModel"],
        }

    # ----------------------------------------------------------------------
    # Perform full multi-resolution Gauss–Newton joint reconstruction
    # ----------------------------------------------------------------------
    def run(self):
        # Initialize multi-resolution schedule and per-level GN iteration counts.
        ResLevels = self.params.ResolutionLevels
        gn_iters_per_level = _parse_gn_iterations_per_level(self.params, ResLevels)
        run_log = _init_run_logging(self.params, len(ResLevels), gn_iters_per_level)
        run_t0 = time.perf_counter() # total run timer
        best_image = None
        best_motion = None

        # Loop over each configured resolution level.
        for idx_res, r in enumerate(ResLevels):
            self._current_level_idx = idx_res
            GN_iter = gn_iters_per_level[idx_res]
            level_t0 = time.perf_counter()  # Level timer.
            # Build level-specific data
            Data_res = self._prepare_resolution_level(idx_res, r)
            if idx_res != 0:
                self._upsample_data(Data_prev, Data_res)
            level_init_time = time.perf_counter() - level_t0 # Compute setup time for this level.
            s_res = Data_res["KspaceData"].flatten()
            s_res_norm = torch.linalg.norm(s_res).item()

            # Log level header and initialization timing.
            _append_run_log(
                run_log,
                (
                    f"Resolution level {idx_res} ({Data_res['Nx']}x{Data_res['Ny']}x{Data_res.get('Nz', 1)}, "
                    f"{Data_res['Ny']} views, {self.params.N_motion_states} virtual times)\n"
                    f"    lambda_r : {self._lambda_r_for_level():.6e}\n"
                    f"    Resolution level initializations : {level_init_time:.6f} s\n"
                ),
            )

            # Initialize per-level residual tracking and best-at-level image and motion snapshots.
            (residual_recon_norms, residual_motion_norms, best_relres, best_image, best_motion) = _initialize_level_tracking()
            
            # Enable and display progress-bars (only for notebook mode).
            show_bar = self.params.jupyter_notebook_flag
            bar_ctx = tqdm(total=GN_iter, desc=f"Resolution level {idx_res + 1}/{len(ResLevels)}", \
                disable=(not show_bar), leave=True, dynamic_ncols=True, position=0) if GN_iter > 0 else nullcontext()

            # Gauss–Newton iterations
            with bar_ctx as pbar: # Enter progress-bar context
                for it in range(GN_iter):
                    _console(self.params, f"  GN iteration {it+1}/{GN_iter}")                    
                    fp_t0 = time.perf_counter() # Fixed-point iteration timer.

                    # -------------------------------IMAGE RECONSTRUCTION STEP -------------------------

                    # 1) Build motion and encoding operators
                    Data_res["MotionOperator"] = self._build_motion_operator(Data_res)
                    Data_res["E"] = self._build_encoding_operator(Data_res)

                    # 2) Solve for image
                    _console(self.params, "    Solving for image...")
                    t_img = time.perf_counter() # Image-solve timer.
                    img = self._solve_image(Data_res)
                    img_elapsed = time.perf_counter() - t_img
                    Data_res["ReconstructedImage"] = img
                    _append_run_log(
                        run_log,
                        f"    Reconstruction step : {_format_cg_info(self._last_image_cg_info)}, elapsed time = {img_elapsed:.6f} s",
                    )

                    # 3) Compute residual
                    x = img.flatten()
                    y = Data_res["E"].forward(x) # Predict k-space from current image and motion estimate.
                    residual = s_res - y # Compute data residual in k-space.
                    res_norm = torch.linalg.norm(residual).item() # Residual L2 norm.
                    rel_res = res_norm / (s_res_norm + 1e-12) # Relative residual
                    residual_recon_norms.append(rel_res)
                    if pbar is not None:
                        pbar.set_postfix(recon=f"{rel_res:.2e}") # Show residual value in progress bar.

                    # Early-stop current level if residual worsens after at least one update.
                    if it > 0 and rel_res > best_relres:
                        _console(self.params, "    Relative residual increased — restoring best solution at this level.")
                        _append_run_log(run_log, \
                            "    Relative residual increased - restoring best solution at this level.",)
                        if pbar is not None:
                            pbar.update(1) # Advance progress bar for this iteration before exiting.
                        del x
                        del y
                        del residual
                        break

                    # Residual improved: store current image/motion as level-best.
                    best_relres = rel_res
                    best_image = Data_res["ReconstructedImage"].clone()
                    best_motion = Data_res["MotionModel"].clone()

                    # Final resolution level acts as an image-only polish stage.
                    if idx_res == len(ResLevels) - 1:
                        fp_elapsed = time.perf_counter() - fp_t0
                        _append_run_log(
                            run_log,
                            (
                                f"    Fixed point iter {it}: "
                                f"recon_rel_residual = {rel_res:.6e}, "
                                f"image_only = True : {fp_elapsed:.6f} s\n"
                            ),
                        )
                        Data_res.pop("E", None)
                        Data_res.pop("MotionOperator", None)
                        if pbar is not None:
                            pbar.update(1)
                        del x
                        del y
                        del residual
                        continue

                    # ------------------------------- MOTION MODEL RECONSTRUCTION STEP -------------------------

                    # 4) Build linearized motion-perturbation simulator around current estimate ∇_u(E)·δu = δkspace
                    Data_res["J"] = self._build_motion_perturbation_simulator(Data_res)

                    # 5) Solve for motion update
                    _console(self.params, "    Solving for motion update...")
                    t_mot = time.perf_counter() # Motion-solve timer.
                    motion_update = self._solve_motion(Data_res, residual)
                    mot_elapsed = time.perf_counter() - t_mot
                    Data_res["MotionModel"] += motion_update.real
                    Data_res.pop("J", None)
                    Data_res.pop("E", None)
                    Data_res.pop("MotionOperator", None)

                    # 6) Compute and log motion update norms.
                    motion_update_norm = torch.linalg.norm(motion_update.flatten()).item() # Motion update L2 norm.
                    alpha_norm = torch.linalg.norm(Data_res["MotionModel"].flatten()).item()
                    motion_update_rel_norm = motion_update_norm / (alpha_norm + 1e-12) # Relative motion update norm
                    residual_motion_norms.append(motion_update_rel_norm)
                    fp_elapsed = time.perf_counter() - fp_t0
                    _append_run_log(run_log, \
                        (
                            f"    Model optimization step: {_format_cg_info(self._last_motion_cg_info)}, "
                            f"elapsed time = {mot_elapsed:.6f} s\n"
                            f"    Fixed point iter {it}: "
                            f"recon_rel_residual = {rel_res:.6e}, "
                            f"motion_rel_residual = {motion_update_rel_norm:.6e}, "
                            f"motion_norm = {motion_update_norm:.6e} : {fp_elapsed:.6f} s\n"
                        ),)
                    
                    # Advance progress bar after completing both image and motion steps.
                    if pbar is not None:
                        pbar.update(1)

                    del x
                    del y
                    del residual
                    del motion_update

            # ------------------------------- SAVE DEBUG OUTPUTS AND LOGS -------------------------
            if self.params.debug_flag:
                # Restore level-best solution before saving debug images.
                if best_image is not None and best_motion is not None:
                    Data_res["ReconstructedImage"] = best_image
                    Data_res["MotionModel"] = best_motion
                # Save reconstructed image at current level.
                show_and_save_image(
                    Data_res["ReconstructedImage"][0],
                    f"image_resolution_level{idx_res + 1}",
                    self.params.debug_folder,
                    flip_for_display=self.params.flip_for_display,
                )
                # Save non-rigid motion debug plots when applicable.
                _save_nonrigid_motion_debug(
                    Data_res,
                    idx_res + 1,
                    self.params.reconstruction_motion_type,
                    self.params.debug_folder,
                    self.params.flip_for_display,
                )

            # Store per-level reconstruction and motion residual curves in run log object.
            run_log["recon_residuals_by_level"][idx_res] = residual_recon_norms
            run_log["motion_residuals_by_level"][idx_res] = residual_motion_norms
            level_elapsed = time.perf_counter() - level_t0
            _append_run_log(run_log, \
                f"    Total time of resolution level {idx_res}: {level_elapsed:.6f} s\n",)
            
            # Keep only the tensors needed to initialize the next level.
            self._strip_level_runtime_state(Data_res)
            Data_prev = self._make_next_level_initializer(Data_res)
            del Data_res
            del s_res

        _append_run_log(
            run_log,
            f"Total time of reconstruction run: {time.perf_counter() - run_t0:.6f} s",
        )
        _save_run_residual_plots(self.params.logs_folder, run_log)

        # If no valid best was tracked in the last level, fallback to the last level initializer.
        if best_image is None or best_motion is None:
            if "Data_prev" not in locals() or "ReconstructedImage" not in Data_prev or "MotionModel" not in Data_prev:
                raise RuntimeError("Reconstruction did not produce a valid image/motion solution.")
            best_image = Data_prev["ReconstructedImage"].clone()
            best_motion = Data_prev["MotionModel"].clone()

        # Rescale reconstructed image back to original k-space magnitude scale.
        best_image_unscaled = best_image * self.kspace_scale
        # Save final reconstructed image(s): one file per Nex when Nex > 1.
        if best_image_unscaled.shape[0] == 1:
            show_and_save_image(
                best_image_unscaled[0],
                "image_reconstructed",
                self.params.results_folder,
                flip_for_display=self.params.flip_for_display,
            )
        else:
            # Save the average across Nex as the default "corrected" image for notebook display.
            mean_img = best_image_unscaled.mean(dim=0)
            show_and_save_image(
                mean_img,
                "image_reconstructed",
                self.params.results_folder,
                flip_for_display=self.params.flip_for_display,
            )
            # Also save each Nex image separately for inspection/debugging.
            for nex_idx in range(best_image_unscaled.shape[0]):
                show_and_save_image(
                    best_image_unscaled[nex_idx],
                    f"image_reconstructed_nex{nex_idx + 1}",
                    self.params.results_folder,
                    flip_for_display=self.params.flip_for_display,
                )

        if self.params.reconstruction_motion_type == "rigid":
            save_final_rigid_motion_plots(best_motion, self.motion_plot_context, self.params.results_folder,
                                          self.params.N_motion_states, self.params.ResolutionLevels, self.params.data_type)
        elif self.params.reconstruction_motion_type == "non-rigid":
            save_final_nonrigid_alpha_maps(best_motion, best_image_unscaled[0], self.params.results_folder,
                                           flip_for_display=self.params.flip_for_display, motion_plot_context=self.motion_plot_context)

        return best_image_unscaled, best_motion

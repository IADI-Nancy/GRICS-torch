import torch
import torch.nn.functional as F
import time
from contextlib import nullcontext
from tqdm.auto import tqdm

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.utils.show_and_save_image import show_and_save_image
from src.utils.save_final_motion_plots import save_final_nonrigid_alpha_maps, save_final_rigid_motion_plots
from src.utils.joint_reconstructor_utils import (
    _assign_cached_reg_scale,
    _append_run_log,
    _console,
    _format_cg_info,
    _init_run_logging,
    _initialize_level_tracking,
    _resolve_gn_iterations_per_level,
    _save_nonrigid_motion_debug,
    _save_run_residual_plots,
)

# --------------------------------------------------------------------------
# Class that performs joint image–motion reconstruction
# --------------------------------------------------------------------------
class JointReconstructor:

    def __init__(self, KspaceData, smaps, SamplingIndices, motion_signal, params, kspace_scale=1.0, motion_plot_context=None):
        Ncoils, Nx_full, Ny_full, Nsli = smaps.shape

        # Parameters constant for all resolutions        
        self.params = params
        self.Ncoils = Ncoils
        self.device = KspaceData.device
        self.Nalpha = 3 if self.params.motion_type == "rigid" else 2
        self.kspace_scale = float(kspace_scale)
        if motion_signal is None:
            raise ValueError("motion_signal must be provided.")
        self.motion_signal = motion_signal.to(self.device)
        self.motion_plot_context = motion_plot_context or {}
        self._last_image_cg_info = None
        self._last_motion_cg_info = None

        # Data changing with resolution
        self.Data_full = {}
        self.Data_full["Nx"] = Nx_full
        self.Data_full["Ny"] = Ny_full
        self.Data_full["SensitivityMaps"] = smaps
        self.Data_full["KspaceData"] = KspaceData
        self.Data_full["Nsamples"] = sum(
            SamplingIndices[0][ms].numel()
            for ms in range(len(SamplingIndices[0]))
        )
        self.Data_full["SamplingIndices"] = SamplingIndices

    def resize_img_2D(self, img, new_size):
        is_complex = img.is_complex()

        # ---------- Helper: interpolate real/imag ----------
        def interp_part(x):
            """Interpolate real-valued tensor of shape [H,W] or [C,H,W]."""
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

            else:
                raise ValueError(f"Unexpected shape {x.shape}")

        # ---------- Real tensor case ----------
        if not is_complex:
            return interp_part(img)

        # ---------- Complex case ----------
        real = interp_part(img.real)
        imag = interp_part(img.imag)
        return torch.complex(real, imag)
        

    def downsample_sampling_indices(self, Sampling_full, Nx_res, Ny_res):
        Nx_full, Ny_full, = self.Data_full["Nx"], self.Data_full["Ny"]

        # central crop coordinates
        x0 = (Nx_full - Nx_res) // 2
        y0 = (Ny_full - Ny_res) // 2

        Sampling_res = []

        for nex in range(self.params.Nex):
            Sampling_res.append([])
            for indices in Sampling_full[nex]:
                # compute x,y coordinates
                x = indices // Ny_full
                y = indices % Ny_full

                # mask inside central region
                mask = (x >= x0) & (x < x0 + Nx_res) & (y >= y0) & (y < y0 + Ny_res)

                # keep only those indices
                x_crop = x[mask] - x0
                y_crop = y[mask] - y0

                # re-flatten for Nx_res × Ny_res grid
                new_inds = x_crop * Ny_res + y_crop

                Sampling_res[nex].append(new_inds)

        return Sampling_res

    def downsample_kspace(self, Nx_res, Ny_res):
        Nx_full, Ny_full = self.Data_full["Nx"], self.Data_full["Ny"]
        kspace_full = self.Data_full["KspaceData"]

        # central crop coordinates
        x0 = (Nx_full - Nx_res) // 2
        y0 = (Ny_full - Ny_res) // 2

        # create mask
        mask = torch.zeros((Nx_full, Ny_full), dtype=torch.bool, device=kspace_full.device)
        mask[x0:x0+Nx_res, y0:y0+Ny_res] = True
        kspace_res = kspace_full[:, :, mask, :].reshape(kspace_full.shape[0], kspace_full.shape[1], -1)

        return kspace_res   

    # ----------------------------------------------------------------------
    # Build low-resolution data structure at given resolution level
    # ----------------------------------------------------------------------
    def downsample_data(self, res_factor):    
        Nx = int(round(self.Data_full["Nx"] * res_factor))
        Ny = int(round(self.Data_full["Ny"] * res_factor))

        Data_res = {}
        Data_res["Nx"] = Nx
        Data_res["Ny"] = Ny
        
        Data_res["SensitivityMaps"] = self.resize_img_2D(self.Data_full["SensitivityMaps"].squeeze(), (Nx, Ny)).unsqueeze(-1)
        Data_res["SamplingIndices"] = self.downsample_sampling_indices(self.Data_full["SamplingIndices"], Nx, Ny)
        Data_res["KspaceData"] = self.downsample_kspace(Nx, Ny)
        Data_res["Nsamples"] = Data_res["KspaceData"].shape[2]

        return Data_res
    
    def upsample_data(self, Data_prev, Data_res):
        img_prev = Data_prev["ReconstructedImage"]
        img_res = self.resize_img_2D(img_prev, (Data_res["Nx"], Data_res["Ny"]))
        Data_res["ReconstructedImage"] = img_res

        mot_prev = Data_prev["MotionModel"]
        if self.params.motion_type == "rigid":
            Data_res["MotionModel"] = torch.zeros((self.Nalpha, self.params.N_motion_states), device=self.device)
            Data_res["MotionModel"][0,:] = mot_prev[0,:] * Data_res["Nx"] / Data_prev["Nx"]  # scale translations
            Data_res["MotionModel"][1,:] = mot_prev[1,:] * Data_res["Ny"] / Data_prev["Ny"]  # scale translations
            Data_res["MotionModel"][2,:] = mot_prev[2,:]  # rotations remain the same
        else:
            mot_res = self.resize_img_2D(mot_prev, (Data_res["Nx"], Data_res["Ny"]))
            mot_res[0] = mot_res[0] * Data_res["Nx"] / Data_prev["Nx"]
            mot_res[1] = mot_res[1] * Data_res["Ny"] / Data_prev["Ny"]
            Data_res["MotionModel"] = mot_res

    def build_motion_operator(self, Data_res):
        Nx, Ny = Data_res["Nx"], Data_res["Ny"]
        alpha = Data_res["MotionModel"]
        if self.params.motion_type == "rigid":
            motionOperator = MotionOperator(Nx, Ny, alpha, self.params.motion_type)
        else:
            motion_signal = self.motion_signal
            motionOperator = MotionOperator(Nx, Ny, alpha, self.params.motion_type, motion_signal=motion_signal.to(dtype=alpha.dtype))
        return motionOperator

    def build_encoding_operator(self, Data_res):
        E = EncodingOperator(Data_res["SensitivityMaps"], Data_res["Nsamples"], Data_res["SamplingIndices"],
                             self.params.Nex, Data_res["MotionOperator"])
        return E
    
    def build_motion_perturbation_simulator(self, Data_res):
        J = MotionPerturbationSimulator(Data_res["SensitivityMaps"], Data_res["Nsamples"], Data_res["SamplingIndices"],
                                        self.params.Nex, Data_res["ReconstructedImage"], Data_res["MotionOperator"])
        return J

    def solve_image(self, Data_res):
        x0 = Data_res["ReconstructedImage"]
        x0 = x0.to(self.device)
        E = Data_res["E"]

        b = E.adjoint(Data_res["KspaceData"])
        solver = ConjugateGradientSolver(
            E, reg_lambda=self.params.lambda_r, verbose=self.params.verbose, early_stopping=self.params.cg_early_stopping,
            true_residual_interval=self.params.cg_true_residual_interval, max_stag_steps=self.params.cg_max_stag_steps,
            max_more_steps=self.params.cg_max_more_steps, use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
            reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
        )
        _assign_cached_reg_scale(self.params, Data_res, "image", solver, b.flatten())

        img_vec = solver.cg(b.flatten(), x0=x0.flatten(), max_iter=self.params.max_iter_recon, tol=self.params.tol_recon)
        self._last_image_cg_info = solver.last_info

        img = img_vec.reshape(self.params.Nex, Data_res["Nx"], Data_res["Ny"])
        return img

    # # ----------------------------------------------------------------------
    # # Solve for motion model update
    # # ----------------------------------------------------------------------
    def _n_motion_params(self, Data_res):
        if self.params.motion_type == "rigid":
            return self.Nalpha * self.params.N_motion_states
        return self.Nalpha * Data_res["Nx"] * Data_res["Ny"]

    def solve_motion(self, Data_res, residual):
        Nparams = self._n_motion_params(Data_res)
        J = Data_res["J"]
        b_data = J.adjoint(residual)
        x0 = torch.zeros(Nparams, dtype=b_data.dtype, device=residual.device)

        if self.params.motion_type == "non-rigid":
            solver = ConjugateGradientSolver(
                J, reg_lambda=self.params.lambda_m, regularizer="Tikhonov_gradient",
                regularization_shape=(self.Nalpha, Data_res["Nx"], Data_res["Ny"]), verbose=self.params.verbose,
                early_stopping=self.params.cg_early_stopping, true_residual_interval=self.params.cg_true_residual_interval,
                max_stag_steps=self.params.cg_max_stag_steps, max_more_steps=self.params.cg_max_more_steps,
                use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy, reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
            )
            # Unscaled regularization:
            # A(dm) = J^H J dm + mu * GhG(dm)
            # b     = J^H r    - mu * GhG(alpha_current)
            _assign_cached_reg_scale(self.params, Data_res, "motion_nonrigid", solver, b_data.flatten())
            b = b_data - solver.effective_lambda() * solver.regularization(Data_res["MotionModel"].flatten())
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

        if self.params.motion_type == "rigid":
            motion_perturb = mot_pert_vec.reshape(self.Nalpha, self.params.N_motion_states)
        else:
            motion_perturb = mot_pert_vec.reshape(self.Nalpha, Data_res["Nx"], Data_res["Ny"])
        return motion_perturb
    
    def _prepare_resolution_level(self, idx_res, r):
        _console(self.params, f"\n=== Resolution level {idx_res+1}: factor {r} ===")

        # Prepare low-resolution dataset
        Data_res = self.downsample_data(r)

        # Initialize image and motion model
        if idx_res == 0:
            Data_res["ReconstructedImage"] = torch.zeros((self.params.Nex, Data_res["Nx"], Data_res["Ny"]), dtype=torch.complex128, device=self.device)
            
            if self.params.motion_type == "rigid":
                Data_res["MotionModel"] = torch.zeros((self.Nalpha, self.params.N_motion_states), device=self.device)
            elif self.params.motion_type == "non-rigid":
                Data_res["MotionModel"] = torch.zeros((self.Nalpha, Data_res["Nx"], Data_res["Ny"]), device=self.device)
        return Data_res

    # ----------------------------------------------------------------------
    # Perform full multi-resolution Gauss–Newton joint reconstruction
    # ----------------------------------------------------------------------
    def run(self):
        ResLevels = self.params.ResolutionLevels
        gn_iters_per_level = _resolve_gn_iterations_per_level(self.params, ResLevels)

        run_log = _init_run_logging(self.params, len(ResLevels), gn_iters_per_level)
        run_t0 = time.perf_counter()
        global_best_metric = float("inf")
        global_best_image = None
        global_best_motion = None
        last_image = None
        last_motion = None

        for idx_res, r in enumerate(ResLevels):
            GN_iter = gn_iters_per_level[idx_res]
            level_t0 = time.perf_counter()
            Data_res = self._prepare_resolution_level(idx_res, r)
            if idx_res != 0:
                self.upsample_data(Data_prev, Data_res)
            level_init_time = time.perf_counter() - level_t0

            _append_run_log(
                run_log,
                (
                    f"Resolution level {idx_res} ({Data_res['Nx']}x{Data_res['Ny']}x1, "
                    f"{Data_res['Ny']} views, {self.params.N_motion_states} virtual times)\n"
                    f"    Resolution level initializations : {level_init_time:.6f} s\n"
                ),
            )

            (residual_recon_norms, residual_motion_norms, best_relres, best_image, best_motion) = _initialize_level_tracking()
            s_res = Data_res["KspaceData"].flatten()
            show_bar = self.params.jupyter_notebook_flag
            # Use a tqdm progress-bar context when there are GN iterations; otherwise use a no-op context manager.
            bar_ctx = tqdm(total=GN_iter, desc=f"Resolution level {idx_res + 1}/{len(ResLevels)}", \
                disable=(not show_bar), leave=True, dynamic_ncols=True, position=0) if GN_iter > 0 else nullcontext()

            # Gauss–Newton iterations
            with bar_ctx as pbar:
                for it in range(GN_iter):
                    _console(self.params, f"  GN iteration {it+1}/{GN_iter}")
                    fp_t0 = time.perf_counter()

                    # -------------------------------IMAGE RECONSTRUCTION STEP -------------------------

                    # 1) Build motion and encoding operators
                    Data_res["MotionOperator"] = self.build_motion_operator(Data_res)
                    Data_res["E"] = self.build_encoding_operator(Data_res)

                    # 2) Solve for image
                    _console(self.params, "    Solving for image...")
                    t_img = time.perf_counter()
                    img = self.solve_image(Data_res)
                    img_elapsed = time.perf_counter() - t_img
                    Data_res["ReconstructedImage"] = img
                    _append_run_log(
                        run_log,
                        f"    Reconstruction step : {_format_cg_info(self._last_image_cg_info)}, elapsed time = {img_elapsed:.6f} s",
                    )

                    if idx_res == len(ResLevels) - 1 and it == GN_iter - 1:
                        if pbar is not None:
                            pbar.update(1)
                        break

                    # 3) Compute residual
                    x = img.flatten()
                    y = Data_res["E"].forward(x)
                    residual = s_res - y
                    res_norm = torch.linalg.norm(residual).item()
                    rel_res = res_norm / (torch.linalg.norm(s_res).item() + 1e-12)
                    residual_recon_norms.append(rel_res)
                    if pbar is not None:
                        pbar.set_postfix(recon=f"{rel_res:.2e}")

                    if it > 0 and rel_res > best_relres:
                        _console(self.params, "    Relative residual increased — restoring best solution at this level.")
                        _append_run_log(
                            run_log,
                            "    Relative residual increased - restoring best solution at this level.",
                        )
                        if pbar is not None:
                            pbar.update(1)
                        break

                    # Residual improved: store current image/motion as level-best.
                    best_relres = rel_res
                    best_image = Data_res["ReconstructedImage"].clone()
                    best_motion = Data_res["MotionModel"].clone()

                    # ------------------------------- MOTION MODEL RECONSTRUCTION STEP -------------------------

                    # 4) Build Jacobian encoding operator for solving ∇_u(E)·δu = δkspace
                    Data_res["J"] = self.build_motion_perturbation_simulator(Data_res)

                    # 5) Solve for motion update
                    _console(self.params, "    Solving for motion update...")
                    t_mot = time.perf_counter()
                    dm = self.solve_motion(Data_res, residual)
                    mot_elapsed = time.perf_counter() - t_mot

                    Data_res["MotionModel"] += dm.real
                    dm_norm = torch.linalg.norm(dm.flatten()).item()
                    alpha_norm = torch.linalg.norm(Data_res["MotionModel"].flatten()).item()
                    dm_rel_norm = dm_norm / (alpha_norm + 1e-12)
                    residual_motion_norms.append(dm_rel_norm)
                    fp_elapsed = time.perf_counter() - fp_t0
                    _append_run_log(
                        run_log,
                        (
                            f"    Model optimization step: {_format_cg_info(self._last_motion_cg_info)}, "
                            f"elapsed time = {mot_elapsed:.6f} s\n"
                            f"    Fixed point iter {it}: "
                            f"recon_rel_residual = {rel_res:.6e}, "
                            f"motion_rel_residual = {dm_rel_norm:.6e}, "
                            f"motion_norm = {dm_norm:.6e} : {fp_elapsed:.6f} s\n"
                        ),
                    )
                    if pbar is not None:
                        pbar.update(1)

            if self.params.debug_flag:
                if best_image is not None and best_motion is not None:
                    Data_res["ReconstructedImage"] = best_image
                    Data_res["MotionModel"] = best_motion
                show_and_save_image(
                    Data_res["ReconstructedImage"][0],
                    f"image_resolution_level{idx_res + 1}",
                    self.params.debug_folder,
                    flip_for_display=self.params.flip_for_display,
                )
                _save_nonrigid_motion_debug(
                    Data_res,
                    idx_res + 1,
                    self.params.motion_type,
                    self.params.debug_folder,
                    self.params.flip_for_display,
                )

            run_log["recon_residuals_by_level"][idx_res] = residual_recon_norms
            run_log["motion_residuals_by_level"][idx_res] = residual_motion_norms
            level_elapsed = time.perf_counter() - level_t0
            _append_run_log(
                run_log,
                f"    Total time of resolution level {idx_res}: {level_elapsed:.6f} s\n",
            )

            Data_prev = Data_res
            if "ReconstructedImage" in Data_res and "MotionModel" in Data_res:
                last_image = Data_res["ReconstructedImage"].clone()
                last_motion = Data_res["MotionModel"].clone()
            if best_image is not None and best_motion is not None and best_relres < global_best_metric:
                global_best_metric = best_relres
                global_best_image = best_image.clone()
                global_best_motion = best_motion.clone()

        _append_run_log(
            run_log,
            f"Total time of reconstruction run: {time.perf_counter() - run_t0:.6f} s",
        )
        _save_run_residual_plots(self.params.logs_folder, run_log)

        # Robust fallback for degenerate short-run settings (e.g. one GN step).
        if global_best_image is None or global_best_motion is None:
            if last_image is None or last_motion is None:
                raise RuntimeError("Reconstruction did not produce a valid image/motion solution.")
            global_best_image = last_image
            global_best_motion = last_motion

        global_best_image_unscaled = global_best_image * self.kspace_scale
        show_and_save_image(
            global_best_image_unscaled[0],
            'image_reconstructed',
            self.params.results_folder,
            flip_for_display=self.params.flip_for_display,
        )
        if self.params.motion_type == "rigid":
            save_final_rigid_motion_plots(global_best_motion, self.motion_plot_context, self.params.results_folder,
                                          self.params.N_motion_states, self.params.ResolutionLevels, self.params.data_type)
        if self.params.motion_type == "non-rigid":
            save_final_nonrigid_alpha_maps(global_best_motion, global_best_image_unscaled[0], self.params.results_folder,
                                           flip_for_display=self.params.flip_for_display, motion_plot_context=self.motion_plot_context)

        return global_best_image_unscaled, global_best_motion

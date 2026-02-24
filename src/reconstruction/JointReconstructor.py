import torch
import torch.nn.functional as F
import os
import time
from contextlib import nullcontext

from tqdm.auto import tqdm

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.utils.show_and_save_image import show_and_save_image
from src.utils.save_alpha_component_map import save_alpha_component_map
from src.utils.save_nonrigid_quiver_with_contours import save_nonrigid_quiver_with_contours
from src.utils.save_residual_subplots import save_residual_subplots
from src.utils.save_clustered_motion_plots import save_clustered_motion_plots

def test_J_singularity(motionSimulator):
    Nalpha = motionSimulator.Nalpha
    N_motion_states = len(motionSimulator.SamplingIndices[0])

    dim = Nalpha * N_motion_states

    # Build JHJ matrix explicitly (small system only!)
    JHJ = torch.zeros((dim, dim), device=motionSimulator.device)

    for i in range(dim):
        e = torch.zeros(dim, device=motionSimulator.device)
        e[i] = 1.0

        Je = motionSimulator.forward(e)
        JHJe = motionSimulator.adjoint(Je)

        JHJ[:, i] = JHJe

    # SVD
    U, S, V = torch.linalg.svd(JHJ)

    print("Singular values of JHJ:")
    print(S)

    print("Smallest singular value:", S[-1])
    print("Condition number:", S[0] / S[-1])

    print("Null-space direction (last right singular vector):")
    print(V[-1])

    return S, V

def _format_cg_info(cg_info):
    if cg_info is None:
        return "flag = -1, relres = nan, iter = 0"
    return (
        f"flag = {cg_info.get('flag', -1)}, "
        f"relres = {cg_info.get('relres', float('nan')):.6e}, "
        f"iter = {cg_info.get('iterations', 0)}"
    )

# --------------------------------------------------------------------------
# Class that performs joint image–motion reconstruction
# --------------------------------------------------------------------------
class JointReconstructor:

    def __init__(
        self,
        KspaceData,
        smaps,
        SamplingIndices,
        motion_signal,
        params,
        kspace_scale=1.0,
        motion_plot_context=None,
    ):
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

    def _console(self, message):
        if getattr(self.params, "print_to_console", True):
            print(message)

    def resize_img_2D(self, img, new_size):
        """
        Bilinear resize for complex or real images.
        Supports:
            - img: [H, W] (real or complex)
            - img: [C, H, W] (real or complex)
        """
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
        Nex = self.params.Nex

        # central crop coordinates
        x0 = (Nx_full - Nx_res) // 2
        y0 = (Ny_full - Ny_res) // 2

        # create mask
        mask = torch.zeros((Nx_full, Ny_full), dtype=torch.bool, device=kspace_full.device)
        mask[x0:x0+Nx_res, y0:y0+Ny_res] = True
        # kspace_reshaped = kspace_full.reshape(Nex, Nx_full, Ny_full, -1)
        # kspace_res = kspace_reshaped[:, mask, :].reshape(-1, kspace_full.shape[-1])
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

    # ----------------------------------------------------------------------
    # Build Ux, Uy fields and Motion Operators
    # ----------------------------------------------------------------------
    def build_motion_operator(self, Data_res):
        Nx, Ny = Data_res["Nx"], Data_res["Ny"]
        alpha = Data_res["MotionModel"]
        if self.params.motion_type == "rigid":
            motionOperator = MotionOperator(Nx, Ny, alpha, self.params.motion_type)
        else:
            motion_signal = self.motion_signal
            if motion_signal.dtype != alpha.dtype:
                motion_signal = motion_signal.to(dtype=alpha.dtype)
            motionOperator = MotionOperator(Nx, Ny, alpha, self.params.motion_type, motion_signal=motion_signal)
        return motionOperator

    
    def build_encoding_operator(self, Data_res):
        E = EncodingOperator(
            Data_res["SensitivityMaps"],
            Data_res["Nsamples"],
            Data_res["SamplingIndices"],
            self.params.Nex,
            Data_res["MotionOperator"]
        )
        return E
    
    def build_motion_perturbation_simulator(self, Data_res):
        J = MotionPerturbationSimulator(
            Data_res["SensitivityMaps"],
            Data_res["Nsamples"],
            Data_res["SamplingIndices"],
            self.params.Nex,
            Data_res["ReconstructedImage"],
            Data_res["MotionOperator"]
        )
        return J

    def _assign_cached_reg_scale(self, Data_res, cache_key, solver, reference_vec):
        if not self.params.cg_use_reg_scale_proxy:
            solver.reg_scale = 1.0
            return

        cache = Data_res.setdefault("_reg_scale_cache", {})
        if cache_key not in cache:
            cache[cache_key] = solver.update_regularization_scale(reference_vec)
        solver.reg_scale = cache[cache_key]

    # ----------------------------------------------------------------------
    # Solve linear system for image
    # ----------------------------------------------------------------------
    def solve_image(self, Data_res):
        x0 = Data_res["ReconstructedImage"]
        x0 = x0.to(self.device)
        E = Data_res["E"]

        b = E.adjoint(Data_res["KspaceData"])
        solver = ConjugateGradientSolver(
            E,
            reg_lambda=self.params.lambda_r,
            verbose=self.params.verbose,
            early_stopping=self.params.cg_early_stopping,
            max_stag_steps=self.params.cg_max_stag_steps,
            max_more_steps=self.params.cg_max_more_steps,
            use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
            reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
        )
        self._assign_cached_reg_scale(Data_res, "image", solver, b.flatten())

        img_vec = solver.cg(
            b.flatten(),
            x0=x0.flatten(),
            max_iter=self.params.max_iter_recon,
            tol=self.params.tol_recon,
            M=None,
        )
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
                J,
                reg_lambda=self.params.lambda_m,
                regularizer="Tikhonov_gradient",
                regularization_shape=(self.Nalpha, Data_res["Nx"], Data_res["Ny"]),
                verbose=self.params.verbose,
                early_stopping=self.params.cg_early_stopping,
                max_stag_steps=self.params.cg_max_stag_steps,
                max_more_steps=self.params.cg_max_more_steps,
                use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
                reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
            )
            # Unscaled regularization:
            # A(dm) = J^H J dm + mu * GhG(dm)
            # b     = J^H r    - mu * GhG(alpha_current)
            self._assign_cached_reg_scale(Data_res, "motion_nonrigid", solver, b_data.flatten())
            b = b_data - solver.effective_lambda() * solver.regularization(Data_res["MotionModel"].flatten())
            mot_pert_vec = solver.cg(
                b.flatten(),
                x0=x0.flatten(),
                max_iter=self.params.max_iter_motion,
                tol=self.params.tol_motion,
                M=None,
            )
        else:
            solver = ConjugateGradientSolver(
                J,
                reg_lambda=self.params.lambda_m,
                verbose=self.params.verbose,
                early_stopping=self.params.cg_early_stopping,
                max_stag_steps=self.params.cg_max_stag_steps,
                max_more_steps=self.params.cg_max_more_steps,
                use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
                reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
            )
            self._assign_cached_reg_scale(Data_res, "motion_rigid", solver, b_data.flatten())
            mot_pert_vec = solver.cg(
                b_data.flatten(),
                x0=x0.flatten(),
                max_iter=self.params.max_iter_motion,
                tol=self.params.tol_motion,
                M=None,
            )
        self._last_motion_cg_info = solver.last_info

        if self.params.motion_type == "rigid":
            motion_perturb = mot_pert_vec.reshape(self.Nalpha, self.params.N_motion_states)
        else:
            motion_perturb = mot_pert_vec.reshape(self.Nalpha, Data_res["Nx"], Data_res["Ny"])
        return motion_perturb
    
    def solve_motion_scaled(self, Data_res, residual):
        if self.params.motion_type != "rigid":
            return self.solve_motion(Data_res, residual)

        Nparams = self.Nalpha * self.params.N_motion_states
        with torch.no_grad():
            diag_JHJ = torch.zeros(Nparams, dtype=torch.complex128, device=self.device)

            for i in range(Nparams):
                e = torch.zeros(Nparams, device=self.device)
                e[i] = 1.0
                Je = Data_res["J"].forward(e)
                JHJe = Data_res["J"].adjoint(Je)
                diag_JHJ[i] = JHJe[i]

            # Avoid division by zero
            eps = 1e-8
            scale = torch.sqrt(diag_JHJ + eps)

        # Nparams = self.Nalpha * self.params.N_motion_states
        device = residual.device

        J = Data_res["J"]

        # ---- Define scaled operator J̃ = J S^{-1} ----
        class ScaledJacobian:
            def __init__(self, J, scale):
                self.J = J
                self.scale = scale
                self.device = J.device

            def forward(self, v):
                # J (S^{-1} v)
                return self.J.forward(v / self.scale)

            def adjoint(self, w):
                # S^{-1} J^H w
                return self.J.adjoint(w) / self.scale

            def normal(self, v):
                return self.adjoint(self.forward(v))

        J_scaled = ScaledJacobian(J, scale)

        # ---- Build RHS in scaled space ----
        b_scaled = J_scaled.adjoint(residual)

        # ---- Initial guess ----
        x0 = torch.zeros(Nparams, dtype=b_scaled.dtype, device=device)

        # ---- Solve in scaled variables ----
        solver = ConjugateGradientSolver(
            J_scaled,
            reg_lambda=self.params.lambda_m,
            verbose=self.params.verbose,
            early_stopping=self.params.cg_early_stopping,
            max_stag_steps=self.params.cg_max_stag_steps,
            max_more_steps=self.params.cg_max_more_steps,
            use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
            reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
        )
        self._assign_cached_reg_scale(Data_res, "motion_scaled", solver, b_scaled.flatten())

        delta_tilde = solver.cg(
            b_scaled.flatten(),
            x0=x0,
            max_iter=self.params.max_iter_motion,
            tol=self.params.tol_motion,
            M=None,
        )
        self._last_motion_cg_info = solver.last_info

        # ---- Unscale back to physical parameters ----
        delta = delta_tilde / scale

        motion_perturb = delta.reshape(self.Nalpha, self.params.N_motion_states)

        return motion_perturb
    
    def random_motion_init(self):
        if self.params.motion_type != "rigid":
            raise RuntimeError("random_motion_init() is only for rigid motion.")
        Nalpha = self.Nalpha                    # should be 3
        Nshots = self.params.N_motion_states            # number of motion states

        alpha = torch.zeros((Nalpha, Nshots), device=self.device)

        # Translation X ∈ [-max_tx, +max_tx]
        alpha[0, :] = self.params.max_tx * (2 * torch.rand(Nshots, device=self.device) - 1)

        # Translation Y ∈ [-max_ty, +max_ty]
        alpha[1, :] = self.params.max_ty * (2 * torch.rand(Nshots, device=self.device) - 1)

        # Rotation ∈ [-max_phi, +max_phi] degrees → radians
        alpha[2, :] = (self.params.max_phi *(2 * torch.rand(Nshots, device=self.device) - 1)* (torch.pi / 180.0))

        return alpha

    def _initialize_global_tracking(self):
        global_best_metric = float("inf")
        global_best_image = None
        global_best_motion = None
        global_converged = False
        return global_best_metric, global_best_image, global_best_motion, global_converged

    def _resolve_gn_iterations_per_level(self, res_levels):
        gn_cfg = self.params.GN_iterations_per_level
        if isinstance(gn_cfg, int):
            return [gn_cfg] * len(res_levels)
        if isinstance(gn_cfg, (list, tuple)):
            if len(gn_cfg) == 0:
                raise ValueError("GN_iterations_per_level list/tuple cannot be empty.")
            gn_list = [int(v) for v in gn_cfg]
            if len(gn_list) != len(res_levels):
                raise ValueError(
                    "Inconsistent config: "
                    f"GN_iterations_per_level has {len(gn_list)} values, "
                    f"but ResolutionLevels has {len(res_levels)} values."
                )
            return gn_list
        raise ValueError("GN_iterations_per_level must be int, list, or tuple.")

    def _init_restart_logging(self, restart_idx, n_levels, gn_iters_per_level):
        os.makedirs(self.params.logs_folder, exist_ok=True)
        log_path = os.path.join(self.params.logs_folder, f"restart_{restart_idx}.log")
        param_items = {}
        simulation_param_keys = {
            "motion_simulation_type",
            "num_motion_events",
            "max_tx",
            "max_ty",
            "max_phi",
            "max_center_x",
            "max_center_y",
            "seed",
            "motion_tau",
            "nonrigid_motion_amplitude",
            "displacementfield_size",
        }
        for key in dir(self.params):
            if key.startswith("_"):
                continue
            if key in simulation_param_keys:
                continue
            value = getattr(self.params, key)
            if callable(value):
                continue
            param_items[key] = value

        with open(log_path, "w") as f:
            f.write(f"Restart {restart_idx}\n")
            f.write(f"Motion type: {self.params.motion_type}\n")
            f.write(f"GN iterations per level: {gn_iters_per_level}\n\n")
            f.write("Parameters (excluding simulation parameters):\n")
            for key in sorted(param_items.keys()):
                f.write(f"  {key} = {param_items[key]}\n")
            f.write("\n")
        return {
            "path": log_path,
            "recon_residuals_by_level": [[] for _ in range(n_levels)],
            "motion_residuals_by_level": [[] for _ in range(n_levels)],
        }

    def _append_restart_log(self, restart_log, line=""):
        with open(restart_log["path"], "a") as f:
            f.write(line + "\n")

    def _save_restart_residual_plots(self, restart_log, restart_idx):
        recon_path = os.path.join(self.params.logs_folder, f"residual_recon_restart_{restart_idx}.png")
        motion_path = os.path.join(self.params.logs_folder, f"residual_motion_restart_{restart_idx}.png")
        save_residual_subplots(
            restart_log["recon_residuals_by_level"],
            title=f"Reconstruction residuals - restart {restart_idx}",
            y_label="Relative residual",
            out_path=recon_path,
        )
        save_residual_subplots(
            restart_log["motion_residuals_by_level"],
            title=f"Motion normalized residuals - restart {restart_idx}",
            y_label="||dm||2 / (||alpha||2 + eps)",
            out_path=motion_path,
        )

    def _prepare_resolution_level(self, idx_res, r, restart):
        self._console(f"\n=== Resolution level {idx_res+1}: factor {r} ===")

        # Prepare low-resolution dataset
        Data_res = self.downsample_data(r)

        # Initialize image and motion model
        if idx_res == 0:
            Data_res["ReconstructedImage"] = torch.zeros((self.params.Nex, Data_res["Nx"], Data_res["Ny"]), dtype=torch.complex128, device=self.device)
            
            if self.params.motion_type == "rigid":
                if restart == 0:
                    Data_res["MotionModel"] = torch.zeros((self.Nalpha, self.params.N_motion_states), device=self.device)
                else:
                    Data_res["MotionModel"] = self.random_motion_init()
            elif self.params.motion_type == "non-rigid":
                Data_res["MotionModel"] = torch.zeros((self.Nalpha, Data_res["Nx"], Data_res["Ny"]), device=self.device)
        return Data_res

    def _initialize_level_tracking(self):
        residual_recon_norms = []
        residual_motion_norms = []
        best_relres = float("inf")
        best_image = None
        best_motion = None
        return residual_recon_norms, residual_motion_norms, best_relres, best_image, best_motion

    def _update_global_best(self, best_relres, best_image, best_motion, global_best_metric, global_best_image, global_best_motion):
        # ----------------------------------------------------------
        # Compare restart result with global best
        # ----------------------------------------------------------
        if best_relres < global_best_metric:
            global_best_metric = best_relres
            global_best_image = best_image.clone()
            global_best_motion = best_motion.clone()
        return global_best_metric, global_best_image, global_best_motion

    def _save_nonrigid_motion_debug(self, Data_res, restart_idx, level_idx):
        if self.params.motion_type != "non-rigid":
            return

        alpha = Data_res["MotionModel"]
        if alpha.ndim != 3 or alpha.shape[0] < 2:
            return

        os.makedirs(self.params.debug_folder, exist_ok=True)

        alpha_x = alpha[0].detach().cpu()
        alpha_y = alpha[1].detach().cpu()

        if torch.is_complex(alpha_x) or torch.is_complex(alpha_y):
            components = (
                ("alpha_x_real", alpha_x.real),
                ("alpha_y_real", alpha_y.real),
                ("alpha_x_imag", alpha_x.imag),
                ("alpha_y_imag", alpha_y.imag),
            )
            alpha_x_for_quiver = alpha_x.real
            alpha_y_for_quiver = alpha_y.real
        else:
            components = (
                ("alpha_x", alpha_x),
                ("alpha_y", alpha_y),
            )
            alpha_x_for_quiver = alpha_x
            alpha_y_for_quiver = alpha_y

        for comp_name, comp in components:
            save_alpha_component_map(
                comp,
                f"{comp_name} restart {restart_idx} level {level_idx}",
                os.path.join(
                    self.params.debug_folder,
                    f"{comp_name}_restart_{restart_idx}_level{level_idx}.png",
                ),
            )

        save_nonrigid_quiver_with_contours(
            alpha_x_for_quiver,
            alpha_y_for_quiver,
            Data_res["ReconstructedImage"][0],
            f"motion field restart {restart_idx} level {level_idx}",
            os.path.join(
                self.params.debug_folder,
                f"motion_quiver_restart_{restart_idx}_level{level_idx}.png",
            ),
        )

    def _save_final_nonrigid_alpha_maps(self, motion_model, reconstructed_image):
        if self.params.motion_type != "non-rigid":
            return
        if motion_model is None or motion_model.ndim != 3 or motion_model.shape[0] < 2:
            return

        os.makedirs(self.params.results_folder, exist_ok=True)

        alpha_x = motion_model[0].detach().cpu()
        alpha_y = motion_model[1].detach().cpu()
        scale = self.motion_plot_context.get("alpha_visual_scale", None)
        alpha_abs_max_x = None if scale is None else scale.get("alpha_abs_max_x")
        alpha_abs_max_y = None if scale is None else scale.get("alpha_abs_max_y")
        amp_max = None if scale is None else scale.get("amp_max")

        if torch.is_complex(alpha_x) or torch.is_complex(alpha_y):
            components = (
                ("final_alpha_x_real", alpha_x.real),
                ("final_alpha_y_real", alpha_y.real),
                ("final_alpha_x_imag", alpha_x.imag),
                ("final_alpha_y_imag", alpha_y.imag),
            )
        else:
            components = (
                ("final_alpha_x", alpha_x),
                ("final_alpha_y", alpha_y),
            )

        for name, comp in components:
            if "alpha_x" in name:
                abs_max = alpha_abs_max_x
            elif "alpha_y" in name:
                abs_max = alpha_abs_max_y
            else:
                abs_max = None
            save_alpha_component_map(
                comp,
                name,
                os.path.join(self.params.results_folder, f"{name}.png"),
                abs_max=abs_max,
            )

        save_nonrigid_quiver_with_contours(
            alpha_x if not torch.is_complex(alpha_x) else alpha_x.real,
            alpha_y if not torch.is_complex(alpha_y) else alpha_y.real,
            reconstructed_image,
            "final_motion_quiver",
            os.path.join(self.params.results_folder, "final_motion_quiver.png"),
            flip_vertical=getattr(
                self.params, "flip_for_display", self.params.data_type in {"real-world", "raw-data"}
            ),
            amp_vmax=amp_max,
        )

    def _save_final_rigid_motion_plots(self, motion_model):
        if self.params.motion_type != "rigid":
            return
        if motion_model is None or motion_model.ndim != 2 or motion_model.shape[0] < 3:
            return
        motion_curve = self.motion_plot_context.get("motion_curve")
        labels_in = self.motion_plot_context.get("labels")
        ky_idx = self.motion_plot_context.get("ky_idx")
        nex_idx = self.motion_plot_context.get("nex_idx")
        if motion_curve is None or labels_in is None or ky_idx is None or nex_idx is None:
            return

        labels = labels_in.to(dtype=torch.long, device=motion_model.device)
        tx = motion_model[0, labels]
        ty = motion_model[1, labels]
        phi = motion_model[2, labels]

        save_clustered_motion_plots(
            motion_curve=motion_curve,
            labels=labels_in,
            ky_idx=ky_idx,
            nex_idx=nex_idx,
            nbins=self.params.N_motion_states,
            output_folder=self.params.results_folder,
            resolution_levels=self.motion_plot_context.get(
                "resolution_levels", getattr(self.params, "ResolutionLevels", None)
            ),
            tx=tx,
            ty=ty,
            phi=phi,
            data_type=self.motion_plot_context.get("data_type", getattr(self.params, "data_type", None)),
            y_limits=self.motion_plot_context.get("y_limits"),
        )

    # ----------------------------------------------------------------------
    # Perform full multi-resolution Gauss–Newton joint reconstruction
    # ----------------------------------------------------------------------
    def run(self):
        ResLevels = self.params.ResolutionLevels
        gn_iters_per_level = self._resolve_gn_iterations_per_level(ResLevels)

        global_best_metric, global_best_image, global_best_motion, global_converged = self._initialize_global_tracking()
        last_image = None
        last_motion = None

        for restart in range(self.params.max_restarts):
            restart_idx = restart + 1
            restart_log = self._init_restart_logging(restart_idx, len(ResLevels), gn_iters_per_level)
            restart_t0 = time.perf_counter()
            restart_converged = False

            for idx_res, r in enumerate(ResLevels):
                GN_iter = gn_iters_per_level[idx_res]
                level_t0 = time.perf_counter()
                Data_res = self._prepare_resolution_level(idx_res, r, restart)
                if idx_res != 0:
                    self.upsample_data(Data_prev, Data_res)
                level_init_time = time.perf_counter() - level_t0

                self._append_restart_log(
                    restart_log,
                    f"Resolution level {idx_res} ({Data_res['Nx']}x{Data_res['Ny']}x1, {Data_res['Ny']} views, {self.params.N_motion_states} virtual times)",
                )
                self._append_restart_log(
                    restart_log,
                    f"    Resolution level initializations : {level_init_time:.6f} s",
                )
                self._append_restart_log(restart_log, "")

                (
                    residual_recon_norms,
                    residual_motion_norms,
                    best_relres,
                    best_image,
                    best_motion,
                ) = self._initialize_level_tracking()
                s_res = Data_res["KspaceData"].flatten()
                show_bar = getattr(self.params, "jupyter_notebook_flag", False)
                bar_ctx = tqdm(
                    total=GN_iter,
                    desc=f"Resolution level {idx_res + 1}/{len(ResLevels)}",
                    disable=(not show_bar),
                    leave=True,
                    dynamic_ncols=True,
                ) if GN_iter > 0 else nullcontext()

                # Gauss–Newton iterations
                with bar_ctx as pbar:
                    for it in range(GN_iter):
                        self._console(f"  GN iteration {it+1}/{GN_iter}")
                        fp_t0 = time.perf_counter()

                        # -------------------------------IMAGE RECONSTRUCTION STEP -------------------------

                        # 1) Build motion and encoding operators
                        Data_res["MotionOperator"] = self.build_motion_operator(Data_res)
                        Data_res["E"] = self.build_encoding_operator(Data_res)

                        # 2) Solve for image
                        self._console("    Solving for image...")
                        t_img = time.perf_counter()
                        img = self.solve_image(Data_res)
                        img_elapsed = time.perf_counter() - t_img
                        Data_res["ReconstructedImage"] = img
                        self._append_restart_log(
                            restart_log,
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
                            self._console("    Relative residual increased — restoring best solution at this level.")
                            self._append_restart_log(
                                restart_log,
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

                        # 4) Solve for motion update
                        self._console("    Solving for motion update...")
                        t_mot = time.perf_counter()
                        if self.params.use_scaled_motion_update:
                            dm = self.solve_motion_scaled(Data_res, residual)
                        else:
                            dm = self.solve_motion(Data_res, residual)
                        mot_elapsed = time.perf_counter() - t_mot

                        Data_res["MotionModel"] += dm.real
                        dm_norm = torch.linalg.norm(dm.flatten()).item()
                        alpha_norm = torch.linalg.norm(Data_res["MotionModel"].flatten()).item()
                        dm_rel_norm = dm_norm / (alpha_norm + 1e-12)
                        residual_motion_norms.append(dm_rel_norm)
                        self._append_restart_log(
                            restart_log,
                            f"    Model optimization step: {_format_cg_info(self._last_motion_cg_info)}, elapsed time = {mot_elapsed:.6f} s",
                        )
                        fp_elapsed = time.perf_counter() - fp_t0
                        self._append_restart_log(
                            restart_log,
                            (
                                f"    Fixed point iter {it}: "
                                f"recon_rel_residual = {rel_res:.6e}, "
                                f"motion_rel_residual = {dm_rel_norm:.6e}, "
                                f"motion_norm = {dm_norm:.6e} : {fp_elapsed:.6f} s"
                            ),
                        )
                        self._append_restart_log(restart_log, "")
                        if pbar is not None:
                            pbar.update(1)

                    # ------------------------- LOGGING and PATIENCE CHECK -------------------------

                if self.params.debug_flag: 
                    if best_image is not None and best_motion is not None:
                        Data_res["ReconstructedImage"] = best_image
                        Data_res["MotionModel"] = best_motion
                    show_and_save_image(
                        Data_res["ReconstructedImage"][0],
                        'image_restart_' + str(restart + 1) + '_resolution_level' + str(idx_res+1),
                        self.params.debug_folder,
                        flip_for_display=getattr(
                            self.params,
                            "flip_for_display",
                            self.params.data_type in {"real-world", "raw-data"},
                        ),
                    )
                    self._save_nonrigid_motion_debug(Data_res, restart + 1, idx_res + 1)
                restart_log["recon_residuals_by_level"][idx_res] = residual_recon_norms
                restart_log["motion_residuals_by_level"][idx_res] = residual_motion_norms
                level_elapsed = time.perf_counter() - level_t0
                self._append_restart_log(
                    restart_log,
                    f"    Total time of resolution level {idx_res}: {level_elapsed:.6f} s",
                )
                self._append_restart_log(restart_log, "")

                if restart_converged:
                    break
                Data_prev = Data_res
                if "ReconstructedImage" in Data_res and "MotionModel" in Data_res:
                    last_image = Data_res["ReconstructedImage"].clone()
                    last_motion = Data_res["MotionModel"].clone()
            if best_image is not None and best_motion is not None:
                global_best_metric, global_best_image, global_best_motion = self._update_global_best(
                    best_relres, best_image, best_motion, global_best_metric, global_best_image, global_best_motion
                )

            self._append_restart_log(
                restart_log,
                f"Total time of restart {restart_idx}: {time.perf_counter() - restart_t0:.6f} s",
            )
            self._save_restart_residual_plots(restart_log, restart_idx)

            if restart_converged:
                global_converged = True
                self._console("Stopping restarts: true convergence achieved.")
                break

        if not global_converged:
            self._console("⚠ WARNING: No restart reached tolerance.")

        # Robust fallback for degenerate short-run settings (e.g. one GN step).
        if global_best_image is None or global_best_motion is None:
            if last_image is None or last_motion is None:
                raise RuntimeError("Reconstruction did not produce a valid image/motion solution.")
            global_best_image = last_image
            global_best_motion = last_motion

        global_best_image_unscaled = global_best_image * self.kspace_scale
        show_and_save_image(
            global_best_image_unscaled[0],
            'reconstructed_image',
            self.params.results_folder,
            flip_for_display=getattr(
                self.params,
                "flip_for_display",
                self.params.data_type in {"real-world", "raw-data"},
            ),
        )
        if self.params.motion_type == "rigid":
            self._save_final_rigid_motion_plots(global_best_motion)
        self._save_final_nonrigid_alpha_maps(global_best_motion, global_best_image_unscaled[0])

        return global_best_image_unscaled, global_best_motion

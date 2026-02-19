import torch
import torch.nn.functional as F
import os

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.utils.show_and_save_image import show_and_save_image
from src.utils.save_alpha_component_map import save_alpha_component_map
from src.utils.save_nonrigid_quiver_with_contours import save_nonrigid_quiver_with_contours
from src.utils.save_residual_convergence import save_residual_convergence

from Parameters import Parameters
params = Parameters()

def test_J_singularity(motionSimulator):
    Nalpha = motionSimulator.Nalpha
    N_mot_states = len(motionSimulator.SamplingIndices[0])

    dim = Nalpha * N_mot_states

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

def log_motion_parameters(alpha, res_level, gn_iter):
    """
    alpha: [Nalpha, N_mot_states]
    """
    fname = f"{params.logs_folder}motion_params_res{res_level}.txt"
    with open(fname, "a") as f:
        f.write(f"\nGN iteration {gn_iter}\n")
        f.write("alpha shape: {}\n".format(tuple(alpha.shape)))
        for a in range(alpha.shape[0]):
            vals = alpha[a].detach().cpu().numpy()
            f.write(f"  alpha[{a}]: {vals}\n")

# --------------------------------------------------------------------------
# Class that performs joint image–motion reconstruction
# --------------------------------------------------------------------------
class JointReconstructor:

    def __init__(self, KspaceData, smaps, SamplingIndices, motion_signal, kspace_scale=1.0):
        Ncoils, Nx_full, Ny_full, Nsli = smaps.shape

        # Parameters constant for all resolutions        
        self.Ncoils = Ncoils
        self.device = KspaceData.device
        self.Nalpha = 3 if params.motion_type == "rigid" else 2
        self.kspace_scale = float(kspace_scale)
        if motion_signal is None:
            raise ValueError("motion_signal must be provided.")
        self.motion_signal = motion_signal.to(self.device)

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

        for nex in range(params.Nex):
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
        Nex = params.Nex

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
        if params.motion_type == "rigid":
            Data_res["MotionModel"] = torch.zeros((self.Nalpha, params.N_mot_states), device=self.device)
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
        if params.motion_type == "rigid":
            motionOperator = MotionOperator(Nx, Ny, alpha, params.motion_type)
        else:
            motion_signal = self.motion_signal
            if motion_signal.dtype != alpha.dtype:
                motion_signal = motion_signal.to(dtype=alpha.dtype)
            motionOperator = MotionOperator(Nx, Ny, alpha, params.motion_type, motion_signal=motion_signal)
        return motionOperator

    
    def build_encoding_operator(self, Data_res):
        E = EncodingOperator(
            Data_res["SensitivityMaps"],
            Data_res["Nsamples"],
            Data_res["SamplingIndices"],
            params.Nex,
            Data_res["MotionOperator"]
        )
        return E
    
    def build_motion_perturbation_simulator(self, Data_res):
        J = MotionPerturbationSimulator(
            Data_res["SensitivityMaps"],
            Data_res["Nsamples"],
            Data_res["SamplingIndices"],
            params.Nex,
            Data_res["ReconstructedImage"],
            Data_res["MotionOperator"]
        )
        return J

    def _assign_cached_reg_scale(self, Data_res, cache_key, solver, reference_vec):
        if not params.cg_use_reg_scale_proxy:
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
            reg_lambda=params.lambda_r,
            verbose=params.verbose,
            early_stopping=params.cg_early_stopping,
            max_stag_steps=params.cg_max_stag_steps,
            max_more_steps=params.cg_max_more_steps,
            use_reg_scale_proxy=params.cg_use_reg_scale_proxy,
            reg_scale_num_probes=params.cg_reg_scale_num_probes,
        )
        self._assign_cached_reg_scale(Data_res, "image", solver, b.flatten())

        img_vec = solver.cg(
            b.flatten(),
            x0=x0.flatten(),
            max_iter=params.max_iter_recon,
            tol=params.tol_recon,
            M=None,
        )

        img = img_vec.reshape(params.Nex, Data_res["Nx"], Data_res["Ny"])
        return img

    # # ----------------------------------------------------------------------
    # # Solve for motion model update
    # # ----------------------------------------------------------------------
    def _n_motion_params(self, Data_res):
        if params.motion_type == "rigid":
            return self.Nalpha * params.N_mot_states
        return self.Nalpha * Data_res["Nx"] * Data_res["Ny"]

    def solve_motion(self, Data_res, residual):
        Nparams = self._n_motion_params(Data_res)
        J = Data_res["J"]
        b_data = J.adjoint(residual)
        x0 = torch.zeros(Nparams, dtype=b_data.dtype, device=residual.device)

        if params.motion_type == "non-rigid":
            solver = ConjugateGradientSolver(
                J,
                reg_lambda=params.lambda_m,
                regularizer="Tikhonov_gradient",
                regularization_shape=(self.Nalpha, Data_res["Nx"], Data_res["Ny"]),
                verbose=params.verbose,
                early_stopping=params.cg_early_stopping,
                max_stag_steps=params.cg_max_stag_steps,
                max_more_steps=params.cg_max_more_steps,
                use_reg_scale_proxy=params.cg_use_reg_scale_proxy,
                reg_scale_num_probes=params.cg_reg_scale_num_probes,
            )
            # Unscaled regularization:
            # A(dm) = J^H J dm + mu * GhG(dm)
            # b     = J^H r    - mu * GhG(alpha_current)
            self._assign_cached_reg_scale(Data_res, "motion_nonrigid", solver, b_data.flatten())
            b = b_data - solver.effective_lambda() * solver.regularization(Data_res["MotionModel"].flatten())
            mot_pert_vec = solver.cg(
                b.flatten(),
                x0=x0.flatten(),
                max_iter=params.max_iter_motion,
                tol=params.tol_motion,
                M=None,
            )
        else:
            solver = ConjugateGradientSolver(
                J,
                reg_lambda=params.lambda_m,
                verbose=params.verbose,
                early_stopping=params.cg_early_stopping,
                max_stag_steps=params.cg_max_stag_steps,
                max_more_steps=params.cg_max_more_steps,
                use_reg_scale_proxy=params.cg_use_reg_scale_proxy,
                reg_scale_num_probes=params.cg_reg_scale_num_probes,
            )
            self._assign_cached_reg_scale(Data_res, "motion_rigid", solver, b_data.flatten())
            mot_pert_vec = solver.cg(
                b_data.flatten(),
                x0=x0.flatten(),
                max_iter=params.max_iter_motion,
                tol=params.tol_motion,
                M=None,
            )

        if params.motion_type == "rigid":
            motion_perturb = mot_pert_vec.reshape(self.Nalpha, params.N_mot_states)
        else:
            motion_perturb = mot_pert_vec.reshape(self.Nalpha, Data_res["Nx"], Data_res["Ny"])
        return motion_perturb
    
    def solve_motion_scaled(self, Data_res, residual):
        if params.motion_type != "rigid":
            return self.solve_motion(Data_res, residual)

        Nparams = self.Nalpha * params.N_mot_states
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

        # Nparams = self.Nalpha * params.N_mot_states
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
            reg_lambda=params.lambda_m,
            verbose=True,
            early_stopping=params.cg_early_stopping,
            max_stag_steps=params.cg_max_stag_steps,
            max_more_steps=params.cg_max_more_steps,
            use_reg_scale_proxy=params.cg_use_reg_scale_proxy,
            reg_scale_num_probes=params.cg_reg_scale_num_probes,
        )
        self._assign_cached_reg_scale(Data_res, "motion_scaled", solver, b_scaled.flatten())

        delta_tilde = solver.cg(
            b_scaled.flatten(),
            x0=x0,
            max_iter=params.max_iter_motion,
            tol=params.tol_motion,
            M=None,
        )

        # ---- Unscale back to physical parameters ----
        delta = delta_tilde / scale

        motion_perturb = delta.reshape(self.Nalpha, params.N_mot_states)

        return motion_perturb
    
    def random_motion_init(self):
        if params.motion_type != "rigid":
            raise RuntimeError("random_motion_init() is only for rigid motion.")
        Nalpha = self.Nalpha                    # should be 3
        Nshots = params.N_mot_states            # number of motion states

        alpha = torch.zeros((Nalpha, Nshots), device=self.device)

        # Translation X ∈ [-max_tx, +max_tx]
        alpha[0, :] = params.max_tx * (2 * torch.rand(Nshots, device=self.device) - 1)

        # Translation Y ∈ [-max_ty, +max_ty]
        alpha[1, :] = params.max_ty * (2 * torch.rand(Nshots, device=self.device) - 1)

        # Rotation ∈ [-max_phi, +max_phi] degrees → radians
        alpha[2, :] = (params.max_phi *(2 * torch.rand(Nshots, device=self.device) - 1)* (torch.pi / 180.0))

        return alpha

    def _initialize_global_tracking(self):
        global_best_metric = float("inf")
        global_best_image = None
        global_best_motion = None
        global_converged = False
        return global_best_metric, global_best_image, global_best_motion, global_converged

    def _prepare_resolution_level(self, idx_res, r, restart):
        if params.verbose:
            fname = f"{params.logs_folder}motion_params_res{idx_res+1}.txt"
            open(fname, "w").close()
        print(f"\n=== Resolution level {idx_res+1}: factor {r} ===")

        # Prepare low-resolution dataset
        Data_res = self.downsample_data(r)

        # Initialize image and motion model
        if idx_res == 0:
            Data_res["ReconstructedImage"] = torch.zeros((params.Nex, Data_res["Nx"], Data_res["Ny"]), dtype=torch.complex128, device=self.device)
            
            if params.motion_type == "rigid":
                if restart == 0:
                    Data_res["MotionModel"] = torch.zeros((self.Nalpha, params.N_mot_states), device=self.device)
                else:
                    Data_res["MotionModel"] = self.random_motion_init()
            elif params.motion_type == "non-rigid":
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
        if params.motion_type != "non-rigid":
            return

        alpha = Data_res["MotionModel"]
        if alpha.ndim != 3 or alpha.shape[0] < 2:
            return

        os.makedirs(params.debug_folder, exist_ok=True)

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
                    params.debug_folder,
                    f"{comp_name}_restart_{restart_idx}_level{level_idx}.png",
                ),
            )

        save_nonrigid_quiver_with_contours(
            alpha_x_for_quiver,
            alpha_y_for_quiver,
            Data_res["ReconstructedImage"][0],
            f"motion field restart {restart_idx} level {level_idx}",
            os.path.join(
                params.debug_folder,
                f"motion_quiver_restart_{restart_idx}_level{level_idx}.png",
            ),
        )

    def _save_final_nonrigid_alpha_maps(self, motion_model, reconstructed_image):
        if params.motion_type != "non-rigid":
            return
        if motion_model is None or motion_model.ndim != 3 or motion_model.shape[0] < 2:
            return

        os.makedirs(params.results_folder, exist_ok=True)

        alpha_x = motion_model[0].detach().cpu()
        alpha_y = motion_model[1].detach().cpu()

        if torch.is_complex(alpha_x) or torch.is_complex(alpha_y):
            components = (
                ("final_alpha_x_real", alpha_x.real),
                ("final_alpha_y_real", alpha_y.real),
                ("final_alpha_x_imag", alpha_x.imag),
                ("final_alpha_y_imag", alpha_y.imag),
            )
            amp = torch.sqrt(alpha_x.real * alpha_x.real + alpha_y.real * alpha_y.real)
        else:
            components = (
                ("final_alpha_x", alpha_x),
                ("final_alpha_y", alpha_y),
            )
            amp = torch.sqrt(alpha_x * alpha_x + alpha_y * alpha_y)

        for name, comp in components:
            save_alpha_component_map(
                comp,
                name,
                os.path.join(params.results_folder, f"{name}.png"),
            )

        save_alpha_component_map(
            amp,
            "final_alpha_amplitude",
            os.path.join(params.results_folder, "final_alpha_amplitude.png"),
        )
        save_nonrigid_quiver_with_contours(
            alpha_x if not torch.is_complex(alpha_x) else alpha_x.real,
            alpha_y if not torch.is_complex(alpha_y) else alpha_y.real,
            reconstructed_image,
            "final_motion_quiver",
            os.path.join(params.results_folder, "final_motion_quiver.png"),
        )

    # ----------------------------------------------------------------------
    # Perform full multi-resolution Gauss–Newton joint reconstruction
    # ----------------------------------------------------------------------
    def run(self):
        ResLevels = params.ResolutionLevels
        GN_iter = params.GN_iterations_per_level

        global_best_metric, global_best_image, global_best_motion, global_converged = self._initialize_global_tracking()

        for restart in range(params.max_restarts):
            restart_converged = False

            for idx_res, r in enumerate(ResLevels):
                Data_res = self._prepare_resolution_level(idx_res, r, restart)
                if idx_res != 0:
                    self.upsample_data(Data_prev, Data_res)
                residual_recon_norms, residual_motion_norms, best_relres, best_image, best_motion = self._initialize_level_tracking()
                s_res = Data_res["KspaceData"].flatten()

                # Gauss–Newton iterations
                for it in range(GN_iter):
                    print(f"  GN iteration {it+1}/{GN_iter}")

                    # -------------------------------IMAGE RECONSTRUCTION STEP -------------------------

                    # 1) Build motion and encoding operators
                    Data_res["MotionOperator"] = self.build_motion_operator(Data_res)
                    Data_res["E"] = self.build_encoding_operator(Data_res)

                    # 2) Solve for image
                    print("    Solving for image...")
                    img = self.solve_image(Data_res)
                    Data_res["ReconstructedImage"] = img

                    if idx_res == len(ResLevels) - 1 and it == GN_iter - 1:
                        break

                    # 3) Compute residual
                    x = img.flatten()
                    y = Data_res["E"].forward(x)
                    residual = s_res - y
                    res_norm = torch.linalg.norm(residual).item()
                    rel_res = res_norm / (torch.linalg.norm(s_res).item() + 1e-12)
                    residual_recon_norms.append(rel_res)

                    if it > 0 and rel_res > best_relres:
                        print("    Relative residual increased — restoring best solution at this level.")
                        break

                    # Residual improved: store current image/motion as level-best.
                    best_relres = rel_res
                    best_image = Data_res["ReconstructedImage"].clone()
                    best_motion = Data_res["MotionModel"].clone()

                    # ------------------------------- MOTION MODEL RECONSTRUCTION STEP -------------------------

                    # 4) Build Jacobian encoding operator for solving ∇_u(E)·δu = δkspace
                    Data_res["J"] = self.build_motion_perturbation_simulator(Data_res)

                    # 4) Solve for motion update
                    print("    Solving for motion update...")
                    if params.use_scaled_motion_update:
                        dm = self.solve_motion_scaled(Data_res, residual)
                    else:
                        dm = self.solve_motion(Data_res, residual)

                    Data_res["MotionModel"] += dm.real
                    dm_norm = torch.linalg.norm(dm.flatten()).item()
                    residual_motion_norms.append(dm_norm)

                    # ------------------------- LOGGING and PATIENCE CHECK -------------------------

                    if params.verbose:
                        log_motion_parameters(
                            Data_res["MotionModel"],
                            res_level=idx_res + 1,
                            gn_iter=it + 1
                        )

                if params.debug_flag: 
                    if best_image is not None and best_motion is not None:
                        Data_res["ReconstructedImage"] = best_image
                        Data_res["MotionModel"] = best_motion
                    show_and_save_image(Data_res["ReconstructedImage"][0], \
                        'image_restart_' + str(restart + 1) + '_resolution_level' + str(idx_res+1), params.debug_folder)
                    self._save_nonrigid_motion_debug(Data_res, restart + 1, idx_res + 1)
                if params.verbose:
                    save_residual_convergence(
                        residual_recon_norms,
                        'recon_' + str(restart + 1) + '_',
                        idx_res + 1,
                        params.logs_folder,
                    )
                    save_residual_convergence(
                        residual_motion_norms,
                        'motion_' + str(restart + 1) + '_',
                        idx_res + 1,
                        params.logs_folder,
                    )

                if restart_converged:
                    break
                Data_prev = Data_res
            if best_image is not None and best_motion is not None:
                global_best_metric, global_best_image, global_best_motion = self._update_global_best(
                    best_relres, best_image, best_motion, global_best_metric, global_best_image, global_best_motion
                )

            if restart_converged:
                global_converged = True
                print("Stopping restarts: true convergence achieved.")
                break

        if not global_converged:
            print("⚠ WARNING: No restart reached tolerance.")

        global_best_image_unscaled = global_best_image * self.kspace_scale
        show_and_save_image(global_best_image_unscaled[0], 'reconstructed_image', params.results_folder)
        torch.save(global_best_motion, f"{params.results_folder}motion_model.pt")
        self._save_final_nonrigid_alpha_maps(global_best_motion, global_best_image_unscaled[0])

        return global_best_image_unscaled, global_best_motion

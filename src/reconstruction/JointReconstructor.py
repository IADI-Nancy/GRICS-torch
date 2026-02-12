import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.utils.show_and_save_image import show_and_save_image

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

        JHJ[:, i] = JHJe.real  # JHJ should be real symmetric

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

def save_residual_convergence(residual_norms, title, res_level):
    plt.figure()
    plt.plot(residual_norms, marker="o")
    plt.xlabel("GN iteration")
    plt.ylabel("||residual||₂")
    plt.title(f"Residual convergence ({title}, resolution level {res_level})")
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(bottom=0)
    plt.savefig(f"{params.logs_folder}residual_convergence_{title}_res{res_level}.png")
    plt.close()

    # Optional: save raw values
    with open(f"{params.logs_folder}residual_convergence_{title}_res{res_level}.txt", "w") as f:
        for i, v in enumerate(residual_norms):
            f.write(f"{i+1}\t{v}\n")



# --------------------------------------------------------------------------
# Class that performs joint image–motion reconstruction
# --------------------------------------------------------------------------
class JointReconstructor:

    def __init__(self, KspaceData, smaps, SamplingIndices):
        Ncoils, Nx_full, Ny_full, Nsli = smaps.shape

        # Parameters constant for all resolutions        
        self.Ncoils = Ncoils
        self.device = KspaceData.device
        self.Nalpha = 3  # number of motion parameters (t_x, t_y, phi)

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
        Data_res["MotionModel"] = torch.zeros((self.Nalpha, params.N_mot_states), device=self.device)
        Data_res["MotionModel"][0,:] = mot_prev[0,:] * Data_res["Nx"] / Data_prev["Nx"]  # scale translations
        Data_res["MotionModel"][1,:] = mot_prev[1,:] * Data_res["Ny"] / Data_prev["Ny"]  # scale translations
        Data_res["MotionModel"][2,:] = mot_prev[2,:]  # rotations remain the same

    # ----------------------------------------------------------------------
    # Build Ux, Uy fields and Motion Operators
    # ----------------------------------------------------------------------
    def build_motion_operator(self, Data_res):
        Nx, Ny = Data_res["Nx"], Data_res["Ny"]
        alpha = Data_res["MotionModel"]
        motionOperator = MotionOperator(Nx, Ny, alpha)
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

    # ----------------------------------------------------------------------
    # Solve linear system for image
    # ----------------------------------------------------------------------
    def solve_image(self, Data_res):
        x0 = Data_res["ReconstructedImage"]
        x0 = x0.to(self.device)
        E = Data_res["E"]

        b = E.adjoint(Data_res["KspaceData"])
        solver = ConjugateGradientSolver(E, reg_lambda=params.lambda_r, verbose=True)

        img_vec = solver.solve_cg(
            b.flatten(),
            x0=x0.flatten(),
            max_iter=params.max_iter_recon,
            tol=params.tol_recon,
        )

        img = img_vec.reshape(params.Nex, Data_res["Nx"], Data_res["Ny"])
        return img

    # # ----------------------------------------------------------------------
    # # Solve for motion model update
    # # ----------------------------------------------------------------------
    def solve_motion(self, Data_res, residual):
        x0 = torch.zeros(self.Nalpha * params.N_mot_states, dtype=torch.float32, device=residual.device)
        J = Data_res["J"]
        b = J.adjoint(residual)
        
        solver = ConjugateGradientSolver(J, reg_lambda=params.lambda_m, verbose=True)

        mot_pert_vec = solver.solve_cg_keep_best(
            b.flatten(),
            x0=x0.flatten(),
            max_iter=params.max_iter_motion,
            tol=params.tol_motion,
        )

        motion_perturb = mot_pert_vec.reshape(self.Nalpha, params.N_mot_states)
        return motion_perturb
    
    def solve_motion_scaled(self, Data_res, residual):
        Nparams = self.Nalpha * params.N_mot_states
        with torch.no_grad():
            diag_JHJ = torch.zeros(Nparams, device=self.device)

            for i in range(Nparams):
                e = torch.zeros(Nparams, device=self.device)
                e[i] = 1.0
                Je = Data_res["J"].forward(e)
                JHJe = Data_res["J"].adjoint(Je)
                diag_JHJ[i] = torch.real(JHJe[i])

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
        x0 = torch.zeros(Nparams, dtype=torch.float32, device=device)

        # ---- Solve in scaled variables ----
        solver = ConjugateGradientSolver(
            J_scaled,
            reg_lambda=params.lambda_m,
            verbose=True
        )

        delta_tilde = solver.solve_cg_keep_best(
            b_scaled.flatten(),
            x0=x0,
            max_iter=params.max_iter_motion,
            tol=params.tol_motion,
        )

        # ---- Unscale back to physical parameters ----
        delta = delta_tilde / scale

        motion_perturb = delta.reshape(self.Nalpha, params.N_mot_states)

        return motion_perturb

    # ----------------------------------------------------------------------
    # Perform full multi-resolution Gauss–Newton joint reconstruction
    # ----------------------------------------------------------------------
    def run(self):
        ResLevels = params.ResolutionLevels
        GN_iter = params.GN_iterations_per_level

        for idx_res, r in enumerate(ResLevels):
            if params.verbose:
                fname = f"{params.logs_folder}motion_params_res{idx_res+1}.txt"
                open(fname, "w").close()
            print(f"\n=== Resolution level {idx_res+1}: factor {r} ===")

            # Prepare low-resolution dataset
            Data_res = self.downsample_data(r)

            # Initialize image and motion model
            if idx_res == 0:
                Data_res["ReconstructedImage"] = torch.zeros((params.Nex, Data_res["Nx"], Data_res["Ny"]), dtype=torch.complex64, device=self.device)
                Data_res["MotionModel"] = torch.zeros((self.Nalpha, params.N_mot_states), device=self.device)
            else:
                self.upsample_data(Data_prev, Data_res)

            residual_recon_norms = []
            residual_motion_norms = []

            # Gauss–Newton iterations
            for it in range(GN_iter):
                print(f"  GN iteration {it+1}/{GN_iter}")

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
                residual = Data_res["KspaceData"].flatten() - y
                res_norm = torch.linalg.norm(residual).item()
                residual_recon_norms.append(res_norm)

                # 4) Build Jacobian encoding operator for solving ∇_u(E)·δu = δkspace
                Data_res["J"] = self.build_motion_perturbation_simulator(Data_res)
                # S, V = test_J_singularity(Data_res["J"])

                # 4) Solve for motion update
                print("    Solving for motion update...")
                # dm = self.solve_motion(Data_res, residual)
                if params.use_scaled_motion_update:
                    dm = self.solve_motion_scaled(Data_res, residual)
                else:
                    dm = self.solve_motion(Data_res, residual)
                
                # dm = self.solve_motion_regularized(Data_res, residual)

                Data_res["MotionModel"] += dm.real

                dm_norm = torch.linalg.norm(dm.flatten()).item()
                residual_motion_norms.append(dm_norm)

                if params.verbose:
                    log_motion_parameters(
                        Data_res["MotionModel"],
                        res_level=idx_res + 1,
                        gn_iter=it + 1
                    )
                if it > 1 and (residual_recon_norms[-1] > residual_recon_norms[-2] or residual_motion_norms[-1] > residual_motion_norms[-2]):
                    print("    Residual increased — stopping GN at this resolution")
                    break

                Data_prev = Data_res

            if params.debug_flag: 
                show_and_save_image(Data_res["ReconstructedImage"][0], 'image_resolution_level_'+str(idx_res+1), params.debug_folder)
            if params.verbose:
                save_residual_convergence(residual_recon_norms, 'recon', idx_res + 1)
                save_residual_convergence(residual_motion_norms, 'motion', idx_res + 1)

        return Data_res["ReconstructedImage"], Data_res["MotionModel"]

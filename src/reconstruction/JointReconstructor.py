import torch
import matplotlib.pyplot as plt

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.utils.Helpers import resize_img_2D
from src.utils.show_slice import show_slice

def show_slice_and_save(image, image_name):
    show_slice(image, max_images=1, headline=image_name)
    plt.savefig('debug_outputs/' + image_name + '.png')

# --------------------------------------------------------------------------
# Class that performs joint image–motion reconstruction
# --------------------------------------------------------------------------
class JointReconstructor:

    def __init__(self, KspaceData, smaps, Nsamples, SamplingIndices, KspaceOffset, params):
        Nx_full, Ny_full, Nsli, Ncoils = smaps.shape

        # Parameters constant for all resolutions        
        self.Ncoils = Ncoils
        self.params = params
        self.device = KspaceData.device
        self.Nalpha = 3  # number of motion parameters (t_x, t_y, phi)

        # Data changing with resolution
        self.Data_full = {}
        self.Data_full["Nx"] = Nx_full
        self.Data_full["Ny"] = Ny_full
        self.Data_full["SensitivityMaps"] = smaps
        self.Data_full["KspaceData"] = KspaceData
        self.Data_full["Nsamples"] = Nsamples
        self.Data_full["SamplingIndices"] = SamplingIndices
        self.Data_full["KspaceOffset"] = KspaceOffset
        

    def downsample_sampling_indices(self, Sampling_full, Nx_res, Ny_res):
        Nx_full, Ny_full, = self.Data_full["Nx"], self.Data_full["Ny"]

        # central crop coordinates
        x0 = (Nx_full - Nx_res) // 2
        y0 = (Ny_full - Ny_res) // 2

        Sampling_res = []

        for indices in Sampling_full:
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

            Sampling_res.append(new_inds)

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
        kspace_reshaped = kspace_full.reshape(Nex, Nx_full, Ny_full, -1)
        kspace_res = kspace_reshaped[:, mask, :].reshape(-1, kspace_full.shape[-1])

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
        
        Data_res["SensitivityMaps"] = resize_img_2D(self.Data_full["SensitivityMaps"].squeeze(), (Nx, Ny)).unsqueeze(2)
        Data_res["SamplingIndices"] = self.downsample_sampling_indices(self.Data_full["SamplingIndices"], Nx, Ny)
        Data_res["KspaceData"] = self.downsample_kspace(Nx, Ny)
        Data_res["KspaceOffset"] = []
        for shot in range(len(Data_res["SamplingIndices"])):
            Data_res["KspaceOffset"].append(int(self.Data_full["KspaceOffset"][shot]/self.Data_full["Nx"]/self.Data_full["Ny"] * (Nx*Ny)))
        Data_res["Nsamples"] = Data_res["KspaceData"].shape[0]

        return Data_res
    
    def upsample_data(self, Data_prev, Data_res):
                img_prev = Data_prev["ReconstructedImage"]
                img_res = resize_img_2D(img_prev.unsqueeze(-1), (Data_res["Nx"], Data_res["Ny"])).squeeze(-1)
                Data_res["ReconstructedImage"] = img_res

                mot_prev = Data_prev["MotionModel"]
                Data_res["MotionModel"] = torch.zeros((self.Nalpha, self.params.N_mot_states), device=self.device)
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
            Data_res["KspaceOffset"],
            Data_res["MotionOperator"]
        )
        return E
    
    def build_motion_perturbation_simulator(self, Data_res):
        J = MotionPerturbationSimulator(
            Data_res["SensitivityMaps"],
            Data_res["Nsamples"],
            Data_res["SamplingIndices"],
            Data_res["KspaceOffset"],
            Data_res["ReconstructedImage"],
            Data_res["MotionOperator"]
        )
        return J

    # ----------------------------------------------------------------------
    # Solve linear system for image
    # ----------------------------------------------------------------------
    def solve_image(self, Data_res):
        x0 = Data_res["ReconstructedImage"].flatten()
        x0 = x0.to(self.device)
        E = Data_res["E"]

        b = E.adjoint(Data_res["KspaceData"])
        solver = ConjugateGradientSolver(E, reg_lambda=self.params.lambda_r, verbose=True)

        img_vec = solver.solve_cg(
            b.flatten(),
            x0=x0,
            max_iter=self.params.max_iter_recon,
            tol=self.params.tol_recon,
        )

        img = img_vec.reshape(Data_res["Nx"], Data_res["Ny"])
        return img

    # # ----------------------------------------------------------------------
    # # Solve for motion model update
    # # ----------------------------------------------------------------------
    def solve_motion(self, Data_res, residual):
        x0 = torch.zeros(self.Nalpha * self.params.N_mot_states, dtype=torch.float32, device=residual.device)
        J = Data_res["J"]
        b = J.adjoint(residual)
        
        solver = ConjugateGradientSolver(J, reg_lambda=self.params.lambda_m, verbose=True)

        mot_pert_vec = solver.solve_cg_keep_best(
            b.flatten(),
            x0=x0,
            max_iter=self.params.max_iter_motion,
            tol=self.params.tol_motion,
        )

        motion_perturb = mot_pert_vec.reshape(self.Nalpha, self.params.N_mot_states)
        return motion_perturb


    # ----------------------------------------------------------------------
    # Perform full multi-resolution Gauss–Newton joint reconstruction
    # ----------------------------------------------------------------------
    def run(self):
        ResLevels = self.params.ResolutionLevels
        GN_iter = self.params.GN_iterations_per_level

        for idx_res, r in enumerate(ResLevels):
            print(f"\n=== Resolution level {idx_res+1}: factor {r} ===")

            # Prepare low-resolution dataset
            Data_res = self.downsample_data(r)

            # Initialize image and motion model
            if idx_res == 0:
                Data_res["ReconstructedImage"] = torch.zeros((Data_res["Nx"], Data_res["Ny"]), dtype=torch.complex64, device=self.device)
                Data_res["MotionModel"] = torch.zeros((self.Nalpha, self.params.N_mot_states), device=self.device)
            else:
                self.upsample_data(Data_prev, Data_res)

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

                # 4) Build Jacobian encoding operator for solving ∇_u(E)·δu = δkspace
                Data_res["J"] = self.build_motion_perturbation_simulator(Data_res)

                # 4) Solve for motion update
                print("    Solving for motion update...")
                dm = self.solve_motion(Data_res, residual)
                Data_res["MotionModel"] += dm

                Data_prev = Data_res

            show_slice_and_save(Data_res["ReconstructedImage"].unsqueeze(-1), 'image_name_resolution_level_'+str(idx_res+1))   

        return Data_res["ReconstructedImage"], Data_res["MotionModel"]

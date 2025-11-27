import torch

from iadi.ReconstructionSolver import ReconstructionSolver
from iadi.MotionOperator import MotionOperator
from iadi.EncodingOperator import EncodingOperator
from iadi.Helpers import resize_img_2D

from utils.show_slice import show_slice
import matplotlib.pyplot as plt

def show_slice_and_save(image, image_name):
    show_slice(image, max_images=1, headline=image_name)
    plt.savefig('debug_outputs/' + image_name + '.png')


# --------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------
import torch
import torch.nn.functional as F

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

    # ----------------------------------------------------------------------
    # Build Ux, Uy fields and Motion Operators
    # ----------------------------------------------------------------------
    def build_motion_operators(self, Data_res):
        Nx, Ny = Data_res["Nx"], Data_res["Ny"]
        # Mflag = Data_res["motion_type_flag"]

        Data_res["MotionOperator"] = []

        for shot in range(self.params.Nshots):

            # if Mflag == 0:   # translations only
            tx = Data_res["MotionModel"][0, shot]
            ty = Data_res["MotionModel"][1, shot]
            Ux = torch.full((Nx,Ny), tx, device=self.device)
            Uy = torch.full((Nx,Ny), ty, device=self.device)

            # elif Mflag == 2:   # constrained nonrigid
            #     S = self.Data.XTranslationVector[shot]
            #     Ux = Data_res["MotionModel"][:,:,0] * S
            #     Uy = Data_res["MotionModel"][:,:,1] * S

            # else:
            #     raise NotImplementedError("Only motion types 0 and 2 supported")

            # user-supplied function:
            MotionOp = MotionOperator.create_sparse_motion_operator(Ux, Uy)
            Data_res["MotionOperator"].append(MotionOp)

        return Data_res

    # ----------------------------------------------------------------------
    # Solve linear system for image
    # ----------------------------------------------------------------------
    def solve_image(self, Data_res):
        x0 = Data_res["ReconstructedImage"].flatten()
        x0 = x0.to(self.device)

        E = EncodingOperator(
            Data_res["SensitivityMaps"],
            Data_res["Nsamples"],
            Data_res["SamplingIndices"],
            Data_res["KspaceOffset"],
            Data_res["MotionOperator"],
        )
        b = E.adjoint(Data_res["KspaceData"])
        solver = ReconstructionSolver(E, reg_lambda=self.params.lambda_r, verbose=True)

        img_vec = solver.solve_cg(
            b.flatten(),
            x0=x0,
            max_iter=self.params.max_iter_recon,
            tol=self.params.tol,
        )

        img = img_vec.reshape(Data_res["Nx"], Data_res["Ny"])
        return img

    # # ----------------------------------------------------------------------
    # # Solve for motion model update
    # # ----------------------------------------------------------------------
    # def solve_motion(self, Data_res, residual):

    #     b = motion_perturbation_simulator_transpose(residual, Data_res)
    #     mu_scaled = self.Params.mu * torch.linalg.norm(b)

    #     flag = Data_res["motion_type_flag"]

    #     if flag == 0:
    #         A = lambda x: (
    #             motion_perturbation_simulator(x, Data_res)
    #             .to(self.device)
    #             + mu_scaled * x
    #         )

    #     elif flag == 2:
    #         # add Laplacian regularizer G'G
    #         def laplace_maps(vec):
    #             Nx, Ny = Data_res["Nx"], Data_res["Ny"]
    #             v = vec.reshape(Nx, Ny, 2)
    #             Lx = -torch.linalg.laplacian(v[:,:,0])
    #             Ly = -torch.linalg.laplacian(v[:,:,1])
    #             return torch.stack([Lx, Ly], dim=-1).reshape(-1)

    #         A = lambda x: (
    #             motion_perturbation_simulator(x, Data_res)
    #             + mu_scaled * laplace_maps(x)
    #         )

    #     solver = ReconstructionSolver(encoding_operator=None)
    #     dm = solver.cg(b, max_iter=128, tol=1e-3, M=None)

    #     if flag == 0:
    #         return dm.reshape(2, Data_res["Nshots"])

    #     elif flag == 2:
    #         return dm.reshape(Data_res["Nx"], Data_res["Ny"], 2)





    # ----------------------------------------------------------------------
    # Perform full multi-resolution Gauss–Newton joint reconstruction
    # ----------------------------------------------------------------------
    def run(self):
        ResLevels = self.params.ResolutionLevels
        GN_iter = self.params.GN_iterations_per_level

        # Data_res = None

        for idx_res, r in enumerate(ResLevels):

            print(f"\n=== Resolution level {idx_res+1}: factor {r} ===")

            # ---------------------------------------------------------
            # Prepare low-resolution dataset
            # ---------------------------------------------------------
            Data_res = self.downsample_data(r)

            # Initialize image and motion model
            # if idx_res == 0:
            Data_res["ReconstructedImage"] = torch.zeros(
                (Data_res["Nx"], Data_res["Ny"]),
                dtype=torch.complex64, device=self.device)

            # if Data_res["motion_type_flag"] == 0:
            Data_res["MotionModel"] = torch.zeros((2, self.params.Nshots), device=self.device)

                # elif Data_res["motion_type_flag"] == 2:
                #     Data_res["MotionModel"] = torch.zeros(
                #         (Data_res["Nx"], Data_res["Ny"], 2),
                #         device=self.device)

            # ---------------------------------------------------------
            # Gauss–Newton iterations
            # ---------------------------------------------------------
            for it in range(GN_iter):
                print(f"  GN iteration {it+1}/{GN_iter}")

                # 1) Build motion operators
                self.build_motion_operators(Data_res)

                # 2) Solve for image
                img = self.solve_image(Data_res)
                Data_res["ReconstructedImage"] = img

                # 3) Compute residual
                # x_vec = img.flatten()
                # y = self.E_forward(x_vec, Data_res)
                # residual = Data_res["KspaceData"] - y

                # # 4) Solve for motion update
                # dm = self.solve_motion(Data_res, residual)
                # Data_res["MotionModel"] += dm
            show_slice_and_save(Data_res["ReconstructedImage"].unsqueeze(-1), 'image_name_resolution_level_'+str(idx_res+1))   

        return Data_res["ReconstructedImage"], Data_res["MotionModel"]

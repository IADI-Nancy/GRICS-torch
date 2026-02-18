import os
import torch

class Parameters:
    debug_flag = True
    verbose = True # whether to print detailed logs during reconstruction
    debug_folder = "debug_outputs/"
    logs_folder = "logs/"
    results_folder = "results/"
    Nex = 3 # number of excitations (repetitions) per k-space acquisition
    motion_type = 'non-rigid'  # 'rigid', 'non-rigid'
    N_mot_states = 4

    # Data loading/generation parameters
    data_type = 'shepp-logan'  # 'shepp-logan', 'fastMRI', 'real-world', 'raw-data'
    path_to_fastMRI_data = 'data/kspace.npz'
    path_to_realworld_data = 'data/breast_motion_data.h5'
    saec_file = 'data/2008-003 01-1724_S11_20210323_151329.h5'
    ismrmrd_file = 'data/t2_1724.h5'
    N_SheppLogan = 128
    Ncoils_SheppLogan = 4
    Nz_SheppLogan = 1

    # Sampling simulation parameters
    NshotsPerNex = 8    
    kspace_sampling_type = 'interleaved' # 'linear', 'interleaved' or 'random'

    # Motion simulation parameters
    simulation_type = 'discrete-non-rigid'  # 'discrete-rigid', 'rigid', 'discrete-non-rigid', 'no-motion' or 'as-it-is'
    num_motion_events = 4
    max_tx = 4.0  # maximum translation in x (pixels)
    max_ty = 3.0  # maximum translation in y (pixels)
    max_phi = 10.0  # maximum rotation (degrees)
    max_center_x = 0 # 60.0  # maximum variation in center x (pixels)
    max_center_y = 0 # 10.0  # maximum variation in center y (pixels)
    seed = 1
    motion_tau = 2  # transition width of motion events (in ky lines)
    nonrigid_motion_amplitude = 1.0
    displacementfield_size = 1

    # Espirits sensitivity map calculation parameters
    acs = 48
    kernel_width = 12

    # General reconstruction parameters
    max_restarts = 3
    use_scaled_motion_update = False  # whether to scale motion updates by the diagonal of J^H J
    ResolutionLevels = [0.25, 0.5, 1.0]  # multi-resolution levels (as fraction of full res)
    GN_iterations_per_level = 8
    patience = 3
    residual_metric_type = "motion"  # "recon", "motion" or "combined"
    motion_weight = 1.0             # used only if combined

    # Image reconstruction parameters
    lambda_r = 1e-3
    max_iter_recon = 128
    tol_recon = 1e-3
    cg_early_stopping = True  # MATLAB-like stagnation/more-steps stopping inside CG
    cg_max_stag_steps = 6     # lighter than MATLAB default (3)
    cg_max_more_steps = 0    # allow more post-stagnation recovery iterations
    cg_use_reg_scale_proxy = True   # scales regularizer by ||A_data v||/||R v|| proxy
    cg_reg_scale_num_probes = 8

    # Motion model parameters
    lambda_m = 1
    max_iter_motion = 128
    tol_motion = 1e-3

    def __init__(self):
        # Enforce double precision globally for tensors created without explicit dtype.
        torch.set_default_dtype(torch.float64)

        if self.debug_flag and not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)
        self.Nshots = self.NshotsPerNex * self.Nex
        if self.simulation_type == 'discrete-rigid':
            self.N_mot_states = self.Nshots  # each shot is a separate motion state
        elif self.simulation_type in ['rigid']:
            self.N_mot_states = self.num_motion_events + 1
        elif self.simulation_type in ['discrete-non-rigid']:
            self.N_mot_states = self.Nshots
        elif self.simulation_type in ['no-motion']:
            self.N_mot_states = 1
        
        if self.motion_type == 'non-rigid':
            self.max_restarts = 1  # random and non-smooth motion is hard to optimize, so we do only one reconstruction run without restarts
        
        os.makedirs(self.debug_folder, exist_ok=True)
        os.makedirs(self.logs_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)

import os

class Parameters:
    debug_flag = True
    debug_folder = "debug_outputs/"
    debug_convergence_folder = "debug_outputs/convergence/"

    # Data loading/generation parameters
    data_type = 'fastMRI'  # 'shepp-logan', 'fastMRI', 'real-world', 'raw-data'
    path_to_fastMRI_data = 'data/kspace.npz'
    path_to_realworld_data = 'data/breast_motion_data.h5'
    saec_file = 'data/2008-003 01-1724_S11_20210323_151329.h5'
    ismrmrd_file = 'data/t2_1724.h5'
    N_SheppLogan = 128
    Ncoils_SheppLogan = 16
    Nz_SheppLogan = 1

    # Sampling simulation parameters
    NshotsPerNex = 4
    Nex = 1 # TODO : add multiple excitations support
    kspace_sampling_type = 'interleaved' # 'linear' or 'interleaved'

    # Motion simulation parameters
    simulation_type = 'discrete-rigid'  # 'discrete-rigid', 'rigid', 'non-rigid', 'no-motion' or 'as-it-is'
    num_motion_events = 4
    max_tx = 4.0  # maximum translation in x (pixels)
    max_ty = 3.0  # maximum translation in y (pixels)
    max_phi = 10.0  # maximum rotation (degrees)
    max_center_x = 60.0  # maximum variation in center x (pixels)
    max_center_y = 10.0  # maximum variation in center y (pixels)
    seed = 1
    motion_tau = 2  # transition width of motion events (in ky lines)

    # Espirits sensitivity map calculation parameters
    acs = 48
    kernel_width = 12

    # General reconstruction parameters
    ResolutionLevels = [0.25, 0.5, 1.0]  # multi-resolution levels (as fraction of full res)
    GN_iterations_per_level = 16

    # Image reconstruction parameters
    lambda_r = 2e-3
    max_iter_recon = 20
    tol_recon = 1e-3

    # Motion model parameters
    lambda_m = 1e-3
    max_iter_motion = 20
    tol_motion = 1e-3

    def __init__(self):
        if self.debug_flag and not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)
        self.Nshots = self.NshotsPerNex * self.Nex
        if self.simulation_type == 'discrete-rigid':
            self.N_mot_states = self.Nshots  # each shot is a separate motion state
        elif self.simulation_type in ['rigid', 'non-rigid', 'as-it-is']:
            self.N_mot_states = self.num_motion_events + 1
        elif self.simulation_type in ['no-motion']:
            self.N_mot_states = 1

        
        os.makedirs("debug_outputs", exist_ok=True)
        os.makedirs("debug_outputs/convergence", exist_ok=True)

import os

class Parameters:
    debug_flag = True
    debug_folder = "debug_outputs/"

    # Initial data parameters
    data_type = 'fastMRI'  # 'shepp-logan', 'fastMRI' or 'real-world'
    path_to_data = 'data/kspace.npz'  # used if data_type is 'fastMRI' or 'real-world'
    # path_to_data = 'data/breast_motion_data.h5'
    N_SheppLogan = 128
    Ncoils_SheppLogan = 16
    Nz_SheppLogan = 1

    # Simulation parameters
    simulation_type = 'discrete-rigid'  # 'discrete-rigid', 'rigid', 'non-rigid' or 'none'

    # Sampling simulation parameters
    NshotsPerNex = 4
    Nex = 1 # TODO : add multiple excitations support
    kspace_sampling_type = 'interleaved' # 'linear' or 'interleaved'

    # Motion simulation parameters
    num_motion_events = 4
    max_tx = 4.0  # maximum translation in x (pixels)
    max_ty = 3.0  # maximum translation in y (pixels)
    max_phi = 10.0  # maximum rotation (degrees)
    max_center_x = 60.0  # maximum variation in center x (pixels)
    max_center_y = 10.0  # maximum variation in center y (pixels)
    seed = 3
    motion_tau = 2  # transition width of motion events (in ky lines)

    # Espirits parameters
    acs = 48
    kernel_width = 12

    # Reconstruction parameters
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
        self.N_mot_states = self.num_motion_events + 1
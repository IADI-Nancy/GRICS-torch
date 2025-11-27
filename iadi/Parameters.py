import os

class Parameters:
    debug_flag = True
    debug_folder = "debug_outputs/"

    # Sampling simulation parameters
    NshotsPerNex = 8
    Nex = 2
    kspace_sampling_type = 'interleaved' # can be also 'linear'

    # Motion simulation parameters
    max_motion = 4 * 3/2  # ~~ in pixels# in cm
    seed = 3

    # Espirits parameters
    acs = 48
    kernel_width = 12

    # Reconstruction parameters
    ResolutionLevels = [0.25, 0.5, 1.0]  # multi-resolution levels (as fraction of full res)
    GN_iterations_per_level = 2 #8
    # the regularization term
    lambda_r = 1e-3
    max_iter_recon = 20
    tol = 1e-3

    def __init__(self):
        if self.debug_flag and not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)
        self.Nshots = self.NshotsPerNex * self.Nex
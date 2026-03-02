import numpy as np
import torch
import os

from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.preprocessing.SamplingSimulator import SamplingSimulator
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions
from src.utils.save_alpha_component_map import save_alpha_component_map
from src.utils.save_nonrigid_quiver_with_contours import save_nonrigid_quiver_with_contours
from src.utils.save_motion_debug_plots import save_motion_debug_plots
from src.utils.nonrigid_display import to_cartesian_components

class MotionSimulator:
    def __init__(self, image, smaps, ky_idx, nex_idx, ky_per_motion_state, params, sp_device=None, t_device=None):
        self.image = image
        self.smaps = smaps
        self.ky_idx = ky_idx
        self.nex_idx = nex_idx
        self.ky_per_motion_state = ky_per_motion_state
        self.params = params
        self.sp_device = sp_device
        self.t_device = t_device
        self.Ncha, self.Nx, self.Ny, self.Nsli = smaps.shape  
    
    def get_corrupted_kspace(self):
        return self.kspace
    
    def get_corrupted_image(self):
        return self.image_no_moco

    def get_motion_information(self):
        return self.navigator, self.tx, self.ty, self.phi

    def _build_sampling_per_line_global_states(self):
        """
        Build sampling indices with one global motion state per acquired ky line.
        Output shape is [Nex][Ny_total], where Ny_total = Nex * Ny.
        """
        ky_flat = torch.cat([k.reshape(-1) for k in self.ky_idx], dim=0)
        nex_flat = torch.cat([n.reshape(-1) for n in self.nex_idx], dim=0).to(torch.int64)
        ny_total = int(ky_flat.numel())
        nex_total = int(self.params.Nex)

        sampling = [
            [torch.empty(0, dtype=torch.int64, device=self.t_device) for _ in range(ny_total)]
            for _ in range(nex_total)
        ]
        kx = torch.arange(self.Nx, device=self.t_device, dtype=torch.int64)

        for state in range(ny_total):
            nex = int(nex_flat[state].item())
            ky = ky_flat[state].to(torch.int64).reshape(1)
            samp = (ky[:, None] + self.Ny * kx[None, :]).reshape(-1)
            sampling[nex][state] = samp

        return sampling

    def _apply_motion(self, alpha, centers=None, motion_signal=None, motion_type=None):
        if motion_type is None:
            motion_type = self.params.motion_type

        self.MotionOperator = MotionOperator(
            self.Nx, self.Ny, alpha, motion_type, centers=centers, motion_signal=motion_signal
        )

        E = EncodingOperator(
            self.smaps,
            self.TotalKspaceSamples,
            self.sampling_idx,
            self.params.Nex,
            self.MotionOperator
        )
        kspace_corruped = E.forward(self.image)
        self.kspace = kspace_corruped.reshape(self.Ncha, self.params.Nex, self.Nx, self.Ny, self.Nsli)

        img_cplx = ifftnc(self.kspace, dims=(-3, -2, -1)).to(self.t_device)
        self.image_no_moco = torch.sum(img_cplx * self.smaps.conj().unsqueeze(1).expand(-1, self.params.Nex, -1, -1, -1), dim=0)

    # -------------------------------------------------------
    #------------------ Simulate zero motion ----------------
    # -------------------------------------------------------
    
    def simulate_no_motion(self):
        """
        Simulate acquisition with NO motion:
        tx = ty = phi = 0 everywhere
        navigator = 0
        """
        # Single motion state per Nex; retain full acquired ky set for each Nex.
        if isinstance(self.ky_idx, list):
            ky_per_mot_state_idx = [[ky] for ky in self.ky_idx]
            ny_total = sum(ky.numel() for ky in self.ky_idx)
        else:
            ky_per_mot_state_idx = [[self.ky_idx]]
            ny_total = int(self.ky_idx.numel())

        self.sampling_idx_per_nex = SamplingSimulator.build_sampling_per_nex_per_motion(ky_per_mot_state_idx, self.Nx, self.Ny, self.t_device)
        
        self.TotalKspaceSamples = self.Nx * self.Ny

        # Expand zero motion to ky (chronological)
        self.tx        = torch.zeros(ny_total, device=self.t_device)
        self.ty        = torch.zeros(ny_total, device=self.t_device)
        self.phi       = torch.zeros(ny_total, device=self.t_device)
        self.navigator = torch.zeros(ny_total, device=self.t_device)


    # -------------------------------------------------------
    #---------- Realistic motion curves generation ----------
    # -------------------------------------------------------
    
    def _create_realistic_motion_curves(self):
        # Time axis: one value per k-space line (Ny)
        t = torch.arange(self.Ny*self.params.Nex, device=self.t_device, dtype=torch.float64)

        # 1) Generate random motion event times over the full acquisition.
        total_lines = self.Ny * self.params.Nex
        event_times = torch.sort(
            torch.randint(0, total_lines, (self.params.num_motion_events,), device=self.t_device)
        ).values

        # Motion transition sharpness (smaller tau -> faster motion)
        tau = self.params.motion_tau

        # 2) Generate random amplitudes for each event
        A_tx  = self.params.max_tx  * (2 * torch.rand(self.params.num_motion_events, device=self.t_device) - 1)
        A_ty  = self.params.max_ty  * (2 * torch.rand(self.params.num_motion_events, device=self.t_device) - 1)
        A_phi = self.params.max_phi * (2 * torch.rand(self.params.num_motion_events, device=self.t_device) - 1) * (torch.pi/180)

        # 3) Build independent motion curves
        tx  = torch.zeros(self.Ny*self.params.Nex, device=self.t_device)
        ty  = torch.zeros(self.Ny*self.params.Nex, device=self.t_device)
        phi = torch.zeros(self.Ny*self.params.Nex, device=self.t_device)

        for i, ti in enumerate(event_times):
            ti = event_times[i].item()
            t_end = min(ti + tau, self.Ny*self.params.Nex)
            # Normalized time for the transition
            alpha = np.linspace(0.0, 1.0, t_end - ti)
            # Raised cosine ramp (finite, smooth)
            s = 0.5 * (1.0 - np.cos(np.pi * alpha))

            f = torch.zeros(self.Ny*self.params.Nex, device=self.t_device)
            f[ti:t_end] = torch.from_numpy(s).to(self.t_device)
            if t_end < self.Ny*self.params.Nex:
                f[t_end:] = 1.0

            # Add motion contribution (same logic as tanh version)
            tx  += A_tx[i]  * f
            ty  += A_ty[i]  * f
            phi += A_phi[i] * f

        # 4) Compute PCA to obtain a single 1D motion curve (simulation of MRI navigators)
        M = torch.stack([tx, ty, phi], dim=0)
        # Mean-center each row (parameter) before PCA
        M_centered = M - M.mean(dim=1, keepdim=True)
        # Perform SVD (PCA)
        # U: 3x3, S: singular values, Vh: NyxNy
        U, _, _ = torch.linalg.svd(M_centered, full_matrices=False)
        # First principal component direction (in parameter space)
        u1 = U[:, 0]
        # Project the centered data onto the first PC
        # score1 shape: (1 x Ny)
        score1 = torch.matmul(u1.unsqueeze(0), M_centered)
        # Convert to 1D curve
        navigator = score1.squeeze(0)
        # Normalize navigator to [-1, 1] for stability
        navigator = navigator / navigator.abs().max()
        # save debug plots
        if self.params.debug_flag:
            save_motion_debug_plots(navigator, tx, ty, phi, self.params.debug_folder, event_times)
        # Return the motion curve, parameter curves, and event times
        return navigator, tx, ty, phi

    def simulate_realistic_rigid_motion(self):
        # One global motion state per acquired ky line (Ny * Nex states).
        self.sampling_idx = self._build_sampling_per_line_global_states()

        self.TotalKspaceSamples = self.Ny * self.Nx
        # generate motion curves and parameters
        self.navigator, self.tx, self.ty, self.phi = self._create_realistic_motion_curves()

        # Apply simulated motion
        alpha = torch.zeros(3, self.Ny * self.params.Nex, device=self.t_device)
        alpha[0, :] = self.tx
        alpha[1, :] = self.ty
        alpha[2, :] = self.phi

        centers = torch.zeros((2, self.Ny * self.params.Nex), device=self.t_device)
        centers[0, :] = self.Nx / 2 + self.params.max_center_x * torch.ones(self.Ny * self.params.Nex, device=self.t_device)
        centers[1, :] = self.Ny / 2 + self.params.max_center_y * torch.randn(self.Ny * self.params.Nex, device=self.t_device)

        self._apply_motion(alpha, centers)

    # -------------------------------------------------------
    # ---- Generation of discrtete motion states per shot ---
    # -------------------------------------------------------

    def _expand_motion_to_ky(self, ky_per_mot_state_idx):
        """
        Expand per-motion-state parameters into per-ky vectors
        in chronological acquisition order.
        """

        # Total number of ky lines
        # Ny_total = sum(ky.numel() for ky in ky_per_mot_state_idx)
        Ny_total = 0
        for ky_list in ky_per_mot_state_idx:
            for ky in ky_list:
                Ny_total += ky.numel()

        # Allocate
        self.tx        = torch.empty(Ny_total, device=self.t_device)
        self.ty        = torch.empty(Ny_total, device=self.t_device)
        self.phi       = torch.empty(Ny_total, device=self.t_device)
        self.navigator = torch.empty(Ny_total, device=self.t_device)

        ptr = 0  # write pointer (chronological)

        m = 0  # motion state index
        for ky_list in ky_per_mot_state_idx:      # loop over Nex
            for ky in ky_list:                    # loop over shots
                n = ky.numel()

                self.tx[ptr:ptr+n]        = self.tx_mot_state[m]
                self.ty[ptr:ptr+n]        = self.ty_mot_state[m]
                self.phi[ptr:ptr+n]       = self.phi_mot_state[m]
                self.navigator[ptr:ptr+n] = self.navigator_mot_state[m]

                ptr += n
                m += 1

    def _create_discrete_motion_curves(self, ky_per_mot_state_idx):
        Nshots = self.params.Nshots

        alpha = torch.zeros((3, Nshots), device=self.t_device)

        # Translation X: [-max_tx, +max_tx]
        self.tx_mot_state = self.params.max_tx * (2 * torch.rand(Nshots, device=self.t_device) - 1)
        alpha[0, :] = self.tx_mot_state

        # Translation Y: [-max_ty, +max_ty]
        self.ty_mot_state = self.params.max_ty * (2 * torch.rand(Nshots, device=self.t_device) - 1)
        alpha[1, :] = self.ty_mot_state

        # Rotation: [-max_phi, +max_phi] degrees → radians
        self.phi_mot_state = self.params.max_phi * (2 * torch.rand(Nshots, device=self.t_device) - 1) * (torch.pi / 180)
        alpha[2, :] = self.phi_mot_state

        # Centers
        centers = torch.zeros((2, Nshots), device=self.t_device)
        centers[0, :] = self.Nx / 2 + self.params.max_center_x * torch.ones(Nshots, device=self.t_device)
        centers[1, :] = self.Ny / 2 + self.params.max_center_y * torch.ones(Nshots, device=self.t_device) # torch.rand(Nshots, device=self.t_device)

        # -------------------------------------------------
        # Build navigator = first principal component
        # -------------------------------------------------
        # Stack motion parameters: (Nshots, 3)
        motion_mat = torch.stack([self.tx_mot_state, self.ty_mot_state, self.phi_mot_state], dim=1)  # (Nshots, 3)

        # Center the data
        motion_mat = motion_mat - motion_mat.mean(dim=0, keepdim=True)

        # PCA via SVD
        # motion_mat = U S V^T
        _, _, Vh = torch.linalg.svd(motion_mat, full_matrices=False)

        # First principal component direction
        pc1 = Vh[0]  # (3,)

        # Project motion onto PC1 → navigator
        self.navigator_mot_state = motion_mat @ pc1  # (Nshots,)

        self._expand_motion_to_ky(ky_per_mot_state_idx)
        # save debug plots
        if self.params.debug_flag:
            save_motion_debug_plots(self.navigator, self.tx, self.ty, self.phi, self.params.debug_folder)

        return self.navigator, alpha, centers

    def simulate_discrete_rigid_motion(self):
        # Each shot is its own motion state
        ky_per_mot_state_idx = self.ky_per_motion_state

        # self.sampling_idx = \
        #     build_sampling_from_motion_states(ky_per_mot_state_idx, self.ky_idx, self.nex_idx, self.Nx, self.Ny, self.t_device)
        self.sampling_idx = SamplingSimulator.build_sampling_per_nex_per_motion(ky_per_mot_state_idx, self.Nx, self.Ny, self.t_device) # ← for debugging only, ignore output
        
        self.TotalKspaceSamples = self.Ny * self.Nx

        self.navigator, alpha, centers = self._create_discrete_motion_curves(ky_per_mot_state_idx)

        self._apply_motion(alpha, centers)


    

    def _create_discrete_non_rigid_alpha_fields(self):
        spatial_model = self.params.nonrigid_spatial_model
        if spatial_model == "gaussian":
            alpha_x, alpha_y, alpha_maps = self._create_gaussian_non_rigid_alpha_fields()
        elif spatial_model == "respiratory":
            alpha_x, alpha_y, alpha_maps = self._create_respiratory_non_rigid_alpha_fields()
        else:
            raise ValueError(
                f"Unknown nonrigid_spatial_model: {spatial_model}. "
                "Supported: 'respiratory', 'gaussian'."
            )

        # Keep synthetic motion region consistent with display orientation:
        # if the image is vertically flipped for display, flip motion maps too.
        if self.params.flip_for_display:
            alpha_x = torch.flip(alpha_x, dims=[0])
            alpha_y = torch.flip(alpha_y, dims=[0])
            alpha_maps = torch.stack([alpha_x, alpha_y], dim=0)

        return alpha_x, alpha_y, alpha_maps

    def _create_gaussian_non_rigid_alpha_fields(self):
        # Legacy Gaussian field kept as an optional fallback.
        x = torch.arange(1, self.Nx + 1, device=self.t_device, dtype=torch.float64)
        y = torch.arange(1, self.Ny + 1, device=self.t_device, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        mu_xx = self.Nx / 2.0
        mu_yy = self.Ny / 2.0
        mu_yx = self.Nx / 2.0
        mu_xy = self.Ny / 2.0

        # MATLAB uses sigma = N/4.
        sigma_x = self.Nx / 4.0
        sigma_y = self.Ny / 4.0

        alpha_x = (
            torch.exp(- (X - mu_xx) ** 2 / (2 * sigma_x ** 2))
            * torch.exp(- (Y - mu_xy) ** 2 / (2 * sigma_y ** 2))
        )
        alpha_y = (
            2 * torch.exp(- (X - mu_yx) ** 2 / (2 * sigma_x ** 2))
            * torch.exp(- (Y - mu_yy) ** 2 / (2 * sigma_y ** 2))
        )
        alpha_maps = torch.stack([alpha_x, alpha_y], dim=0)
        return alpha_x, alpha_y, alpha_maps

    def _create_respiratory_non_rigid_alpha_fields(self):
        # Respiration-like 2D field:
        # - dominant superior-inferior displacement near inferior (diaphragm-like) region
        # - weaker left-right component with opposite direction across the midline
        x = torch.linspace(-1.0, 1.0, self.Nx, device=self.t_device, dtype=torch.float64)
        y = torch.linspace(-1.0, 1.0, self.Ny, device=self.t_device, dtype=torch.float64)
        SI, LR = torch.meshgrid(x, y, indexing="ij")

        diaphragm_level = float(self.params.nonrigid_diaphragm_level)
        diaphragm_sharpness = float(self.params.nonrigid_diaphragm_sharpness)
        lateral_sigma = float(self.params.nonrigid_lateral_sigma)
        ap_fraction = float(self.params.nonrigid_ap_fraction)
        inferior_gain = float(self.params.nonrigid_inferior_gain)
        top_decay = float(self.params.nonrigid_top_decay)

        # Smooth mask that activates motion predominantly in the upper anatomy.
        region_mask = torch.sigmoid((-SI - diaphragm_level) * diaphragm_sharpness)
        # Concentrate motion near the central body region laterally.
        lateral_envelope = torch.exp(-0.5 * (LR / max(lateral_sigma, 1e-6)) ** 2)

        # Convention in this codebase:
        # alpha[0] -> Ux -> displacement along axis 0 (rows, SI-like direction)
        # alpha[1] -> Uy -> displacement along axis 1 (cols, LR-like direction)
        # We keep SI in alpha_x and use a negative sign so simulated respiratory
        # motion direction is inverted (as requested) without changing display logic.
        top_coord = torch.clamp(-SI, min=0.0)
        top_taper = torch.clamp(1.0 - top_decay * top_coord, min=0.05)
        region_gain = top_taper * (1.0 + inferior_gain * torch.clamp(-SI, min=0.0))
        si_profile = region_mask * lateral_envelope * region_gain
        si_profile = si_profile / torch.clamp(torch.max(torch.abs(si_profile)), min=1e-12)
        alpha_x = -si_profile

        # Weaker LR displacement: opposite directions on left/right sides.
        lr_profile = -LR * region_mask * lateral_envelope
        lr_profile = lr_profile / torch.clamp(torch.max(torch.abs(lr_profile)), min=1e-12)
        alpha_y = ap_fraction * lr_profile

        alpha_maps = torch.stack([alpha_x, alpha_y], dim=0)
        return alpha_x, alpha_y, alpha_maps

    def simulate_discrete_non_rigid_motion(self):
        print("Simulating non-rigid motion fields...")

        ky_per_mot_state_idx = self.ky_per_motion_state
        self.sampling_idx = SamplingSimulator.build_sampling_per_nex_per_motion(
            ky_per_mot_state_idx, self.Nx, self.Ny, self.t_device
        )
        self.TotalKspaceSamples = self.Ny * self.Nx

        # Total number of motion states across all Nex acquisitions.
        Nshots = sum(len(shot_list) for shot_list in ky_per_mot_state_idx)

        # MATLAB-equivalent random motion vectors:
        # XTranslationVector = 4 * randn(1,Nshots)
        # YTranslationVector = 2 * randn(1,Nshots)
        # RotationVector     = 3 * randn(1,Nshots) [deg]
        tx_vec = 4.0 * torch.randn(Nshots, device=self.t_device)
        ty_vec = 2.0 * torch.randn(Nshots, device=self.t_device)
        phi_vec = (3.0 * torch.randn(Nshots, device=self.t_device)) * (torch.pi / 180.0)
        # For non-rigid motion_type_flag==2, MATLAB uses S = XTranslationVector.
        S = tx_vec

        alpha_x, alpha_y, alpha_maps = self._create_discrete_non_rigid_alpha_fields()
        self.alpha_maps = alpha_maps

        if self.params.debug_flag:
            print("Visualizing non-rigid alpha fields (alpha_x, alpha_y)...")
            os.makedirs(self.params.debug_folder, exist_ok=True)
            flip_for_display = self.params.flip_for_display
            alpha_x_cart, alpha_y_cart = to_cartesian_components(alpha_x, alpha_y)
            save_alpha_component_map(
                alpha_x_cart,
                "simulated_alpha_x",
                os.path.join(self.params.debug_folder, "simulated_alpha_x.png"),
                flip_vertical=flip_for_display,
            )
            save_alpha_component_map(
                alpha_y_cart,
                "simulated_alpha_y",
                os.path.join(self.params.debug_folder, "simulated_alpha_y.png"),
                flip_vertical=flip_for_display,
            )
            save_nonrigid_quiver_with_contours(
                alpha_x,
                alpha_y,
                self.image[0],
                "simulated_motion_quiver",
                os.path.join(self.params.debug_folder, "simulated_motion_quiver.png"),
                flip_vertical=flip_for_display,
            )

        self._apply_motion(alpha_maps, centers=None, motion_signal=S, motion_type='non-rigid')

        self.navigator_mot_state = S
        self.tx_mot_state = tx_vec
        self.ty_mot_state = ty_vec
        self.phi_mot_state = phi_vec
        self._expand_motion_to_ky(ky_per_mot_state_idx)

    def _create_realistic_non_rigid_motion_curve(self):
        """
        Create a respiratory-like sinusoidal motion curve with unit amplitude.
        Frequency (cycles per image/Nex block) and phase are randomized.
        """
        n_lines_total = self.Ny * self.params.Nex
        line_idx = torch.arange(n_lines_total, device=self.t_device, dtype=torch.float64)

        cycles_min = float(self.params.nonrigid_resp_cycles_min)
        cycles_max = float(self.params.nonrigid_resp_cycles_max)
        if cycles_min <= 0 or cycles_max <= 0:
            raise ValueError("nonrigid_resp_cycles_min/max must be > 0.")
        if cycles_min > cycles_max:
            cycles_min, cycles_max = cycles_max, cycles_min

        cycles = cycles_min + (cycles_max - cycles_min) * torch.rand(1, device=self.t_device).item()
        phase = 2.0 * np.pi * torch.rand(1, device=self.t_device).item()
        # Angular increment per acquired ky line: cycles are expressed per image.
        # This keeps the same respiration frequency per image when Nex changes,
        # while remaining continuous across Nex boundaries.
        omega = 2.0 * np.pi * cycles / max(float(self.Ny), 1.0)
        signal = torch.sin(omega * line_idx + phase)

        # Numerical guard for exact unit-amplitude scaling.
        signal = signal / torch.clamp(torch.max(torch.abs(signal)), min=1e-12)
        return signal

    def simulate_realistic_non_rigid_motion(self):
        print("Simulating realistic non-rigid motion fields...")

        # One global motion state per acquired ky line (Ny * Nex states).
        self.sampling_idx = self._build_sampling_per_line_global_states()
        self.TotalKspaceSamples = self.Ny * self.Nx

        alpha_x, alpha_y, alpha_maps = self._create_discrete_non_rigid_alpha_fields()
        amp = float(self.params.nonrigid_motion_amplitude)
        alpha_maps = alpha_maps * amp
        self.alpha_maps = alpha_maps

        self.navigator = self._create_realistic_non_rigid_motion_curve()
        self._apply_motion(alpha_maps, centers=None, motion_signal=self.navigator, motion_type='non-rigid')

        # Keep compatibility with downstream plotting/debug interfaces.
        # TODO to remove this part and to make a proper separation between rigid and non-rigid motion information in the codebase.
        self.tx = self.navigator.clone()
        self.ty = torch.zeros_like(self.navigator)
        self.phi = torch.zeros_like(self.navigator)

        if self.params.debug_flag:
            os.makedirs(self.params.debug_folder, exist_ok=True)
            flip_for_display = self.params.flip_for_display
            alpha_x_cart, alpha_y_cart = to_cartesian_components(alpha_x * amp, alpha_y * amp)
            save_alpha_component_map(
                alpha_x_cart,
                "simulated_alpha_x",
                os.path.join(self.params.debug_folder, "simulated_alpha_x.png"),
                flip_vertical=flip_for_display,
            )
            save_alpha_component_map(
                alpha_y_cart,
                "simulated_alpha_y",
                os.path.join(self.params.debug_folder, "simulated_alpha_y.png"),
                flip_vertical=flip_for_display,
            )
            save_nonrigid_quiver_with_contours(
                alpha_x * amp,
                alpha_y * amp,
                self.image[0],
                "simulated_motion_quiver",
                os.path.join(self.params.debug_folder, "simulated_motion_quiver.png"),
                flip_vertical=flip_for_display,
            )



        




    








        
        
        

    

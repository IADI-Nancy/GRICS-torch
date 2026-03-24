import numpy as np
import torch

from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.preprocessing.SamplingSimulator import SamplingSimulator
from src.utils.fftnc import ifftnc # normalised ifft for n dimensions
from src.utils.plotting import (
    save_motion_debug_plots, save_nonrigid_alpha_plots,
)
from src.utils.nonrigid_display import (
    flip_nonrigid_alpha_for_display,
)
from src.utils.motion_simulator_utils import (
    build_event_transition_curve,
    build_navigator_from_motion_matrix,
    build_rigid_rotation_centers,
    build_sampling_per_line_global_states,
    compress_consecutive_rigid_states,
    expand_motion_states_to_readouts,
    globalize_per_shot_readout_layout,
    num_motion_readouts,
    require_motion_param,
    rigid_motion_amplitude_scale,
    translation_limits_px,
)

class MotionSimulator:
    def __init__(
        self,
        image,
        smaps,
        ky_idx,
        nex_idx,
        ky_per_motion_state,
        params,
        sp_device=None,
        t_device=None,
        kz_idx=None,
        kz_per_motion_state=None,
    ):
        self.image = image
        self.smaps = smaps
        self.ky_idx = ky_idx
        self.nex_idx = nex_idx
        self.ky_per_motion_state = ky_per_motion_state
        self.kz_idx = kz_idx
        self.kz_per_motion_state = kz_per_motion_state
        self.params = params
        self.sp_device = sp_device
        self.t_device = t_device
        self.Ncha, self.Nx, self.Ny, self.Nz = smaps.shape
    
    def get_corrupted_kspace(self):
        return self.kspace
    
    def get_corrupted_image(self):
        return self.image_no_moco

    def get_rigid_motion_information_2d(self):
        return self.navigator, self.tx, self.ty, self.phi

    def get_rigid_motion_information_3d(self):
        return self.navigator, self.tx, self.ty, self.tz, self.rx, self.ry, self.rz

    def get_nonrigid_motion_information(self):
        return self.navigator, getattr(self, "alpha_maps", None)

    # =============================================================================
    # =========================== SHARED CORE UTILITIES ============================
    # =============================================================================
    # Utilities in this block are motion-type agnostic and are reused by rigid
    # and non-rigid simulation paths.
    # =============================================================================

    def _apply_motion(self, alpha, centers=None, motion_signal=None, motion_type=None):
        if motion_type is None:
            motion_type = self.params.simulated_motion_type

        self.MotionOperator = MotionOperator(self.Nx, self.Ny, alpha, motion_type, centers=centers, motion_signal=motion_signal, Nz=self.Nz)
        E = EncodingOperator(self.smaps, self.TotalKspaceSamples, self.sampling_idx, self.params.Nex, self.MotionOperator)
        kspace_corruped = E.forward(self.image)
        self.kspace = kspace_corruped.reshape(self.Ncha, self.params.Nex, self.Nx, self.Ny, self.Nz)

        img_cplx = ifftnc(self.kspace, dims=(-3, -2, -1)).to(self.t_device)
        self.image_no_moco = torch.sum(img_cplx * self.smaps.conj().unsqueeze(1).expand(-1, self.params.Nex, -1, -1, -1), dim=0)

    # =============================================================================
    # ============================== RIGID MOTION =================================
    # =============================================================================
    # This block handles rigid-motion simulation:
    # 1) realistic per-line temporal curves
    # 2) discrete per-shot rigid states
    # =============================================================================

    # ---------------------------------------------------------------------------
    # ------------------------ RIGID: REALISTIC CURVES --------------------------
    # ---------------------------------------------------------------------------

    def _create_realistic_motion_curves_2d(self):
        # Time axis: one value per k-space line (Ny)
        total_lines = num_motion_readouts(self.ky_idx)
        n_events = int(require_motion_param(self.params, "num_motion_events"))
        n_events = max(1, min(n_events, total_lines))
        tau = int(require_motion_param(self.params, "motion_tau"))
        tau = max(1, tau)

        # 1) Generate unique event times over the full acquisition.
        # Each event represents a bounded state-to-state change.
        event_times = torch.sort(torch.randperm(total_lines, device=self.t_device)[:n_events]).values

        # 2) Generate random bounded state increments for each event
        lim = translation_limits_px(self.params, self.Nx, self.Ny, self.Nz)
        A_tx  = lim["max_tx_px"]  * (2 * torch.rand(n_events, device=self.t_device) - 1)
        A_ty  = lim["max_ty_px"]  * (2 * torch.rand(n_events, device=self.t_device) - 1)
        rigid_amp_scale = rigid_motion_amplitude_scale(self.params)
        A_phi = (
            float(require_motion_param(self.params, "max_phi"))
            * rigid_amp_scale
            * (2 * torch.rand(n_events, device=self.t_device) - 1)
            * (torch.pi / 180)
        )

        # 3) Build independent motion curves
        tx  = torch.zeros(total_lines, device=self.t_device)
        ty  = torch.zeros(total_lines, device=self.t_device)
        phi = torch.zeros(total_lines, device=self.t_device)

        for i, ti in enumerate(event_times):
            ti = event_times[i].item()
            f = build_event_transition_curve(ti, tau, total_lines, device=self.t_device)

            # Add bounded increment so config amplitudes act as per-state changes.
            tx  += A_tx[i]  * f
            ty  += A_ty[i]  * f
            phi += A_phi[i] * f

        M = torch.stack([tx, ty, phi], dim=0)
        navigator = build_navigator_from_motion_matrix(M)
        # save debug plots
        if self.params.debug_flag:
            save_motion_debug_plots(navigator, tx, ty, phi, self.params.debug_folder, event_times)
        # Return the motion curve, parameter curves, and event times
        return navigator, tx, ty, phi

    def _create_realistic_motion_curves_3d(self):
        # Time axis: one value per acquired 3D readout.
        n_states = num_motion_readouts(self.ky_idx)
        n_events = int(require_motion_param(self.params, "num_motion_events"))
        n_events = max(1, min(n_events, n_states))
        tau = int(require_motion_param(self.params, "motion_tau"))
        tau = max(1, tau)

        # Random unique event times over the full acquisition.
        # Each event applies a bounded state-to-state increment.
        event_times = torch.sort(torch.randperm(n_states, device=self.t_device)[:n_events]).values

        lim = translation_limits_px(self.params, self.Nx, self.Ny, self.Nz)
        max_tx_3d = lim["max_tx_3d_px"]
        max_ty_3d = lim["max_ty_3d_px"]
        max_tz_3d = lim["max_tz_3d_px"]
        rigid_amp_scale = rigid_motion_amplitude_scale(self.params)
        max_rx_3d = float(require_motion_param(self.params, "max_rx_3d")) * rigid_amp_scale
        max_ry_3d = float(require_motion_param(self.params, "max_ry_3d")) * rigid_amp_scale
        max_rz_3d = float(require_motion_param(self.params, "max_rz_3d")) * rigid_amp_scale

        A_tx = max_tx_3d * (2 * torch.rand(n_events, device=self.t_device) - 1)
        A_ty = max_ty_3d * (2 * torch.rand(n_events, device=self.t_device) - 1)
        A_tz = max_tz_3d * (2 * torch.rand(n_events, device=self.t_device) - 1)
        A_rx = max_rx_3d * (2 * torch.rand(n_events, device=self.t_device) - 1) * (torch.pi / 180)
        A_ry = max_ry_3d * (2 * torch.rand(n_events, device=self.t_device) - 1) * (torch.pi / 180)
        A_rz = max_rz_3d * (2 * torch.rand(n_events, device=self.t_device) - 1) * (torch.pi / 180)

        tx = torch.zeros(n_states, device=self.t_device)
        ty = torch.zeros(n_states, device=self.t_device)
        tz = torch.zeros(n_states, device=self.t_device)
        rx = torch.zeros(n_states, device=self.t_device)
        ry = torch.zeros(n_states, device=self.t_device)
        rz = torch.zeros(n_states, device=self.t_device)

        for i, _ in enumerate(event_times):
            ti = event_times[i].item()
            f = build_event_transition_curve(ti, tau, n_states, device=self.t_device)

            tx += A_tx[i] * f
            ty += A_ty[i] * f
            tz += A_tz[i] * f
            rx += A_rx[i] * f
            ry += A_ry[i] * f
            rz += A_rz[i] * f

        M = torch.stack([tx, ty, tz, rx, ry, rz], dim=0)
        navigator = build_navigator_from_motion_matrix(M)

        if self.params.debug_flag:
            # Keep existing debug plot signature by visualizing rz as the rotational surrogate.
            save_motion_debug_plots(navigator, tx, ty, rz, self.params.debug_folder, event_times)

        return navigator, tx, ty, tz, rx, ry, rz

    def simulate_realistic_rigid_motion(self):
        self.TotalKspaceSamples = self.Ny * self.Nx * self.Nz
        n_states = num_motion_readouts(self.ky_idx)

        if self.Nz > 1:
            # Generate 3D realistic rigid curves and parameters.
            self.navigator, self.tx, self.ty, self.tz, self.rx, self.ry, self.rz = self._create_realistic_motion_curves_3d()

            alpha = torch.zeros(6, n_states, device=self.t_device)
            alpha[0, :] = self.tx
            alpha[1, :] = self.ty
            alpha[2, :] = self.tz
            alpha[3, :] = self.rx
            alpha[4, :] = self.ry
            alpha[5, :] = self.rz

            lim = translation_limits_px(self.params, self.Nx, self.Ny, self.Nz)
            centers = build_rigid_rotation_centers(lim, n_states, device=self.t_device, Nx=self.Nx, Ny=self.Ny, Nz=self.Nz)
        else:
            # 2D realistic rigid (existing behavior).
            self.navigator, self.tx, self.ty, self.phi = self._create_realistic_motion_curves_2d()

            alpha = torch.zeros(3, n_states, device=self.t_device)
            alpha[0, :] = self.tx
            alpha[1, :] = self.ty
            alpha[2, :] = self.phi

            lim = translation_limits_px(self.params, self.Nx, self.Ny, self.Nz)
            centers = build_rigid_rotation_centers(lim, n_states, device=self.t_device, Nx=self.Nx, Ny=self.Ny, Nz=self.Nz)

        self.sampling_idx, alpha_compressed, centers_compressed = compress_consecutive_rigid_states(alpha, self.ky_idx, self.nex_idx, device=self.t_device, Nx=self.Nx, Ny=self.Ny, Nz=self.Nz, Nex=self.params.Nex, centers=centers, kz_idx=self.kz_idx)
        self._apply_motion(alpha_compressed, centers_compressed)

    # -----------------------------------------------------------------------------
    # -------------------- Discrete Rigid States (Per-Shot) ----------------------
    # -----------------------------------------------------------------------------

    def _create_discrete_motion_curves(self, ky_per_mot_state_idx):
        if len(ky_per_mot_state_idx) == 0:
            raise ValueError("ky_per_mot_state_idx cannot be empty.")
        Nshots = len(ky_per_mot_state_idx[0])

        if self.Nz > 1:
            alpha = torch.zeros((6, Nshots), device=self.t_device)

            lim = translation_limits_px(self.params, self.Nx, self.Ny, self.Nz)
            max_tx_3d = lim["max_tx_3d_px"]
            max_ty_3d = lim["max_ty_3d_px"]
            max_tz_3d = lim["max_tz_3d_px"]
            rigid_amp_scale = rigid_motion_amplitude_scale(self.params)
            max_rx_3d = float(require_motion_param(self.params, "max_rx_3d")) * rigid_amp_scale
            max_ry_3d = float(require_motion_param(self.params, "max_ry_3d")) * rigid_amp_scale
            max_rz_3d = float(require_motion_param(self.params, "max_rz_3d")) * rigid_amp_scale

            self.tx_mot_state = max_tx_3d * (2 * torch.rand(Nshots, device=self.t_device) - 1)
            self.ty_mot_state = max_ty_3d * (2 * torch.rand(Nshots, device=self.t_device) - 1)
            self.tz_mot_state = max_tz_3d * (2 * torch.rand(Nshots, device=self.t_device) - 1)
            self.rx_mot_state = max_rx_3d * (2 * torch.rand(Nshots, device=self.t_device) - 1) * (torch.pi / 180)
            self.ry_mot_state = max_ry_3d * (2 * torch.rand(Nshots, device=self.t_device) - 1) * (torch.pi / 180)
            self.rz_mot_state = max_rz_3d * (2 * torch.rand(Nshots, device=self.t_device) - 1) * (torch.pi / 180)

            alpha[0, :] = self.tx_mot_state
            alpha[1, :] = self.ty_mot_state
            alpha[2, :] = self.tz_mot_state
            alpha[3, :] = self.rx_mot_state
            alpha[4, :] = self.ry_mot_state
            alpha[5, :] = self.rz_mot_state

            centers = build_rigid_rotation_centers(lim, Nshots, device=self.t_device, Nx=self.Nx, Ny=self.Ny, Nz=self.Nz)
            motion_mat = torch.stack([self.tx_mot_state, self.ty_mot_state, self.tz_mot_state, self.rx_mot_state, self.ry_mot_state, self.rz_mot_state], dim=1)
        else:
            alpha = torch.zeros((3, Nshots), device=self.t_device)

            # Translation X: [-max_tx, +max_tx]
            lim = translation_limits_px(self.params, self.Nx, self.Ny, self.Nz)
            self.tx_mot_state = lim["max_tx_px"] * (2 * torch.rand(Nshots, device=self.t_device) - 1)
            alpha[0, :] = self.tx_mot_state

            # Translation Y: [-max_ty, +max_ty]
            self.ty_mot_state = lim["max_ty_px"] * (2 * torch.rand(Nshots, device=self.t_device) - 1)
            alpha[1, :] = self.ty_mot_state

            # Rotation: [-max_phi, +max_phi] degrees → radians
            rigid_amp_scale = rigid_motion_amplitude_scale(self.params)
            self.phi_mot_state = float(require_motion_param(self.params, "max_phi")) * rigid_amp_scale * (2 * torch.rand(Nshots, device=self.t_device) - 1) * (torch.pi / 180)
            alpha[2, :] = self.phi_mot_state

            # Centers
            centers = build_rigid_rotation_centers(lim, Nshots, device=self.t_device, Nx=self.Nx, Ny=self.Ny, Nz=self.Nz)

            # Stack motion parameters: (Nshots, 3)
            motion_mat = torch.stack([self.tx_mot_state, self.ty_mot_state, self.phi_mot_state], dim=1)  # (Nshots, 3)

        self.navigator_mot_state = build_navigator_from_motion_matrix(motion_mat.T)

        state_curves = {"tx": self.tx_mot_state, "ty": self.ty_mot_state, "navigator": self.navigator_mot_state}
        if self.Nz > 1:
            state_curves["tz"] = self.tz_mot_state
            state_curves["rx"] = self.rx_mot_state
            state_curves["ry"] = self.ry_mot_state
            state_curves["rz"] = self.rz_mot_state
        else:
            state_curves["phi"] = self.phi_mot_state

        expanded_curves = expand_motion_states_to_readouts(ky_per_mot_state_idx, state_curves, device=self.t_device)
        self.tx = expanded_curves["tx"]
        self.ty = expanded_curves["ty"]
        self.navigator = expanded_curves["navigator"]
        if self.Nz > 1:
            self.tz = expanded_curves["tz"]
            self.rx = expanded_curves["rx"]
            self.ry = expanded_curves["ry"]
            self.rz = expanded_curves["rz"]
        else:
            self.phi = expanded_curves["phi"]

        # save debug plots
        if self.params.debug_flag:
            save_motion_debug_plots(self.navigator, self.tx, self.ty, self.rz if self.Nz > 1 else self.phi, self.params.debug_folder)

        return self.navigator, alpha, centers

    def simulate_discrete_rigid_motion(self):
        # Each shot is its own motion state
        ky_readout_layout = globalize_per_shot_readout_layout(self.ky_per_motion_state, device=self.t_device)
        kz_readout_layout = None
        if self.kz_per_motion_state is not None:
            kz_readout_layout = globalize_per_shot_readout_layout(self.kz_per_motion_state, device=self.t_device)

        # self.sampling_idx = \
        #     build_sampling_from_motion_states(ky_per_mot_state_idx, self.ky_idx, self.nex_idx, self.Nx, self.Ny, self.t_device)
        self.sampling_idx = SamplingSimulator._build_sampling_per_nex_per_motion(ky_readout_layout, self.t_device, self.Nx, self.Ny, Nz=self.Nz, binned_kz_indices=kz_readout_layout) # ← for debugging only, ignore output
        
        self.TotalKspaceSamples = self.Ny * self.Nx * self.Nz

        self.navigator, alpha, centers = self._create_discrete_motion_curves(ky_readout_layout)

        self._apply_motion(alpha, centers)

    # =============================================================================
    # ============================ NON-RIGID MOTION ===============================
    # =============================================================================
    # This block handles non-rigid-motion simulation:
    # 1) shared spatial alpha-field generators
    # 2) discrete non-rigid simulation
    # 3) realistic non-rigid simulation
    # =============================================================================

    # ---------------------------------------------------------------------------
    # -------------------- NON-RIGID: SHARED BLOCK (ALPHA FIELDS) ---------------
    # ---------------------------------------------------------------------------

    def _create_discrete_non_rigid_alpha_fields(self):
        _, _, alpha_maps = self._create_respiratory_non_rigid_alpha_fields()
        alpha_maps = flip_nonrigid_alpha_for_display(alpha_maps, self.params.flip_for_display)
        return alpha_maps

    def _create_respiratory_non_rigid_alpha_fields(self):
        # Respiration-like non-rigid field with 2D and 3D support.
        # Axis convention used by MotionOperator:
        # - alpha[0] -> displacement along axis 0 (SI-like)
        # - alpha[1] -> displacement along axis 1 (LR-like)
        # - alpha[2] -> displacement along axis 2 (AP-like, 3D only)
        x = torch.linspace(-1.0, 1.0, self.Nx, device=self.t_device, dtype=torch.float64)
        y = torch.linspace(-1.0, 1.0, self.Ny, device=self.t_device, dtype=torch.float64)

        diaphragm_level = float(self.params.nonrigid_diaphragm_level)
        diaphragm_sharpness = float(self.params.nonrigid_diaphragm_sharpness)
        inferior_gain = float(self.params.nonrigid_inferior_gain)
        top_decay = float(self.params.nonrigid_top_decay)

        # Per-axis envelope sigmas.
        sigma_lr = float(require_motion_param(self.params, "nonrigid_lateral_sigma_lr"))
        sigma_ap = float(require_motion_param(self.params, "nonrigid_lateral_sigma_ap"))
        if sigma_lr <= 0 or sigma_ap <= 0:
            raise ValueError("nonrigid_lateral_sigma_lr and nonrigid_lateral_sigma_ap must be positive.")

        # Per-axis displacement fractions relative to SI.
        lr_fraction = float(require_motion_param(self.params, "nonrigid_lr_fraction"))
        ap_fraction = float(require_motion_param(self.params, "nonrigid_ap_fraction"))

        # Anterior bias: 0 = symmetric, 1 = fully anterior (supine table).
        anterior_bias = float(require_motion_param(self.params, "nonrigid_anterior_bias"))

        if self.Nz > 1:
            z = torch.linspace(-1.0, 1.0, self.Nz, device=self.t_device, dtype=torch.float64)
            SI, LR, AP = torch.meshgrid(x, y, z, indexing="ij")

            region_mask = torch.sigmoid((-SI - diaphragm_level) * diaphragm_sharpness)
            lateral_envelope = torch.exp(-0.5 * ((LR / sigma_lr) ** 2 + (AP / sigma_ap) ** 2))

            # Suppress posterior motion (AP < 0 in supine = posterior / table side).
            # anterior_weight ranges from (1 - bias) at AP=-1 to 1.0 at AP=+1.
            anterior_weight = 1.0 - anterior_bias * 0.5 * (1.0 - AP)

            top_coord = torch.clamp(-SI, min=0.0)
            top_taper = torch.clamp(1.0 - top_decay * top_coord, min=0.05)
            region_gain = top_taper * (1.0 + inferior_gain * torch.clamp(-SI, min=0.0))

            si_profile = region_mask * lateral_envelope * region_gain * anterior_weight
            si_profile = si_profile / torch.clamp(torch.max(torch.abs(si_profile)), min=1e-12)
            alpha_x = -si_profile

            lr_profile = -LR * region_mask * lateral_envelope * anterior_weight
            lr_profile = lr_profile / torch.clamp(torch.max(torch.abs(lr_profile)), min=1e-12)
            alpha_y = lr_fraction * lr_profile

            ap_profile = -AP * region_mask * lateral_envelope * anterior_weight
            ap_profile = ap_profile / torch.clamp(torch.max(torch.abs(ap_profile)), min=1e-12)
            alpha_z = ap_fraction * ap_profile

            alpha_maps = torch.stack([alpha_x, alpha_y, alpha_z], dim=0)
            return alpha_x, alpha_y, alpha_maps

        SI, LR = torch.meshgrid(x, y, indexing="ij")

        region_mask = torch.sigmoid((-SI - diaphragm_level) * diaphragm_sharpness)
        lateral_envelope = torch.exp(-0.5 * (LR / sigma_lr) ** 2)

        top_coord = torch.clamp(-SI, min=0.0)
        top_taper = torch.clamp(1.0 - top_decay * top_coord, min=0.05)
        region_gain = top_taper * (1.0 + inferior_gain * torch.clamp(-SI, min=0.0))
        si_profile = region_mask * lateral_envelope * region_gain
        si_profile = si_profile / torch.clamp(torch.max(torch.abs(si_profile)), min=1e-12)
        alpha_x = -si_profile

        lr_profile = -LR * region_mask * lateral_envelope
        lr_profile = lr_profile / torch.clamp(torch.max(torch.abs(lr_profile)), min=1e-12)
        alpha_y = lr_fraction * lr_profile

        alpha_maps = torch.stack([alpha_x, alpha_y], dim=0)
        return alpha_x, alpha_y, alpha_maps

    # ---------------------------------------------------------------------------
    # ------------------ NON-RIGID: DISCRETE SIMULATION -------------------------
    # ---------------------------------------------------------------------------

    def simulate_discrete_non_rigid_motion(self):
        print("Simulating non-rigid motion fields...")

        ky_readout_layout = globalize_per_shot_readout_layout(self.ky_per_motion_state, device=self.t_device)
        kz_readout_layout = None
        if self.kz_per_motion_state is not None:
            kz_readout_layout = globalize_per_shot_readout_layout(self.kz_per_motion_state, device=self.t_device)
        self.sampling_idx = SamplingSimulator._build_sampling_per_nex_per_motion(ky_readout_layout, self.t_device, self.Nx, self.Ny, Nz=self.Nz, binned_kz_indices=kz_readout_layout)
        self.TotalKspaceSamples = self.Ny * self.Nx * self.Nz

        # One unique motion state per total shot across the acquisition.
        if len(ky_readout_layout) == 0:
            raise ValueError("ky_readout_layout cannot be empty.")
        n_shots = len(ky_readout_layout[0])

        # One Gaussian temporal coefficient per shot drives the non-rigid field amplitude.
        s_scale = float(require_motion_param(self.params, "nonrigid_discrete_s_scale"))
        s = s_scale * torch.randn(n_shots, device=self.t_device)

        alpha_maps = self._create_discrete_non_rigid_alpha_fields()
        self.alpha_maps = alpha_maps

        if self.params.debug_flag:
            save_nonrigid_alpha_plots(alpha_maps, self.image[0], "simulated", self.params.debug_folder, flip_vertical=self.params.flip_for_display)

        self._apply_motion(alpha_maps, centers=None, motion_signal=s, motion_type='non-rigid')
        expanded_curves = expand_motion_states_to_readouts(ky_readout_layout, {"navigator": s}, device=self.t_device)
        self.navigator = expanded_curves["navigator"]

    # ---------------------------------------------------------------------------
    # ------------------ NON-RIGID: REALISTIC SIMULATION  -----------------------
    # ---------------------------------------------------------------------------

    def _create_realistic_non_rigid_motion_curve(self):
        """
        Create a respiratory-like sinusoidal motion curve with unit amplitude.
        Frequency (cycles per image/Nex block) and phase are randomized.
        """
        n_lines_total = num_motion_readouts(self.ky_idx)
        line_idx = torch.arange(n_lines_total, device=self.t_device, dtype=torch.float64)

        cycles_min = float(self.params.nonrigid_resp_cycles_min)
        cycles_max = float(self.params.nonrigid_resp_cycles_max)

        cycles = cycles_min + (cycles_max - cycles_min) * torch.rand(1, device=self.t_device).item()
        phase = 2.0 * np.pi * torch.rand(1, device=self.t_device).item()
        # Angular increment per acquired readout: cycles are expressed per full
        # image/volume repeat so the respiratory rate stays stable when Nz changes.
        readouts_per_repeat = max(float(n_lines_total) / max(float(self.params.Nex), 1.0), 1.0)
        omega = 2.0 * np.pi * cycles / readouts_per_repeat
        signal = torch.sin(omega * line_idx + phase)

        # Numerical guard for exact unit-amplitude scaling.
        signal = signal / torch.clamp(torch.max(torch.abs(signal)), min=1e-12)
        return signal

    def simulate_realistic_non_rigid_motion(self):
        print("Simulating realistic non-rigid motion fields...")

        # One global motion state per acquired readout.
        self.sampling_idx = build_sampling_per_line_global_states(self.ky_idx, self.nex_idx, self.kz_idx, device=self.t_device, Nx=self.Nx, Ny=self.Ny, Nz=self.Nz, Nex=self.params.Nex)
        self.TotalKspaceSamples = self.Ny * self.Nx * self.Nz

        alpha_maps = self._create_discrete_non_rigid_alpha_fields()
        amp = float(self.params.nonrigid_motion_amplitude)
        alpha_maps = alpha_maps * amp
        self.alpha_maps = alpha_maps

        self.navigator = self._create_realistic_non_rigid_motion_curve()
        self._apply_motion(alpha_maps, centers=None, motion_signal=self.navigator, motion_type='non-rigid')

        if self.params.debug_flag:
            save_nonrigid_alpha_plots(alpha_maps, self.image[0], "simulated", self.params.debug_folder, flip_vertical=self.params.flip_for_display)

    

import numpy as np
import torch
import os

from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionOperator import MotionOperator
from src.preprocessing.SamplingSimulator import SamplingSimulator
from src.utils.fftnc import fftnc, ifftnc # normalised fft and ifft for n dimensions
from src.utils.plotting import (
    save_motion_debug_plots, save_nonrigid_alpha_plots,
)
from src.utils.nonrigid_display import (
    to_cartesian_components,
    flip_nonrigid_alpha_for_display,
    split_nonrigid_alpha_components,
)

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
        self.Ncha, self.Nx, self.Ny, self.Nz = smaps.shape
    
    def _get_corrupted_kspace(self):
        return self.kspace
    
    def _get_corrupted_image(self):
        return self.image_no_moco

    def _get_motion_information(self):
        return self.navigator, self.tx, self.ty, self.phi

    # =============================================================================
    # =========================== SHARED CORE UTILITIES ============================
    # =============================================================================
    # Utilities in this block are motion-type agnostic and are reused by rigid,
    # non-rigid, and no-motion simulation paths.
    # =============================================================================

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
        kz = torch.arange(self.Nz, device=self.t_device, dtype=torch.int64) if self.Nz > 1 else None

        for state in range(ny_total):
            nex = int(nex_flat[state].item())
            ky = ky_flat[state].to(torch.int64).reshape(1)
            samp_xy = (ky[:, None] + self.Ny * kx[None, :])
            if self.Nz > 1:
                samp = (samp_xy[:, :, None] * self.Nz + kz[None, None, :]).reshape(-1)
            else:
                samp = samp_xy.reshape(-1)
            sampling[nex][state] = samp

        return sampling

    def _globalize_per_shot_states(self, ky_per_mot_state_idx):
        """
        Convert a per-Nex shot layout [Nex][NshotsPerNex] into a global shot-state
        layout [Nex][NshotsTotal], where each total shot has its own unique motion
        state and all other states are empty for that Nex.
        """
        if len(ky_per_mot_state_idx) == 0:
            raise ValueError("ky_per_mot_state_idx cannot be empty.")

        first_nonempty = None
        for ky_list in ky_per_mot_state_idx:
            if len(ky_list) > 0:
                first_nonempty = ky_list[0]
                break
        if first_nonempty is None:
            raise ValueError("ky_per_mot_state_idx must contain at least one shot.")

        total_shots = sum(len(ky_list) for ky_list in ky_per_mot_state_idx)
        empty = lambda: torch.empty(0, dtype=first_nonempty.dtype, device=self.t_device)

        global_layout = []
        offset = 0
        for ky_list in ky_per_mot_state_idx:
            row = [empty() for _ in range(total_shots)]
            for local_shot_idx, ky in enumerate(ky_list):
                row[offset + local_shot_idx] = ky
            global_layout.append(row)
            offset += len(ky_list)

        return global_layout

    def _apply_motion(self, alpha, centers=None, motion_signal=None, motion_type=None):
        if motion_type is None:
            motion_type = self.params.simulated_motion_type

        self.MotionOperator = MotionOperator(
            self.Nx, self.Ny, alpha, motion_type, centers=centers, motion_signal=motion_signal, Nz=self.Nz
        )

        E = EncodingOperator(
            self.smaps,
            self.TotalKspaceSamples,
            self.sampling_idx,
            self.params.Nex,
            self.MotionOperator
        )
        kspace_corruped = E.forward(self.image)
        self.kspace = kspace_corruped.reshape(self.Ncha, self.params.Nex, self.Nx, self.Ny, self.Nz)

        img_cplx = ifftnc(self.kspace, dims=(-3, -2, -1)).to(self.t_device)
        self.image_no_moco = torch.sum(img_cplx * self.smaps.conj().unsqueeze(1).expand(-1, self.params.Nex, -1, -1, -1), dim=0)

    def _translation_limits_px(self):
        """
        Convert translation and rotation-center amplitudes from mm (config)
        to pixels/voxels.
        x/y use FoVxy_mm, z uses FoVz_mm.
        """
        fovxy_mm = float(getattr(self.params, "FoVxy_mm", 250.0))
        fovz_mm = float(getattr(self.params, "FoVz_mm", 200.0))
        fovxy_mm = max(fovxy_mm, 1e-12)
        fovz_mm = max(fovz_mm, 1e-12)

        sx = float(self.Nx) / fovxy_mm
        sy = float(self.Ny) / fovxy_mm
        sz = float(self.Nz) / fovz_mm

        rigid_amp_scale = float(getattr(self.params, "rigid_motion_amplitude_scale", 1.0))
        if rigid_amp_scale < 0:
            raise ValueError("rigid_motion_amplitude_scale must be >= 0.")

        max_tx_mm = float(getattr(self.params, "max_tx", 0.0)) * rigid_amp_scale
        max_ty_mm = float(getattr(self.params, "max_ty", 0.0)) * rigid_amp_scale
        max_tx_3d_mm = float(getattr(self.params, "max_tx_3d", max_tx_mm)) * rigid_amp_scale
        max_ty_3d_mm = float(getattr(self.params, "max_ty_3d", max_ty_mm)) * rigid_amp_scale
        max_tz_3d_mm = float(getattr(self.params, "max_tz_3d", 0.0)) * rigid_amp_scale
        max_center_x_mm = float(getattr(self.params, "max_center_x", 0.0))
        max_center_y_mm = float(getattr(self.params, "max_center_y", 0.0))
        max_center_x_3d_mm = float(getattr(self.params, "max_center_x_3d", max_center_x_mm))
        max_center_y_3d_mm = float(getattr(self.params, "max_center_y_3d", max_center_y_mm))
        max_center_z_3d_mm = float(getattr(self.params, "max_center_z_3d", 0.0))

        return {
            "max_tx_px": max_tx_mm * sx,
            "max_ty_px": max_ty_mm * sy,
            "max_tx_3d_px": max_tx_3d_mm * sx,
            "max_ty_3d_px": max_ty_3d_mm * sy,
            "max_tz_3d_px": max_tz_3d_mm * sz,
            "max_center_x_px": max_center_x_mm * sx,
            "max_center_y_px": max_center_y_mm * sy,
            "max_center_x_3d_px": max_center_x_3d_mm * sx,
            "max_center_y_3d_px": max_center_y_3d_mm * sy,
            "max_center_z_3d_px": max_center_z_3d_mm * sz,
        }

    # =============================================================================
    # ================================ NO MOTION ==================================
    # =============================================================================
    # This block handles the strict zero-motion baseline simulation.
    # =============================================================================
    
    def _simulate_no_motion(self):
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

        self.sampling_idx_per_nex = SamplingSimulator._build_sampling_per_nex_per_motion(
            ky_per_mot_state_idx, self.t_device, self.Nx, self.Ny,
            Nz=self.Nz, kspace_sampling_type=self.params.kspace_sampling_type
        )
        
        self.TotalKspaceSamples = self.Nx * self.Ny * self.Nz

        # Expand zero motion to ky (chronological)
        self.tx        = torch.zeros(ny_total, device=self.t_device)
        self.ty        = torch.zeros(ny_total, device=self.t_device)
        self.phi       = torch.zeros(ny_total, device=self.t_device)
        self.navigator = torch.zeros(ny_total, device=self.t_device)


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

    def _create_realistic_motion_curves(self):
        # Time axis: one value per k-space line (Ny)
        t = torch.arange(self.Ny*self.params.Nex, device=self.t_device, dtype=torch.float64)
        n_events = int(getattr(self.params, "num_motion_events", 3))
        n_events = max(1, min(n_events, self.Ny * self.params.Nex))
        tau = int(getattr(self.params, "motion_tau", max(1, self.Ny // 8)))
        tau = max(1, tau)

        # 1) Generate unique event times over the full acquisition.
        # Each event represents a bounded state-to-state change.
        total_lines = self.Ny * self.params.Nex
        event_times = torch.sort(
            torch.randperm(total_lines, device=self.t_device)[:n_events]
        ).values

        # 2) Generate random bounded state increments for each event
        lim = self._translation_limits_px()
        A_tx  = lim["max_tx_px"]  * (2 * torch.rand(n_events, device=self.t_device) - 1)
        A_ty  = lim["max_ty_px"]  * (2 * torch.rand(n_events, device=self.t_device) - 1)
        rigid_amp_scale = float(getattr(self.params, "rigid_motion_amplitude_scale", 1.0))
        A_phi = (
            float(getattr(self.params, "max_phi", 0.0))
            * rigid_amp_scale
            * (2 * torch.rand(n_events, device=self.t_device) - 1)
            * (torch.pi / 180)
        )

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

            # Add bounded increment so config amplitudes act as per-state changes.
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

    def _create_realistic_motion_curves_3d(self):
        # Time axis: one value per k-space line (Ny * Nex global states).
        n_states = self.Ny * self.params.Nex
        t = torch.arange(n_states, device=self.t_device, dtype=torch.float64)
        n_events = int(getattr(self.params, "num_motion_events", 3))
        n_events = max(1, min(n_events, n_states))
        tau = int(getattr(self.params, "motion_tau", max(1, self.Ny // 8)))
        tau = max(1, tau)

        # Random unique event times over the full acquisition.
        # Each event applies a bounded state-to-state increment.
        event_times = torch.sort(
            torch.randperm(n_states, device=self.t_device)[:n_events]
        ).values

        lim = self._translation_limits_px()
        max_tx_3d = lim["max_tx_3d_px"]
        max_ty_3d = lim["max_ty_3d_px"]
        max_tz_3d = lim["max_tz_3d_px"]
        rigid_amp_scale = float(getattr(self.params, "rigid_motion_amplitude_scale", 1.0))
        max_rx_3d = float(getattr(self.params, "max_rx_3d", getattr(self.params, "max_phi", 0.0))) * rigid_amp_scale
        max_ry_3d = float(getattr(self.params, "max_ry_3d", getattr(self.params, "max_phi", 0.0))) * rigid_amp_scale
        max_rz_3d = float(getattr(self.params, "max_rz_3d", getattr(self.params, "max_phi", 0.0))) * rigid_amp_scale

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
            t_end = min(ti + tau, n_states)
            alpha = np.linspace(0.0, 1.0, t_end - ti)
            s = 0.5 * (1.0 - np.cos(np.pi * alpha))

            f = torch.zeros(n_states, device=self.t_device)
            f[ti:t_end] = torch.from_numpy(s).to(self.t_device)
            if t_end < n_states:
                f[t_end:] = 1.0

            tx += A_tx[i] * f
            ty += A_ty[i] * f
            tz += A_tz[i] * f
            rx += A_rx[i] * f
            ry += A_ry[i] * f
            rz += A_rz[i] * f

        M = torch.stack([tx, ty, tz, rx, ry, rz], dim=0)
        M_centered = M - M.mean(dim=1, keepdim=True)
        U, _, _ = torch.linalg.svd(M_centered, full_matrices=False)
        u1 = U[:, 0]
        navigator = torch.matmul(u1.unsqueeze(0), M_centered).squeeze(0)
        navigator = navigator / torch.clamp(navigator.abs().max(), min=1e-12)

        if self.params.debug_flag:
            # Keep existing debug plot signature by visualizing rz as the rotational surrogate.
            save_motion_debug_plots(navigator, tx, ty, rz, self.params.debug_folder, event_times)

        return navigator, tx, ty, tz, rx, ry, rz

    def _simulate_realistic_rigid_motion(self):
        # One global motion state per acquired ky line (Ny * Nex states).
        self.sampling_idx = self._build_sampling_per_line_global_states()

        self.TotalKspaceSamples = self.Ny * self.Nx * self.Nz
        n_states = self.Ny * self.params.Nex

        if self.Nz > 1:
            # Generate 3D realistic rigid curves and parameters.
            (
                self.navigator,
                self.tx,
                self.ty,
                self.tz,
                self.rx,
                self.ry,
                self.rz,
            ) = self._create_realistic_motion_curves_3d()

            alpha = torch.zeros(6, n_states, device=self.t_device)
            alpha[0, :] = self.tx
            alpha[1, :] = self.ty
            alpha[2, :] = self.tz
            alpha[3, :] = self.rx
            alpha[4, :] = self.ry
            alpha[5, :] = self.rz

            # Keep compatibility with existing plotting/debug interfaces.
            self.phi = self.rz

            lim = self._translation_limits_px()
            max_center_x_3d = lim["max_center_x_3d_px"]
            max_center_y_3d = lim["max_center_y_3d_px"]
            max_center_z_3d = lim["max_center_z_3d_px"]
            centers = torch.zeros((3, n_states), device=self.t_device)
            centers[0, :] = self.Nx / 2 + max_center_x_3d * torch.ones(n_states, device=self.t_device)
            centers[1, :] = self.Ny / 2 + max_center_y_3d * torch.ones(n_states, device=self.t_device)
            centers[2, :] = self.Nz / 2 + max_center_z_3d * torch.ones(n_states, device=self.t_device)
        else:
            # 2D realistic rigid (existing behavior).
            self.navigator, self.tx, self.ty, self.phi = self._create_realistic_motion_curves()

            alpha = torch.zeros(3, n_states, device=self.t_device)
            alpha[0, :] = self.tx
            alpha[1, :] = self.ty
            alpha[2, :] = self.phi

            lim = self._translation_limits_px()
            centers = torch.zeros((2, n_states), device=self.t_device)
            centers[0, :] = self.Nx / 2 + lim["max_center_x_px"] * torch.ones(n_states, device=self.t_device)
            centers[1, :] = self.Ny / 2 + lim["max_center_y_px"] * torch.randn(n_states, device=self.t_device)

        self._apply_motion(alpha, centers)

    # -----------------------------------------------------------------------------
    # -------------------- Discrete Rigid States (Per-Shot) ----------------------
    # -----------------------------------------------------------------------------

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
        is_3d_rigid_states = all(
            hasattr(self, name)
            for name in ("tz_mot_state", "rx_mot_state", "ry_mot_state", "rz_mot_state")
        )
        if is_3d_rigid_states:
            self.tz = torch.empty(Ny_total, device=self.t_device)
            self.rx = torch.empty(Ny_total, device=self.t_device)
            self.ry = torch.empty(Ny_total, device=self.t_device)
            self.rz = torch.empty(Ny_total, device=self.t_device)

        ptr = 0  # write pointer (chronological)

        for ky_list in ky_per_mot_state_idx:      # loop over Nex
            for shot_idx, ky in enumerate(ky_list):  # loop over shot index within Nex
                n = ky.numel()

                # Motion states are indexed globally across the full acquisition.
                self.tx[ptr:ptr+n]        = self.tx_mot_state[shot_idx]
                self.ty[ptr:ptr+n]        = self.ty_mot_state[shot_idx]
                self.phi[ptr:ptr+n]       = self.phi_mot_state[shot_idx]
                self.navigator[ptr:ptr+n] = self.navigator_mot_state[shot_idx]
                if is_3d_rigid_states:
                    self.tz[ptr:ptr+n] = self.tz_mot_state[shot_idx]
                    self.rx[ptr:ptr+n] = self.rx_mot_state[shot_idx]
                    self.ry[ptr:ptr+n] = self.ry_mot_state[shot_idx]
                    self.rz[ptr:ptr+n] = self.rz_mot_state[shot_idx]

                ptr += n

    def _create_discrete_motion_curves(self, ky_per_mot_state_idx):
        if len(ky_per_mot_state_idx) == 0:
            raise ValueError("ky_per_mot_state_idx cannot be empty.")
        Nshots = len(ky_per_mot_state_idx[0])

        if self.Nz > 1:
            alpha = torch.zeros((6, Nshots), device=self.t_device)

            lim = self._translation_limits_px()
            max_tx_3d = lim["max_tx_3d_px"]
            max_ty_3d = lim["max_ty_3d_px"]
            max_tz_3d = lim["max_tz_3d_px"]
            rigid_amp_scale = float(getattr(self.params, "rigid_motion_amplitude_scale", 1.0))
            max_rx_3d = float(getattr(self.params, "max_rx_3d", getattr(self.params, "max_phi", 0.0))) * rigid_amp_scale
            max_ry_3d = float(getattr(self.params, "max_ry_3d", getattr(self.params, "max_phi", 0.0))) * rigid_amp_scale
            max_rz_3d = float(getattr(self.params, "max_rz_3d", getattr(self.params, "max_phi", 0.0))) * rigid_amp_scale

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

            # Keep legacy plotting outputs populated.
            self.phi_mot_state = self.rz_mot_state

            max_center_x_3d = lim["max_center_x_3d_px"]
            max_center_y_3d = lim["max_center_y_3d_px"]
            max_center_z_3d = lim["max_center_z_3d_px"]
            centers = torch.zeros((3, Nshots), device=self.t_device)
            centers[0, :] = self.Nx / 2 + max_center_x_3d * torch.ones(Nshots, device=self.t_device)
            centers[1, :] = self.Ny / 2 + max_center_y_3d * torch.ones(Nshots, device=self.t_device)
            centers[2, :] = self.Nz / 2 + max_center_z_3d * torch.ones(Nshots, device=self.t_device)

            motion_mat = torch.stack(
                [
                    self.tx_mot_state,
                    self.ty_mot_state,
                    self.tz_mot_state,
                    self.rx_mot_state,
                    self.ry_mot_state,
                    self.rz_mot_state,
                ],
                dim=1,
            )
        else:
            alpha = torch.zeros((3, Nshots), device=self.t_device)

            # Translation X: [-max_tx, +max_tx]
            lim = self._translation_limits_px()
            self.tx_mot_state = lim["max_tx_px"] * (2 * torch.rand(Nshots, device=self.t_device) - 1)
            alpha[0, :] = self.tx_mot_state

            # Translation Y: [-max_ty, +max_ty]
            self.ty_mot_state = lim["max_ty_px"] * (2 * torch.rand(Nshots, device=self.t_device) - 1)
            alpha[1, :] = self.ty_mot_state

            # Rotation: [-max_phi, +max_phi] degrees → radians
            rigid_amp_scale = float(getattr(self.params, "rigid_motion_amplitude_scale", 1.0))
            self.phi_mot_state = (
                float(getattr(self.params, "max_phi", 0.0))
                * rigid_amp_scale
                * (2 * torch.rand(Nshots, device=self.t_device) - 1)
                * (torch.pi / 180)
            )
            alpha[2, :] = self.phi_mot_state

            # Centers
            centers = torch.zeros((2, Nshots), device=self.t_device)
            centers[0, :] = self.Nx / 2 + lim["max_center_x_px"] * torch.ones(Nshots, device=self.t_device)
            centers[1, :] = self.Ny / 2 + lim["max_center_y_px"] * torch.ones(Nshots, device=self.t_device) # torch.rand(Nshots, device=self.t_device)

            # Stack motion parameters: (Nshots, 3)
            motion_mat = torch.stack([self.tx_mot_state, self.ty_mot_state, self.phi_mot_state], dim=1)  # (Nshots, 3)

        # -------------------------------------------------
        # Build navigator = first principal component
        # -------------------------------------------------
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

    def _simulate_discrete_rigid_motion(self):
        # Each shot is its own motion state
        ky_per_mot_state_idx = self._globalize_per_shot_states(self.ky_per_motion_state)

        # self.sampling_idx = \
        #     build_sampling_from_motion_states(ky_per_mot_state_idx, self.ky_idx, self.nex_idx, self.Nx, self.Ny, self.t_device)
        self.sampling_idx = SamplingSimulator._build_sampling_per_nex_per_motion(
            ky_per_mot_state_idx, self.t_device, self.Nx, self.Ny,
            Nz=self.Nz, kspace_sampling_type=self.params.kspace_sampling_type
        ) # ← for debugging only, ignore output
        
        self.TotalKspaceSamples = self.Ny * self.Nx * self.Nz

        self.navigator, alpha, centers = self._create_discrete_motion_curves(ky_per_mot_state_idx)

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
    # -------------------- NON-RIGID: SHARED ALPHA FIELDS ----------------------
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

        # Per-axis envelope sigmas (fall back to legacy single sigma).
        _legacy_sigma = float(getattr(self.params, "nonrigid_lateral_sigma", 0.35))
        sigma_lr = max(float(getattr(self.params, "nonrigid_lateral_sigma_lr",
                                     _legacy_sigma)), 1e-6)
        sigma_ap = max(float(getattr(self.params, "nonrigid_lateral_sigma_ap",
                                     _legacy_sigma)), 1e-6)

        # Per-axis displacement fractions relative to SI.
        _legacy_frac = float(getattr(self.params, "nonrigid_ap_fraction", 0.2))
        lr_fraction = float(getattr(self.params, "nonrigid_lr_fraction", _legacy_frac))
        ap_fraction = float(getattr(self.params, "nonrigid_ap_fraction", _legacy_frac))

        # Anterior bias: 0 = symmetric, 1 = fully anterior (supine table).
        anterior_bias = float(getattr(self.params, "nonrigid_anterior_bias", 0.0))

        if self.Nz > 1:
            z = torch.linspace(-1.0, 1.0, self.Nz, device=self.t_device, dtype=torch.float64)
            SI, LR, AP = torch.meshgrid(x, y, z, indexing="ij")

            region_mask = torch.sigmoid((-SI - diaphragm_level) * diaphragm_sharpness)
            lateral_envelope = torch.exp(
                -0.5 * ((LR / sigma_lr) ** 2 + (AP / sigma_ap) ** 2)
            )

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

    def _simulate_discrete_non_rigid_motion(self):
        print("Simulating non-rigid motion fields...")

        ky_per_mot_state_idx = self._globalize_per_shot_states(self.ky_per_motion_state)
        self.sampling_idx = SamplingSimulator._build_sampling_per_nex_per_motion(
            ky_per_mot_state_idx, self.t_device, self.Nx, self.Ny,
            Nz=self.Nz, kspace_sampling_type=self.params.kspace_sampling_type
        )
        self.TotalKspaceSamples = self.Ny * self.Nx * self.Nz

        # One unique motion state per total shot across the acquisition.
        if len(ky_per_mot_state_idx) == 0:
            raise ValueError("ky_per_mot_state_idx cannot be empty.")
        Nshots = len(ky_per_mot_state_idx[0])

        # MATLAB-equivalent random motion vectors:
        # XTranslationVector = 4 * randn(1,Nshots)
        # YTranslationVector = 2 * randn(1,Nshots)
        # RotationVector     = 3 * randn(1,Nshots) [deg]
        tx_vec = 4.0 * torch.randn(Nshots, device=self.t_device)
        ty_vec = 2.0 * torch.randn(Nshots, device=self.t_device)
        phi_vec = (3.0 * torch.randn(Nshots, device=self.t_device)) * (torch.pi / 180.0)
        # For non-rigid motion_type_flag==2, MATLAB uses S = XTranslationVector.
        S = tx_vec

        alpha_maps = self._create_discrete_non_rigid_alpha_fields()
        self.alpha_maps = alpha_maps

        if self.params.debug_flag:
            save_nonrigid_alpha_plots(
                alpha_maps, self.image[0],
                "simulated", self.params.debug_folder,
                flip_vertical=self.params.flip_for_display,
            )

        self._apply_motion(alpha_maps, centers=None, motion_signal=S, motion_type='non-rigid')

        self.navigator_mot_state = S
        self.tx_mot_state = tx_vec
        self.ty_mot_state = ty_vec
        self.phi_mot_state = phi_vec
        self._expand_motion_to_ky(ky_per_mot_state_idx)

    # ---------------------------------------------------------------------------
    # ------------------ NON-RIGID: REALISTIC SIMULATION  -----------------------
    # ---------------------------------------------------------------------------

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

    def _simulate_realistic_non_rigid_motion(self):
        print("Simulating realistic non-rigid motion fields...")

        # One global motion state per acquired ky line (Ny * Nex states).
        self.sampling_idx = self._build_sampling_per_line_global_states()
        self.TotalKspaceSamples = self.Ny * self.Nx * self.Nz

        alpha_maps = self._create_discrete_non_rigid_alpha_fields()
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
            save_nonrigid_alpha_plots(
                alpha_maps, self.image[0],
                "simulated", self.params.debug_folder,
                flip_vertical=self.params.flip_for_display,
            )

    

import torch
import torch.nn.functional as F
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.MotionOperator import MotionOperator
from src.reconstruction.EncodingOperator import EncodingOperator
from src.reconstruction.MotionPerturbationSimulator import MotionPerturbationSimulator
from src.utils.plotting import show_and_save_image
from src.utils.save_final_motion_plots import save_final_nonrigid_alpha_maps, save_final_rigid_motion_plots
from src.utils.joint_reconstructor_utils import (
    _JointReconstructionLogger, _assign_cached_reg_scale, _console,
    _initialize_level_tracking, _parse_gn_iterations_per_level, _save_nonrigid_motion_debug)



@dataclass
class _GaussNewtonIterationResult:
    image: torch.Tensor
    motion: torch.Tensor
    predicted_kspace: torch.Tensor
    residual: torch.Tensor
    motion_for_residual: torch.Tensor
    motion_update: torch.Tensor | None
    image_elapsed: float
    motion_elapsed: float | None


# --------------------------------------------------------------------------
# Class that performs joint image–motion reconstruction
# --------------------------------------------------------------------------
class JointReconstructor:

    def __init__(
        self, KspaceData: torch.Tensor, smaps: torch.Tensor,
        SamplingIndices: list[list[torch.Tensor]], motion_signal: torch.Tensor,
        params: SimpleNamespace, kspace_scale: float = 1.0,
        motion_plot_context: dict[str, Any] | None = None,
        initial_image: torch.Tensor | None = None,
        initial_motion: torch.Tensor | None = None,
        external_image_regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None):
        """Initialize joint reconstruction.

        Args:
            KspaceData: Complex ``[Nc, Ne, Nx, Ny, Nz]`` tensor.
            smaps: Complex ``[Nc, Nx, Ny, Nz]`` tensor; ``Nz=1`` for 2D.
            SamplingIndices: Nested ``[Ne][Nm]`` lists of 1D flattened
                integer k-space-index tensors.
            motion_signal: Real ``[Nm, Ns]`` tensor.
            params: Validated flat configuration from ``data.params`` or
                ``prepared.params``; do not pass a dict or TOML filename.
            kspace_scale: Scalar used to restore output-image magnitude.
            motion_plot_context: Optional plotting metadata.
            initial_image: Optional complex ``[Ne, Nx, Ny, (Nz)]`` tensor;
                the ``Ne`` axis may be omitted only when ``Ne=1``.
            initial_motion: Optional real ``[Nalpha, Nm]`` rigid tensor or
                ``[Nalpha, Nx, Ny, (Nz), Ns]`` non-rigid tensor.
            external_image_regularizer: Optional callable mapping an image to
                a same-shape, same-device image prior.
        """
        Ncoils, Nx_full, Ny_full, Nz_full = smaps.shape

        # Parameters constant for all resolutions        
        self.params = params
        self.Ncoils = Ncoils
        self.Nz_full = int(Nz_full)
        self.device = KspaceData.device
        if self.params.reconstruction_motion_type == "rigid":
            self.Nalpha = 6 if self.Nz_full > 1 else 3
        else:
            self.Nalpha = 3 if self.Nz_full > 1 else 2
        self.kspace_scale = float(kspace_scale)
        if motion_signal is None:
            raise ValueError("motion_signal must be provided.")
        self.motion_signal = motion_signal.to(self.device)
        if self.motion_signal.ndim != 2:
            raise ValueError(
                "motion_signal must have shape [Nstate, Nsensor]. "
                f"Got {tuple(self.motion_signal.shape)}."
            )
        self.Nphysio = int(self.motion_signal.shape[1])
        self.motion_plot_context = motion_plot_context or {}
        self.initial_image = initial_image
        self.initial_motion = initial_motion
        self._last_image_cg_info = None
        self._last_motion_cg_info = None
        self.external_image_regularizer = external_image_regularizer
        self._current_level_idx = 0

        # Data changing with resolution
        self.Data_full = {}
        self.Data_full["Nx"] = Nx_full
        self.Data_full["Ny"] = Ny_full
        self.Data_full["Nz"] = self.Nz_full
        self.Data_full["SensitivityMaps"] = smaps
        self.Data_full["KspaceData"] = KspaceData
        self.Data_full["Nsamples"] = sum(
            SamplingIndices[0][ms].numel()
            for ms in range(len(SamplingIndices[0]))
        )
        self.Data_full["SamplingIndices"] = SamplingIndices
        self._initialize_motion_state_schedule()

    def _initialize_motion_state_schedule(self):
        full_states = int(self.params.N_motion_states)
        schedule = getattr(self.params, "N_motion_states_per_level", None)
        if schedule is None:
            schedule = [full_states] * len(self.params.ResolutionLevels)
        self.motion_states_per_level = [int(value) for value in schedule]
        if len(self.motion_states_per_level) != len(self.params.ResolutionLevels):
            raise ValueError(
                "N_motion_states_per_level must have one entry per ResolutionLevels entry."
            )
        if any(value < 1 or value > full_states for value in self.motion_states_per_level):
            raise ValueError(
                f"N_motion_states_per_level values must be between 1 and {full_states}."
            )
        if int(self.motion_signal.shape[0]) != full_states:
            raise ValueError(f"motion_signal has {self.motion_signal.shape[0]} states; expected {full_states}.")
        if self.params.reconstruction_motion_type == "rigid" and any(
            value != full_states for value in self.motion_states_per_level
        ):
            raise ValueError("Per-level motion-state reduction is supported only for non-rigid reconstruction.")

    def _resize_img_xy(self, img, new_size):
        is_complex = img.is_complex()
        target_3d = len(new_size) == 3

        # ---------- Helper: interpolate real/imag ----------
        def interp_part(x):
            """Interpolate real-valued tensor in 2D or 3D spatial coordinates."""
            if target_3d:
                nx_new, ny_new, nz_new = new_size
                if x.ndim == 3:
                    # [Nx, Ny, Nz] -> [1, 1, Nz, Nx, Ny]
                    xv = x.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                    out = F.interpolate(xv, size=(nz_new, nx_new, ny_new), mode="trilinear", align_corners=False)
                    return out[0, 0].permute(1, 2, 0)  # [Nx, Ny, Nz]
                elif x.ndim == 4:
                    # [C, Nx, Ny, Nz] -> [1, C, Nz, Nx, Ny]
                    xv = x.permute(0, 3, 1, 2).unsqueeze(0)
                    out = F.interpolate(xv, size=(nz_new, nx_new, ny_new), mode="trilinear", align_corners=False)
                    return out[0].permute(0, 2, 3, 1)  # [C, Nx, Ny, Nz]
                elif x.ndim == 5:
                    # [C, Nx, Ny, Nz, S] -> [C, Nx_new, Ny_new, Nz_new, S]
                    c, s = x.shape[0], x.shape[-1]
                    xv = x.permute(0, 4, 1, 2, 3).reshape(c * s, x.shape[1], x.shape[2], x.shape[3])
                    xv = xv.permute(0, 3, 1, 2).unsqueeze(0)
                    out = F.interpolate(xv, size=(nz_new, nx_new, ny_new), mode="trilinear", align_corners=False)
                    return out[0].permute(0, 2, 3, 1).reshape(c, s, nx_new, ny_new, nz_new).permute(0, 2, 3, 4, 1)
                else:
                    raise ValueError(f"Unexpected shape {x.shape} for 3D resize.")

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
            elif x.ndim == 4 and x.shape[-1] != 1:
                # [C, Nx, Ny, S] -> [C, Nx_new, Ny_new, S]
                c, s = x.shape[0], x.shape[-1]
                xv = x.permute(0, 3, 1, 2).reshape(c * s, x.shape[1], x.shape[2])
                out_list = []
                for idx in range(c * s):
                    xc = xv[idx].unsqueeze(0).unsqueeze(0)
                    rc = F.interpolate(xc, size=new_size, mode="bilinear", align_corners=False)
                    out_list.append(rc[0, 0])
                return torch.stack(out_list, dim=0).reshape(c, s, new_size[0], new_size[1]).permute(0, 2, 3, 1)
            elif x.ndim == 4 and x.shape[-1] == 1:
                # 2D-with-single-z convention: [C, Nx, Ny, 1] -> [C, Nx_new, Ny_new, 1]
                C = x.shape[0]
                out_list = []
                for c in range(C):
                    xc = x[c, :, :, 0].unsqueeze(0).unsqueeze(0)
                    rc = F.interpolate(xc, size=new_size, mode="bilinear", align_corners=False)
                    out_list.append(rc[0, 0])
                return torch.stack(out_list, dim=0).unsqueeze(-1)
            else:
                raise ValueError(f"Unexpected shape {x.shape}")

        # ---------- Real tensor case ----------
        if not is_complex:
            return interp_part(img)

        # ---------- Complex case ----------
        real = interp_part(img.real)
        imag = interp_part(img.imag)
        return torch.complex(real, imag)

    def _downsample_sampling_indices(self, Sampling_full, Nx_res, Ny_res, Nz_res=1):
        Nx_full, Ny_full = self.Data_full["Nx"], self.Data_full["Ny"]
        Nz_full = int(self.Data_full.get("Nz", 1))

        # central crop coordinates
        x0 = (Nx_full - Nx_res) // 2
        y0 = (Ny_full - Ny_res) // 2
        z0 = (Nz_full - Nz_res) // 2

        Sampling_res = []

        for nex in range(self.params.Nex):
            Sampling_res.append([])
            for indices in Sampling_full[nex]:
                if Nz_full > 1:
                    # Decode flattened 3D index: idx = ((x * Ny) + y) * Nz + z
                    z = indices % Nz_full
                    xy = indices // Nz_full
                    x = xy // Ny_full
                    y = xy % Ny_full
                else:
                    # compute x,y coordinates for 2D flattening
                    x = indices // Ny_full
                    y = indices % Ny_full

                # mask inside central region
                if Nz_full > 1:
                    mask = (
                        (x >= x0) & (x < x0 + Nx_res)
                        & (y >= y0) & (y < y0 + Ny_res)
                        & (z >= z0) & (z < z0 + Nz_res)
                    )
                else:
                    mask = (x >= x0) & (x < x0 + Nx_res) & (y >= y0) & (y < y0 + Ny_res)

                # keep only those indices
                x_crop = x[mask] - x0
                y_crop = y[mask] - y0

                if Nz_full > 1:
                    z_keep = z[mask] - z0
                    # re-flatten for Nx_res × Ny_res × Nz_full grid
                    new_inds = (x_crop * Ny_res + y_crop) * Nz_res + z_keep
                else:
                    # re-flatten for Nx_res × Ny_res grid
                    new_inds = x_crop * Ny_res + y_crop

                Sampling_res[nex].append(new_inds)

        return Sampling_res

    def _downsample_kspace(self, Nx_res, Ny_res, Nz_res=1):
        Nx_full, Ny_full = self.Data_full["Nx"], self.Data_full["Ny"]
        Nz_full = int(self.Data_full.get("Nz", 1))
        kspace_full = self.Data_full["KspaceData"]

        # central crop coordinates
        x0 = (Nx_full - Nx_res) // 2
        y0 = (Ny_full - Ny_res) // 2
        z0 = (Nz_full - Nz_res) // 2

        if Nz_full > 1:
            kspace_res = kspace_full[:, :, x0:x0 + Nx_res, y0:y0 + Ny_res, z0:z0 + Nz_res]
        else:
            kspace_res = kspace_full[:, :, x0:x0 + Nx_res, y0:y0 + Ny_res, :]
        kspace_res = kspace_res.reshape(kspace_full.shape[0], kspace_full.shape[1], -1)

        return kspace_res   

    def _reduce_motion_states(self, sampling_indices, target_states, kspace=None):
        full_states = int(self.motion_signal.shape[0])
        if target_states == full_states:
            return sampling_indices, self.motion_signal

        weights = torch.tensor(
            [sum(sampling_indices[nex][state].numel() for nex in range(self.params.Nex))
             for state in range(full_states)],
            dtype=self.motion_signal.dtype, device=self.device,
        )

        binning_mode = str(
            getattr(self.params, "motion_binning_mode", "kmeans")
        ).strip().lower()
        if binning_mode == "kspace_energy":
            if kspace is None:
                raise ValueError(
                    "Resolution-specific k-space is required for kspace_energy reduction."
                )
            # Recompute state energy after the resolution crop, exactly where
            # GRICS++ selects its resolution-specific virtual times.
            weights.zero_()
            for nex in range(self.params.Nex):
                for state in range(full_states):
                    indices = sampling_indices[nex][state].long()
                    if indices.numel() > 0:
                        weights[state] += kspace[:, nex, indices].abs().square().sum()
            # GRICS++ keeps the highest-energy states at each resolution and
            # attaches every remaining state to its nearest retained state.
            selected = torch.argsort(weights, descending=True, stable=True)[:target_states]
            centers = self.motion_signal[selected].clone()
            labels = torch.cdist(self.motion_signal, centers).argmin(dim=1)
        else:
            # Preserve the original deterministic weighted K-means reduction.
            selected = [int(torch.argmax(weights).item())]
            min_distance = torch.cdist(
                self.motion_signal, self.motion_signal[selected]
            ).squeeze(1)
            while len(selected) < target_states:
                next_idx = int(torch.argmax(min_distance).item())
                selected.append(next_idx)
                distance = torch.cdist(
                    self.motion_signal, self.motion_signal[[next_idx]]
                ).squeeze(1)
                min_distance = torch.minimum(min_distance, distance)

            centers = self.motion_signal[selected].clone()
            for _ in range(20):
                distances = torch.cdist(self.motion_signal, centers)
                labels = distances.argmin(dim=1)
                updated = []
                for cluster in range(target_states):
                    mask = labels == cluster
                    if not mask.any():
                        updated.append(centers[cluster])
                        continue
                    cluster_weights = weights[mask]
                    denominator = torch.clamp(cluster_weights.sum(), min=1.0)
                    updated.append(
                        (self.motion_signal[mask] * cluster_weights[:, None]).sum(dim=0)
                        / denominator
                    )
                new_centers = torch.stack(updated)
                if torch.allclose(new_centers, centers):
                    centers = new_centers
                    break
                centers = new_centers

        reduced = []
        for nex in range(self.params.Nex):
            nex_bins = []
            for cluster in range(target_states):
                members = torch.nonzero(labels == cluster, as_tuple=False).reshape(-1).tolist()
                pieces = [sampling_indices[nex][state] for state in members]
                nex_bins.append(torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long, device=self.device))
            reduced.append(nex_bins)
        return reduced, centers

    def _downsample_data(self, res_factor):    
        Nx = int(round(self.Data_full["Nx"] * res_factor))
        Ny = int(round(self.Data_full["Ny"] * res_factor))
        Nz_full = int(self.Data_full.get("Nz", 1))
        Nz = int(round(Nz_full * res_factor)) if Nz_full > 1 else 1
        Nz = max(Nz, 1)

        Data_res = {}
        Data_res["Nx"] = Nx
        Data_res["Ny"] = Ny
        Data_res["Nz"] = Nz

        resize_shape = (Nx, Ny, Nz) if Nz > 1 else (Nx, Ny)
        Data_res["SensitivityMaps"] = self._resize_img_xy(self.Data_full["SensitivityMaps"], resize_shape)
        sampling_indices = self._downsample_sampling_indices(
            self.Data_full["SamplingIndices"], Nx, Ny, Nz_res=Nz
        )
        Data_res["KspaceData"] = self._downsample_kspace(Nx, Ny, Nz_res=Nz)
        target_states = self.motion_states_per_level[self._current_level_idx]
        Data_res["SamplingIndices"], Data_res["MotionSignal"] = self._reduce_motion_states(
            sampling_indices, target_states, kspace=Data_res["KspaceData"])
        Data_res["Nsamples"] = Data_res["KspaceData"].shape[2]

        return Data_res
    
    def _upsample_data(self, Data_prev, Data_res):
        img_prev = Data_prev["ReconstructedImage"]
        resize_shape = (
            (Data_res["Nx"], Data_res["Ny"], Data_res["Nz"])
            if int(Data_res.get("Nz", 1)) > 1 else
            (Data_res["Nx"], Data_res["Ny"])
        )
        img_res = self._resize_img_xy(img_prev, resize_shape)
        Data_res["ReconstructedImage"] = img_res

        mot_prev = Data_prev["MotionModel"]
        if self.params.reconstruction_motion_type == "rigid":
            Data_res["MotionModel"] = torch.zeros((self.Nalpha, self.params.N_motion_states), device=self.device)
            Data_res["MotionModel"][0,:] = mot_prev[0,:] * Data_res["Nx"] / Data_prev["Nx"]  # scale translations
            Data_res["MotionModel"][1,:] = mot_prev[1,:] * Data_res["Ny"] / Data_prev["Ny"]  # scale translations
            if self.Nalpha > 3:
                Data_res["MotionModel"][2,:] = mot_prev[2,:] * Data_res.get("Nz", 1) / max(1, Data_prev.get("Nz", 1))
                Data_res["MotionModel"][3:,:] = mot_prev[3:,:]
            else:
                Data_res["MotionModel"][2,:] = mot_prev[2,:]  # rotations remain the same
        else:
            resize_shape = (
                (Data_res["Nx"], Data_res["Ny"], Data_res["Nz"])
                if int(Data_res.get("Nz", 1)) > 1 else
                (Data_res["Nx"], Data_res["Ny"])
            )
            mot_res = self._resize_img_xy(mot_prev, resize_shape)
            mot_res[0] = mot_res[0] * Data_res["Nx"] / Data_prev["Nx"]
            mot_res[1] = mot_res[1] * Data_res["Ny"] / Data_prev["Ny"]
            if mot_res.shape[0] > 2 and int(Data_res.get("Nz", 1)) > 1:
                mot_res[2] = mot_res[2] * Data_res["Nz"] / max(1, Data_prev["Nz"])
            Data_res["MotionModel"] = mot_res

    def _build_motion_operator(self, Data_res):
        Nx, Ny = Data_res["Nx"], Data_res["Ny"]
        alpha = Data_res["MotionModel"]
        if self.params.reconstruction_motion_type == "rigid":
            motionOperator = MotionOperator(
                Nx, Ny, alpha, self.params.reconstruction_motion_type, Nz=Data_res.get("Nz", 1)
            )
        else:
            motion_signal = Data_res["MotionSignal"]
            motionOperator = MotionOperator(
                Nx, Ny, alpha, self.params.reconstruction_motion_type,
                motion_signal=motion_signal.to(dtype=alpha.dtype), Nz=Data_res.get("Nz", 1)
            )
        return motionOperator

    def _build_encoding_operator(self, Data_res):
        E = EncodingOperator(Data_res["SensitivityMaps"], Data_res["Nsamples"], Data_res["SamplingIndices"],
                             self.params.Nex, Data_res["MotionOperator"])
        return E
    
    def _build_motion_perturbation_simulator(self, Data_res):
        J = MotionPerturbationSimulator(Data_res["SensitivityMaps"], Data_res["Nsamples"], Data_res["SamplingIndices"],
                                        self.params.Nex, Data_res["ReconstructedImage"], Data_res["MotionOperator"])
        return J

    def _lambda_r_for_level(self):
        lambda_r = self.params.lambda_r
        if isinstance(lambda_r, (list, tuple)):
            if len(lambda_r) == 0:
                raise ValueError("lambda_r list/tuple cannot be empty.")
            if len(lambda_r) != len(self.params.ResolutionLevels):
                raise ValueError(
                    "Inconsistent config: "
                    f"lambda_r has {len(lambda_r)} values, "
                    f"but ResolutionLevels has {len(self.params.ResolutionLevels)} values."
                )
            return float(lambda_r[self._current_level_idx])
        return float(lambda_r)

    def _solve_image(
        self, Data_res, *, image_prior=None, regularization_weight=None,
        differentiable=False, max_iterations=None,
    ):
        x0 = Data_res["ReconstructedImage"].to(
            self.device, dtype=torch.complex128
        )
        E = Data_res["E"]

        b = E.adjoint(Data_res["KspaceData"])
        if regularization_weight is None:
            regularization_weight = self._lambda_r_for_level()
        if max_iterations is None:
            max_iterations = self.params.max_iter_recon
        solver = ConjugateGradientSolver(
            E, reg_lambda=regularization_weight, verbose=self.params.verbose, early_stopping=self.params.cg_early_stopping,
            true_residual_interval=self.params.cg_true_residual_interval, max_stag_steps=self.params.cg_max_stag_steps,
            max_more_steps=self.params.cg_max_more_steps, use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
            reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
        )
        _assign_cached_reg_scale(self.params, Data_res, "image", solver, b.flatten())

        # A supplied prior implements lambda * ||x - prior||_2^2.
        if image_prior is not None:
            if image_prior.shape != x0.shape:
                raise ValueError(
                    "image_regularizer must preserve the image shape; "
                    f"got {tuple(image_prior.shape)}, expected {tuple(x0.shape)}."
                )
            b = b.flatten() + solver._effective_lambda() * image_prior.to(b.dtype).flatten()

        img_vec = solver.cg(
            b.flatten(), x0=x0.flatten(), max_iter=max_iterations,
            tol=self.params.tol_recon, differentiable=differentiable,
        )
        self._last_image_cg_info = solver.last_info

        if int(Data_res.get("Nz", 1)) > 1:
            img = img_vec.reshape(self.params.Nex, Data_res["Nx"], Data_res["Ny"], Data_res["Nz"])
        else:
            img = img_vec.reshape(self.params.Nex, Data_res["Nx"], Data_res["Ny"])
        return img

    def _n_motion_params(self, Data_res):
        if self.params.reconstruction_motion_type == "rigid":
            return self.Nalpha * self.params.N_motion_states
        return self.Nalpha * self.Nphysio * Data_res["Nx"] * Data_res["Ny"] * int(Data_res.get("Nz", 1))

    def _solve_motion(self, Data_res, residual, *, max_iterations=None):
        if max_iterations is None:
            max_iterations = self.params.max_iter_motion
        Nparams = self._n_motion_params(Data_res)
        J = Data_res["J"]
        b_data = J.adjoint(residual)
        x0 = torch.zeros(Nparams, dtype=b_data.dtype, device=residual.device)

        if self.params.reconstruction_motion_type == "non-rigid":
            reg_shape = (
                (self.Nalpha, Data_res["Nx"], Data_res["Ny"], int(Data_res.get("Nz", 1)), self.Nphysio)
                if int(Data_res.get("Nz", 1)) > 1
                else (self.Nalpha, Data_res["Nx"], Data_res["Ny"], self.Nphysio)
            )
            solver = ConjugateGradientSolver(
                J, reg_lambda=self.params.lambda_m, regularizer="Tikhonov_gradient",
                regularization_shape=reg_shape, regularization_spatial_dims=(1, 2, 3) if int(Data_res.get("Nz", 1)) > 1 else (1, 2), verbose=self.params.verbose,
                early_stopping=self.params.cg_early_stopping, true_residual_interval=self.params.cg_true_residual_interval,
                max_stag_steps=self.params.cg_max_stag_steps, max_more_steps=self.params.cg_max_more_steps,
                use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy, reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
            )
            # Unscaled _regularization:
            # _A(dm) = J^H J dm + mu * GhG(dm)
            # b     = J^H r    - mu * GhG(alpha_current)
            _assign_cached_reg_scale(self.params, Data_res, "motion_nonrigid", solver, b_data.flatten())
            b = b_data - solver._effective_lambda() * solver._regularization(Data_res["MotionModel"].flatten())
            mot_pert_vec = solver.cg(b.flatten(), x0=x0.flatten(), max_iter=max_iterations, tol=self.params.tol_motion)
        else:
            solver = ConjugateGradientSolver(
                J, reg_lambda=self.params.lambda_m, verbose=self.params.verbose, early_stopping=self.params.cg_early_stopping,
                true_residual_interval=self.params.cg_true_residual_interval, max_stag_steps=self.params.cg_max_stag_steps,
                max_more_steps=self.params.cg_max_more_steps, use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
                reg_scale_num_probes=self.params.cg_reg_scale_num_probes,
            )
            _assign_cached_reg_scale(self.params, Data_res, "motion_rigid", solver, b_data.flatten())
            mot_pert_vec = solver.cg(b_data.flatten(), x0=x0.flatten(), max_iter=max_iterations, tol=self.params.tol_motion)
        self._last_motion_cg_info = solver.last_info

        if self.params.reconstruction_motion_type == "rigid":
            motion_perturb = mot_pert_vec.reshape(self.Nalpha, self.params.N_motion_states)
        else:
            if int(Data_res.get("Nz", 1)) > 1:
                motion_perturb = mot_pert_vec.reshape(self.Nalpha, Data_res["Nx"], Data_res["Ny"], Data_res["Nz"], self.Nphysio)
            else:
                motion_perturb = mot_pert_vec.reshape(self.Nalpha, Data_res["Nx"], Data_res["Ny"], self.Nphysio)
        return motion_perturb
    

    @property
    def last_image_cg_info(self):
        """Diagnostics from the most recent image CG solve."""
        return self._last_image_cg_info

    @property
    def last_motion_cg_info(self):
        """Diagnostics from the most recent motion CG solve."""
        return self._last_motion_cg_info

    def gauss_newton_iteration(
        self, data, *, image_regularizer=None, regularization_weight=None,
        update_motion=True, image_cg_iterations=None, motion_cg_iterations=None):
        # ------------------------------- IMAGE RECONSTRUCTION STEP -------------------------

        # 1) Build motion and encoding operators
        data["MotionModel"] = data["MotionModel"].detach()
        data["MotionOperator"] = self._build_motion_operator(data)
        data["E"] = self._build_encoding_operator(data)

        # Evaluate the optional CNN prior z = D(image).
        prior = None
        differentiable_image = image_regularizer is not None and torch.is_grad_enabled()
        if image_regularizer is not None:
            prior = image_regularizer(data["ReconstructedImage"])
            if not torch.is_tensor(prior):
                raise TypeError("image_regularizer must return a torch.Tensor.")
            if prior.device != data["ReconstructedImage"].device:
                raise ValueError("image_regularizer must preserve the image device.")

        # 2) Solve for image
        # When a prior z exists, solve with lambda ||x - z||_2^2.
        image_t0 = time.perf_counter()
        image = self._solve_image(
            data, image_prior=prior, regularization_weight=regularization_weight,
            differentiable=differentiable_image, max_iterations=image_cg_iterations)
        image_elapsed = time.perf_counter() - image_t0
        data["ReconstructedImage"] = image

        # 3) Compute the residual used by the forward-only motion update.
        # The learned image remains differentiable through the CG solution,
        # but motion is deliberately detached in GRICS-Net. Building this
        # additional encoding pass with gradient recording would retain a
        # second full-resolution operator graph until the motion step ends and
        # can nearly double the peak memory of an unrolled level.
        with torch.no_grad():
            predicted = data["E"].forward(image.detach().flatten())
            residual = data["KspaceData"].flatten() - predicted
        motion_for_residual = data["MotionModel"]
        motion_update = None
        motion_elapsed = None

        # ------------------------------- MOTION MODEL RECONSTRUCTION STEP -------------------------
        if update_motion:
            # 4) Build linearized motion-perturbation simulator around current
            # estimate: ∇_u(E)·δu = δkspace
            motion_t0 = time.perf_counter()
            with torch.no_grad():
                motion_data = dict(data)
                motion_data["ReconstructedImage"] = image.detach()
                motion_data["MotionModel"] = motion_for_residual.detach()
                motion_data["J"] = self._build_motion_perturbation_simulator(motion_data)

                # 5) Solve for motion update
                motion_update = self._solve_motion(
                    motion_data, residual.detach(), max_iterations=motion_cg_iterations)
                motion = (motion_for_residual.detach() + motion_update.real).detach()
            motion_elapsed = time.perf_counter() - motion_t0
        else:
            motion = motion_for_residual.detach()

        # Discard iteration-only operators before carrying the state forward.
        self._strip_level_runtime_state(data)
        data["ReconstructedImage"] = image
        data["MotionModel"] = motion
        return _GaussNewtonIterationResult(
            image, motion, predicted, residual, motion_for_residual, motion_update, image_elapsed, motion_elapsed)

    def _prepare_resolution_level(self, idx_res, r):
        _console(self.params, f"\n=== Resolution level {idx_res+1}: factor {r} ===")

        # Prepare low-resolution dataset
        Data_res = self._downsample_data(r)

        # Initialize image and motion model
        if idx_res == 0:
            if int(Data_res.get("Nz", 1)) > 1:
                Data_res["ReconstructedImage"] = torch.zeros(
                    (self.params.Nex, Data_res["Nx"], Data_res["Ny"], Data_res["Nz"]),
                    dtype=torch.complex128, device=self.device)
            else:
                Data_res["ReconstructedImage"] = torch.zeros((self.params.Nex, Data_res["Nx"], Data_res["Ny"]), dtype=torch.complex128, device=self.device)
            
            if self.params.reconstruction_motion_type == "rigid":
                Data_res["MotionModel"] = torch.zeros((self.Nalpha, self.params.N_motion_states), device=self.device)
            elif self.params.reconstruction_motion_type == "non-rigid":
                if int(Data_res.get("Nz", 1)) > 1:
                    Data_res["MotionModel"] = torch.zeros(
                        (self.Nalpha, Data_res["Nx"], Data_res["Ny"], Data_res["Nz"], self.Nphysio),
                        device=self.device)
                else:
                    Data_res["MotionModel"] = torch.zeros((self.Nalpha, Data_res["Nx"], Data_res["Ny"], self.Nphysio), device=self.device)
            self._apply_external_initializer(Data_res)
        return Data_res

    def _apply_external_initializer(self, data):
        """Initialize the coarsest level from optional full-resolution estimates."""
        spatial = ((data["Nx"], data["Ny"], data["Nz"]) if int(data.get("Nz", 1)) > 1 else (data["Nx"], data["Ny"]))
        if self.initial_image is not None:
            image = torch.as_tensor(self.initial_image, device=self.device)
            if image.ndim == len(spatial):
                image = image.unsqueeze(0)
            if image.shape[0] == 1 and self.params.Nex > 1:
                image = image.expand(self.params.Nex, *image.shape[1:])
            if image.ndim != len(spatial) + 1 or image.shape[0] != self.params.Nex:
                raise ValueError(f"Invalid initial_image shape {tuple(image.shape)}.")
            data["ReconstructedImage"] = self._resize_img_xy(image.to(torch.complex128), spatial)
        if self.initial_motion is not None:
            motion = torch.as_tensor(self.initial_motion, device=self.device, dtype=torch.float64)
            if self.params.reconstruction_motion_type == "non-rigid":
                if motion.ndim == len(spatial) + 1:
                    motion = motion.unsqueeze(-1)
                if motion.shape[0] != self.Nalpha or motion.shape[-1] != self.Nphysio:
                    raise ValueError(f"Invalid initial_motion shape {tuple(motion.shape)}; expected Nalpha={self.Nalpha}, Nsensor={self.Nphysio}.")
                data["MotionModel"] = self._resize_img_xy(motion, spatial)
            else:
                expected = (self.Nalpha, self.params.N_motion_states)
                if tuple(motion.shape) != expected:
                    raise ValueError(f"Rigid initial_motion must have shape {expected}.")
                data["MotionModel"] = motion

    @staticmethod
    def _strip_level_runtime_state(data):
        if data is None:
            return None
        for key in ("MotionOperator", "E", "J"):
            data.pop(key, None)
        return data

    @staticmethod
    def _make_next_level_initializer(data):
        if data is None:
            return None
        return {"Nx": data["Nx"], "Ny": data["Ny"], "Nz": data.get("Nz", 1),
                "ReconstructedImage": data["ReconstructedImage"], "MotionModel": data["MotionModel"]}

    def _run_resolution_level(
        self, data, *, level_index, level_iterations, level_count,
        update_final_motion, gn_early_stopping, logger):
        """Run all configured GN iterations for one prepared resolution."""
        measured_norm = torch.linalg.norm(data["KspaceData"].flatten()).item()
        reconstruction_residuals, motion_residuals, best_relative_residual, best_image, best_motion = _initialize_level_tracking()

        with logger.progress(level_index, level_iterations, level_count) as progress:
            for iteration_index in range(level_iterations):
                logger.announce_iteration(iteration_index, level_iterations)
                iteration_t0 = time.perf_counter()
                is_final_iteration = (level_index == level_count - 1 and iteration_index == level_iterations - 1)
                update_motion = not is_final_iteration or update_final_motion

                result = self.gauss_newton_iteration(data, image_regularizer=self.external_image_regularizer,
                    regularization_weight=self._lambda_r_for_level(), update_motion=update_motion)
                relative_residual = torch.linalg.norm(result.residual).item() / (measured_norm + 1e-12)
                reconstruction_residuals.append(relative_residual)
                logger.show_residual(progress, relative_residual)

                # Stop a diverging level and restore its best completed state.
                if gn_early_stopping and iteration_index > 0 and relative_residual > best_relative_residual:
                    data["ReconstructedImage"] = best_image
                    data["MotionModel"] = best_motion
                    logger.iteration_stopped_early(progress)
                    break

                best_relative_residual = relative_residual
                best_image = result.image.clone()
                best_motion = result.motion_for_residual.clone()

                relative_motion_update = None
                motion_update_norm = None
                if update_motion:
                    # 6) Compute the normalized motion-update residual.
                    motion_update_norm = torch.linalg.norm(result.motion_update.flatten()).item()
                    motion_norm = torch.linalg.norm(result.motion.flatten()).item()
                    relative_motion_update = motion_update_norm / (motion_norm + 1e-12)
                    motion_residuals.append(relative_motion_update)

                logger.iteration_finished(
                    iteration_index=iteration_index, result=result, relative_residual=relative_residual,
                    image_cg_info=self._last_image_cg_info, motion_cg_info=self._last_motion_cg_info,
                    relative_motion_update=relative_motion_update, motion_update_norm=motion_update_norm,
                    elapsed=time.perf_counter() - iteration_t0)
                logger.update_progress(progress)

        return reconstruction_residuals, motion_residuals, best_image, best_motion

    def _save_final_outputs(self, image, motion):
        """Save final reconstructed images and motion diagnostics."""
        if image.shape[0] == 1:
            show_and_save_image(image[0], "image_reconstructed", self.params.results_folder,
                flip_for_display=self.params.flip_for_display)
        else:
            show_and_save_image(image.mean(dim=0), "image_reconstructed", self.params.results_folder,
                flip_for_display=self.params.flip_for_display)
            for nex_index in range(image.shape[0]):
                show_and_save_image(image[nex_index], f"image_reconstructed_nex{nex_index + 1}",
                    self.params.results_folder, flip_for_display=self.params.flip_for_display)

        if self.params.reconstruction_motion_type == "rigid":
            save_final_rigid_motion_plots(motion, self.motion_plot_context, self.params.results_folder,
                self.params.N_motion_states, self.params.ResolutionLevels, self.params.data_type)
        elif self.params.reconstruction_motion_type == "non-rigid":
            save_final_nonrigid_alpha_maps(motion, image[0], self.params.results_folder,
                flip_for_display=self.params.flip_for_display, motion_plot_context=self.motion_plot_context)

    # ----------------------------------------------------------------------
    # Perform full multi-resolution Gauss-Newton joint reconstruction
    # ----------------------------------------------------------------------
    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image, motion)`` from the configured multi-resolution run.

        The image is ``[Ne, Nx, Ny, (Nz)]``. Motion is ``[Nalpha, Nm]`` for
        rigid or ``[Nalpha, Nx, Ny, (Nz), Ns]`` for non-rigid reconstruction.
        """
        resolution_levels = self.params.ResolutionLevels
        iterations_per_level = _parse_gn_iterations_per_level(self.params, resolution_levels)
        save_outputs = bool(getattr(self.params, "save_reconstruction_outputs", True))
        gn_early_stopping = bool(getattr(self.params, "gn_early_stopping", True))
        update_final_motion = bool(getattr(self.params, "update_motion_on_final_iteration", False))
        logger = _JointReconstructionLogger(self.params, iterations_per_level)
        run_t0 = time.perf_counter()
        previous = None
        final_best_image = None
        final_best_motion = None

        # Advance through the configured spatial resolutions, coarse to fine.
        for level_index, resolution in enumerate(resolution_levels):
            self._current_level_idx = level_index
            level_t0 = time.perf_counter()

            # Prepare this resolution and initialize it from the preceding level.
            data = self._prepare_resolution_level(level_index, resolution)
            if previous is not None:
                self._upsample_data(previous, data)
            logger.level_started(level_index, data, self._lambda_r_for_level(),
                time.perf_counter() - level_t0)

            # Run the alternating image/motion updates at this resolution.
            reconstruction_residuals, motion_residuals, best_image, best_motion = self._run_resolution_level(
                data, level_index=level_index, level_iterations=iterations_per_level[level_index],
                level_count=len(resolution_levels), update_final_motion=update_final_motion,
                gn_early_stopping=gn_early_stopping, logger=logger)

            # Save optional level diagnostics without mixing them into GN logic.
            if save_outputs and self.params.debug_flag:
                show_and_save_image(data["ReconstructedImage"][0],
                    f"image_resolution_level{level_index + 1}", self.params.debug_folder,
                    flip_for_display=self.params.flip_for_display)
                _save_nonrigid_motion_debug(data, level_index + 1, self.params.reconstruction_motion_type,
                    self.params.debug_folder, self.params.flip_for_display)

            logger.level_finished(level_index, reconstruction_residuals, motion_residuals,
                time.perf_counter() - level_t0)

            # Retain only the image and motion needed to initialize the next level.
            self._strip_level_runtime_state(data)
            previous = self._make_next_level_initializer(data)
            final_best_image = best_image
            final_best_motion = best_motion

        logger.run_finished(time.perf_counter() - run_t0)
        if previous is None:
            raise RuntimeError("Reconstruction did not produce a valid image/motion solution.")

        # A requested final motion update returns the completed final GN state.
        if update_final_motion:
            final_image = previous["ReconstructedImage"]
            final_motion = previous["MotionModel"]
        else:
            final_image = final_best_image if final_best_image is not None else previous["ReconstructedImage"]
            final_motion = final_best_motion if final_best_motion is not None else previous["MotionModel"]

        image_unscaled = final_image * self.kspace_scale
        if save_outputs:
            self._save_final_outputs(image_unscaled, final_motion)
        return image_unscaled, final_motion

    # ----------------------------------------------------------------------
    # External integration API
    # ----------------------------------------------------------------------
    def _full_resolution_iteration_data(self, image, motion, sampling_indices=None):
        spatial_shape = ((self.Data_full["Nx"], self.Data_full["Ny"], self.Data_full["Nz"])
                         if self.Nz_full > 1 else (self.Data_full["Nx"], self.Data_full["Ny"]))
        image = torch.as_tensor(image, device=self.device)
        squeeze_nex = image.ndim == len(spatial_shape)
        if squeeze_nex:
            image = image.unsqueeze(0)
        expected_image_shape = (self.params.Nex, *spatial_shape)
        if tuple(image.shape) != expected_image_shape:
            raise ValueError(f"image must have shape {expected_image_shape} or {spatial_shape}; got {tuple(image.shape)}.")

        self._current_level_idx = len(self.params.ResolutionLevels) - 1
        return {
            "Nx": self.Data_full["Nx"], "Ny": self.Data_full["Ny"], "Nz": self.Data_full["Nz"],
            "SensitivityMaps": self.Data_full["SensitivityMaps"],
            "KspaceData": self.Data_full["KspaceData"].reshape(self.Ncoils, self.params.Nex, -1),
            "Nsamples": self.Data_full["Nx"] * self.Data_full["Ny"] * self.Data_full["Nz"],
            "SamplingIndices": self.Data_full["SamplingIndices"] if sampling_indices is None else sampling_indices,
            "MotionSignal": self.motion_signal, "ReconstructedImage": image,
            "MotionModel": torch.as_tensor(motion, device=self.device), "_squeeze_nex": squeeze_nex,
        }

    def full_resolutions_gauss_newton_iteration_api(
        self, image: torch.Tensor, motion: torch.Tensor, *,
        image_regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        regularization_weight: float | None = None, update_motion: bool = True,
        image_cg_iterations: int | None = None,
        motion_cg_iterations: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one full-resolution image update and optional motion update.

        Args:
            image: Complex ``[Ne, Nx, Ny, (Nz)]`` tensor; the ``Ne`` axis may
                be omitted only when ``Ne=1``.
            motion: Real ``[Nalpha, Nm]`` rigid tensor or
                ``[Nalpha, Nx, Ny, (Nz), Ns]`` non-rigid tensor.
            image_regularizer: Optional same-shape image-prior callable.
            regularization_weight: Non-negative prior weight, or ``None`` for
                configured ``lambda_r``.
            update_motion: Whether to run motion after the image step.
            image_cg_iterations: Positive limit, or ``None`` for the config.
            motion_cg_iterations: Positive limit, or ``None`` for the config.
        """
        input_image_dtype = torch.as_tensor(image).dtype
        data = self._full_resolution_iteration_data(image, motion)
        result = self.gauss_newton_iteration(
            data, image_regularizer=image_regularizer, regularization_weight=regularization_weight,
            update_motion=update_motion, image_cg_iterations=image_cg_iterations,
            motion_cg_iterations=motion_cg_iterations)
        output_image = result.image[0] if data["_squeeze_nex"] else result.image
        return output_image.to(input_image_dtype), result.motion

    def predict_kspace_api(
        self, image: torch.Tensor, motion: torch.Tensor, *,
        sampling_indices: list[list[torch.Tensor]] | None = None) -> torch.Tensor:
        """Return flattened complex predicted k-space.

        Args:
            image: Complex ``[Ne, Nx, Ny, (Nz)]`` tensor; the ``Ne`` axis may
                be omitted only when ``Ne=1``.
            motion: Real ``[Nalpha, Nm]`` rigid tensor or
                ``[Nalpha, Nx, Ny, (Nz), Ns]`` non-rigid tensor.
            sampling_indices: Optional nested ``[Ne][Nm]`` lists of 1D integer
                tensors; ``None`` reuses the constructor layout.

        Returns:
            Complex vector with ``Nc * Ne * Nx * Ny * Nz`` elements.
        """
        data = self._full_resolution_iteration_data(image, motion, sampling_indices)
        data["MotionModel"] = data["MotionModel"].detach()
        data["MotionOperator"] = self._build_motion_operator(data)
        data["E"] = self._build_encoding_operator(data)
        prediction = data["E"].forward(data["ReconstructedImage"].to(torch.complex128).flatten())
        self._strip_level_runtime_state(data)
        return prediction

import math
import torch
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from src.reconstruction.joint_reconstructor_utils.resampling import (
    downsample_data, upsample_data, resize_img_xy,
)
from src.reconstruction.ConjugateGadientSolver import ConjugateGradientSolver
from src.reconstruction.joint_reconstructor_utils.configuration import (
    _parse_gn_iterations_per_level, configure_motion_states_per_resolution_level,
    image_regularization_weight_for_level,
)
from src.reconstruction.joint_reconstructor_utils.initialization import (
    apply_external_initializer, initialize_zero_image_and_motion,
)
from src.reconstruction.joint_reconstructor_utils.operators import (
    build_motion_operator, build_encoding_operator, build_motion_perturbation_simulator,
)
from src.reconstruction.joint_reconstructor_utils.state import (
    remove_temporary_operators, extract_image_and_motion_for_next_level,
)
from src.reconstruction.joint_reconstructor_utils.regularization import _assign_cached_reg_scale
from src.reconstruction.joint_reconstructor_utils.calibration_prior import CalibrationPriorEncodingOperator
from src.reconstruction.joint_reconstructor_utils.logging import JointReconstructionLogger
from src.reconstruction.joint_reconstructor_utils.timing import Timer



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
        external_image_regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        calibration_image_prior: torch.Tensor | None = None,
        voxel_spacing_mm: tuple[float, ...] | None = None):
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
            calibration_image_prior: Optional nonnegative calibration magnitude
                used as the multiplicative GRICS++ image constraint.
            voxel_spacing_mm: Physical spacing of the full-resolution encoded
                image grid; required by GRICS++ motion scaling.
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
        self.regularization_scaling = getattr(params, 'regularization_scaling', 'direct')
        self.use_calibration_image_prior = getattr(params, 'use_calibration_image_prior', False)
        if self.regularization_scaling not in ('direct', 'grics_cpp'):
            raise ValueError('regularization_scaling must be direct or grics_cpp.')
        if self.use_calibration_image_prior and external_image_regularizer is not None:
            raise ValueError('Calibration image prior cannot be combined with an external image regularizer.')
        spatial_shape = (Nx_full, Ny_full, self.Nz_full) if self.Nz_full > 1 else (Nx_full, Ny_full)
        if self.use_calibration_image_prior:
            if calibration_image_prior is None:
                raise ValueError('Calibration image prior is enabled, but no calibration image was supplied.')
            calibration = torch.as_tensor(calibration_image_prior, device=self.device)
            if self.Nz_full == 1 and tuple(calibration.shape) == (Nx_full, Ny_full, 1):
                calibration = calibration[..., 0]
            if (tuple(calibration.shape) != spatial_shape or calibration.is_complex()
                    or not torch.isfinite(calibration).all() or torch.any(calibration < 0)):
                raise ValueError(f'Calibration image prior must be a finite nonnegative real image of shape {spatial_shape}.')
            self.calibration_image_prior = calibration.to(torch.float64)
        else:
            self.calibration_image_prior = None
        if self.regularization_scaling == 'grics_cpp':
            if (voxel_spacing_mm is None or len(voxel_spacing_mm) != len(spatial_shape)
                    or any(not math.isfinite(float(v)) or float(v) <= 0 for v in voxel_spacing_mm)):
                raise ValueError('GRICS++ scaling requires positive voxel_spacing_mm.')
        self.voxel_spacing_mm = tuple(float(v) for v in voxel_spacing_mm) if voxel_spacing_mm is not None else None
        self.kspace_scale = float(kspace_scale)
        if not math.isfinite(self.kspace_scale) or self.kspace_scale <= 0:
            raise ValueError('kspace_scale must be positive and finite.')
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
        if self.calibration_image_prior is not None:
            self.Data_full['CalibrationImagePrior'] = self.calibration_image_prior
        self.motion_states_per_level = configure_motion_states_per_resolution_level(
            self.params, self.motion_signal)

    def _solve_image(
        self, Data_res, *, image_prior=None, regularization_weight=None,
        differentiable=False, max_iterations=None,
    ):
        """Solve the regularized image problem with motion held fixed.

        Without calibration, CG solves (E^H E + lambda I)x = E^H y,
        optionally with a centered prior on the RHS. With calibration C,
        solve for p using E*C, then return x=C*p as in GRICS++.
        """
        # Start CG from the current image estimate at this resolution.
        x0 = Data_res["ReconstructedImage"].to(
            self.device, dtype=torch.complex128
        )
        E = Data_res["E"]
        calibration = Data_res.get("CalibrationImagePrior")
        if calibration is not None:
            if image_prior is not None:
                raise ValueError("Calibration image prior cannot be combined with a centered image prior.")
            calibration = calibration.unsqueeze(0).expand(self.params.Nex, *calibration.shape)
            E = CalibrationPriorEncodingOperator(E, calibration)
            x0 = torch.where(calibration > 0, x0 / calibration.clamp_min(1e-12), torch.zeros_like(x0))

        # Back-project measured k-space to form the normal-equation RHS.
        b = E.adjoint(Data_res["KspaceData"])
        if regularization_weight is None:
            regularization_weight = image_regularization_weight_for_level(self.params, self._current_level_idx)
        if max_iterations is None:
            max_iterations = self.params.max_iter_recon
        solver = ConjugateGradientSolver(
            E, reg_lambda=regularization_weight, regularizer="Tikhonov", regularization_shape=None,
            regularization_spatial_dims=None, verbose=self.params.verbose, stop_on_stagnation=self.params.cg_stop_on_stagnation,
            true_residual_interval=self.params.cg_true_residual_interval, stagnation_consecutive_steps=self.params.cg_stagnation_consecutive_steps,
            stagnation_countdown_steps=self.params.cg_stagnation_countdown_steps, use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
            reg_scale_num_probes=(self.params.cg_reg_scale_num_probes
                                  if self.params.cg_use_reg_scale_proxy else None),
        )
        # Reuse the level's scale so regularization stays consistent across solves.
        _assign_cached_reg_scale(self.params, Data_res, "image", solver, b.flatten())
        if self.regularization_scaling == "grics_cpp":
            n_pixels = Data_res["Nx"] * Data_res["Ny"] * int(Data_res.get("Nz", 1))
            full_pixels = self.Data_full["Nx"] * self.Data_full["Ny"] * self.Data_full["Nz"]
            # C++ uses the raw-data adjoint norm; y here was divided by kspace_scale.
            solver.reg_scale = ((n_pixels / full_pixels) ** 0.5 * self.kspace_scale
                                * torch.linalg.norm(b.flatten()).item())

        # A supplied prior implements lambda * ||x - prior||_2^2.
        if image_prior is not None:
            if image_prior.shape != x0.shape:
                raise ValueError(
                    "image_regularizer must preserve the image shape; "
                    f"got {tuple(image_prior.shape)}, expected {tuple(x0.shape)}."
                )
            b = b.flatten() + solver._effective_lambda() * image_prior.to(b.dtype).flatten()

        # Keep gradients through CG when training with an external image prior.
        img_vec = solver.cg(
            b.flatten(), x0=x0.flatten(), max_iter=max_iterations,
            tol=self.params.tol_recon, differentiable=differentiable,
        )
        self._last_image_cg_info = solver.last_info
        if calibration is not None:
            img_vec = calibration.flatten() * img_vec

        # Convert the flattened CG solution back to the image's spatial layout.
        if int(Data_res.get("Nz", 1)) > 1:
            img = img_vec.reshape(self.params.Nex, Data_res["Nx"], Data_res["Ny"], Data_res["Nz"])
        else:
            img = img_vec.reshape(self.params.Nex, Data_res["Nx"], Data_res["Ny"])
        return img

    def _grics_cpp_level_spacing_and_ratio(self, data):
        level_shape = (data["Nx"], data["Ny"]) if self.Nz_full == 1 else (data["Nx"], data["Ny"], data["Nz"])
        full_shape = (self.Data_full["Nx"], self.Data_full["Ny"]) if self.Nz_full == 1 else (self.Data_full["Nx"], self.Data_full["Ny"], self.Data_full["Nz"])
        spacing = tuple(self.voxel_spacing_mm[i] * full_shape[i] / level_shape[i]
                        for i in range(len(level_shape)))
        return spacing, (math.prod(level_shape) / math.prod(full_shape)) ** 0.5

    def _n_motion_params(self, Data_res):
        if self.params.reconstruction_motion_type == "rigid":
            return self.Nalpha * self.params.N_motion_states
        return self.Nalpha * self.Nphysio * Data_res["Nx"] * Data_res["Ny"] * int(Data_res.get("Nz", 1))

    def _solve_motion(self, Data_res, residual, *, max_iterations=None):
        """Solve for a motion increment using the current linearized encoding.

        J maps motion increments to k-space changes. The caller adds the
        returned increment to the current motion estimate.
        """
        if max_iterations is None:
            max_iterations = self.params.max_iter_motion
        Nparams = self._n_motion_params(Data_res)
        J = Data_res["J"]
        # Project the k-space mismatch into motion-parameter space. Start from
        # zero because CG estimates an increment, not the full motion model.
        b_data = J.adjoint(residual)
        x0 = torch.zeros(Nparams, dtype=b_data.dtype, device=residual.device)

        if self.params.reconstruction_motion_type == "non-rigid":
            # Penalize spatial variation of the updated motion field; do not
            # differentiate along the component or physiological-sensor axes.
            reg_shape = (
                (self.Nalpha, Data_res["Nx"], Data_res["Ny"], int(Data_res.get("Nz", 1)), self.Nphysio)
                if int(Data_res.get("Nz", 1)) > 1
                else (self.Nalpha, Data_res["Nx"], Data_res["Ny"], self.Nphysio)
            )
            solver = ConjugateGradientSolver(
                J, reg_lambda=self.params.lambda_m, regularizer="Tikhonov_gradient",
                regularization_shape=reg_shape, regularization_spatial_dims=(1, 2, 3) if int(Data_res.get("Nz", 1)) > 1 else (1, 2), verbose=self.params.verbose,
                stop_on_stagnation=self.params.cg_stop_on_stagnation, true_residual_interval=self.params.cg_true_residual_interval,
                stagnation_consecutive_steps=self.params.cg_stagnation_consecutive_steps, stagnation_countdown_steps=self.params.cg_stagnation_countdown_steps,
                use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy, reg_scale_num_probes=(self.params.cg_reg_scale_num_probes
                                  if self.params.cg_use_reg_scale_proxy else None),
            )
            # Regularize the total field alpha_current + dm, which contributes
            # the current field's penalty to the RHS. G is a spatial gradient.
            # _A(dm) = J^H J dm + mu * GhG(dm)
            # b     = J^H r    - mu * GhG(alpha_current)
            _assign_cached_reg_scale(self.params, Data_res, "motion_nonrigid", solver, b_data.flatten())
            if self.regularization_scaling == "grics_cpp":
                spacing, ratio = self._grics_cpp_level_spacing_and_ratio(Data_res)
                solver.regularization_spacing = spacing
                # J and residual scale together, so k-space normalization cancels.
                solver.reg_scale = ratio * min(spacing) ** 4 * torch.linalg.norm(b_data.flatten()).item()
            if getattr(self.params, "use_motion_preconditioner", False):
                diagonal = J.approximate_normal_diagonal()
                diagonal = diagonal + solver._effective_lambda() * solver._gradient_diagonal(
                    dtype=diagonal.dtype, device=diagonal.device)
                if not torch.isfinite(diagonal).all():
                    raise ValueError("Motion preconditioner has non-finite diagonal entries.")
                floor = torch.clamp(diagonal.max() * 1e-8, min=1e-12)
                inverse_diagonal = diagonal.clamp_min(floor).reciprocal()
                solver.preconditioner = lambda residual: inverse_diagonal * residual
            b = b_data - solver._effective_lambda() * solver._regularization(Data_res["MotionModel"].flatten())
            mot_pert_vec = solver.cg(b.flatten(), x0=x0.flatten(), max_iter=max_iterations, tol=self.params.tol_motion)
        else:
            # Rigid motion uses a magnitude penalty on the increment itself:
            # (J^H J + mu I) dm = J^H residual.
            solver = ConjugateGradientSolver(
                J, reg_lambda=self.params.lambda_m, regularizer="Tikhonov", regularization_shape=None,
                regularization_spatial_dims=None, verbose=self.params.verbose, stop_on_stagnation=self.params.cg_stop_on_stagnation,
                true_residual_interval=self.params.cg_true_residual_interval, stagnation_consecutive_steps=self.params.cg_stagnation_consecutive_steps,
                stagnation_countdown_steps=self.params.cg_stagnation_countdown_steps, use_reg_scale_proxy=self.params.cg_use_reg_scale_proxy,
                reg_scale_num_probes=(self.params.cg_reg_scale_num_probes
                                  if self.params.cg_use_reg_scale_proxy else None),
            )
            _assign_cached_reg_scale(self.params, Data_res, "motion_rigid", solver, b_data.flatten())
            if self.regularization_scaling == "grics_cpp":
                spacing, ratio = self._grics_cpp_level_spacing_and_ratio(Data_res)
                solver.reg_scale = ratio * min(spacing) ** 4 * torch.linalg.norm(b_data.flatten()).item()
            mot_pert_vec = solver.cg(b_data.flatten(), x0=x0.flatten(), max_iter=max_iterations, tol=self.params.tol_motion)
        self._last_motion_cg_info = solver.last_info

        # Restore either per-state rigid parameters or spatial motion fields.
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
        data["MotionOperator"] = build_motion_operator(data, self.params)
        data["E"] = build_encoding_operator(data, self.params)

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
        with Timer() as image_timer:
            image = self._solve_image(
                data, image_prior=prior, regularization_weight=regularization_weight,
                differentiable=differentiable_image, max_iterations=image_cg_iterations)
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
            with Timer() as motion_timer, torch.no_grad():
                motion_data = dict(data)
                motion_data["ReconstructedImage"] = image.detach()
                motion_data["MotionModel"] = motion_for_residual.detach()
                motion_data["J"] = build_motion_perturbation_simulator(motion_data, self.params)

                # 5) Solve for motion update
                motion_update = self._solve_motion(
                    motion_data, residual.detach(), max_iterations=motion_cg_iterations)
                motion = (motion_for_residual.detach() + motion_update.real).detach()
            motion_elapsed = motion_timer.elapsed
        else:
            motion = motion_for_residual.detach()

        # Discard iteration-only operators before carrying the state forward.
        remove_temporary_operators(data)
        data["ReconstructedImage"] = image
        data["MotionModel"] = motion
        return _GaussNewtonIterationResult(
            image, motion, predicted, residual, motion_for_residual, motion_update, image_timer.elapsed, motion_elapsed)

    def _prepare_resolution_level(self, idx_res, r):
        # Prepare low-resolution dataset
        Data_res = downsample_data(
            self.Data_full, r,
            target_states=self.motion_states_per_level[self._current_level_idx],
            motion_signal=self.motion_signal, params=self.params, device=self.device,
        )
        if self.calibration_image_prior is not None:
            shape = ((Data_res["Nx"], Data_res["Ny"], Data_res["Nz"]) if self.Nz_full > 1
                     else (Data_res["Nx"], Data_res["Ny"]))
            Data_res["CalibrationImagePrior"] = (
                self.calibration_image_prior if tuple(self.calibration_image_prior.shape) == shape
                else resize_img_xy(self.calibration_image_prior, shape)
            )

        # Initialize image and motion model
        if idx_res == 0:
            initialize_zero_image_and_motion(
                Data_res, params=self.params, device=self.device,
                Nalpha=self.Nalpha, Nphysio=self.Nphysio,
            )
            apply_external_initializer(
                Data_res, initial_image=self.initial_image, initial_motion=self.initial_motion,
                params=self.params, device=self.device, Nalpha=self.Nalpha, Nphysio=self.Nphysio,
            )
        return Data_res

    def _run_resolution_level(
        self, data, *, level_index, gauss_newton_iterations_at_level, level_count,
        update_final_motion, gn_early_stopping, logger):
        """Alternate image/motion updates and return the last accepted estimate pair.

        Updates data in place. If early stopping detects a residual increase,
        restore the saved image/motion pair before leaving this level. Without
        early stopping, accept every iteration regardless of residual changes.
        """
        # Normalize all iteration residuals against the same measured-data norm.
        measured_norm = torch.linalg.norm(data["KspaceData"].flatten()).item()
        best_relative_residual = float("inf")
        best_image = best_motion = None

        with logger.iterations(level_index) as gauss_newton_iteration_indices:
            for gauss_newton_iteration_index in gauss_newton_iteration_indices:
                is_last_at_level = gauss_newton_iteration_index == gauss_newton_iterations_at_level - 1
                if getattr(self.params, "image_only_last_iteration_per_level", False):
                    # GRICS++ does not carry an unevaluated motion update into the next level.
                    update_motion = not is_last_at_level or (level_index == level_count - 1 and update_final_motion)
                else:
                    update_motion = not (level_index == level_count - 1 and is_last_at_level) or update_final_motion

                result = self.gauss_newton_iteration(
                    data, image_regularizer=self.external_image_regularizer,
                    regularization_weight=image_regularization_weight_for_level(
                        self.params, self._current_level_idx),
                    update_motion=update_motion)
                # This residual uses the newly solved image and the motion
                # estimate from before the current motion update.
                relative_residual = torch.linalg.norm(result.residual).item() / (measured_norm + 1e-12)
                logger.record_residual(relative_residual)

                # Stop a diverging level and restore its best completed state.
                if gn_early_stopping and gauss_newton_iteration_index > 0 and relative_residual > best_relative_residual:
                    data["ReconstructedImage"] = best_image
                    data["MotionModel"] = best_motion
                    logger.iteration_stopped_early()
                    break

                # Save the pair that produced this residual. Clone it so later
                # updates cannot overwrite the state used for restoration.
                best_relative_residual = relative_residual
                best_image = result.image.clone()
                best_motion = result.motion_for_residual.clone()

                logger.iteration_finished(
                    result, relative_residual, self._last_image_cg_info, self._last_motion_cg_info)

        return best_image, best_motion

    # ----------------------------------------------------------------------
    # Perform full multi-resolution Gauss-Newton joint reconstruction
    # ----------------------------------------------------------------------
    def run(self, *, defer_tensor_export: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image, motion)`` from the configured multi-resolution run.

        defer_tensor_export leaves enabled tensor exports to the caller without
        changing configuration. Logs, metadata and plots retain their normal behavior.

        The image is ``[Ne, Nx, Ny, (Nz)]``. Motion is ``[Nalpha, Nm]`` for
        rigid or ``[Nalpha, Nx, Ny, (Nz), Ns]`` for non-rigid reconstruction.
        """
        if type(defer_tensor_export) is not bool:
            raise ValueError('defer_tensor_export must be a boolean.')
        resolution_levels = self.params.ResolutionLevels
        iterations_per_level = _parse_gn_iterations_per_level(self.params, resolution_levels)
        gn_early_stopping = bool(self.params.gn_early_stopping)
        update_final_motion = bool(self.params.update_motion_on_final_iteration)
        logger = JointReconstructionLogger(self.params, iterations_per_level, self.motion_plot_context)
        if defer_tensor_export and self.params.save_reconstruction_tensors:
            logger.append("Tensor export: deferred to caller; save_reconstruction_tensors remains enabled.")
        logger.start_run()
        previous = None
        final_best_image = None
        final_best_motion = None

        # Advance through the configured spatial resolutions, coarse to fine.
        for level_index, resolution in enumerate(resolution_levels):
            self._current_level_idx = level_index
            logger.start_level(level_index, resolution)

            # Prepare this resolution and initialize it from the preceding level.
            data = self._prepare_resolution_level(level_index, resolution)
            if previous is not None:
                upsample_data(previous, data, self.params, self.Nalpha, self.device)
            logger.level_prepared(
                data, image_regularization_weight_for_level(self.params, self._current_level_idx))

            # Run the alternating image/motion updates at this resolution.
            best_image, best_motion = self._run_resolution_level(
                data, level_index=level_index, gauss_newton_iterations_at_level=iterations_per_level[level_index],
                level_count=len(resolution_levels), update_final_motion=update_final_motion,
                gn_early_stopping=gn_early_stopping, logger=logger)

            logger.level_finished(data)

            # Retain only the image and motion needed to initialize the next level.
            remove_temporary_operators(data)
            previous = extract_image_and_motion_for_next_level(data)
            final_best_image = best_image
            final_best_motion = best_motion

        logger.run_finished()
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
        logger.save_final_outputs(image_unscaled, final_motion, defer_tensor_export=defer_tensor_export)
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
            "CalibrationImagePrior": self.calibration_image_prior,
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
        data["MotionOperator"] = build_motion_operator(data, self.params)
        data["E"] = build_encoding_operator(data, self.params)
        prediction = data["E"].forward(data["ReconstructedImage"].to(torch.complex128).flatten())
        remove_temporary_operators(data)
        return prediction

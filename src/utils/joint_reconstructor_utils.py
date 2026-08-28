import os
from contextlib import nullcontext

import torch
from tqdm.auto import tqdm

from src.utils.plotting import save_nonrigid_alpha_plots, save_residual_subplots


def _format_cg_info(cg_info):
    if cg_info is None:
        return "flag = -1, relres = nan, iter = 0"
    return (
        f"flag = {cg_info.get('flag', -1)}, "
        f"relres = {cg_info.get('relres', float('nan')):.6e}, "
        f"iter = {cg_info.get('iterations', 0)}"
    )


def _console(params, message):
    if params.print_to_console:
        print(message)


def _assign_cached_reg_scale(params, Data_res, cache_key, solver, reference_vec):
    if not params.cg_use_reg_scale_proxy:
        solver.reg_scale = 1.0
        return

    cache = Data_res.setdefault("_reg_scale_cache", {})
    if cache_key not in cache:
        cache[cache_key] = solver._update_regularization_scale(reference_vec)
    solver.reg_scale = cache[cache_key]


def _parse_gn_iterations_per_level(params, res_levels):
    gn_cfg = params.GN_iterations_per_level
    if isinstance(gn_cfg, int):
        return [gn_cfg] * len(res_levels)
    if isinstance(gn_cfg, (list, tuple)):
        if len(gn_cfg) == 0:
            raise ValueError("GN_iterations_per_level list/tuple cannot be empty.")
        gn_list = [int(v) for v in gn_cfg]
        if len(gn_list) != len(res_levels):
            raise ValueError(
                "Inconsistent config: "
                f"GN_iterations_per_level has {len(gn_list)} values, "
                f"but ResolutionLevels has {len(res_levels)} values."
            )
        return gn_list
    raise ValueError("GN_iterations_per_level must be int, list, or tuple.")


def _init_run_logging(params, n_levels, gn_iters_per_level):
    os.makedirs(params.logs_folder, exist_ok=True)
    log_path = os.path.join(params.logs_folder, "joint_reconstruction.log")
    param_items = {}
    simulation_param_keys = {"simulated_motion_type", "num_motion_events", "max_tx", "max_ty", "max_phi",
                             "max_center_x", "max_center_y", "seed", "motion_tau", "nonrigid_motion_amplitude"}
    for key in dir(params):
        if key.startswith("_"):
            continue
        if key in simulation_param_keys:
            continue
        value = getattr(params, key)
        if callable(value):
            continue
        param_items[key] = value

    with open(log_path, "w") as f:
        f.write("Joint reconstruction run\n")
        f.write(f"Reconstruction motion type: {params.reconstruction_motion_type}\n")
        f.write(f"Simulated motion type: {params.simulated_motion_type}\n")
        f.write(f"GN iterations per level: {gn_iters_per_level}\n\n")
        f.write("Parameters (excluding simulation parameters):\n")
        for key in sorted(param_items.keys()):
            f.write(f"  {key} = {param_items[key]}\n")
        f.write("\n")
    return {
        "path": log_path,
        "recon_residuals_by_level": [[] for _ in range(n_levels)],
        "motion_residuals_by_level": [[] for _ in range(n_levels)],
    }


def _append_run_log(run_log, line=""):
    with open(run_log["path"], "a") as f:
        f.write(line + "\n")


def _save_run_residual_plots(logs_folder, run_log):
    recon_path = os.path.join(logs_folder, "recon_residual.png")
    motion_path = os.path.join(logs_folder, "motion_residual.png")
    save_residual_subplots(run_log["recon_residuals_by_level"], title="Reconstruction residuals",
                           y_label="Relative residual", out_path=recon_path)
    save_residual_subplots(run_log["motion_residuals_by_level"], title="Motion normalized residuals",
                           y_label="||dm||2 / (||alpha||2 + eps)", out_path=motion_path)


def _initialize_level_tracking():
    residual_recon_norms = []
    residual_motion_norms = []
    best_relres = float("inf")
    best_image = None
    best_motion = None
    return residual_recon_norms, residual_motion_norms, best_relres, best_image, best_motion


def _save_nonrigid_motion_debug(Data_res, level_idx, motion_type, debug_folder, flip_for_display):
    if motion_type != "non-rigid":
        return

    alpha = Data_res["MotionModel"]
    if alpha.shape[0] < 2:
        return

    image = Data_res["ReconstructedImage"][0]
    if alpha.ndim in (3, 4) and not (alpha.ndim == 4 and image.ndim == 2):
        save_nonrigid_alpha_plots(
            alpha, image,
            f"level{level_idx}", debug_folder,
            flip_vertical=flip_for_display,
        )
    elif alpha.ndim in (4, 5):
        for sensor_idx in range(alpha.shape[-1]):
            save_nonrigid_alpha_plots(
                alpha[..., sensor_idx], image,
                f"level{level_idx}_sensor{sensor_idx + 1}", debug_folder,
                flip_vertical=flip_for_display,
            )


class _JointReconstructionLogger:
    """Own run logging, progress display, residual history, and residual plots.

    This helper deliberately does not decide whether an iteration should stop,
    whether motion should be updated, or which reconstruction state is kept.
    Those algorithm decisions remain in JointReconstructor.
    """

    def __init__(self, params, iterations_per_level):
        self.params = params
        self.enabled = bool(getattr(params, "save_reconstruction_outputs", True))
        n_levels = len(iterations_per_level)
        self.run_log = (
            _init_run_logging(params, n_levels, iterations_per_level)
            if self.enabled else {
                "recon_residuals_by_level": [[] for _ in range(n_levels)],
                "motion_residuals_by_level": [[] for _ in range(n_levels)],
            }
        )

    def append(self, message=""):
        if self.enabled:
            _append_run_log(self.run_log, message)

    def progress(self, level_index, level_iterations, level_count):
        if level_iterations <= 0:
            return nullcontext()
        return tqdm(
            total=level_iterations,
            desc=f"Resolution level {level_index + 1}/{level_count}",
            disable=not self.params.jupyter_notebook_flag,
            leave=True, dynamic_ncols=True, position=0,
        )

    def announce_iteration(self, iteration_index, level_iterations):
        _console(self.params, f"  GN iteration {iteration_index + 1}/{level_iterations}")

    @staticmethod
    def show_residual(progress, relative_residual):
        if progress is not None:
            progress.set_postfix(recon=f"{relative_residual:.2e}")

    @staticmethod
    def update_progress(progress):
        if progress is not None:
            progress.update(1)

    def level_started(self, level_index, data, regularization_weight, elapsed):
        self.append(
            f"Resolution level {level_index} "
            f"({data['Nx']}x{data['Ny']}x{data.get('Nz', 1)}, "
            f"{data['Ny']} views, "
            f"{len(data['SamplingIndices'][0])} virtual times)\n"
            f"    lambda_r : {regularization_weight:.6e}\n"
            f"    Resolution level initializations : {elapsed:.6f} s\n"
        )

    def iteration_stopped_early(self, progress):
        message = "    Relative residual increased - restoring best solution at this level."
        _console(self.params, message)
        self.append(message)
        self.update_progress(progress)

    def iteration_finished(
        self, *, iteration_index, result, relative_residual,
        image_cg_info, motion_cg_info, relative_motion_update=None,
        motion_update_norm=None, elapsed,
    ):
        self.append(
            "    Reconstruction step : "
            f"{_format_cg_info(image_cg_info)}, "
            f"elapsed time = {result.image_elapsed:.6f} s"
        )
        if relative_motion_update is None:
            self.append(
                f"    Fixed point iter {iteration_index}: "
                f"recon_rel_residual = {relative_residual:.6e}, "
                f"image_only = True : {elapsed:.6f} s\n"
            )
            return
        self.append(
            "    Model optimization step: "
            f"{_format_cg_info(motion_cg_info)}, "
            f"elapsed time = {result.motion_elapsed:.6f} s\n"
            f"    Fixed point iter {iteration_index}: "
            f"recon_rel_residual = {relative_residual:.6e}, "
            f"motion_rel_residual = {relative_motion_update:.6e}, "
            f"motion_norm = {motion_update_norm:.6e} : {elapsed:.6f} s\n"
        )

    def level_finished(self, level_index, reconstruction_residuals, motion_residuals, elapsed):
        self.run_log["recon_residuals_by_level"][level_index] = reconstruction_residuals
        self.run_log["motion_residuals_by_level"][level_index] = motion_residuals
        self.append(f"    Total time of resolution level {level_index}: {elapsed:.6f} s\n")

    def run_finished(self, elapsed):
        self.append(f"Total time of reconstruction run: {elapsed:.6f} s")
        if self.enabled:
            _save_run_residual_plots(self.params.logs_folder, self.run_log)

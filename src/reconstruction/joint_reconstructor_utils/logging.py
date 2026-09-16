"""Run logging, progress display, and diagnostic plots for joint reconstruction."""

import os
import time
from contextlib import contextmanager, nullcontext

import torch
from tqdm.auto import tqdm

from src.utils.plotting import save_nonrigid_alpha_plots, save_residual_subplots, show_and_save_image
from src.utils.save_final_motion_plots import save_final_nonrigid_alpha_maps, save_final_rigid_motion_plots


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


class JointReconstructionLogger:
    """Own run logging, progress display, residual history, and residual plots.

    This helper deliberately does not decide whether an iteration should stop,
    whether motion should be updated, or which reconstruction state is kept.
    Those algorithm decisions remain in JointReconstructor.
    """

    def __init__(self, params, iterations_per_level, motion_plot_context=None):
        self.params = params
        self.motion_plot_context = motion_plot_context or {}
        self.iterations_per_level = iterations_per_level
        self._progress = None
        self.enabled = bool(params.save_reconstruction_outputs)
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

    def iteration_stopped_early(self):
        message = "    Relative residual increased - restoring best solution at this level."
        _console(self.params, message)
        self.append(message)
        self.update_progress(self._progress)

    def _write_iteration(
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

    def start_run(self):
        self._run_started = time.perf_counter()

    def start_level(self, level_index, resolution):
        self._level_started = time.perf_counter()
        self._level_index = level_index
        _console(self.params, f"\n=== Resolution level {level_index + 1}: factor {resolution} ===")

    def level_prepared(self, data, regularization_weight):
        self.level_started(
            self._level_index, data, regularization_weight,
            time.perf_counter() - self._level_started)

    @contextmanager
    def iterations(self, level_index):
        """Own progress lifetime and iteration timing, including early exits."""
        count = self.iterations_per_level[level_index]
        with self.progress(level_index, count, len(self.iterations_per_level)) as progress:
            self._progress = progress
            try:
                yield self._iteration_indices(count)
            finally:
                self._progress = None

    def _iteration_indices(self, count):
        for index in range(count):
            self.announce_iteration(index, count)
            self._iteration_started = time.perf_counter()
            self._iteration_index = index
            yield index

    def record_residual(self, relative_residual):
        self.run_log["recon_residuals_by_level"][self._level_index].append(relative_residual)
        self.show_residual(self._progress, relative_residual)

    def iteration_finished(self, result, relative_residual, image_cg_info, motion_cg_info):
        relative_motion_update = None
        motion_update_norm = None
        if result.motion_update is not None:
            motion_update_norm = torch.linalg.norm(result.motion_update.flatten()).item()
            motion_norm = torch.linalg.norm(result.motion.flatten()).item()
            relative_motion_update = motion_update_norm / (motion_norm + 1e-12)
            self.run_log["motion_residuals_by_level"][self._level_index].append(relative_motion_update)
        self._write_iteration(
            iteration_index=self._iteration_index, result=result, relative_residual=relative_residual,
            image_cg_info=image_cg_info, motion_cg_info=motion_cg_info,
            relative_motion_update=relative_motion_update, motion_update_norm=motion_update_norm,
            elapsed=time.perf_counter() - self._iteration_started)
        self.update_progress(self._progress)

    def level_finished(self, data):
        if self.enabled and self.params.save_debug_plots:
            show_and_save_image(
                data["ReconstructedImage"][0], f"image_resolution_level{self._level_index + 1}",
                self.params.debug_folder, flip_for_display=self.params.flip_for_display)
            _save_nonrigid_motion_debug(
                data, self._level_index + 1, self.params.reconstruction_motion_type,
                self.params.debug_folder, self.params.flip_for_display)
        self.append(
            f"    Total time of resolution level {self._level_index}: "
            f"{time.perf_counter() - self._level_started:.6f} s\n")

    def run_finished(self):
        self.append(f"Total time of reconstruction run: {time.perf_counter() - self._run_started:.6f} s")
        if self.enabled:
            _save_run_residual_plots(self.params.logs_folder, self.run_log)

    def save_final_outputs(self, image, motion):
        """Save final reconstructed images and motion diagnostics."""
        if not self.enabled:
            return
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

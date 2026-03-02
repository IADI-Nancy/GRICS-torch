import os
import torch

from src.utils.plotting import (
    save_alpha_component_map, save_nonrigid_quiver_with_contours, save_residual_subplots,
)
from src.utils.nonrigid_display import to_cartesian_components


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


def _initialize_global_tracking():
    global_best_metric = float("inf")
    global_best_image = None
    global_best_motion = None
    global_converged = False
    return global_best_metric, global_best_image, global_best_motion, global_converged


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
    simulation_param_keys = {"motion_simulation_type", "num_motion_events", "max_tx", "max_ty", "max_phi",
                             "max_center_x", "max_center_y", "seed", "motion_tau", "nonrigid_motion_amplitude",
                             "displacementfield_size"}
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
        f.write(f"Motion type: {params.motion_type}\n")
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


def _update_global_best(best_relres, best_image, best_motion, global_best_metric, global_best_image, global_best_motion):
    if best_relres < global_best_metric:
        global_best_metric = best_relres
        global_best_image = best_image.clone()
        global_best_motion = best_motion.clone()
    return global_best_metric, global_best_image, global_best_motion


def _save_nonrigid_motion_debug(Data_res, level_idx, motion_type, debug_folder, flip_for_display):
    if motion_type != "non-rigid":
        return

    alpha = Data_res["MotionModel"]
    if alpha.ndim != 3 or alpha.shape[0] < 2:
        return

    os.makedirs(debug_folder, exist_ok=True)

    alpha_x = alpha[0].detach().cpu()
    alpha_y = alpha[1].detach().cpu()
    alpha_x_cart, alpha_y_cart = to_cartesian_components(alpha_x, alpha_y)

    if torch.is_complex(alpha_x) or torch.is_complex(alpha_y):
        components = (("alpha_x_real", alpha_x_cart.real), ("alpha_y_real", alpha_y_cart.real),
                      ("alpha_x_imag", alpha_x_cart.imag), ("alpha_y_imag", alpha_y_cart.imag))
        alpha_x_for_quiver = alpha_x.real
        alpha_y_for_quiver = alpha_y.real
    else:
        components = (("alpha_x", alpha_x_cart), ("alpha_y", alpha_y_cart))
        alpha_x_for_quiver = alpha_x
        alpha_y_for_quiver = alpha_y

    for comp_name, comp in components:
        save_alpha_component_map(comp, f"{comp_name} level {level_idx}",
                                 os.path.join(debug_folder, f"{comp_name}_level{level_idx}.png"),
                                 flip_vertical=flip_for_display)

    save_nonrigid_quiver_with_contours(
        alpha_x_for_quiver, alpha_y_for_quiver, Data_res["ReconstructedImage"][0],
        f"motion field level {level_idx}",
        os.path.join(debug_folder, f"motion_quiver_level{level_idx}.png"),
        flip_vertical=flip_for_display,
    )

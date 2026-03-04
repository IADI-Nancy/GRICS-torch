import os
import torch

from src.utils.plotting import (
    save_alpha_component_map, save_nonrigid_quiver_with_contours, save_clustered_motion_plots,
)
from src.utils.nonrigid_display import to_cartesian_components


def save_final_nonrigid_alpha_maps(motion_model, reconstructed_image, results_folder, flip_for_display=True, motion_plot_context=None):
    if motion_model is None or motion_model.ndim != 3 or motion_model.shape[0] < 2:
        return

    os.makedirs(results_folder, exist_ok=True)
    context = motion_plot_context or {}

    alpha_x = motion_model[0].detach().cpu()
    alpha_y = motion_model[1].detach().cpu()
    alpha_x_cart, alpha_y_cart = to_cartesian_components(alpha_x, alpha_y)
    scale = context.get("alpha_visual_scale", None)
    alpha_abs_max_x = None if scale is None else scale.get("alpha_abs_max_x")
    alpha_abs_max_y = None if scale is None else scale.get("alpha_abs_max_y")
    amp_max = None if scale is None else scale.get("amp_max")

    if torch.is_complex(alpha_x) or torch.is_complex(alpha_y):
        components = (("final_alpha_x_real", alpha_x_cart.real), ("final_alpha_y_real", alpha_y_cart.real),
                      ("final_alpha_x_imag", alpha_x_cart.imag), ("final_alpha_y_imag", alpha_y_cart.imag))
    else:
        components = (("final_alpha_x", alpha_x_cart), ("final_alpha_y", alpha_y_cart))

    for name, comp in components:
        if "alpha_x" in name:
            abs_max = alpha_abs_max_x
        elif "alpha_y" in name:
            abs_max = alpha_abs_max_y
        else:
            abs_max = None
        save_alpha_component_map(comp, name, os.path.join(results_folder, f"{name}.png"),
                                 flip_vertical=flip_for_display, abs_max=abs_max)

    save_nonrigid_quiver_with_contours(
        alpha_x if not torch.is_complex(alpha_x) else alpha_x.real,
        alpha_y if not torch.is_complex(alpha_y) else alpha_y.real, reconstructed_image, "final_motion_quiver",
        os.path.join(results_folder, "final_motion_quiver.png"), flip_vertical=flip_for_display, amp_vmax=amp_max,
    )


def save_final_rigid_motion_plots(motion_model, motion_plot_context, results_folder, n_motion_states, resolution_levels, data_type):
    if motion_model is None or motion_model.ndim != 2 or motion_model.shape[0] < 3:
        return

    context = motion_plot_context or {}
    motion_curve = context.get("motion_curve")
    labels_in = context.get("labels")
    ky_idx = context.get("ky_idx")
    nex_idx = context.get("nex_idx")
    if motion_curve is None or labels_in is None or ky_idx is None or nex_idx is None:
        return

    labels = labels_in.to(dtype=torch.long, device=motion_model.device)
    tx = motion_model[0, labels]
    ty = motion_model[1, labels]
    phi = None
    tz = None
    rx = None
    ry = None
    rz = None
    if motion_model.shape[0] >= 6:
        tz = motion_model[2, labels]
        rx = motion_model[3, labels]
        ry = motion_model[4, labels]
        rz = motion_model[5, labels]
    else:
        phi = motion_model[2, labels]

    save_clustered_motion_plots(
        motion_curve=motion_curve, labels=labels_in, ky_idx=ky_idx, nex_idx=nex_idx, nbins=n_motion_states,
        output_folder=results_folder, resolution_levels=context.get("resolution_levels", resolution_levels),
        tx=tx, ty=ty, phi=phi, tz=tz, rx=rx, ry=ry, rz=rz,
        data_type=context.get("data_type", data_type), y_limits=context.get("y_limits"),
    )

import torch

from src.utils.plotting import (
    save_clustered_motion_plots, save_nonrigid_alpha_plots,
)


def save_final_nonrigid_alpha_maps(motion_model, reconstructed_image, results_folder, flip_for_display=True, motion_plot_context=None):
    if motion_model is None or motion_model.ndim not in (3, 4) or motion_model.shape[0] < 2:
        return

    context = motion_plot_context or {}
    scale = context.get("alpha_visual_scale", None)
    save_nonrigid_alpha_plots(
        motion_model, reconstructed_image,
        "final", results_folder,
        flip_vertical=flip_for_display,
        abs_max_x=None if scale is None else scale.get("alpha_abs_max_x"),
        abs_max_y=None if scale is None else scale.get("alpha_abs_max_y"),
        amp_max=None if scale is None else scale.get("amp_max"),
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

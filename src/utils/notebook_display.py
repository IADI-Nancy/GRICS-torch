from pathlib import Path
import warnings

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch


def _load_image(path):
    p = Path(path)
    if not p.exists():
        return None
    return mpimg.imread(p)

def _first_existing_path(*paths):
    for path in paths:
        if Path(path).exists():
            return str(path)
    return str(paths[0])


def _first_existing_glob(folder, *patterns):
    folder = Path(folder)
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return str(matches[0])
    return str(folder / patterns[0])


def _infer_has_ground_truth(params):
    motion_sim_type = params.motion_simulation_type
    if motion_sim_type is None:
        return params.data_type == "shepp-logan"
    return motion_sim_type not in {"as-it-is", "no-motion-data"}


def _display_image_row(image_paths, subtitles, title=None, figsize=None):
    present = []
    for path, subtitle in zip(image_paths, subtitles):
        img = _load_image(path)
        if img is not None:
            present.append((img, subtitle))

    if not present:
        return

    n = len(present)
    if figsize is None:
        if n == 1:
            figsize = (7.0, 7.0)
        elif n == 2:
            figsize = (10.0, 4.8)
        else:
            figsize = (13.0, 4.8)

    fig, axes = plt.subplots(1, n, figsize=figsize, gridspec_kw={"wspace": 0.02})
    if n == 1:
        axes = [axes]

    for ax, (img, subtitle) in zip(axes, present):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(subtitle, fontsize=10, pad=4)

    if title:
        fig.suptitle(title, fontsize=11, y=0.985)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"This figure includes Axes that are not compatible with tight_layout.*",
            category=UserWarning,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), pad=0.15)
    plt.show()


def display_run_panels(params, motion_type, has_ground_truth=None, jupyter_notebook_flag=None):
    if jupyter_notebook_flag is None:
        jupyter_notebook_flag = bool(getattr(params, "jupyter_notebook_flag", False))
    if not jupyter_notebook_flag:
        return
    if has_ground_truth is None:
        has_ground_truth = _infer_has_ground_truth(params)

    logs_folder = Path(params.logs_folder)
    input_folder = Path(params.input_data_folder)
    results_folder = Path(params.results_folder)

    recon_logs = sorted(logs_folder.glob("recon_residual.png"))
    motion_logs = sorted(logs_folder.glob("motion_residual.png"))
    logs_figsize = (13.0, 4.8) if has_ground_truth else (10.0, 4.8)
    if motion_logs:
        _display_image_row(
            [str(motion_logs[-1])],
            [""],
            title=None,
            figsize=logs_figsize,
        )
    if recon_logs:
        _display_image_row(
            [str(recon_logs[-1])],
            [""],
            title=None,
            figsize=logs_figsize,
        )

    image_paths = [
        _first_existing_path(input_folder / "image_corrupted.png", input_folder / "input_distorted.png"),
        _first_existing_path(results_folder / "image_reconstructed.png", results_folder / "image_reconstructed_nex1.png"),
    ]
    subtitles = ["Corrupted", "Corrected"]
    images_figsize = None
    if has_ground_truth:
        image_paths.append(
            _first_existing_path(input_folder / "image_ground_truth.png", input_folder / "input_ground_truth.png")
        )
        subtitles.append("Ground truth")
        images_figsize = (13.0, 4.8)
    else:
        images_figsize = (10.0, 4.8)
    _display_image_row(image_paths, subtitles, title="Images", figsize=images_figsize)

    if motion_type == "rigid":
        _display_image_row(
            [
                str(input_folder / "clustered_motion_curves_chronological.png"),
                str(results_folder / "clustered_motion_curves_chronological.png"),
            ],
            ["Simulated / input motion (chronological)", "Reconstructed motion (chronological)"],
            title="Rigid Motion",
            figsize=(images_figsize[0], 3.4),
        )
    elif motion_type == "non-rigid":
        _display_image_row(
            [
                str(input_folder / "simulated_motion_quiver_input.png"),
                str(results_folder / "final_motion_quiver.png"),
            ],
            ["", ""],
            title="Non-rigid motion model",
        )


def display_input_sampling_motion_panels(params, has_ground_truth=None, jupyter_notebook_flag=None):
    if jupyter_notebook_flag is None:
        jupyter_notebook_flag = bool(getattr(params, "jupyter_notebook_flag", False))
    if not jupyter_notebook_flag:
        return
    if has_ground_truth is None:
        has_ground_truth = _infer_has_ground_truth(params)

    input_folder = Path(params.input_data_folder)
    row_width = 13.0 if has_ground_truth else 10.0
    sampling_path = _first_existing_glob(
        input_folder,
        "ky_order_nex*.png",
        "ky_order_realworld_slice*.png",
        "ky_order_rawdata_slice*.png",
        "ky_sampling_order.png",
    )
    motion_vs_ky_path = str(input_folder / "clustered_motion_curve_sorted_ky.png")

    _display_image_row(
        [sampling_path, motion_vs_ky_path],
        ["Sampling order (ky)", "Motion curve in ky order"],
        title="Sampling And Motion",
        figsize=(row_width, 4.8),
    )


def _central_planes_3d(vol):
    ix = vol.shape[0] // 2
    iy = vol.shape[1] // 2
    iz = vol.shape[2] // 2
    return (vol[:, :, iz], vol[:, iy, :], vol[ix, :, :])


def _normalize_per_volume(vol, p_low=2.0, p_high=98.0):
    vals = vol.reshape(-1)
    lo = float(torch.quantile(vals, p_low / 100.0).item())
    hi = float(torch.quantile(vals, p_high / 100.0).item())
    if hi <= lo:
        hi = lo + 1e-12
    return torch.clamp((vol - lo) / (hi - lo), 0.0, 1.0)


def display_3d_image_matrix(image_uncorrected, image_corrected, image_gt):
    vols = [
        _normalize_per_volume(torch.abs(image_uncorrected).detach().cpu()),
        _normalize_per_volume(torch.abs(image_corrected).detach().cpu()),
        _normalize_per_volume(torch.abs(image_gt).detach().cpu()),
    ]
    col_titles = ["Uncorrected", "Motion-corrected", "Ground truth"]
    row_titles = ["Mid-axial (XY)", "Mid-coronal (XZ)", "Mid-sagittal (YZ)"]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), gridspec_kw={"wspace": 0.02, "hspace": 0.08})
    for col, vol in enumerate(vols):
        planes = _central_planes_3d(vol)
        for row in range(3):
            ax = axes[row, col]
            ax.imshow(torch.flipud(planes[row]).numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, pad=8)
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=10)

    fig.suptitle("3D Image Comparison", fontsize=12, y=0.995)
    plt.show()


def display_logs_and_motion_same_as_2d(params):
    logs_folder = Path(params.logs_folder)
    input_folder = Path(params.input_data_folder)
    results_folder = Path(params.results_folder)

    recon_logs = sorted(logs_folder.glob("recon_residual.png"))
    motion_logs = sorted(logs_folder.glob("motion_residual.png"))
    logs_figsize = (13.0, 4.8)

    if motion_logs:
        _display_image_row([str(motion_logs[-1])], [""], title=None, figsize=logs_figsize)
    if recon_logs:
        _display_image_row([str(recon_logs[-1])], [""], title=None, figsize=logs_figsize)

    _display_image_row(
        [
            str(input_folder / "clustered_motion_curves_chronological.png"),
            str(results_folder / "clustered_motion_curves_chronological.png"),
        ],
        ["Simulated / input motion (chronological)", "Reconstructed motion (chronological)"],
        title="Rigid Motion",
        figsize=(13.0, 3.4),
    )

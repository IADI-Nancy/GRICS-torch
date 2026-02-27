from pathlib import Path
import warnings

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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


def display_image_row(image_paths, subtitles, title=None, figsize=None):
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


def display_run_panels(params, motion_type, has_ground_truth=None, jupyter_notebook_flag=False):
    if not jupyter_notebook_flag:
        return
    if has_ground_truth is None:
        has_ground_truth = _infer_has_ground_truth(params)

    logs_folder = Path(params.logs_folder)
    input_folder = Path(params.input_data_folder)
    results_folder = Path(params.results_folder)

    recon_logs = sorted(logs_folder.glob("residual_recon_restart_*.png"))
    motion_logs = sorted(logs_folder.glob("residual_motion_restart_*.png"))
    logs_figsize = (13.0, 4.8) if has_ground_truth else (10.0, 4.8)
    if motion_logs:
        display_image_row(
            [str(motion_logs[-1])],
            [""],
            title=None,
            figsize=logs_figsize,
        )
    if recon_logs:
        display_image_row(
            [str(recon_logs[-1])],
            [""],
            title=None,
            figsize=logs_figsize,
        )

    image_paths = [
        _first_existing_path(input_folder / "img_corrupted.png", input_folder / "input_distorted.png"),
        str(results_folder / "reconstructed_image.png"),
    ]
    subtitles = ["Corrupted", "Corrected"]
    images_figsize = None
    if has_ground_truth:
        image_paths.append(
            _first_existing_path(input_folder / "img_ground_truth.png", input_folder / "input_ground_truth.png")
        )
        subtitles.append("Ground truth")
        images_figsize = (13.0, 4.8)
    else:
        images_figsize = (10.0, 4.8)
    display_image_row(image_paths, subtitles, title="Images", figsize=images_figsize)

    if motion_type == "rigid":
        display_image_row(
            [
                str(input_folder / "clustered_motion_curves_chronological.png"),
                str(results_folder / "clustered_motion_curves_chronological.png"),
            ],
            ["Simulated / input motion (chronological)", "Reconstructed motion (chronological)"],
            title="Rigid Motion",
            figsize=(images_figsize[0], 3.4),
        )
    elif motion_type == "non-rigid":
        display_image_row(
            [
                str(input_folder / "simulated_motion_quiver_input.png"),
                str(results_folder / "final_motion_quiver.png"),
            ],
            ["", ""],
            title="Non-rigid motion model",
        )


def display_input_sampling_motion_panels(params, has_ground_truth=None, jupyter_notebook_flag=False):
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

    display_image_row(
        [sampling_path, motion_vs_ky_path],
        ["Sampling order (ky)", "Motion curve in ky order"],
        title="Sampling And Motion",
        figsize=(row_width, 4.8),
    )

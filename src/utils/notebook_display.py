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


def display_run_panels(params, motion_type, has_ground_truth=True, jupyter_notebook_flag=False):
    if not jupyter_notebook_flag:
        return

    logs_folder = Path(params.logs_folder)
    input_folder = Path(params.input_data_folder)
    results_folder = Path(params.results_folder)

    recon_global_logs = sorted(logs_folder.glob("residual_recon_global_restart_*.png"))
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
    if recon_global_logs:
        display_image_row(
            [str(recon_global_logs[-1])],
            [""],
            title=None,
            figsize=logs_figsize,
        )
    elif recon_logs:
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
    if has_ground_truth:
        image_paths.append(
            _first_existing_path(input_folder / "img_ground_truth.png", input_folder / "input_ground_truth.png")
        )
        subtitles.append("Ground truth")
    display_image_row(image_paths, subtitles, title="Images")

    if motion_type == "rigid":
        display_image_row(
            [
                str(input_folder / "clustered_motion_curves_chronological.png"),
                str(results_folder / "clustered_motion_curves_chronological.png"),
            ],
            ["Simulated / input motion (chronological)", "Reconstructed motion (chronological)"],
            title="Rigid Motion",
        )
    elif motion_type == "non-rigid":
        display_image_row(
            [
                str(input_folder / "simulated_motion_quiver_input.png"),
                str(results_folder / "final_motion_quiver.png"),
            ],
            ["Simulated / input alpha quiver", "Reconstructed alpha quiver"],
            title="Non-rigid Motion",
        )

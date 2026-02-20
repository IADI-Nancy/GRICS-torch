from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def _load_image(path):
    p = Path(path)
    if not p.exists():
        return None
    return mpimg.imread(p)


def display_image_row(image_paths, subtitles, title=None, figsize=(10, 3)):
    n = len(image_paths)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, path, subtitle in zip(axes, image_paths, subtitles):
        img = _load_image(path)
        if img is None:
            ax.axis("off")
            ax.set_title(f"{subtitle}\n(missing)")
            continue
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(subtitle, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    plt.show()


def display_run_panels(params, motion_type, has_ground_truth=True, jupyter_notebook_flag=False):
    if not jupyter_notebook_flag:
        return

    logs_folder = Path(params.logs_folder)
    input_folder = Path(params.input_data_folder)
    results_folder = Path(params.results_folder)

    recon_logs = sorted(logs_folder.glob("residual_recon_restart_*.png"))
    motion_logs = sorted(logs_folder.glob("residual_motion_restart_*.png"))
    if recon_logs and motion_logs:
        display_image_row(
            [str(recon_logs[-1]), str(motion_logs[-1])],
            ["Reconstruction residuals", "Motion residuals"],
            title="Logs",
            figsize=(8, 3),
        )

    image_paths = [
        str(input_folder / "input_distorted.png"),
        str(results_folder / "reconstructed_image.png"),
    ]
    subtitles = ["Corrupted", "Corrected"]
    if has_ground_truth:
        image_paths.append(str(input_folder / "input_ground_truth.png"))
        subtitles.append("Ground truth")
    display_image_row(image_paths, subtitles, title="Images", figsize=(10, 3))

    if motion_type == "rigid":
        display_image_row(
            [
                str(input_folder / "clustered_motion_curves_chronological.png"),
                str(results_folder / "clustered_motion_curves_chronological.png"),
            ],
            ["Simulated / input motion (chronological)", "Reconstructed motion (chronological)"],
            title="Rigid Motion",
            figsize=(10, 3),
        )
    elif motion_type == "non-rigid":
        display_image_row(
            [
                str(input_folder / "simulated_motion_quiver_input.png"),
                str(results_folder / "final_motion_quiver.png"),
            ],
            ["Simulated / input alpha quiver", "Reconstructed alpha quiver"],
            title="Non-rigid Motion",
            figsize=(10, 3),
        )

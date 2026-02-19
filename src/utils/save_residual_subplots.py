import os
import matplotlib.pyplot as plt


def save_residual_subplots(values_by_level, title, y_label, out_path):
    n_levels = len(values_by_level)
    if n_levels == 0:
        return

    global_max = None
    for vals in values_by_level:
        if len(vals) == 0:
            continue
        local_max = max(vals)
        global_max = local_max if global_max is None else max(global_max, local_max)

    global_min = 0.0
    if global_max is None:
        global_max = 1.0
    if global_max <= global_min:
        global_max = global_min + 1e-12

    fig, axes = plt.subplots(1, n_levels, figsize=(4.4 * n_levels, 3.8), sharey=True)
    if n_levels == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        vals = values_by_level[idx]
        if len(vals) > 0:
            ax.plot(vals, marker="o")
            ax.set_xlim(0, max(1, len(vals) - 1))
        else:
            ax.text(0.5, 0.5, "No iterations", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(global_min, global_max)
        ax.grid(True)
        ax.set_ylabel(y_label)
        ax.set_title(f"Resolution level {idx + 1}")
        ax.set_xlabel("GN iteration")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

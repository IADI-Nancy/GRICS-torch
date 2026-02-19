import os
import matplotlib.pyplot as plt


def save_residual_convergence(residual_norms, title, res_level, logs_folder):
    os.makedirs(logs_folder, exist_ok=True)

    plt.figure()
    plt.plot(residual_norms, marker="o")
    plt.xlabel("GN iteration")
    plt.ylabel("||residual||2")
    plt.title(f"Residual convergence ({title}, resolution level {res_level})")
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(logs_folder, f"residual_convergence_{title}_res{res_level}.png"))
    plt.close()

    with open(os.path.join(logs_folder, f"residual_convergence_{title}_res{res_level}.txt"), "w") as f:
        for i, v in enumerate(residual_norms):
            f.write(f"{i+1}\t{v}\n")

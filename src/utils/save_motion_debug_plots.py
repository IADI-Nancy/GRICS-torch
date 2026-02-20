import os
import matplotlib.pyplot as plt


def save_motion_debug_plots(motion_curve, tx, ty, phi, output_folder, event_times=None):
    os.makedirs(output_folder, exist_ok=True)

    plt.figure()
    plt.plot(motion_curve.detach().cpu().numpy())
    plt.title("Motion Curve")
    plt.savefig(os.path.join(output_folder, "motion_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(tx.detach().cpu().numpy())
    plt.title("tx curve")
    plt.savefig(os.path.join(output_folder, "tx_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(ty.detach().cpu().numpy())
    plt.title("ty curve")
    plt.savefig(os.path.join(output_folder, "ty_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(phi.detach().cpu().numpy())
    plt.title("phi curve")
    plt.savefig(os.path.join(output_folder, "phi_curve.png"))
    plt.close()

    # Combined plot with clustered markers is generated in save_clustered_motion_plots.py

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

    plt.figure(figsize=(10, 4))
    plt.plot(motion_curve.detach().cpu().numpy(), label="PC1 Motion Curve", linewidth=2)
    plt.plot(tx.detach().cpu().numpy(), label="tx", alpha=0.8)
    plt.plot(ty.detach().cpu().numpy(), label="ty", alpha=0.8)
    plt.plot(phi.detach().cpu().numpy(), label="phi", alpha=0.8)

    if event_times is not None:
        for e in event_times.detach().cpu().numpy():
            plt.axvline(x=e, color="black", linewidth=1)

    plt.title("All motion curves (superimposed)")
    plt.xlabel("Acquisition line number")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "all_curves.png"))
    plt.close()

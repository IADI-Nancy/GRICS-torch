import argparse
import itertools
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.DataLoader import DataLoader
from src.reconstruction.JointReconstructor import JointReconstructor
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import cleanup_runtime, initialize_runtime


def _parse_float_list(values):
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def _parse_int_list(values):
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def _alpha_rss_error(alpha_final, alpha_true):
    alpha_final = alpha_final.real if torch.is_complex(alpha_final) else alpha_final
    alpha_true = alpha_true.real if torch.is_complex(alpha_true) else alpha_true
    if alpha_final.shape != alpha_true.shape:
        raise ValueError(
            f"Alpha shape mismatch: reconstructed {tuple(alpha_final.shape)} vs true {tuple(alpha_true.shape)}"
        )
    diff = alpha_final - alpha_true
    n = max(int(diff.numel()), 1)
    return torch.sqrt(torch.sum(diff * diff) / n).item()


def _run_demo_c_nonrigid(sim_overrides, run_overrides):
    params = load_config(
        data_type="shepp-logan",
        reconstruction_config="config/reconstruction/nonrigid_fast.toml",
        shepp_logan_config="config/shepp_logan.toml",
        sampling_config="config/sampling_simulation/interleaved.toml",
        motion_simulation_config="config/motion_simulation/discrete_nonrigid.toml",
        overrides={
            "jupyter_notebook_flag": False,
            "print_to_console": False,
            "verbose": False,
            **sim_overrides,
            **run_overrides,
        },
    )
    sp_device, t_device = initialize_runtime(params)
    data = DataLoader(params=params, t_device=t_device, sp_device=sp_device)
    recon = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        kspace_scale=data.kspace_scale,
        motion_plot_context=getattr(data, "motion_plot_context", None),
    )
    _, alpha = recon.run()
    if not hasattr(data, "alpha_maps_true"):
        raise RuntimeError("alpha_maps_true not found in DataLoader output for Demo C.")
    alpha_true = data.alpha_maps_true.to(alpha.device)
    return _alpha_rss_error(alpha, alpha_true)


def _run_demo_d_nonrigid(sim_overrides, run_overrides):
    params = load_config(
        data_type="shepp-logan",
        reconstruction_config="config/reconstruction/nonrigid_fast.toml",
        shepp_logan_config="config/shepp_logan.toml",
        sampling_config="config/sampling_simulation/linear.toml",
        motion_simulation_config="config/motion_simulation/nonrigid.toml",
        overrides={
            "Nex": 3,
            "N_motion_states": 10,
            "jupyter_notebook_flag": False,
            "print_to_console": False,
            "verbose": False,
            **sim_overrides,
            **run_overrides,
        },
    )
    sp_device, t_device = initialize_runtime(params)
    data = DataLoader(params=params, t_device=t_device, sp_device=sp_device)
    recon = JointReconstructor(
        data.kspace,
        data.smaps,
        data.sampling_idx,
        motion_signal=data.motion_signal,
        params=params,
        kspace_scale=data.kspace_scale,
        motion_plot_context=getattr(data, "motion_plot_context", None),
    )
    _, alpha = recon.run()
    if not hasattr(data, "alpha_maps_true"):
        raise RuntimeError("alpha_maps_true not found in DataLoader output for Demo D.")
    alpha_true = data.alpha_maps_true.to(alpha.device)
    return _alpha_rss_error(alpha, alpha_true)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep non-rigid simulation parameters, run Demo C and Demo D pipelines, "
            "and find the parameter set minimizing residual_c + residual_d."
        )
    )
    parser.add_argument("--amplitudes", default="3.0")
    parser.add_argument("--displacement-sizes", default="1,2")
    parser.add_argument("--cycles-min", default="5.0")
    parser.add_argument("--cycles-max", default="7.0")
    parser.add_argument("--diaphragm-levels", default="0.25")
    parser.add_argument("--diaphragm-sharpness", default="12.0")
    parser.add_argument("--lateral-sigmas", default="0.25")
    parser.add_argument("--ap-fractions", default="0.15")
    parser.add_argument("--inferior-gains", default="0.0,0.4")
    parser.add_argument("--top-decays", default="0.6")
    parser.add_argument(
        "--output-txt",
        default="tests/artifacts/nonrigid_param_search/results.txt",
    )
    args = parser.parse_args()

    amplitudes = _parse_float_list(args.amplitudes)
    displacement_sizes = _parse_int_list(args.displacement_sizes)
    cycles_min_values = _parse_float_list(args.cycles_min)
    cycles_max_values = _parse_float_list(args.cycles_max)
    diaphragm_levels = _parse_float_list(args.diaphragm_levels)
    diaphragm_sharpness_values = _parse_float_list(args.diaphragm_sharpness)
    lateral_sigmas = _parse_float_list(args.lateral_sigmas)
    ap_fractions = _parse_float_list(args.ap_fractions)
    inferior_gains = _parse_float_list(args.inferior_gains)
    top_decays = _parse_float_list(args.top_decays)

    output_path = Path(args.output_txt).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "run\tnonrigid_motion_amplitude\tdisplacementfield_size\t"
        "nonrigid_resp_cycles_min\tnonrigid_resp_cycles_max\t"
        "nonrigid_diaphragm_level\tnonrigid_diaphragm_sharpness\t"
        "nonrigid_lateral_sigma\tnonrigid_ap_fraction\t"
        "nonrigid_inferior_gain\tnonrigid_top_decay\t"
        "alpha_rss_demo_c\talpha_rss_demo_d\talpha_rss_sum\n"
    )
    # Reset file on every new script execution.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)

    all_combos = [
        combo
        for combo in itertools.product(
            amplitudes,
            displacement_sizes,
            cycles_min_values,
            cycles_max_values,
            diaphragm_levels,
            diaphragm_sharpness_values,
            lateral_sigmas,
            ap_fractions,
            inferior_gains,
            top_decays,
        )
        if combo[3] >= combo[2]
    ]
    total_runs = len(all_combos)
    print(f"Total parameter combinations to run: {total_runs}")

    results = []
    for run_idx, (
        amp,
        disp_size,
        cmin,
        cmax,
        diaphragm_level,
        diaphragm_sharpness,
        lateral_sigma,
        ap_fraction,
        inferior_gain,
        top_decay,
    ) in enumerate(all_combos, start=1):
        run_name = f"run_{run_idx:04d}"
        run_root = output_path.parent / run_name
        run_root.mkdir(parents=True, exist_ok=True)

        sim_overrides = {
            "nonrigid_motion_amplitude": amp,
            "displacementfield_size": disp_size,
            "nonrigid_resp_cycles_min": cmin,
            "nonrigid_resp_cycles_max": cmax,
            "nonrigid_diaphragm_level": diaphragm_level,
            "nonrigid_diaphragm_sharpness": diaphragm_sharpness,
            "nonrigid_lateral_sigma": lateral_sigma,
            "nonrigid_ap_fraction": ap_fraction,
            "nonrigid_inferior_gain": inferior_gain,
            "nonrigid_top_decay": top_decay,
        }
        run_overrides_c = {
            "debug_flag": False,
            "clean_output_folders_before_run": True,
            "debug_folder": str(run_root / "demo_c" / "debug_outputs"),
            "logs_folder": str(run_root / "demo_c" / "logs"),
            "results_folder": str(run_root / "demo_c" / "results"),
            "input_data_folder": str(run_root / "demo_c" / "input_data"),
        }
        run_overrides_d = {
            "debug_flag": False,
            "clean_output_folders_before_run": True,
            "debug_folder": str(run_root / "demo_d" / "debug_outputs"),
            "logs_folder": str(run_root / "demo_d" / "logs"),
            "results_folder": str(run_root / "demo_d" / "results"),
            "input_data_folder": str(run_root / "demo_d" / "input_data"),
        }

        print(
            f"[run {run_idx}/{total_runs} | {run_name}] "
            f"amp={amp}, displacementfield_size={disp_size}, "
            f"cycles_min={cmin}, cycles_max={cmax}, "
            f"diaphragm_level={diaphragm_level}, diaphragm_sharpness={diaphragm_sharpness}, "
            f"lateral_sigma={lateral_sigma}, ap_fraction={ap_fraction}, "
            f"inferior_gain={inferior_gain}, top_decay={top_decay}"
        )
        try:
            alpha_rss_c = _run_demo_c_nonrigid(sim_overrides, run_overrides_c)
            cleanup_runtime()
            alpha_rss_d = _run_demo_d_nonrigid(sim_overrides, run_overrides_d)
            cleanup_runtime()
        except Exception as exc:
            cleanup_runtime()
            print(f"[{run_name}] FAILED: {exc}")
            continue

        total = alpha_rss_c + alpha_rss_d
        row = {
            "run": run_name,
            "nonrigid_motion_amplitude": amp,
            "displacementfield_size": disp_size,
            "nonrigid_resp_cycles_min": cmin,
            "nonrigid_resp_cycles_max": cmax,
            "nonrigid_diaphragm_level": diaphragm_level,
            "nonrigid_diaphragm_sharpness": diaphragm_sharpness,
            "nonrigid_lateral_sigma": lateral_sigma,
            "nonrigid_ap_fraction": ap_fraction,
            "nonrigid_inferior_gain": inferior_gain,
            "nonrigid_top_decay": top_decay,
            "alpha_rss_demo_c": alpha_rss_c,
            "alpha_rss_demo_d": alpha_rss_d,
            "alpha_rss_sum": total,
        }
        results.append(row)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(
                f"{row['run']}\t{row['nonrigid_motion_amplitude']}\t{row['displacementfield_size']}\t"
                f"{row['nonrigid_resp_cycles_min']}\t{row['nonrigid_resp_cycles_max']}\t"
                f"{row['nonrigid_diaphragm_level']}\t{row['nonrigid_diaphragm_sharpness']}\t"
                f"{row['nonrigid_lateral_sigma']}\t{row['nonrigid_ap_fraction']}\t"
                f"{row['nonrigid_inferior_gain']}\t{row['nonrigid_top_decay']}\t"
                f"{row['alpha_rss_demo_c']:.12e}\t{row['alpha_rss_demo_d']:.12e}\t{row['alpha_rss_sum']:.12e}\n"
            )
        print(
            f"[{run_name}] alpha_rss_demo_c={alpha_rss_c:.6e}, "
            f"alpha_rss_demo_d={alpha_rss_d:.6e}, sum={total:.6e}"
        )

    if not results:
        raise RuntimeError("No successful runs. Check parameter ranges and runtime environment.")

    results.sort(key=lambda r: r["alpha_rss_sum"])
    best = results[0]

    with open(output_path, "a", encoding="utf-8") as f:
        f.write("\nBEST_PARAMETER_SET\n")
        f.write(
            f"run={best['run']}, nonrigid_motion_amplitude={best['nonrigid_motion_amplitude']}, "
            f"displacementfield_size={best['displacementfield_size']}, "
            f"nonrigid_resp_cycles_min={best['nonrigid_resp_cycles_min']}, "
            f"nonrigid_resp_cycles_max={best['nonrigid_resp_cycles_max']}, "
            f"nonrigid_diaphragm_level={best['nonrigid_diaphragm_level']}, "
            f"nonrigid_diaphragm_sharpness={best['nonrigid_diaphragm_sharpness']}, "
            f"nonrigid_lateral_sigma={best['nonrigid_lateral_sigma']}, "
            f"nonrigid_ap_fraction={best['nonrigid_ap_fraction']}, "
            f"nonrigid_inferior_gain={best['nonrigid_inferior_gain']}, "
            f"nonrigid_top_decay={best['nonrigid_top_decay']}, "
            f"alpha_rss_demo_c={best['alpha_rss_demo_c']:.12e}, "
            f"alpha_rss_demo_d={best['alpha_rss_demo_d']:.12e}, "
            f"alpha_rss_sum={best['alpha_rss_sum']:.12e}\n"
        )

    print("\nBest parameter set (min alpha_rss_demo_c + alpha_rss_demo_d):")
    print(best)
    print(f"\nSaved full sweep results to: {output_path}")


if __name__ == "__main__":
    main()

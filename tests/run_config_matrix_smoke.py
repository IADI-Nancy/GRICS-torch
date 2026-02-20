import os
import shutil
import sys
import tomllib
from itertools import product
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.DataLoader import DataLoader
from src.runtime.runtime_config import load_config
from src.runtime.runtime_setup import initialize_runtime


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
ARTIFACT_ROOT = ROOT / "tests" / "artifacts" / "config_matrix_smoke"


def _read_type(path, section, key):
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data[section][key]


def _sanitize(name):
    return name.replace("/", "__").replace(".", "_")


def _snapshot_outputs(params, dst_root):
    for attr in ("debug_folder", "logs_folder", "results_folder"):
        src = Path(getattr(params, attr))
        dst = dst_root / attr
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def _clean_outputs(params):
    for attr in ("debug_folder", "logs_folder", "results_folder"):
        folder = Path(getattr(params, attr))
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def _build_cases():
    recon_cfgs = sorted((CONFIG / "reconstruction").glob("*.toml"))
    samp_cfgs = sorted((CONFIG / "sampling_simulation").glob("*.toml"))
    mot_cfgs = sorted((CONFIG / "motion_simulation").glob("*.toml"))

    cases = []
    for data_type in ("shepp-logan", "fastMRI"):
        for recon, samp, mot in product(recon_cfgs, samp_cfgs, mot_cfgs):
            motion_type = _read_type(recon, "reconstruction", "motion_type")
            mot_sim_type = _read_type(mot, "motion", "motion_simulation_type")

            compatible = (
                (motion_type == "rigid" and mot_sim_type in {"rigid", "discrete-rigid"})
                or (motion_type == "non-rigid" and mot_sim_type == "discrete-non-rigid")
            )
            if not compatible:
                continue

            case_name = (
                f"{data_type}__{recon.stem}__{samp.stem}__{mot.stem}"
            )
            cases.append(
                {
                    "name": case_name,
                    "kwargs": {
                        "data_type": data_type,
                        "reconstruction_config": str(recon.relative_to(ROOT)),
                        "shepp_logan_config": "config/shepp_logan.toml" if data_type == "shepp-logan" else None,
                        "sampling_config": str(samp.relative_to(ROOT)),
                        "motion_simulation_config": str(mot.relative_to(ROOT)),
                        "overrides": {
                            "debug_flag": False,
                            "verbose": False,
                            "N_SheppLogan": 32,
                            "acs": 16,
                            "kernel_width": 6,
                            "espirit_max_iter": 6,
                            "max_restarts": 1,
                            "ResolutionLevels": [0.25],
                            "GN_iterations_per_level": [1],
                            "max_iter_recon": 2,
                            "max_iter_motion": 2,
                        },
                    },
                    "filename": "data/kspace.npz" if data_type == "fastMRI" else None,
                }
            )

    for recon in recon_cfgs:
        case_name = f"real-world__{recon.stem}__from_data"
        cases.append(
            {
                "name": case_name,
                "kwargs": {
                    "data_type": "real-world",
                    "reconstruction_config": str(recon.relative_to(ROOT)),
                    "overrides": {
                        "debug_flag": False,
                        "verbose": False,
                        "acs": 16,
                        "kernel_width": 6,
                        "espirit_max_iter": 6,
                    },
                },
                "filename": "data/breast_motion_data.h5",
            }
        )

    return cases


def run():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = _build_cases()
    print(f"[matrix] running {len(cases)} cases")

    for idx, case in enumerate(cases, start=1):
        print(f"[matrix] ({idx}/{len(cases)}) {case['name']}")
        params = load_config(**case["kwargs"])
        sp_device, t_device = initialize_runtime(params)
        _clean_outputs(params)

        DataLoader(
            params=params,
            t_device=t_device,
            sp_device=sp_device,
            filename=case["filename"],
        )

        log_path = Path(params.logs_folder) / "config_matrix_smoke.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"OK {case['name']}\n")

        case_dir = ARTIFACT_ROOT / f"{idx:03d}_{_sanitize(case['name'])}"
        _snapshot_outputs(params, case_dir)

    print(f"[matrix] done. artifacts at: {ARTIFACT_ROOT}")


if __name__ == "__main__":
    run()

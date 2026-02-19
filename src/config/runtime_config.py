import os
from pathlib import Path
from types import SimpleNamespace
import tomllib
import torch


def _flatten(dct, prefix=""):
    out = {}
    for key, value in dct.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, full_key))
        else:
            out[full_key] = value
    return out


def _load_toml_flat(path):
    with open(path, "rb") as f:
        data = tomllib.load(f)
    flat = _flatten(data)
    return {key.split(".")[-1]: value for key, value in flat.items()}


def refresh_derived(params):
    torch.set_default_dtype(torch.float64)

    params.Nshots = int(params.NshotsPerNex) * int(params.Nex)

    if params.simulation_type in ["discrete-rigid", "discrete-non-rigid"]:
        params.N_mot_states = params.Nshots
    elif params.simulation_type == "rigid":
        params.N_mot_states = int(params.num_motion_events) + 1
    elif params.simulation_type == "no-motion":
        params.N_mot_states = 1

    if params.motion_type == "non-rigid":
        params.max_restarts = 1

    os.makedirs(params.debug_folder, exist_ok=True)
    os.makedirs(params.logs_folder, exist_ok=True)
    os.makedirs(params.results_folder, exist_ok=True)

    return params


def load_config(config_paths=None, overrides=None):
    repo_root = Path(__file__).resolve().parents[2]
    defaults_path = repo_root / "config" / "defaults.toml"

    if not defaults_path.exists():
        raise FileNotFoundError(f"Missing defaults config: {defaults_path}")

    cfg = _load_toml_flat(defaults_path)

    for path in (config_paths or []):
        cfg.update(_load_toml_flat(path))

    for key, value in (overrides or {}).items():
        cfg[key] = value

    params = SimpleNamespace(**cfg)
    return refresh_derived(params)

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


_RIGID_MOTION_KEYS = {
    "num_motion_events",
    "max_tx",
    "max_ty",
    "max_phi",
    "max_center_x",
    "max_center_y",
    "motion_tau",
}

_NONRIGID_MOTION_KEYS = {
    "nonrigid_motion_amplitude",
    "displacementfield_size",
}

_CODE_DEFAULTS = {
    "seed": 1,
    "use_scaled_motion_update": False,
    "espirit_max_iter": 100,
}


def _drop_keys(cfg, keys):
    for key in keys:
        cfg.pop(key, None)


def refresh_derived(params):
    torch.set_default_dtype(torch.float64)

    if not hasattr(params, "flip_for_display"):
        params.flip_for_display = params.data_type in {"real-world", "raw-data"}
    if not hasattr(params, "clean_output_folders_before_run"):
        params.clean_output_folders_before_run = True

    has_sampling_sim = hasattr(params, "NshotsPerNex") and hasattr(params, "Nex")
    if has_sampling_sim:
        params.Nshots = int(params.NshotsPerNex) * int(params.Nex)
    elif not hasattr(params, "Nshots"):
        params.Nshots = 1

    if not hasattr(params, "motion_simulation_type"):
        params.motion_simulation_type = "as-it-is"

    manual_states = int(getattr(params, "N_motion_states", 4))
    if manual_states < 1:
        raise ValueError("N_motion_states must be >= 1.")
    params.N_motion_states = manual_states

    if params.motion_simulation_type in ["discrete-rigid", "discrete-non-rigid"]:
        # Simulated motion active: use simulation-driven state count.
        params.N_motion_states = params.Nshots
    elif params.motion_simulation_type == "rigid":
        # Simulated motion active: use simulation-driven state count.
        if not hasattr(params, "num_motion_events"):
            raise ValueError("Missing 'num_motion_events' for rigid simulation.")
        params.N_motion_states = int(params.num_motion_events) + 1
    elif params.motion_simulation_type in ["no-motion", "as-it-is"]:
        # No simulated motion: keep manual reconstruction value.
        params.N_motion_states = manual_states

    if not hasattr(params, "Nex"):
        params.Nex = 1

    if params.motion_type == "non-rigid":
        params.max_restarts = 1

    os.makedirs(params.debug_folder, exist_ok=True)
    os.makedirs(params.logs_folder, exist_ok=True)
    os.makedirs(params.results_folder, exist_ok=True)
    os.makedirs(params.input_data_folder, exist_ok=True)

    return params


def load_config(
    *,
    data_type,
    motion_type=None,
    reconstruction_config,
    shepp_logan_config=None,
    sampling_config=None,
    motion_simulation_config=None,
    motion_simulation_type=None,
    kspace_sampling_type=None,
    NshotsPerNex=None,
    Nex=None,
    flip_for_display=None,
    overrides=None,
):
    repo_root = Path(__file__).resolve().parents[2]
    general_path = repo_root / "config" / "general.toml"

    if not general_path.exists():
        raise FileNotFoundError(f"Missing general config: {general_path}")

    if not reconstruction_config:
        raise ValueError("reconstruction_config is required.")

    cfg = {}
    cfg.update(_CODE_DEFAULTS)
    cfg.update(_load_toml_flat(general_path))
    cfg["data_type"] = data_type

    cfg.update(_load_toml_flat(reconstruction_config))

    if data_type == "shepp-logan":
        if not shepp_logan_config:
            raise ValueError("shepp_logan_config is required when data_type='shepp-logan'.")
        cfg.update(_load_toml_flat(shepp_logan_config))
    elif data_type == "fastMRI":
        pass
    elif data_type in {"real-world", "raw-data"}:
        pass
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    if sampling_config:
        cfg.update(_load_toml_flat(sampling_config))
    if motion_simulation_config:
        cfg.update(_load_toml_flat(motion_simulation_config))

    if motion_type is not None:
        cfg["motion_type"] = motion_type
    if kspace_sampling_type is not None:
        cfg["kspace_sampling_type"] = kspace_sampling_type
    if motion_simulation_type is not None:
        cfg["motion_simulation_type"] = motion_simulation_type
    if NshotsPerNex is not None:
        cfg["NshotsPerNex"] = int(NshotsPerNex)
    if Nex is not None:
        cfg["Nex"] = int(Nex)
    if flip_for_display is not None:
        cfg["flip_for_display"] = bool(flip_for_display)

    if not hasattr(SimpleNamespace(**cfg), "motion_type"):
        raise ValueError(
            "motion_type must be provided either in reconstruction config or as load_config argument."
        )

    if "kspace_sampling_type" in cfg:
        sampling_from_data = False
    else:
        _drop_keys(cfg, {"kspace_sampling_type", "NshotsPerNex", "Nex", "Nshots"})
        sampling_from_data = True

    if data_type != "real-world" and sampling_from_data:
        raise ValueError(
            "Sampling configuration is required when data_type is not 'real-world'. "
            "Provide sampling_config or kspace_sampling_type (+ Nex/NshotsPerNex)."
        )

    motion_simulation_type_final = cfg.get("motion_simulation_type")
    if motion_simulation_type_final is None and sampling_from_data:
        motion_simulation_type_final = "as-it-is"
    elif motion_simulation_type_final is None:
        motion_simulation_type_final = "no-motion"
    cfg["motion_simulation_type"] = motion_simulation_type_final

    if "kspace_sampling_type" in cfg:
        if "NshotsPerNex" not in cfg or "Nex" not in cfg:
            raise ValueError(
                "NshotsPerNex and Nex are required when kspace_sampling_type is specified."
            )

    if cfg["motion_simulation_type"] in {"as-it-is", "no-motion"}:
        _drop_keys(cfg, _RIGID_MOTION_KEYS | _NONRIGID_MOTION_KEYS)
    elif cfg["motion_simulation_type"] in {"rigid", "discrete-rigid"}:
        _drop_keys(cfg, _NONRIGID_MOTION_KEYS)
    elif cfg["motion_simulation_type"] == "discrete-non-rigid":
        _drop_keys(cfg, _RIGID_MOTION_KEYS)
    else:
        raise ValueError(f"Unsupported motion_simulation_type: {cfg['motion_simulation_type']}")

    for key, value in (overrides or {}).items():
        cfg[key] = value

    if "flip_for_display" not in cfg:
        cfg["flip_for_display"] = data_type in {"real-world", "raw-data"}

    params = SimpleNamespace(**cfg)
    return refresh_derived(params)

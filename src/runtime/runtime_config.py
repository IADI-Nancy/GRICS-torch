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
    "nonrigid_resp_cycles_min",
    "nonrigid_resp_cycles_max",
    "nonrigid_spatial_model",
    "nonrigid_diaphragm_level",
    "nonrigid_diaphragm_sharpness",
    "nonrigid_lateral_sigma",
    "nonrigid_ap_fraction",
    "nonrigid_inferior_gain",
    "nonrigid_top_decay",
}

_CODE_DEFAULTS = {
    "seed": 1,
    "use_scaled_motion_update": False,
    "espirit_max_iter": 100,
    "cg_true_residual_interval": 10,
    "nonrigid_resp_cycles_min": 2.0,
    "nonrigid_resp_cycles_max": 5.0,
    "jupyter_notebook_flag": False,
    "runtime_device": "gpu",
}


def _drop_keys(cfg, keys):
    for key in keys:
        cfg.pop(key, None)


_MOTION_SIM_ALIASES = {
    # Dimension-agnostic legacy names.
    "as-it-is": ("as-it-is", None),
    "as_it_is": ("as-it-is", None),
    "no-motion-data": ("no-motion-data", None),
    "no_motion_data": ("no-motion-data", None),
    "rigid": ("rigid", None),
    "discrete-rigid": ("discrete-rigid", None),
    "discrete_rigid": ("discrete-rigid", None),
    "non-rigid": ("non-rigid", None),
    "non_rigid": ("non-rigid", None),
    "discrete-non-rigid": ("discrete-non-rigid", None),
    "discrete_non_rigid": ("discrete-non-rigid", None),
    # New explicit names.
    "rigid_2d": ("rigid", "2D"),
    "rigid_3d": ("rigid", "3D"),
    "discrete_rigid_2d": ("discrete-rigid", "2D"),
    "discrete_rigid_3d": ("discrete-rigid", "3D"),
    "nonrigid_2d": ("non-rigid", "2D"),
    "nonrigid_3d": ("non-rigid", "3D"),
    "discrete_nonrigid_2d": ("discrete-non-rigid", "2D"),
    "discrete_nonrigid_3d": ("discrete-non-rigid", "3D"),
    "no_motion_data_2d": ("no-motion-data", "2D"),
    "no_motion_data_3d": ("no-motion-data", "3D"),
    "as_it_is_2d": ("as-it-is", "2D"),
    "as_it_is_3d": ("as-it-is", "3D"),
}


def _normalize_data_dimension(dim):
    if dim is None:
        return None
    d = str(dim).strip().upper()
    if d in {"2D", "2"}:
        return "2D"
    if d in {"3D", "3"}:
        return "3D"
    raise ValueError("data_dimension must be '2D' or '3D'.")


def _parse_motion_simulation_type(raw_type):
    key = str(raw_type).strip().lower()
    if key not in _MOTION_SIM_ALIASES:
        raise ValueError(f"Unsupported motion_simulation_type: {raw_type}")
    return _MOTION_SIM_ALIASES[key]


def _refresh_derived(params):
    torch.set_default_dtype(torch.float64)

    if not hasattr(params, "flip_for_display"):
        params.flip_for_display = params.data_type in {"real-world", "raw-data"}
    if not hasattr(params, "clean_output_folders_before_run"):
        params.clean_output_folders_before_run = True
    if not hasattr(params, "jupyter_notebook_flag"):
        params.jupyter_notebook_flag = False
    if not hasattr(params, "runtime_device"):
        params.runtime_device = "gpu"
    params.runtime_device = str(params.runtime_device).lower()
    if params.runtime_device not in {"cpu", "gpu"}:
        raise ValueError("runtime_device must be 'cpu' or 'gpu'.")
    if not hasattr(params, "print_to_console"):
        params.print_to_console = not bool(params.jupyter_notebook_flag)
    if not hasattr(params, "verbose"):
        params.verbose = not bool(params.jupyter_notebook_flag)

    has_sampling_sim = hasattr(params, "NshotsPerNex") and hasattr(params, "Nex")
    if has_sampling_sim:
        params.Nshots = int(params.NshotsPerNex) * int(params.Nex)
    elif not hasattr(params, "Nshots"):
        params.Nshots = 1

    if not hasattr(params, "motion_simulation_type"):
        params.motion_simulation_type = "as-it-is"

    # Infer/normalize global data dimension.
    data_dim_raw = getattr(params, "data_dimension", None)
    inferred_dim = None
    if hasattr(params, "Nz_SheppLogan"):
        inferred_dim = "3D" if int(params.Nz_SheppLogan) > 1 else "2D"
    if data_dim_raw is None:
        params.data_dimension = inferred_dim if inferred_dim is not None else "2D"
    else:
        params.data_dimension = _normalize_data_dimension(data_dim_raw)

    # Canonicalize motion simulation type and track optional type-implied dimension.
    motion_type_canonical, motion_type_dim = _parse_motion_simulation_type(params.motion_simulation_type)
    existing_motion_dim = _normalize_data_dimension(getattr(params, "motion_simulation_dimension", None))
    if existing_motion_dim is not None:
        motion_type_dim = existing_motion_dim
    params.motion_simulation_type = motion_type_canonical
    params.motion_simulation_dimension = motion_type_dim

    if motion_type_dim is not None and motion_type_dim != params.data_dimension:
        raise ValueError(
            f"motion_simulation_type implies {motion_type_dim}, but data_dimension is {params.data_dimension}."
        )

    # Optional explicit dimension tags from config files.
    recon_dim = _normalize_data_dimension(getattr(params, "reconstruction_dimension", None))
    motion_cfg_dim = _normalize_data_dimension(getattr(params, "motion_simulation_config_dimension", None))
    if recon_dim is not None and recon_dim != params.data_dimension:
        raise ValueError(
            f"Reconstruction config is tagged {recon_dim}, but data_dimension is {params.data_dimension}."
        )
    if motion_cfg_dim is not None and motion_cfg_dim != params.data_dimension:
        raise ValueError(
            f"Motion simulation config is tagged {motion_cfg_dim}, but data_dimension is {params.data_dimension}."
        )

    # Shepp-Logan must match declared dimension.
    if hasattr(params, "Nz_SheppLogan"):
        nz_dim = "3D" if int(params.Nz_SheppLogan) > 1 else "2D"
        if nz_dim != params.data_dimension:
            raise ValueError(
                f"Shepp-Logan Nz_SheppLogan={int(params.Nz_SheppLogan)} implies {nz_dim}, "
                f"but data_dimension is {params.data_dimension}."
            )

    manual_states = int(params.N_motion_states)
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
    elif params.motion_simulation_type == "non-rigid":
        # For realistic non-rigid simulation, keep user-defined reconstruction bins.
        params.N_motion_states = manual_states
    elif params.motion_simulation_type in ["no-motion-data", "as-it-is"]:
        # No simulated motion: keep manual reconstruction value.
        params.N_motion_states = manual_states

    if not hasattr(params, "Nex"):
        params.Nex = 1

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
    from_image_config=None,
    sampling_config=None,
    motion_simulation_config=None,
    motion_simulation_type=None,
    data_dimension=None,
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
    elif data_type in {"from_image", "from_dicom"}:
        if from_image_config is None:
            raise ValueError(
                "from_image_config is required when data_type is 'from_image' or 'from_dicom'."
            )
        cfg.update(_load_toml_flat(from_image_config))
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
    if data_dimension is not None:
        cfg["data_dimension"] = data_dimension
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

    if data_type not in {"real-world", "raw-data"} and sampling_from_data:
        raise ValueError(
            "Sampling configuration is required when data_type is not 'real-world'/'raw-data'. "
            "Provide sampling_config or kspace_sampling_type (+ Nex/NshotsPerNex)."
        )

    if data_type in {"from_image", "from_dicom"} and motion_simulation_config is None and motion_simulation_type is None:
        raise ValueError(
            "motion_simulation_config (or motion_simulation_type override) is required "
            "for data_type='from_image'/'from_dicom'."
        )

    motion_simulation_type_final = cfg.get("motion_simulation_type")
    if motion_simulation_type_final is None and sampling_from_data:
        motion_simulation_type_final = "as-it-is"
    elif motion_simulation_type_final is None:
        motion_simulation_type_final = "no-motion-data"
    cfg["motion_simulation_type"] = motion_simulation_type_final

    if "kspace_sampling_type" in cfg:
        if "NshotsPerNex" not in cfg or "Nex" not in cfg:
            raise ValueError(
                "NshotsPerNex and Nex are required when kspace_sampling_type is specified."
            )

    motion_type_canonical, motion_type_dim = _parse_motion_simulation_type(cfg["motion_simulation_type"])
    cfg["motion_simulation_type"] = motion_type_canonical
    cfg["motion_simulation_dimension"] = motion_type_dim

    if cfg["motion_simulation_type"] in {"as-it-is", "no-motion-data"}:
        _drop_keys(cfg, _RIGID_MOTION_KEYS | _NONRIGID_MOTION_KEYS)
    elif cfg["motion_simulation_type"] in {"rigid", "discrete-rigid"}:
        _drop_keys(cfg, _NONRIGID_MOTION_KEYS)
    elif cfg["motion_simulation_type"] in {"discrete-non-rigid", "non-rigid"}:
        _drop_keys(cfg, _RIGID_MOTION_KEYS)
    else:
        raise ValueError(f"Unsupported motion_simulation_type: {cfg['motion_simulation_type']}")

    for key, value in (overrides or {}).items():
        cfg[key] = value

    # Derive console verbosity from notebook mode unless explicitly overridden.
    notebook_flag = bool(cfg.get("jupyter_notebook_flag", False))
    if not overrides or "print_to_console" not in overrides:
        cfg["print_to_console"] = not notebook_flag
    if not overrides or "verbose" not in overrides:
        cfg["verbose"] = not notebook_flag

    if "flip_for_display" not in cfg:
        cfg["flip_for_display"] = data_type in {"real-world", "raw-data"}

    params = SimpleNamespace(**cfg)
    return _refresh_derived(params)

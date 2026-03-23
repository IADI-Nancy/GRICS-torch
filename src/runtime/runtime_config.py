from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import os
import tomllib
import warnings

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


def _rename_cfg_key(cfg, source_key, target_key):
    if source_key not in cfg:
        return
    source_value = cfg.pop(source_key)
    if target_key in cfg and cfg[target_key] != source_value:
        raise ValueError(
            f"Config provides conflicting values for '{source_key}' and '{target_key}'."
        )
    cfg[target_key] = source_value


def _normalize_positive_int(value, name):
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")
    return value


def _normalize_synthetic_coil_count(value, name):
    value = _normalize_positive_int(value, name)
    if value % 4 != 0:
        raise ValueError(
            f"{name} must be divisible by 4 for synthetic coil-map generation."
        )
    return value


_RIGID_MOTION_KEYS = {
    "num_motion_events",
    "max_tx",
    "max_ty",
    "max_phi",
    "max_center_x",
    "max_center_y",
    "motion_tau",
    "rigid_motion_amplitude_scale",
}

_NONRIGID_MOTION_KEYS = {
    "nonrigid_motion_amplitude",
    "nonrigid_resp_cycles_min",
    "nonrigid_resp_cycles_max",
    "nonrigid_diaphragm_level",
    "nonrigid_diaphragm_sharpness",
    "nonrigid_lateral_sigma",
    "nonrigid_lateral_sigma_lr",
    "nonrigid_lateral_sigma_ap",
    "nonrigid_ap_fraction",
    "nonrigid_lr_fraction",
    "nonrigid_anterior_bias",
    "nonrigid_inferior_gain",
    "nonrigid_top_decay",
}

_CODE_DEFAULTS = {
    "use_scaled_motion_update": False,
    "espirit_max_iter": 100,
    "cg_true_residual_interval": 10,
    "jupyter_notebook_flag": False,
}

_MOTION_TYPES = {"rigid", "non-rigid"}
_MOTION_STATE_MODES = {"realistic", "per-shot"}
_SAMPLING_TYPES = {"linear", "interleaved", "random", "from-data"}
_MOTION_SIM_TYPES = {
    "as-it-is",
    "rigid",
    "discrete-rigid",
    "non-rigid",
    "discrete-non-rigid",
}
_SIM_TYPE_TO_MODEL = {
    "rigid": ("rigid", "realistic"),
    "discrete-rigid": ("rigid", "per-shot"),
    "non-rigid": ("non-rigid", "realistic"),
    "discrete-non-rigid": ("non-rigid", "per-shot"),
    "as-it-is": (None, None),
}
_PER_SHOT_SIM_TYPES = {"discrete-rigid", "discrete-non-rigid"}

_PATH_KEYS = {
    "debug_folder",
    "logs_folder",
    "results_folder",
    "initial_data_folder",
}
_RUNTIME_KEYS = {
    "debug_flag",
    "runtime_device",
    "verbose",
    "print_to_console",
    "clean_output_folders_before_run",
    "jupyter_notebook_flag",
    "flip_for_display",
    "seed",
}
_DATA_KEYS = {
    "data_type",
    "data_dimension",
    "reconstruction_dimension",
    "motion_simulation_config_dimension",
}
_DATA_SOURCE_KEYS = {
    "FoVxy_mm",
    "FoVz_mm",
    "N_SheppLogan",
    "Ncoils_SheppLogan",
    "Ncoils_input",
    "Nz_SheppLogan",
    "SheppLoganFillFraction",
    "from_image",
    "image_resize_factor",
    "rawdata_sensor_type",
}
_SAMPLING_KEYS = {
    "kspace_sampling_type",
    "NshotsPerNex",
    "Nex",
    "Nshots",
}
_MOTION_KEYS = {
    "reconstruction_motion_type",
    "simulated_motion_type",
    "motion_state_mode",
    "motion_simulation_type",
} | _RIGID_MOTION_KEYS | _NONRIGID_MOTION_KEYS

_STRUCTURAL_OVERRIDE_KEYS = {"data_type"}


def _drop_keys(cfg, keys):
    for key in keys:
        cfg.pop(key, None)


def _normalize_data_dimension(dim):
    if dim is None:
        return None
    d = str(dim).strip().upper()
    if d in {"2D", "2"}:
        return "2D"
    if d in {"3D", "3"}:
        return "3D"
    raise ValueError("data_dimension must be '2D' or '3D'.")


def _normalize_motion_type(raw_motion_type):
    key = str(raw_motion_type).strip().lower()
    if key not in _MOTION_TYPES:
        raise ValueError(f"Unsupported motion_type: {raw_motion_type}")
    return key


def _normalize_motion_state_mode(raw_mode):
    key = str(raw_mode).strip().lower()
    if key not in _MOTION_STATE_MODES:
        raise ValueError(
            f"Unsupported motion_state_mode: {raw_mode}. Supported: 'realistic', 'per-shot'."
        )
    return key


def _normalize_motion_simulation_type(raw_type):
    if raw_type is None:
        return None
    key = str(raw_type).strip().lower()
    if key not in _MOTION_SIM_TYPES:
        raise ValueError(f"Unsupported motion_simulation_type: {raw_type}")
    return key


def _simulation_type_from_motion_model(motion_type, motion_state_mode):
    mtype = _normalize_motion_type(motion_type)
    mode = _normalize_motion_state_mode(motion_state_mode)
    if mtype == "rigid":
        return "rigid" if mode == "realistic" else "discrete-rigid"
    if mtype == "non-rigid":
        return "non-rigid" if mode == "realistic" else "discrete-non-rigid"
    raise ValueError(f"Unsupported motion_type: {motion_type}")


@dataclass
class PathsConfig:
    debug_folder: str | None = None
    logs_folder: str | None = None
    results_folder: str | None = None
    initial_data_folder: str | None = None

    def to_flat_dict(self):
        out = {}
        for key in _PATH_KEYS:
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


@dataclass
class RuntimeConfig:
    debug_flag: bool | None = None
    runtime_device: str | None = None
    verbose: bool | None = None
    print_to_console: bool | None = None
    clean_output_folders_before_run: bool | None = None
    jupyter_notebook_flag: bool | None = None
    flip_for_display: bool | None = None
    seed: int | None = None

    def to_flat_dict(self):
        out = {}
        for key in _RUNTIME_KEYS:
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


@dataclass
class DataConfig:
    data_type: str | None = None
    data_dimension: str | None = None
    reconstruction_dimension: str | None = None
    motion_simulation_config_dimension: str | None = None
    source_options: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self):
        out = {}
        for key in _DATA_KEYS:
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        out.update(self.source_options)
        return out


@dataclass
class SamplingConfig:
    kspace_sampling_type: str | None = None
    NshotsPerNex: int | None = None
    Nex: int | None = None
    Nshots: int | None = None

    def to_flat_dict(self):
        out = {}
        for key in _SAMPLING_KEYS:
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


@dataclass
class MotionConfig:
    reconstruction_motion_type: str | None = None
    simulated_motion_type: str | None = None
    motion_state_mode: str | None = None
    motion_simulation_type: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self):
        out = {
            "reconstruction_motion_type": self.reconstruction_motion_type,
            "simulated_motion_type": self.simulated_motion_type,
            "motion_state_mode": self.motion_state_mode,
            "motion_simulation_type": self.motion_simulation_type,
        }
        out.update(self.parameters)
        return out


@dataclass
class ReconstructionConfig:
    N_motion_states: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self):
        out = dict(self.options)
        if self.N_motion_states is not None:
            out["N_motion_states"] = self.N_motion_states
        return out


@dataclass
class ConfigBundle:
    paths: PathsConfig
    runtime: RuntimeConfig
    data: DataConfig
    sampling: SamplingConfig
    motion: MotionConfig
    reconstruction: ReconstructionConfig

    @classmethod
    def from_flat_dict(cls, flat_cfg):
        remaining = dict(flat_cfg)

        paths = PathsConfig(**{key: remaining.pop(key, None) for key in _PATH_KEYS})
        runtime = RuntimeConfig(**{key: remaining.pop(key, None) for key in _RUNTIME_KEYS})

        data_kwargs = {key: remaining.pop(key, None) for key in _DATA_KEYS}
        data_source_options = {
            key: remaining.pop(key)
            for key in list(remaining.keys())
            if key in _DATA_SOURCE_KEYS
        }
        data = DataConfig(**data_kwargs, source_options=data_source_options)

        sampling = SamplingConfig(**{key: remaining.pop(key, None) for key in _SAMPLING_KEYS})

        motion_kwargs = {
            key: remaining.pop(key, None)
            for key in (
                "reconstruction_motion_type",
                "simulated_motion_type",
                "motion_state_mode",
                "motion_simulation_type",
            )
        }
        motion_parameters = {
            key: remaining.pop(key)
            for key in list(remaining.keys())
            if key in (_RIGID_MOTION_KEYS | _NONRIGID_MOTION_KEYS)
        }
        motion = MotionConfig(**motion_kwargs, parameters=motion_parameters)

        n_motion_states = remaining.pop("N_motion_states", None)
        reconstruction = ReconstructionConfig(
            N_motion_states=n_motion_states,
            options=remaining,
        )

        return cls(
            paths=paths,
            runtime=runtime,
            data=data,
            sampling=sampling,
            motion=motion,
            reconstruction=reconstruction,
        )

    def to_flat_dict(self):
        flat = {}
        flat.update(self.paths.to_flat_dict())
        flat.update(self.runtime.to_flat_dict())
        flat.update(self.data.to_flat_dict())
        flat.update(self.sampling.to_flat_dict())
        flat.update(self.motion.to_flat_dict())
        flat.update(self.reconstruction.to_flat_dict())
        return flat


def _load_base_config_dict(
    *,
    data_type,
    reconstruction_config,
    shepp_logan_config=None,
    from_image_config=None,
    sampling_config=None,
    motion_simulation_config=None,
):
    repo_root = Path(__file__).resolve().parents[2]
    general_path = repo_root / "config" / "general.toml"
    if not general_path.exists():
        raise FileNotFoundError(f"Missing general config: {general_path}")
    if not reconstruction_config:
        raise ValueError("reconstruction_config is required.")

    cfg = dict(_CODE_DEFAULTS)
    cfg.update(_load_toml_flat(general_path))
    cfg["data_type"] = data_type

    reconstruction_cfg = _load_toml_flat(reconstruction_config)
    _rename_cfg_key(reconstruction_cfg, "motion_type", "reconstruction_motion_type")
    cfg.update(reconstruction_cfg)

    if data_type == "shepp-logan":
        if not shepp_logan_config:
            raise ValueError("shepp_logan_config is required when data_type='shepp-logan'.")
        cfg.update(_load_toml_flat(shepp_logan_config))
    elif data_type == "from_image":
        if from_image_config is None:
            raise ValueError(
                "from_image_config is required when data_type is 'from_image'."
            )
        cfg.update(_load_toml_flat(from_image_config))
    elif data_type in {"real-world", "raw-data"}:
        pass
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    if sampling_config:
        cfg.update(_load_toml_flat(sampling_config))
    if motion_simulation_config:
        motion_cfg = _load_toml_flat(motion_simulation_config)
        _rename_cfg_key(motion_cfg, "motion_type", "simulated_motion_type")
        cfg.update(motion_cfg)

    return cfg


def _apply_direct_arguments(
    cfg,
    *,
    reconstruction_motion_type=None,
    simulated_motion_type=None,
    motion_simulation_type=None,
    motion_state_mode=None,
    data_dimension=None,
    kspace_sampling_type=None,
    NshotsPerNex=None,
    Nex=None,
    N_motion_states=None,
    flip_for_display=None,
):
    if reconstruction_motion_type is not None:
        cfg["reconstruction_motion_type"] = reconstruction_motion_type
    if simulated_motion_type is not None:
        cfg["simulated_motion_type"] = simulated_motion_type
    if motion_simulation_type is not None:
        cfg["motion_simulation_type"] = motion_simulation_type
    if motion_state_mode is not None:
        cfg["motion_state_mode"] = motion_state_mode
    if data_dimension is not None:
        cfg["data_dimension"] = data_dimension
    if kspace_sampling_type is not None:
        cfg["kspace_sampling_type"] = kspace_sampling_type
    if NshotsPerNex is not None:
        cfg["NshotsPerNex"] = int(NshotsPerNex)
    if Nex is not None:
        cfg["Nex"] = int(Nex)
    if N_motion_states is not None:
        cfg["N_motion_states"] = int(N_motion_states)
    if flip_for_display is not None:
        cfg["flip_for_display"] = bool(flip_for_display)


def _resolve_sampling_origin(cfg, data_type):
    if "kspace_sampling_type" in cfg:
        return False

    _drop_keys(cfg, {"kspace_sampling_type", "NshotsPerNex", "Nex", "Nshots"})
    if data_type not in {"real-world", "raw-data"}:
        raise ValueError(
            "Sampling configuration is required when data_type is not 'real-world'/'raw-data'. "
            "Provide sampling_config or kspace_sampling_type (+ Nex/NshotsPerNex)."
        )
    return True


def _require_motion_input_for_simulated_sources(cfg, *, motion_simulation_config):
    if (
        cfg.get("data_type") not in {"real-world", "raw-data"}
        and motion_simulation_config is None
        and cfg.get("motion_simulation_type") is None
        and cfg.get("motion_state_mode") is None
    ):
        raise ValueError(
            "motion_simulation_config (or motion_simulation_type/motion_state_mode override) is required "
            f"for data_type='{cfg.get('data_type')}'."
        )


def _resolve_motion_simulation(cfg, *, sampling_from_data, motion_simulation_config):
    motion_simulation_type_final = cfg.get("motion_simulation_type")
    if motion_simulation_type_final is None and sampling_from_data:
        motion_simulation_type_final = "as-it-is"
    elif motion_simulation_type_final is None:
        motion_state_mode_final = cfg.get("motion_state_mode", "realistic")
        simulated_motion_type_final = cfg.get(
            "simulated_motion_type",
            cfg["reconstruction_motion_type"],
        )
        motion_simulation_type_final = _simulation_type_from_motion_model(
            simulated_motion_type_final,
            motion_state_mode_final,
        )
        cfg["simulated_motion_type"] = _normalize_motion_type(simulated_motion_type_final)
        cfg["motion_state_mode"] = _normalize_motion_state_mode(motion_state_mode_final)

    cfg["motion_simulation_type"] = _normalize_motion_simulation_type(motion_simulation_type_final)
    if (
        cfg["motion_simulation_type"] == "as-it-is"
        and cfg.get("data_type") not in {"real-world", "raw-data"}
    ):
        raise ValueError(
            "motion_simulation_type='as-it-is' is only valid for real-world or raw-data inputs."
        )


def _prune_irrelevant_motion_parameters(cfg):
    sim_type = cfg["motion_simulation_type"]
    if sim_type == "as-it-is":
        _drop_keys(cfg, _RIGID_MOTION_KEYS | _NONRIGID_MOTION_KEYS)
    elif sim_type in {"rigid", "discrete-rigid"}:
        _drop_keys(cfg, _NONRIGID_MOTION_KEYS)
    elif sim_type in {"non-rigid", "discrete-non-rigid"}:
        _drop_keys(cfg, _RIGID_MOTION_KEYS)
    else:
        raise ValueError(f"Unsupported motion_simulation_type: {sim_type}")


def _apply_user_overrides(cfg, overrides):
    for key, value in (overrides or {}).items():
        if key in _STRUCTURAL_OVERRIDE_KEYS:
            raise ValueError(
                f"'{key}' cannot be set via overrides because it determines which source configs are loaded. "
                f"Pass {key}=... directly to load_config(...) instead."
            )
        cfg[key] = value


def _apply_notebook_output_defaults(cfg, overrides):
    notebook_flag = bool(cfg.get("jupyter_notebook_flag", False))
    if not overrides or "print_to_console" not in overrides:
        cfg["print_to_console"] = not notebook_flag
    if not overrides or "verbose" not in overrides:
        cfg["verbose"] = not notebook_flag


def _apply_display_defaults(cfg, data_type):
    if "flip_for_display" not in cfg:
        cfg["flip_for_display"] = data_type in {"real-world", "raw-data"}


def _normalize_runtime_config(runtime, data_type):
    if runtime.flip_for_display is None:
        runtime.flip_for_display = data_type in {"real-world", "raw-data"}
    if runtime.clean_output_folders_before_run is None:
        runtime.clean_output_folders_before_run = True
    if runtime.jupyter_notebook_flag is None:
        runtime.jupyter_notebook_flag = False
    if runtime.runtime_device is None:
        warnings.warn(
            "runtime_device not specified; defaulting to 'cpu'.",
            RuntimeWarning,
        )
        runtime.runtime_device = "cpu"
    runtime.runtime_device = str(runtime.runtime_device).lower()
    if runtime.runtime_device not in {"cpu", "gpu"}:
        raise ValueError("runtime_device must be 'cpu' or 'gpu'.")
    if runtime.print_to_console is None:
        runtime.print_to_console = not bool(runtime.jupyter_notebook_flag)
    if runtime.verbose is None:
        runtime.verbose = not bool(runtime.jupyter_notebook_flag)


def _normalize_sampling_config(sampling, data_type):
    if sampling.kspace_sampling_type is None:
        if data_type in {"real-world", "raw-data"}:
            sampling.kspace_sampling_type = "from-data"
        else:
            sampling.kspace_sampling_type = "linear"
    else:
        sampling.kspace_sampling_type = str(sampling.kspace_sampling_type).strip().lower()

    if sampling.kspace_sampling_type not in _SAMPLING_TYPES:
        raise ValueError(
            "kspace_sampling_type must be one of "
            f"{sorted(_SAMPLING_TYPES)}."
        )

    has_sampling_sim = sampling.NshotsPerNex is not None and sampling.Nex is not None
    if has_sampling_sim:
        sampling.NshotsPerNex = _normalize_positive_int(sampling.NshotsPerNex, "NshotsPerNex")
        sampling.Nex = _normalize_positive_int(sampling.Nex, "Nex")
        sampling.Nshots = int(sampling.NshotsPerNex) * int(sampling.Nex)
    elif sampling.Nshots is None:
        sampling.Nshots = 1


def _normalize_data_config(data):
    recon_dim = _normalize_data_dimension(data.reconstruction_dimension)

    for coil_key in ("Ncoils_SheppLogan", "Ncoils_input"):
        if coil_key in data.source_options:
            data.source_options[coil_key] = _normalize_synthetic_coil_count(
                data.source_options[coil_key],
                coil_key,
            )

    inferred_dim = None
    if "Nz_SheppLogan" in data.source_options:
        nz = int(data.source_options["Nz_SheppLogan"])
        inferred_dim = "3D" if nz > 1 else "2D"

    if data.data_dimension is None:
        if inferred_dim is not None:
            data.data_dimension = inferred_dim
        elif recon_dim is not None:
            data.data_dimension = recon_dim
        else:
            data.data_dimension = "2D"
    else:
        data.data_dimension = _normalize_data_dimension(data.data_dimension)

    if data.data_type == "from_image" and data.data_dimension == "3D":
        raise ValueError(
            f"data_type='{data.data_type}' is only supported for 2D inputs; "
            "3D is not compatible with this source type."
        )

    motion_cfg_dim = _normalize_data_dimension(data.motion_simulation_config_dimension)
    if recon_dim is not None and recon_dim != data.data_dimension:
        raise ValueError(
            f"Reconstruction config is tagged {recon_dim}, but data_dimension is {data.data_dimension}."
        )
    if motion_cfg_dim is not None and motion_cfg_dim != data.data_dimension:
        raise ValueError(
            f"Motion simulation config is tagged {motion_cfg_dim}, but data_dimension is {data.data_dimension}."
        )

    if "Nz_SheppLogan" in data.source_options:
        nz_dim = "3D" if int(data.source_options["Nz_SheppLogan"]) > 1 else "2D"
        if nz_dim != data.data_dimension:
            raise ValueError(
                f"Shepp-Logan Nz_SheppLogan={int(data.source_options['Nz_SheppLogan'])} implies {nz_dim}, "
                f"but data_dimension is {data.data_dimension}."
            )


def _normalize_motion_config(motion):
    if motion.reconstruction_motion_type is None:
        raise ValueError("reconstruction_motion_type must be provided.")
    motion.reconstruction_motion_type = _normalize_motion_type(motion.reconstruction_motion_type)

    if motion.simulated_motion_type is not None:
        motion.simulated_motion_type = _normalize_motion_type(motion.simulated_motion_type)

    if motion.motion_state_mode is not None:
        motion.motion_state_mode = _normalize_motion_state_mode(motion.motion_state_mode)

    if motion.motion_simulation_type is None:
        raise ValueError("motion_simulation_type must be resolved before motion config normalization.")
    motion.motion_simulation_type = _normalize_motion_simulation_type(motion.motion_simulation_type)

    inferred_motion_type, inferred_mode = _SIM_TYPE_TO_MODEL[motion.motion_simulation_type]
    if inferred_motion_type is not None:
        if (
            motion.simulated_motion_type is not None
            and motion.simulated_motion_type != inferred_motion_type
        ):
            raise ValueError(
                f"motion_simulation_type '{motion.motion_simulation_type}' is incompatible with "
                f"simulated_motion_type '{motion.simulated_motion_type}'."
            )
        motion.simulated_motion_type = inferred_motion_type

    if inferred_mode is not None:
        if motion.motion_state_mode is not None and motion.motion_state_mode != inferred_mode:
            raise ValueError(
                f"motion_state_mode='{motion.motion_state_mode}' conflicts with "
                f"motion_simulation_type='{motion.motion_simulation_type}'."
            )
        motion.motion_state_mode = inferred_mode
    else:
        motion.simulated_motion_type = None
        if motion.motion_simulation_type == "as-it-is" and motion.motion_state_mode is not None:
            raise ValueError(
                "motion_state_mode must not be set when motion_simulation_type='as-it-is'."
            )
        motion.motion_state_mode = None

    if motion.motion_simulation_type in {"rigid", "discrete-rigid"}:
        rigid_amp_scale = float(motion.parameters.get("rigid_motion_amplitude_scale", 1.0))
        if rigid_amp_scale < 0:
            raise ValueError("rigid_motion_amplitude_scale must be >= 0.")
        motion.parameters["rigid_motion_amplitude_scale"] = rigid_amp_scale

    if motion.motion_simulation_type == "non-rigid":
        if "nonrigid_resp_cycles_min" not in motion.parameters:
            raise ValueError(
                "nonrigid_resp_cycles_min must be specified for realistic non-rigid motion simulation."
            )
        if "nonrigid_resp_cycles_max" not in motion.parameters:
            raise ValueError(
                "nonrigid_resp_cycles_max must be specified for realistic non-rigid motion simulation."
            )
        cycles_min = float(motion.parameters["nonrigid_resp_cycles_min"])
        cycles_max = float(motion.parameters["nonrigid_resp_cycles_max"])
        if cycles_min <= 0 or cycles_max <= 0:
            raise ValueError("nonrigid_resp_cycles_min/max must be > 0.")
        if cycles_min > cycles_max:
            cycles_min, cycles_max = cycles_max, cycles_min
        motion.parameters["nonrigid_resp_cycles_min"] = cycles_min
        motion.parameters["nonrigid_resp_cycles_max"] = cycles_max


def _normalize_reconstruction_config(reconstruction, motion, sampling):
    if reconstruction.N_motion_states is None:
        raise ValueError("N_motion_states must be provided in the reconstruction config or via override.")

    manual_states = _normalize_positive_int(reconstruction.N_motion_states, "N_motion_states")
    if motion.motion_simulation_type in _PER_SHOT_SIM_TYPES:
        reconstruction.N_motion_states = int(sampling.Nshots)
    else:
        reconstruction.N_motion_states = manual_states


def _ensure_output_folders(paths):
    for folder in (
        paths.debug_folder,
        paths.logs_folder,
        paths.results_folder,
        paths.initial_data_folder,
    ):
        os.makedirs(folder, exist_ok=True)


def _resolve_configs(configs):
    torch.set_default_dtype(torch.float64)

    _normalize_runtime_config(configs.runtime, configs.data.data_type)
    _normalize_sampling_config(configs.sampling, configs.data.data_type)
    _normalize_data_config(configs.data)
    _normalize_motion_config(configs.motion)
    _normalize_reconstruction_config(configs.reconstruction, configs.motion, configs.sampling)
    _ensure_output_folders(configs.paths)
    return configs


def load_config(
    *,
    data_type,
    reconstruction_motion_type=None,
    simulated_motion_type=None,
    reconstruction_config,
    shepp_logan_config=None,
    from_image_config=None,
    sampling_config=None,
    motion_simulation_config=None,
    motion_simulation_type=None,
    motion_state_mode=None,
    data_dimension=None,
    kspace_sampling_type=None,
    NshotsPerNex=None,
    Nex=None,
    N_motion_states=None,
    flip_for_display=None,
    overrides=None,
):
    cfg = _load_base_config_dict(
        data_type=data_type,
        reconstruction_config=reconstruction_config,
        shepp_logan_config=shepp_logan_config,
        from_image_config=from_image_config,
        sampling_config=sampling_config,
        motion_simulation_config=motion_simulation_config,
    )

    _apply_direct_arguments(
        cfg,
        reconstruction_motion_type=reconstruction_motion_type,
        simulated_motion_type=simulated_motion_type,
        motion_simulation_type=motion_simulation_type,
        motion_state_mode=motion_state_mode,
        data_dimension=data_dimension,
        kspace_sampling_type=kspace_sampling_type,
        NshotsPerNex=NshotsPerNex,
        Nex=Nex,
        N_motion_states=N_motion_states,
        flip_for_display=flip_for_display,
    )

    if "reconstruction_motion_type" not in cfg:
        raise ValueError(
            "reconstruction_motion_type must be provided either in reconstruction config "
            "or as a load_config argument."
        )

    _apply_user_overrides(cfg, overrides)

    sampling_from_data = _resolve_sampling_origin(cfg, cfg["data_type"])
    _require_motion_input_for_simulated_sources(
        cfg,
        motion_simulation_config=motion_simulation_config,
    )
    _resolve_motion_simulation(
        cfg,
        sampling_from_data=sampling_from_data,
        motion_simulation_config=motion_simulation_config,
    )

    if "kspace_sampling_type" in cfg and ("NshotsPerNex" not in cfg or "Nex" not in cfg):
        raise ValueError(
            "NshotsPerNex and Nex are required when kspace_sampling_type is specified."
        )

    _prune_irrelevant_motion_parameters(cfg)
    _apply_notebook_output_defaults(cfg, overrides)
    _apply_display_defaults(cfg, cfg["data_type"])

    configs = ConfigBundle.from_flat_dict(cfg)
    configs = _resolve_configs(configs)
    return SimpleNamespace(**configs.to_flat_dict())

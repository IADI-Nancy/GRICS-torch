import tomllib


def _flatten(dct, prefix=""):
    out = {}
    for key, value in dct.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, full_key))
        else:
            out[full_key] = value
    return out


def apply_toml_overrides(params, toml_path):
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    flat = _flatten(data)
    for key, value in flat.items():
        attr = key.split(".")[-1]
        if not hasattr(params, attr):
            raise ValueError(f"Unknown parameter '{attr}' in {toml_path}")
        setattr(params, attr, value)

    if hasattr(params, "refresh_derived"):
        params.refresh_derived()


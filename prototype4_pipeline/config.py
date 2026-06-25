from pathlib import Path

import yaml


def deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    inherited = config.pop("inherits", None)
    if inherited:
        parent = load_config(path.parent / inherited)
        return deep_update(parent, config)

    config.setdefault("project", {})
    config.setdefault("paths", {})
    config.setdefault("models", {})
    return config

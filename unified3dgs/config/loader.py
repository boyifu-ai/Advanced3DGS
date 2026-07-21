from __future__ import annotations

from pathlib import Path
from typing import Dict


def parse_scalar(raw: str) -> object:
    value = raw.strip()
    if value == "":
        return None
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_flat_yaml_fallback(path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-"):
            raise ValueError(
                f"Fallback YAML parser only supports flat key/value files; "
                f"found list item at {path}:{line_number}"
            )
        if ":" not in stripped:
            raise ValueError(f"Invalid config line at {path}:{line_number}: {line}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty config key at {path}:{line_number}")
        data[key] = parse_scalar(raw_value)
    return data


def load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    try:
        import yaml
    except ImportError:
        return _load_flat_yaml_fallback(path)

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return dict(loaded)

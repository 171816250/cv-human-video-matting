"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data


def config_value(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read a nested config value using a dotted path."""
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


_TRUE_STRINGS = {"true", "1", "yes", "y", "on"}
_FALSE_STRINGS = {"false", "0", "no", "n", "off", ""}


def config_int(config: dict[str, Any], path: str, default: int) -> int:
    """Read a nested config value coerced to int, tolerating non-numeric input."""
    value = config_value(config, path, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def config_float(config: dict[str, Any], path: str, default: float) -> float:
    """Read a nested config value coerced to float, tolerating non-numeric input."""
    value = config_value(config, path, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def config_bool(config: dict[str, Any], path: str, default: bool) -> bool:
    """Read a nested config value coerced to bool.

    Handles real YAML booleans as well as quoted strings like "false"/"no",
    which ``bool(str)`` would otherwise treat as truthy.
    """
    value = config_value(config, path, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return default

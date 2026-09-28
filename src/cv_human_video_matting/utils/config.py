"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import re

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    validate_project_config(data)
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


RENDER_MODES = {"alpha", "green", "foreground", "blur", "transparent", "image", "side_by_side"}
DEPTH_MODELS = {"MiDaS_small", "DPT_Large", "DPT_Hybrid"}


def parse_background_color(value: str) -> tuple[int, int, int]:
    """Parse a CSS-style #RRGGBB color and return OpenCV BGR channel order."""
    if not isinstance(value, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None:
        raise ValueError("render.background_color must use #RRGGBB format, e.g. '#00FF00'.")
    red, green, blue = (int(value[index:index + 2], 16) for index in (1, 3, 5))
    return blue, green, red


def validate_project_config(config: dict[str, Any]) -> None:
    """Reject invalid explicit project values before loading models or writing output.

    Missing values retain application defaults. The legacy coercion helpers below
    remain available for callers, but a configuration file must have correct YAML
    types so a typo cannot silently switch an experiment to a different setting.
    """
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping.")
    for section in ("project", "device", "input", "matting", "postprocess", "depth", "render", "paths"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping.")
    depth = config.get("depth", {})
    if "thresholds" in depth and not isinstance(depth["thresholds"], dict):
        raise ValueError("depth.thresholds must be a mapping.")

    def integer(path: str, minimum: int, nullable: bool = False) -> None:
        value = config_value(config, path)
        if value is None and nullable:
            return
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            suffix = " or null" if nullable else ""
            raise ValueError(f"{path} must be an integer >= {minimum}{suffix}.")

    def number(path: str, default: float, minimum: float, maximum: float) -> float:
        value = config_value(config, path, default)
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum
        ):
            raise ValueError(f"{path} must be a finite number in [{minimum}, {maximum}].")
        return float(value)

    def choice(path: str, choices: set[str]) -> None:
        value = config_value(config, path)
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"{path} must be one of: {', '.join(sorted(choices))}.")

    integer("input.resize_short", 2, nullable=True)
    for name in ("edge_feather", "mask_open", "mask_close"):
        if name in config.get("postprocess", {}):
            integer(f"postprocess.{name}", 0)
    if "interval" in depth:
        integer("depth.interval", 1)
    number("postprocess.temporal_alpha", 0.75, 0.0, 1.0)
    near = number("depth.thresholds.near", 0.66, 0.0, 1.0)
    mid = number("depth.thresholds.mid", 0.33, 0.0, 1.0)
    if not mid < near:
        raise ValueError("depth.thresholds must satisfy 0 <= mid < near <= 1.")

    for path in ("depth.enabled", "postprocess.enable_temporal_smooth", "render.preserve_audio"):
        value = config_value(config, path, False)
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a YAML boolean (true or false).")
    for path, choices in (
        ("depth.model", DEPTH_MODELS),
        ("render.mode", RENDER_MODES),
        ("matting.model", {"dummy", "rvm", "rvm_mobilenetv3", "rvm_resnet50", "birefnet",
                           "mobilenetv3", "resnet50", "ultra"}),
    ):
        section, key = path.split(".")
        if key in config.get(section, {}):
            choice(path, choices)

    device = config_value(config, "device.mode", "auto")
    if not isinstance(device, str) or re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", device) is None:
        raise ValueError("device.mode must be auto, cpu, cuda, or cuda:<index>.")

    ratio = config_value(config, "matting.downsample_ratio", "auto")
    if ratio != "auto":
        if number("matting.downsample_ratio", 1.0, 0.0, 1.0) == 0:
            raise ValueError("matting.downsample_ratio must be 'auto' or a number in (0, 1].")
    parse_background_color(config_value(config, "render.background_color", "#00FF00"))


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

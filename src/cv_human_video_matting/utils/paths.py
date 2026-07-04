"""Path helpers used across the project."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(path: str | Path) -> Path:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    return file_path.parent

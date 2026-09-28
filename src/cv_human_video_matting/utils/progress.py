"""Shared cooperative cancellation and frame-progress contracts."""

from __future__ import annotations

from typing import Callable, Optional


ProgressCallback = Callable[[int, Optional[int]], None]
CancelCallback = Callable[[], bool]


class ProcessingCancelled(RuntimeError):
    """A processing job was stopped before its result was published."""


def check_cancelled(should_cancel: CancelCallback | None) -> None:
    """Raise at safe interruption points without storing global job state."""
    if should_cancel is not None and should_cancel():
        raise ProcessingCancelled("Processing was cancelled.")

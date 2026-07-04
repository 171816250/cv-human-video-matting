"""Shared interfaces for matting runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MattingResult:
    alpha: np.ndarray
    foreground: np.ndarray | None = None


class MattingRunner(Protocol):
    def predict(self, frame_bgr: np.ndarray) -> MattingResult:
        """Return alpha matte and optional foreground for one BGR frame."""

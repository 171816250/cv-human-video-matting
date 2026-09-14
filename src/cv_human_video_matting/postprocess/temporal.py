"""Stateful temporal smoothing for alpha mattes."""

from __future__ import annotations

import numpy as np

from cv_human_video_matting.postprocess.alpha_refine import _as_uint8_alpha


class TemporalAlphaSmoother:
    """Blend each alpha matte with the previous smoothed matte."""

    def __init__(self, blend: float = 0.75) -> None:
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"temporal blend must be in [0, 1], got {blend}")
        self.blend = float(blend)
        self._previous: np.ndarray | None = None

    def apply(self, alpha: np.ndarray) -> np.ndarray:
        alpha_u8 = _as_uint8_alpha(alpha)
        current = alpha_u8.astype(np.float32)
        if self._previous is None or self._previous.shape != current.shape:
            self._previous = current
            return alpha_u8

        smoothed = self._previous * self.blend + current * (1.0 - self.blend)
        self._previous = smoothed
        return np.clip(smoothed, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        self._previous = None

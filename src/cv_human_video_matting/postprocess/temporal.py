"""Stateful temporal smoothing for alpha mattes."""

from __future__ import annotations

import numpy as np

from cv_human_video_matting.postprocess.alpha_refine import _as_uint8_alpha


class TemporalAlphaSmoother:
    """Smooth alpha with fixed EMA or change-adaptive per-pixel history weights.

    ``ema`` preserves the original filter. In ``adaptive`` mode, the absolute
    difference between consecutive *unsmoothed* alpha inputs reduces the history
    weight with a fourth-power ramp. Changes at or above ``motion_threshold`` (in normalized
    alpha units) use the current alpha directly. Small changes keep nearly the
    same noise reduction as EMA. This is an alpha-change heuristic, not optical
    flow: large segmentation errors also reduce smoothing and can still flicker.
    """

    def __init__(
        self,
        blend: float = 0.75,
        mode: str = "ema",
        motion_threshold: float = 0.15,
    ) -> None:
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"temporal blend must be in [0, 1], got {blend}")
        if mode not in {"ema", "adaptive"}:
            raise ValueError("temporal mode must be 'ema' or 'adaptive'.")
        if not 0.0 < motion_threshold <= 1.0:
            raise ValueError("temporal motion_threshold must be in (0, 1].")
        self.blend = float(blend)
        self.mode = mode
        self.motion_threshold = float(motion_threshold)
        self._previous: np.ndarray | None = None
        self._previous_raw: np.ndarray | None = None

    def apply(self, alpha: np.ndarray) -> np.ndarray:
        if np.issubdtype(alpha.dtype, np.floating) and not np.isfinite(alpha).all():
            raise ValueError("Alpha must contain only finite values.")
        alpha_u8 = _as_uint8_alpha(alpha)
        current = alpha_u8.astype(np.float32)
        if self._previous is None or self._previous.shape != current.shape:
            self._previous = current
            self._previous_raw = current if self.mode == "adaptive" else None
            return alpha_u8

        if self.mode == "adaptive":
            # Inputs have already been quantized to uint8. A threshold below
            # one alpha level is equivalent to one level, and the lower bound
            # avoids division underflow for very small positive thresholds.
            threshold_levels = max(1.0, 255.0 * self.motion_threshold)
            change = np.abs(current - self._previous_raw) / threshold_levels
            history_weight = self.blend * (1.0 - np.minimum(change, 1.0) ** 4)
            smoothed = self._previous * history_weight + current * (1.0 - history_weight)
            self._previous_raw = current
        else:
            smoothed = self._previous * self.blend + current * (1.0 - self.blend)
        self._previous = smoothed
        return np.clip(smoothed, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        self._previous = None
        self._previous_raw = None
